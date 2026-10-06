"""Part 8 D3: ablations (design 19), zero-shot TG->DL and DL->TG, HGB with my part 1-5 settings (HGB_PARAMS), seeds
0-2, 3-seed average, clipped to [0, 1]. Guarded token tables (validate_ha2_guard_tokens_{DL,TG}.parquet).
  TDonly  the 18 TD tokens only (A2).
  A3      77 tokens, but the 6 x 3 apparent-impedance tokens zr / zi / absz in OHM (V/I of the loop, last-cycle
          phasors, unclipped; 0 when |I| <= 1e-9, as in tokens.py) instead of per Z1; the other 59 unchanged.
  A1raw   HGB on the raw 480 x 6 local window (2,880 sample values, my cached windows).
  A1Z     A1raw + (R1, X1, R0, X0) of the faulted line.
Outputs: results/validate_ha2_part8_abl_summary.csv, validate_ha2_part8_abl_pred_<cell>_<dir>.csv,
validate_ha2_part8_A3_ohm_{DL,TG}.parquet.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from physics_hgb import HGB_PARAMS, assert_allow_list  # noqa: E402
from tokens import LOCAL_TOKENS, LOOPS, PH, TD_TOKENS, phasor  # noqa: E402

R = C.RESULTS
NW = int(os.environ.get("VAL_WORKERS", "6"))
A = C.A_OP
ZTOK = [f"{t}_{lp}" for lp in LOOPS for t in ("zr", "zi", "absz")]
ALL77 = list(LOCAL_TOKENS) + list(TD_TOKENS)


def ohm_work(task: tuple) -> list[dict]:
    grid, idx, line, s1s = task
    sig = np.load(C.CACHE / grid / f"ep{idx}.npy")
    Z1, Z0 = C.line_z(grid, line)
    k0 = (Z0 - Z1) / (3.0 * Z1)
    rows = []
    for s1 in s1s:
        L = phasor(sig, s1 - C.N_CYC)
        LI, LV = L[0:3], L[3:6]
        I0 = LI.sum() / 3.0
        row = {"sim_idx": idx, "s1": s1}
        for lp in LOOPS:
            if lp.endswith("g"):
                p = PH[lp[0]]
                V, I = LV[p], LI[p] + k0 * 3.0 * I0
            else:
                p, q = PH[lp[0]], PH[lp[1]]
                V, I = LV[p] - LV[q], LI[p] - LI[q]
            zz = V / I if abs(I) > 1e-9 else 0j
            row[f"ohm_zr_{lp}"], row[f"ohm_zi_{lp}"], row[f"ohm_absz_{lp}"] = zz.real, zz.imag, abs(zz)
        rows.append(row)
    return rows


def a3_tables(T: dict) -> dict:
    out = {}
    for g, d in T.items():
        f = R / f"validate_ha2_part8_A3_ohm_{g}.parquet"
        if not f.exists():
            tasks = [(g, int(i), gg["line"].iloc[0], gg["s1"].tolist()) for i, gg in d.groupby("sim_idx")]
            with ProcessPoolExecutor(max_workers=NW) as ex:
                res = list(ex.map(ohm_work, tasks, chunksize=8))
            pd.DataFrame([x for rr in res for x in rr]).to_parquet(f, index=False)
        o = pd.read_parquet(f)
        m = d.merge(o, on=["sim_idx", "s1"], how="left", validate="1:1")
        assert len(m) == len(d) and m["ohm_zr_ag"].notna().all()
        # sanity: per-unit token = clip(ohm / Z1) where not clipped
        z1 = (m["R1"] + 1j * m["X1"]).to_numpy()
        zz = (m["ohm_zr_ag"] + 1j * m["ohm_zi_ag"]).to_numpy() / z1
        ok = np.abs(zz.real) < 4.9
        print(g, "A3 sanity zr_ag max diff:", float(np.abs(zz.real[ok] - m["zr_ag"].to_numpy()[ok]).max()), flush=True)
        for c in ZTOK:
            m[c] = m[f"ohm_{c}"]
        out[g] = m
    return out


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, tag: str) -> tuple[np.ndarray, list[np.ndarray]]:
    P = []
    for seed in (0, 1, 2):
        t0 = time.time()
        mdl = HistGradientBoostingRegressor(random_state=seed, **HGB_PARAMS).fit(Xtr, ytr)
        P.append(mdl.predict(Xte))
        print(tag, "seed", seed, f"{time.time() - t0:.0f}s", flush=True)
    return np.clip(np.mean(P, axis=0), 0, 1), P


def summary_row(cell: str, direction: str, te: pd.DataFrame, p: np.ndarray, P: list) -> dict:
    y = te["y"].to_numpy()
    e = np.abs(p - y) * 100
    tau = te["tau"].to_numpy()
    single = [float(np.abs(np.clip(q, 0, 1) - y).mean() * 100) for q in P]
    return {"cell": cell, "direction": direction, "n": len(te), "MAE": e.mean(), "median": np.median(e),
            "MAE_tau_le15": e[tau <= 15].mean(), "MAE_tau_ge21": e[tau >= 21].mean(),
            "single_seed_mean": np.mean(single), "single_seed_sd": np.std(single, ddof=1)}


def main() -> None:
    T = {g: pd.read_parquet(R / f"validate_ha2_guard_tokens_{g}.parquet").sort_values(["sim_idx", "tau"])
         .reset_index(drop=True) for g in ("DL", "TG")}
    T3 = a3_tables(T)
    W = {}
    for g in ("DL", "TG"):
        w = np.load(R / f"validate_ha2_windows_{g}.npz")
        raw = pd.read_parquet(R / f"validate_ha2_tokens_{g}.parquet", columns=["sim_idx", "tau"])
        assert (w["sim_idx"] == raw["sim_idx"].to_numpy()).all() and (w["tau"] == raw["tau"].to_numpy()).all()
        # align windows to the sorted guard table
        key = pd.MultiIndex.from_arrays([raw["sim_idx"], raw["tau"]])
        pos = key.get_indexer(pd.MultiIndex.from_arrays([T[g]["sim_idx"], T[g]["tau"]]))
        assert (pos >= 0).all()
        W[g] = w["X"].reshape(len(raw), -1)[pos]
    rows = []
    out_csv = R / "validate_ha2_part8_abl_summary.csv"
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        d = f"{src}to{tgt}"
        tr, te = T[src], T[tgt]
        cells = {}
        assert_allow_list(list(TD_TOKENS))
        cells["TDonly"] = (tr[TD_TOKENS].to_numpy(np.float64), te[TD_TOKENS].to_numpy(np.float64))
        assert_allow_list(ALL77)
        cells["HA2_ref"] = (tr[ALL77].to_numpy(np.float64), te[ALL77].to_numpy(np.float64))
        cells["A3"] = (T3[src][ALL77].to_numpy(np.float64), T3[tgt][ALL77].to_numpy(np.float64))
        cells["A1raw"] = (W[src], W[tgt])
        zc = ["R1", "X1", "R0", "X0"]
        cells["A1Z"] = (np.hstack([W[src], tr[zc].to_numpy(np.float32)]), np.hstack([W[tgt], te[zc].to_numpy(np.float32)]))
        for cell, (Xtr, Xte) in cells.items():
            f = R / f"validate_ha2_part8_abl_pred_{cell}_{d}.csv"
            if f.exists():
                pr = pd.read_csv(f)
                assert (pr["sim_idx"].to_numpy() == te["sim_idx"].to_numpy()).all()
                P = [pr[f"p_s{k}"].to_numpy() for k in range(3)]
                p = pr["pred"].to_numpy()
            else:
                p, P = fit_predict(Xtr, tr["y"].to_numpy(), Xte, f"{cell} {d}")
                te[["sim_idx", "tau", "s1", "y"]].assign(pred=p, p_s0=P[0], p_s1=P[1], p_s2=P[2]).to_csv(f, index=False)
            rows.append(summary_row(cell, d, te, p, P))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
