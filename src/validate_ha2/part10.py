"""Part 10 (claudedocs/validator_spec_part10.md; design 19 / 19a): the two CIGRE MV ablation cells, zero-shot
DL + TG (pooled, my guarded official windows, 14,640 + 32,940) -> all 56,340 CIGRE MV line-fault windows (LOCAL
terminal: bus X, bus 14 for MainLn8-14; target y_local). CPU only, sklearn HGB with HGB_PARAMS, seeds 0-2, 3-seed
mean prediction, clipped to [0, 1], MAE in % of line length.

  HA2_ref  77 guarded tokens (reproduces my part-5 MV cell, 32.49 %), same run.
  A3       77 tokens, the 6 x 3 apparent-impedance tokens zr / zi / absz = Re, Im, |V/I| in OHM of the loop
           (last-cycle phasors, k0-compensated ground loops as in tokens.py), unclipped, 0 when |I| <= 1e-9;
           the other 59 unchanged. DL / TG ohm tables: my part-8 files; MV: computed here.
  A1raw    HGB on the raw 480 x 6 local window (2,880 sample values), the same input as my part-8 A1raw
           (no per-window scaling: part 8 used the raw float32 samples, which I keep here).
Reuses my own code: tokens.phasor / LOOPS / PH, mv.line_z / CACHE, part8_abl.ZTOK, physics_hgb.HGB_PARAMS.
Outputs: results/validate_ha2_part10_{summary.csv, A3_ohm_MV.parquet, pred_<cell>_MV.csv}.
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
import mv as MVM  # noqa: E402
from part8_abl import ZTOK  # noqa: E402
from physics_hgb import HGB_PARAMS, assert_allow_list  # noqa: E402
from tokens import LOCAL_TOKENS, LOOPS, PH, TD_TOKENS, phasor  # noqa: E402

R = C.RESULTS
NW = int(os.environ.get("VAL_WORKERS", "6"))
ALL77 = list(LOCAL_TOKENS) + list(TD_TOKENS)
OHM_MV = R / "validate_ha2_part10_A3_ohm_MV.parquet"
OUT = R / "validate_ha2_part10_summary.csv"


def ohm_rows(loc: np.ndarray, Z1: complex, Z0: complex, idx: int, s1s: list[int]) -> list[dict]:
    k0 = (Z0 - Z1) / (3.0 * Z1)
    rows = []
    for s1 in s1s:
        L = phasor(loc, s1 - C.N_CYC)
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


def ohm_mv_work(task: tuple) -> list[dict]:
    idx, line, flipped, s1s = task
    sig = np.load(MVM.CACHE / f"ep{idx}.npy")
    loc = sig[:, 6:12] if flipped else sig[:, 0:6]
    assert np.isfinite(loc).all()
    Z1, Z0 = MVM.line_z(line)
    return ohm_rows(loc, Z1, Z0, idx, s1s)


def with_ohm(d: pd.DataFrame, o: pd.DataFrame, z1: np.ndarray, tag: str) -> pd.DataFrame:
    m = d.merge(o, on=["sim_idx", "s1"], how="left", validate="1:1")
    assert len(m) == len(d) and m["ohm_zr_ag"].notna().all()
    assert (m["sim_idx"].to_numpy() == d["sim_idx"].to_numpy()).all() and (m["s1"].to_numpy() == d["s1"].to_numpy()).all()
    # sanity: per-unit token = clip(ohm / Z1) on unclipped windows, every loop
    for lp in LOOPS:
        zz = (m[f"ohm_zr_{lp}"] + 1j * m[f"ohm_zi_{lp}"]).to_numpy() / z1
        ok = (np.abs(zz.real) < 4.9) & (np.abs(zz.imag) < 4.9)
        dr = float(np.abs(zz.real[ok] - m[f"zr_{lp}"].to_numpy()[ok]).max())
        di = float(np.abs(zz.imag[ok] - m[f"zi_{lp}"].to_numpy()[ok]).max())
        assert dr < 1e-6 and di < 1e-6, (tag, lp, dr, di)
    clipped = np.zeros(len(m), bool)
    for lp in LOOPS:
        clipped |= (np.abs(m[f"zr_{lp}"]) >= 5).to_numpy() | (np.abs(m[f"zi_{lp}"]) >= 5).to_numpy() \
            | (m[f"absz_{lp}"] >= 10).to_numpy()
    print(tag, f"A3 sanity ok; windows with >= 1 clipped per-unit z token: {clipped.mean() * 100:.1f} %", flush=True)
    for c in ZTOK:
        m[c] = m[f"ohm_{c}"]
    return m


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, tag: str) -> tuple[np.ndarray, list[np.ndarray]]:
    P = []
    for seed in (0, 1, 2):
        t0 = time.time()
        mdl = HistGradientBoostingRegressor(random_state=seed, **HGB_PARAMS).fit(Xtr, ytr)
        P.append(np.concatenate([mdl.predict(Xte[i:i + 8192]) for i in range(0, len(Xte), 8192)]))
        print(tag, "seed", seed, f"{time.time() - t0:.0f}s", flush=True)
    return np.clip(np.mean(P, axis=0), 0, 1), P


def summary_row(cell: str, te: pd.DataFrame, p: np.ndarray, P: list) -> dict:
    y = te["y_local"].to_numpy()
    e = np.abs(p - y) * 100
    t = te["pft_ms"].to_numpy()
    ft = te["ftype"]
    sc = (ft.str.endswith("shc") | ft.str.endswith("shc_w_arc")).to_numpy()
    single = [float(np.abs(np.clip(q, 0, 1) - y).mean() * 100) for q in P]
    row = {"cell": cell, "direction": "DLTGtoMV", "n": len(te), "MAE": e.mean(), "median": np.median(e),
           "MAE_post_le15": e[t <= 15].mean(), "MAE_post_ge21": e[t >= 21].mean(), "MAE_shortcircuit": e[sc].mean(),
           "single_seed_mean": np.mean(single), "single_seed_sd": np.std(single, ddof=1),
           "share_pred_at_0_or_1": float(((p <= 0) | (p >= 1)).mean())}
    for f in sorted(ft.unique()):
        row[f"MAE_{f}"] = e[(ft == f).to_numpy()].mean()
    return row


def main() -> None:
    T = {g: pd.read_parquet(R / f"validate_ha2_guard_tokens_{g}.parquet") for g in ("DL", "TG", "MV")}
    mv = T["MV"]
    assert len(mv) == 56340 and len(T["DL"]) == 14640 and len(T["TG"]) == 32940
    # A3 ohm tables
    if not OHM_MV.exists():
        tasks = [(int(i), g["line"].iloc[0], bool(g["flipped"].iloc[0]), g["s1"].tolist())
                 for i, g in mv.groupby("sim_idx", sort=True)]
        with ProcessPoolExecutor(max_workers=NW) as ex:
            res = list(ex.map(ohm_mv_work, tasks, chunksize=8))
        pd.DataFrame([x for rr in res for x in rr]).to_parquet(OHM_MV, index=False)
    T3 = {}
    for g in ("DL", "TG"):
        o = pd.read_parquet(R / f"validate_ha2_part8_A3_ohm_{g}.parquet")
        d = T[g]
        T3[g] = with_ohm(d, o, (d["R1"] + 1j * d["X1"]).to_numpy(), g)
    z1mv = np.array([MVM.line_z(ln)[0] for ln in mv["line"]])
    T3["MV"] = with_ohm(mv, pd.read_parquet(OHM_MV), z1mv, "MV")

    # raw windows, row-aligned with the (guarded) token tables
    W = {}
    for g in ("DL", "TG"):
        w = np.load(R / f"validate_ha2_windows_{g}.npz")
        assert (w["sim_idx"] == T[g]["sim_idx"].to_numpy()).all() and (w["tau"] == T[g]["tau"].to_numpy()).all()
        W[g] = w["X"].reshape(len(T[g]), -1)
    w = np.load(R / "validate_ha2_part8_mv_windows.npz")
    assert (w["sim_idx"] == mv["sim_idx"].to_numpy()).all() and (w["s1"] == mv["s1"].to_numpy()).all()
    W["MV"] = w["X"].reshape(len(mv), -1)
    assert W["MV"].shape[1] == 2880 and np.isfinite(W["MV"]).all()

    tr = pd.concat([T["DL"], T["TG"]], ignore_index=True)  # same order as my part-5 guard.py MV cell
    tr3 = pd.concat([T3["DL"], T3["TG"]], ignore_index=True)
    ytr = tr["y"].to_numpy()
    assert (tr3["y"].to_numpy() == ytr).all()
    assert_allow_list(ALL77)
    cells = {
        "HA2_ref": lambda: (tr[ALL77].to_numpy(np.float64), mv[ALL77].to_numpy(np.float64)),
        "A3": lambda: (tr3[ALL77].to_numpy(np.float64), T3["MV"][ALL77].to_numpy(np.float64)),
        "A1raw": lambda: (np.concatenate([W["DL"], W["TG"]]), W["MV"]),
    }
    rows = []
    for cell, mk in cells.items():
        f = R / f"validate_ha2_part10_pred_{cell}_MV.csv"
        if f.exists():
            pr = pd.read_csv(f)
            assert (pr["sim_idx"].to_numpy() == mv["sim_idx"].to_numpy()).all()
            P = [pr[f"p_s{k}"].to_numpy() for k in range(3)]
            p = pr["pred"].to_numpy()
        else:
            Xtr, Xte = mk()
            print(cell, "X", Xtr.shape, Xte.shape, Xtr.dtype, flush=True)
            p, P = fit_predict(Xtr, ytr, Xte, cell)
            del Xtr, Xte
            mv[["sim_idx", "s1", "pft_ms", "ftype", "line", "flipped", "y_local"]].assign(
                pred=p, p_s0=P[0], p_s1=P[1], p_s2=P[2]).to_csv(f, index=False)
        rows.append(summary_row(cell, mv, p, P))
        print({k: rows[-1][k] for k in ("cell", "MAE", "median", "single_seed_mean", "single_seed_sd")}, flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
