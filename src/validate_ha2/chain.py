"""Part 6B: measurement-chain robustness (design section 22), written from the spec only. CPU only.

Chain elements, applied to the FULL record of every channel of BOTH terminals (independently) before windowing;
each filter is started from steady state by prepending the first pre-fault cycle repeated for 0.2 s (10 cycles,
1,920 samples), which is dropped afterwards.
  AA    2nd-order Butterworth low-pass, fc = 2 kHz, causal lfilter, all six channels.
  CT    PSRC 'CT SAT' model: i_e = sgn(l) 10 (w|l| / (sqrt2 Vs))^S, dl/dt = Rt i2 (Lb = 0), i2 = i1/N - i_e;
        backward Euler + safeguarded Newton per sample; S = 22, N = 1200/5 = 240, Rw = 0.6 ohm, Rt = Rw + Rb.
        CT-mild: Vs 800 V, Rb 1.0, remanence 0; CT-severe: Vs 400 V, Rb 2.0, remanence +0.6 x sqrt2 Vs / w
        (all phases, both ends). Output mapped back to primary: i2 x N.
        (The design text writes '(10/RP)'; RP is not defined there. I use the standard C37.110 / Swift form with
        10 A at Vs, i.e. RP = 1. With S = 22 a factor sqrt2 in RP would shift the knee flux by only 1.6 %.)
        Saturation flag per window: max|i_e| > 0.05 max|i2| in the window on any LOCAL phase.
  CVT   H(s) = s Ce R / (s^2 Lc Ce + s Ce R + 1) with w0^2 Lc Ce = 1 and tau = 2 Lc / R
        -> H(s) = (2/tau) s / (s^2 + (2/tau) s + w0^2); bilinear transform at fs. CVT-5 / CVT-15 (tau 5 / 15 ms).
  FULL  = CT-severe + CVT-15, then AA on all six channels.
Per scenario and window (DL 14,640, TG 32,940): 59 + 18 local tokens (my tokens.py, I0 guard applied), two-ended
aerial-mode R-L locator (my mv.two_ended, identical to part 4), oracle-loop reactance and Takagi-I2 (3ph: = R, 21a),
saturation flag. H-A2: 77-token HGB (part 1-5 settings, seeds 0-2, 3-seed average) trained on the CLEAN source grid and
tested on each distorted target; and trained AND tested with FULL (matched chain).
"""
from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import bilinear, butter, lfilter, savgol_filter
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from guard import GUARD_HALF, GUARD_ZERO, THR  # noqa: E402
from mv import two_ended  # noqa: E402
from physics_hgb import HGB_PARAMS, assert_allow_list  # noqa: E402
from tokens import LOCAL_TOKENS, LOOPS, TD_TOKENS, local_tokens, td_tokens  # noqa: E402

R = C.RESULTS
NW = int(os.environ.get("VAL_WORKERS", "6"))
FS = C.FS
W0 = 2 * np.pi * C.F0
PRE = 10 * C.N_CYC
NCT = 240.0
RW = 0.6
S_EXP = 22.0
CT = {"CTm": dict(Vs=800.0, Rb=1.0, rem=0.0), "CTs": dict(Vs=400.0, Rb=2.0, rem=0.6)}
SCEN = ["clean", "AA", "CTm", "CTs", "CVT5", "CVT15", "FULL"]
B_AA, A_AA = butter(2, 2000.0, btype="low", fs=FS)
COLS = list(LOCAL_TOKENS) + list(TD_TOKENS)


def cvt_ba(tau: float) -> tuple[np.ndarray, np.ndarray]:
    return bilinear([2.0 / tau, 0.0], [1.0, 2.0 / tau, W0 ** 2], fs=FS)


CVT = {"CVT5": cvt_ba(0.005), "CVT15": cvt_ba(0.015)}


def ext(x: np.ndarray) -> np.ndarray:
    """Prepend 0.2 s of the first pre-fault cycle (steady-state start)."""
    return np.concatenate([np.tile(x[:C.N_CYC], (PRE // C.N_CYC,) + (1,) * (x.ndim - 1)), x], axis=0)


def filt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    return lfilter(b, a, ext(x), axis=0)[PRE:]


@njit(cache=True)
def ct_sim(i1: np.ndarray, Vs: float, Rb: float, rem: float) -> tuple[np.ndarray, np.ndarray]:
    h = 1.0 / 9600.0
    Rt = RW + Rb
    K = W0 / (math.sqrt(2.0) * Vs)
    lam = rem * math.sqrt(2.0) * Vs / W0
    n = i1.shape[0]
    i2 = np.empty(n)
    ie = np.empty(n)
    for k in range(n):
        ip = i1[k] / NCT
        g = lam
        ieg = math.copysign(10.0 * (K * abs(g)) ** S_EXP, g)
        other = g + h * Rt * (ip - ieg)
        lo, hi = (g, other) if other >= g else (other, g)
        x = other
        for _ in range(100):
            iex = math.copysign(10.0 * (K * abs(x)) ** S_EXP, x)
            f = x - g - h * Rt * (ip - iex)
            if abs(f) <= 1e-14 * (1.0 + abs(x)):
                break
            if f > 0:
                hi = x
            else:
                lo = x
            fp = 1.0 + h * Rt * S_EXP * abs(iex) / max(abs(x), 1e-300)
            xn = x - f / fp
            if not (lo < xn < hi):
                xn = 0.5 * (lo + hi)
            if abs(xn - x) <= 1e-15 * (1.0 + abs(x)):
                x = xn
                break
            x = xn
        lam = x
        ie[k] = math.copysign(10.0 * (K * abs(x)) ** S_EXP, x)
        i2[k] = ip - ie[k]
    return i2, ie


def ct_apply(I: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ie = ext(I)
    out, ies, i2s = np.empty_like(I), np.empty_like(I), np.empty_like(I)
    for k in range(3):
        i2, ie = ct_sim(np.ascontiguousarray(Ie[:, k]), p["Vs"], p["Rb"], p["rem"])
        out[:, k], ies[:, k], i2s[:, k] = i2[PRE:] * NCT, ie[PRE:], i2[PRE:]
    return out, ies, i2s


def chain(sig: np.ndarray, scen: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    I, V = sig[:, 0:3], sig[:, 3:6]
    ie = i2 = None
    if scen == "clean":
        return sig, None, None
    if scen == "AA":
        return filt(B_AA, A_AA, sig), None, None
    if scen in CT:
        I2, ie, i2 = ct_apply(I, CT[scen])
        return np.column_stack([I2, V]), ie, i2
    if scen in CVT:
        return np.column_stack([I, filt(*CVT[scen], V)]), None, None
    if scen == "FULL":
        I2, ie, i2 = ct_apply(I, CT["CTs"])
        V2 = filt(*CVT["CVT15"], V)
        return filt(B_AA, A_AA, np.column_stack([I2, V2])), ie, i2
    raise ValueError(scen)


def work(task: tuple) -> list[dict]:
    grid, r = task
    idx = int(r["sim_idx"])
    loc0 = np.load(C.CACHE / grid / f"ep{idx}.npy")
    rem0 = np.load(C.CACHE / f"{grid}_R" / f"ep{idx}.npy")
    Z1, Z0 = C.line_z(grid, r["line"])
    nf = int(round((r["start"] - 1.0) * FS))
    lp = C.oracle_loop(r["etype"], r["ph1"], r["ph2"])
    ft = C.ftype(r["etype"])
    rows = []
    for scen in SCEN:
        loc, ie, i2 = chain(loc0, scen)
        rem, _, _ = chain(rem0, scen)
        dloc = savgol_filter(loc, 11, 2, deriv=1, delta=1.0 / FS, axis=0)
        dS = dloc[:, 0:3]
        dR = savgol_filter(rem[:, 0:3], 11, 2, deriv=1, delta=1.0 / FS, axis=0)
        for tau in C.taus_for(r["etype"]):
            s1 = nf + int(round(tau / 1000.0 * FS))
            row = {"scen": scen, "grid": grid, "sim_idx": idx, "tau": tau, "s1": s1, "ftype": ft, "line": r["line"],
                   "res": r["res"], "y": r["loc"] / 100.0, "oracle": lp}
            row.update(local_tokens(loc, s1, Z1, Z0))
            row.update(td_tokens(loc, dloc, s1, Z1, Z0))
            row["two_ended"] = two_ended(loc, rem, dS, dR, Z1.real, Z1.imag / W0, slice(s1 - C.WIN, s1))
            if ie is not None:
                w = slice(s1 - C.WIN, s1)
                row["sat"] = bool((np.abs(ie[w]).max(axis=0) > 0.05 * np.abs(i2[w]).max(axis=0)).any())
            else:
                row["sat"] = False
            rows.append(row)
    return rows


def guard(df: pd.DataFrame) -> pd.DataFrame:
    m = (df["i0r"] < THR).to_numpy()
    df.loc[m, GUARD_ZERO] = 0.0
    df.loc[m, GUARD_HALF] = 0.5
    return df


def tables() -> dict[str, pd.DataFrame]:
    out = {}
    for g in ("DL", "TG"):
        f = R / f"validate_ha2_step3_chain_tokens_{g}.parquet"
        if not f.exists():
            t0 = time.time()
            tasks = [(g, r) for r in C.episodes(g).sort_values("sim_idx").to_dict("records")]
            with ProcessPoolExecutor(max_workers=NW) as ex:
                res = list(ex.map(work, tasks, chunksize=4))
            df = pd.DataFrame([x for rr in res for x in rr])
            assert np.isfinite(df[COLS].to_numpy()).all()
            guard(df).to_parquet(f, index=False)
            print(g, "tokens", len(df), f"{time.time() - t0:.0f}s", flush=True)
        out[g] = pd.read_parquet(f)
    return out


def cvt_residual() -> pd.DataFrame:
    rows = []
    for name, (b, a) in CVT.items():
        for T in (10, 20, 40):
            worst = 0.0
            for ph in np.arange(0, 180, 5):
                n0 = 4 * C.N_CYC
                t = np.arange(n0 + 960) / FS
                v = np.sin(W0 * t + np.deg2rad(ph))
                v[n0:] = 0.0
                y = filt(b, a, v)
                k1 = n0 + int(round(T / 1000 * FS))
                worst = max(worst, np.abs(y[k1:k1 + C.N_CYC // 2]).max())  # peak in the half-cycle after T
            rows.append({"cvt": name, "ms_after_collapse": T, "residual_pct_of_peak": 100 * worst})
        # 50 Hz gain check
        w, hh = __import__("scipy.signal", fromlist=["freqz"]).freqz(b, a, worN=[C.F0], fs=FS)
        rows.append({"cvt": name, "ms_after_collapse": "gain_50Hz", "residual_pct_of_peak": 100 * abs(hh[0])})
    return pd.DataFrame(rows)


def main() -> None:
    T = tables()
    cache = {g: pd.read_parquet(R / f"validate_ha2_guard_tokens_{g}.parquet") for g in ("DL", "TG")}
    rows, sat = [], []
    # clean recomputation equals my cached guarded tables (sanity)
    for g in ("DL", "TG"):
        c = T[g][T[g]["scen"] == "clean"].reset_index(drop=True)
        k = cache[g].sort_values(["sim_idx", "tau"]).reset_index(drop=True)
        c = c.sort_values(["sim_idx", "tau"]).reset_index(drop=True)
        assert (c["s1"].to_numpy() == k["s1"].to_numpy()).all()
        print(g, "clean recomputed vs cached guarded tokens, max abs diff:",
              float(np.abs(c[COLS].to_numpy() - k[COLS].to_numpy()).max()), flush=True)
    idxmap = {lp: k for k, lp in enumerate(LOOPS)}
    for g in ("DL", "TG"):
        for scen, d in T[g].groupby("scen", sort=False):
            ii = d["oracle"].map(idxmap).to_numpy().astype(int)
            rr = np.arange(len(d))
            react = d[[f"react_{lp}" for lp in LOOPS]].to_numpy()[rr, ii]
            tak2 = d[[f"tak2_{lp}" for lp in LOOPS]].to_numpy()[rr, ii]
            tak2 = np.where(d["ftype"].str.startswith("3ph").to_numpy(), react, tak2)
            y = d["y"].to_numpy()
            for m, p in (("two_ended", d["two_ended"].to_numpy()), ("react_oracle", react), ("tak2_oracle", tak2)):
                e = np.abs(np.clip(p, 0, 1) - y) * 100
                rows.append({"grid": g, "scen": scen, "method": m, "n": len(d), "MAE": e.mean(), "median": np.median(e)})
            sat.append({"grid": g, "scen": scen, "n": len(d), "sat_windows": int(d["sat"].sum()),
                        "sat_fraction": d["sat"].mean(),
                        "sat_fraction_shc": d.loc[d["ftype"].str.contains("shc"), "sat"].mean()})
    # H-A2
    assert_allow_list(COLS)
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        for train_scen, test_scens in (("clean", SCEN), ("FULL", ["FULL"])):
            tr = T[src][T[src]["scen"] == train_scen]
            P = {s: [] for s in test_scens}
            for seed in (0, 1, 2):
                t0 = time.time()
                mdl = HistGradientBoostingRegressor(random_state=seed, **HGB_PARAMS).fit(
                    tr[COLS].to_numpy(np.float64), tr["y"].to_numpy())
                for s in test_scens:
                    te = T[tgt][T[tgt]["scen"] == s]
                    P[s].append(mdl.predict(te[COLS].to_numpy(np.float64)))
                print(src, "->", tgt, "train", train_scen, "seed", seed, f"{time.time() - t0:.0f}s", flush=True)
            for s in test_scens:
                te = T[tgt][T[tgt]["scen"] == s]
                p = np.clip(np.mean(P[s], axis=0), 0, 1)
                e = np.abs(p - te["y"].to_numpy()) * 100
                single = [float(np.abs(np.clip(q, 0, 1) - te["y"].to_numpy()).mean() * 100) for q in P[s]]
                rows.append({"grid": tgt, "scen": s, "method": f"HA2_train_{train_scen}", "n": len(te), "MAE": e.mean(),
                             "median": np.median(e), "single_seed_sd": np.std(single, ddof=1)})
                te[["sim_idx", "tau", "s1", "y"]].assign(pred=p).to_csv(
                    R / f"validate_ha2_step3_chain_pred_HA2_{src}to{tgt}_train{train_scen}_test{s}.csv", index=False)
    s = pd.DataFrame(rows)
    clean = s[s["scen"] == "clean"].set_index(["grid", "method"])["MAE"]
    s["delta_vs_clean"] = [r.MAE - clean.get((r.grid, r.method if r.method != "HA2_train_FULL" else "HA2_train_clean"),
                                             np.nan) for r in s.itertuples()]
    s["robust_rule"] = [("<=1.0" if r.method == "two_ended" else "<=2.0") for r in s.itertuples()]
    s["robust"] = [(r.delta_vs_clean <= (1.0 if r.method == "two_ended" else 2.0)) for r in s.itertuples()]
    s.to_csv(R / "validate_ha2_step3_chain_summary.csv", index=False)
    sat = pd.DataFrame(sat)
    sat.to_csv(R / "validate_ha2_step3_chain_saturation.csv", index=False)
    cv = cvt_residual()
    cv.to_csv(R / "validate_ha2_step3_chain_cvt_residual.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s.round(3).to_string(index=False))
        print(sat.round(4).to_string(index=False))
        print(cv.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
