"""Part 6A: classical one-ended locators (design sections 21 / 21a), written from the spec only.

Methods on the ORACLE loop (labels; favours the baselines, disclosed):
  R   reactance          d = Im(V/I) / Im(Z1)
  T   Takagi             d = Im(V conj(dIl)) / Im(Z1 I conj(dIl)),   dIl = loop superimposed current
  T2  Takagi-I2          polarised by I2 (same rotation as my tak2 token); 3ph faults: T2 = R (21a)
  MT  modified Takagi    ground loops: polarised by 3I0 exp(-jT), T = angle((Z0SR + (1-d) Z0L) / (Z0SL + Z0L + Z0SR)),
                         3 iterations from d = 0.5; Z0SL = -dV0/dI0 (local), Z0SR = -dV0R/dI0R (REMOTE, oracle);
                         phase loops: = T
  E   Eriksson (1985)    d^2 - K1 d + K2 - K3 Rf = 0; Rf eliminated through Im(. conj K3) = 0; root in [-0.1, 1.1],
                         closest to 0.5 if two; ZSL = -dV1/dI1 local, ZSR = -dV1R/dI1R REMOTE (oracle)
Loop V, I with k0 compensation exactly as in my tokens.py. Phasors: one-cycle DFT (absolute n mod N), last cycle of the
window. Pre-fault memory: the cycle ending 10 samples before inception, [nf - 10 - N, nf - 10).
Scoring: estimate clipped to [0, 1]; non-finite (undefined) -> 0.5 and counted.
Grids: DL, TG (tau windows), adapt TEST line faults (rule windows), CIGRE MV (rule windows; MainLn8-14 local = bus 14,
y_local = 1 - y, no remote -> E and MT undefined).
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import mv as MVM  # noqa: E402
from ingrid import adapt_episodes, rule_windows  # noqa: E402
from tokens import PH, phasor  # noqa: E402

R = C.RESULTS
NW = int(os.environ.get("VAL_WORKERS", "6"))
A = C.A_OP
N = C.N_CYC
PRE_GAP = 10
METHODS = ["R", "T", "T2", "MT", "E"]
R_G = {"ag": 1.0 + 0j, "bg": A ** 2, "cg": A}
R_P = {"ab": 1.0 - A ** 2, "bc": A ** 2 - A, "ca": A - 1.0}


def seq(p3: np.ndarray) -> tuple[complex, complex, complex]:
    return (p3.sum() / 3.0, (p3[0] + A * p3[1] + A ** 2 * p3[2]) / 3.0, (p3[0] + A ** 2 * p3[1] + A * p3[2]) / 3.0)


def polest(V: complex, I: complex, P: complex, Z1: complex) -> float:
    den = (Z1 * I * np.conj(P)).imag
    return (V * np.conj(P)).imag / den if den != 0 else np.nan


def eriksson(V: complex, I: complex, dI: complex, ZL: complex, ZSL: complex, ZSR: complex) -> float:
    """Derivation (checked): V = d ZL I + Rf If, dI = If (ZSR + (1-d) ZL) / (ZSL + ZL + ZSR)
    -> (V/(ZL I) - d)(1 + ZSR/ZL - d) = Rf dI/(ZL I) (1 + (ZSL + ZSR)/ZL), i.e. d^2 - K1 d + K2 - K3 Rf = 0."""
    K1 = 1 + ZSR / ZL + V / (ZL * I)
    K2 = V / (ZL * I) * (1 + ZSR / ZL)
    K3 = dI / (ZL * I) * (1 + (ZSL + ZSR) / ZL)
    c3 = np.conj(K3)
    a, b, c = c3.imag, -(K1 * c3).imag, (K2 * c3).imag  # Im((d^2 - K1 d + K2) conj K3) = 0, d real
    if not np.isfinite([a, b, c]).all():
        return np.nan
    if abs(a) < 1e-12 * (abs(b) + abs(c) + 1e-300):
        roots = [-c / b] if b != 0 else []
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return np.nan
        s = math.sqrt(disc)
        roots = [(-b + s) / (2 * a), (-b - s) / (2 * a)]
    roots = [r for r in roots if -0.1 <= r <= 1.1]
    if not roots:
        return np.nan
    return min(roots, key=lambda r: abs(r - 0.5))


def window_methods(loc: np.ndarray, rem: np.ndarray | None, s1: int, nf: int, lp: str, ftype: str,
                   Z1: complex, Z0: complex) -> dict:
    pre = nf - PRE_GAP - N
    assert pre >= 0
    L, P = phasor(loc, s1 - N), phasor(loc, pre)
    LI, LV, PI, PV = L[0:3], L[3:6], P[0:3], P[3:6]
    dI, dV = LI - PI, LV - PV
    I0, I1, I2 = seq(LI)
    dI0, dI1, _ = seq(dI)
    dV0, dV1, _ = seq(dV)
    k0 = (Z0 - Z1) / (3.0 * Z1)
    if lp.endswith("g"):
        p = PH[lp[0]]
        V, I, dIl, pol2 = LV[p], LI[p] + k0 * 3 * I0, dI[p] + k0 * 3 * dI0, I2 / R_G[lp]
    else:
        p, q = PH[lp[0]], PH[lp[1]]
        V, I, dIl, pol2 = LV[p] - LV[q], LI[p] - LI[q], dI[p] - dI[q], I2 / R_P[lp]
    out = {}
    with np.errstate(all="ignore"):
        out["R"] = (V / I).imag / Z1.imag if abs(I) > 1e-9 else np.nan
        out["T"] = polest(V, I, dIl, Z1)
        out["T2"] = out["R"] if ftype.startswith("3ph") else polest(V, I, pol2, Z1)
        ZSL = -dV1 / dI1 if abs(dI1) > 1e-9 else np.nan
        ZSR = np.nan
        Z0SL = -dV0 / dI0 if abs(dI0) > 1e-6 * abs(I1) + 1e-9 else np.nan
        Z0SR = np.nan
        if rem is not None:
            LR, PR = phasor(rem, s1 - N), phasor(rem, pre)
            dIR, dVR = LR[0:3] - PR[0:3], LR[3:6] - PR[3:6]
            dI0R, dI1R, _ = seq(dIR)
            dV0R, dV1R, _ = seq(dVR)
            I1R = seq(LR[0:3])[1]
            ZSR = -dV1R / dI1R if abs(dI1R) > 1e-9 else np.nan
            Z0SR = -dV0R / dI0R if abs(dI0R) > 1e-6 * abs(I1R) + 1e-9 else np.nan
        if lp.endswith("g"):
            if np.isfinite(Z0SL) and np.isfinite(Z0SR):
                d = 0.5
                for _ in range(3):
                    Ds = (Z0SR + (1 - d) * Z0) / (Z0SL + Z0 + Z0SR)
                    d = polest(V, I, 3 * I0 * np.exp(-1j * np.angle(Ds)), Z1)
                    if not np.isfinite(d):
                        break
                out["MT"] = d
            else:
                out["MT"] = np.nan
        else:
            out["MT"] = out["T"]
        if np.isfinite(ZSL) and np.isfinite(ZSR) and abs(I) > 1e-9:
            out["E"] = eriksson(V, I, dIl, Z1, ZSL, ZSR)
        else:
            out["E"] = np.nan
        out["ZSL_r"], out["ZSL_x"] = (complex(ZSL).real, complex(ZSL).imag) if np.isfinite(ZSL) else (np.nan, np.nan)
        out["ZSR_r"], out["ZSR_x"] = (complex(ZSR).real, complex(ZSR).imag) if np.isfinite(ZSR) else (np.nan, np.nan)
    return {k: float(v) if np.isfinite(v) else np.nan for k, v in out.items()}


def work(task: tuple) -> list[dict]:
    grid, r = task
    idx = int(r["sim_idx"])
    if grid in ("DL", "TG"):
        loc = np.load(C.CACHE / grid / f"ep{idx}.npy")
        rem = np.load(C.CACHE / f"{grid}_R" / f"ep{idx}.npy")
        Z1, Z0 = C.line_z(grid, r["line"])
        nf = int(round((r["start"] - 1.0) * C.FS))
        ends = [nf + int(round(t / 1000.0 * C.FS)) for t in C.taus_for(r["etype"])]
        y_loc, flip = r["loc"] / 100.0, False
    elif grid == "AD":
        loc = np.load(C.CACHE / "AD" / f"ep{idx}.npy")
        rem = np.load(C.CACHE / "AD_R" / f"ep{idx}.npy")
        Z1, Z0 = C.line_z("AD", r["line"])
        nf = math.floor((r["start"] - 1.0) * C.FS)
        ends = rule_windows(r["start"], r["etype"], r["dur"], len(loc))
        y_loc, flip = r["loc"] / 100.0, False
    else:  # MV
        sig = np.load(MVM.CACHE / f"ep{idx}.npy")
        Z1, Z0 = MVM.line_z(r["line"])
        has_x = not np.isnan(sig[:, 0]).any()
        has_y = not np.isnan(sig[:, 6]).any()
        if has_x:
            loc, rem, flip = sig[:, 0:6], (sig[:, 6:12] if has_y else None), False
        else:
            loc, rem, flip = sig[:, 6:12], None, True
        nf = math.floor((r["start"] - 1.0) * C.FS)
        D = math.floor(r["dur"] * C.FS) if "incipient" in r["etype"] else 289
        ends = [s1 for s1 in range(C.WIN, len(sig) + 1, 48) if s1 > nf and s1 - C.WIN < nf + D]
        y_loc = 1.0 - r["loc"] / 100.0 if flip else r["loc"] / 100.0
    lp = C.oracle_loop(r["etype"], r["ph1"], r["ph2"])
    ft = C.ftype(r["etype"])
    rows = []
    for s1 in ends:
        row = {"grid": grid, "sim_idx": idx, "s1": s1, "nf": nf, "pft_ms": (s1 - nf) / 9.6, "ftype": ft,
               "line": r["line"], "res": r["res"], "y_local": y_loc, "flipped": flip, "oracle": lp,
               "has_remote": rem is not None}
        row.update(window_methods(loc, rem, s1, nf, lp, ft, Z1, Z0))
        rows.append(row)
    return rows


def tasks() -> list[tuple]:
    t = []
    for g in ("DL", "TG"):
        for r in C.episodes(g).sort_values("sim_idx").to_dict("records"):
            t.append((g, r))
    a = adapt_episodes()
    for r in a[a["split"] == "test"].to_dict("records"):
        t.append(("AD", r))
    for r in MVM.episodes().to_dict("records"):
        t.append(("MV", r))
    return t


def pft_bin(p: np.ndarray) -> np.ndarray:
    return np.select([p <= 15, p <= 30, p <= 50], ["<=15", "20-30", "35-50"], ">=55")


def rf_bin(res: np.ndarray) -> np.ndarray:
    return np.select([res <= 1, res <= 10, res <= 25], ["<=1", "(1,10]", "(10,25]"], ">25")


def main() -> None:
    out = R / "validate_ha2_step3_classical_perwindow.parquet"
    if out.exists():
        df = pd.read_parquet(out)
    else:
        with ProcessPoolExecutor(max_workers=NW) as ex:
            res = list(ex.map(work, tasks(), chunksize=8))
        df = pd.DataFrame([x for rr in res for x in rr])
        df.to_parquet(out, index=False)
    print(df.groupby("grid").size().to_dict(), flush=True)
    # scored estimates
    for m in METHODS:
        v = df[m].to_numpy()
        df[f"nan_{m}"] = ~np.isfinite(v)
        df[f"e_{m}"] = np.abs(np.where(np.isfinite(v), np.clip(v, 0, 1), 0.5) - df["y_local"]) * 100
    # my H-A2 (guarded, parts 1-5)
    ha2 = {"DL": ("validate_ha2_guard_pred_zeroshot_HA2_TGtoDL.csv", "y"),
           "TG": ("validate_ha2_guard_pred_zeroshot_HA2_DLtoTG.csv", "y"),
           "AD": ("validate_ha2_guard_pred_ingrid_HA2_adapt_test.csv", "y"),
           "MV": ("validate_ha2_guard_pred_mv_HA2_MV.csv", "y_local")}
    parts = []
    for g, (f, ycol) in ha2.items():
        p = pd.read_csv(R / f)[["sim_idx", "s1", "pred", ycol]].rename(columns={ycol: "y_ha2", "pred": "HA2"})
        d = df[df["grid"] == g].merge(p, on=["sim_idx", "s1"], how="left", validate="1:1")
        assert d["HA2"].notna().all() and len(d) == len(p), g
        assert np.allclose(d["y_ha2"], d["y_local"]), g
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["e_HA2"] = np.abs(df["HA2"] - df["y_local"]) * 100
    df["pbin"] = pft_bin(df["pft_ms"].to_numpy())
    df["rfbin"] = rf_bin(df["res"].to_numpy())
    df["shc"] = df["ftype"].str.contains("shc")
    allm = METHODS + ["HA2"]
    rows = []
    for g, d in df.groupby("grid", sort=False):
        for subset, dd in (("all", d), ("shc", d[d["shc"]])):
            for m in allm:
                e = dd[f"e_{m}"]
                row = {"grid": g, "subset": subset, "method": m, "n": len(dd), "MAE": e.mean(), "median": e.median(),
                       "n_undefined": int(dd[f"nan_{m}"].sum()) if m != "HA2" else 0}
                for b in ("<=15", "20-30", "35-50", ">=55"):
                    row[f"MAE_pft_{b}"] = e[dd["pbin"] == b].mean()
                for b in ("<=1", "(1,10]", "(10,25]", ">25"):
                    row[f"MAE_rf_{b}"] = e[dd["rfbin"] == b].mean()
                rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(R / "validate_ha2_step3_classical_summary.csv", index=False)
    # bolted check (DL, R_f <= 1, short circuits, post >= 25 ms)
    b = df[(df["grid"] == "DL") & (df["res"] <= 1) & df["shc"] & (df["pft_ms"] >= 25)]
    bolt = pd.DataFrame([{"method": m, "n": len(b), "MAE": b[f"e_{m}"].mean(), "pass_le3": b[f"e_{m}"].mean() <= 3,
                          **{f"MAE_{ft}": b.loc[b["ftype"] == ft, f"e_{m}"].mean() for ft in sorted(b["ftype"].unique())}}
                         for m in METHODS])
    bolt.to_csv(R / "validate_ha2_step3_classical_bolted.csv", index=False)
    # H-A2 vs best classical per grid
    cmp = []
    for g in ("DL", "TG", "AD", "MV"):
        sa = s[(s["grid"] == g) & (s["subset"] == "all")].set_index("method")
        best = sa.loc[METHODS, "MAE"].idxmin()
        cmp.append({"grid": g, "best_classical": best, "best_MAE": sa.loc[best, "MAE"], "HA2_MAE": sa.loc["HA2", "MAE"],
                    "ratio": sa.loc["HA2", "MAE"] / sa.loc[best, "MAE"],
                    "HA2_beats_classical(<=0.8x)": sa.loc["HA2", "MAE"] <= 0.8 * sa.loc[best, "MAE"]})
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(R / "validate_ha2_step3_classical_vs_ha2.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(bolt.round(3).to_string(index=False))
        print(s[s.subset == "all"].round(2).to_string(index=False))
        print(s[s.subset == "shc"][["grid", "method", "n", "MAE", "median"]].round(2).to_string(index=False))
        print(cmp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
