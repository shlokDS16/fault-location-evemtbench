"""Part 8 (claudedocs/validator_spec_part8.md; design 24 / 24a), parts A, B, C, D1. CPU only. Written from the spec
and the design text only; reuses my own earlier code (mv.two_ended, tokens.phasor, ingrid.rule_windows).

A  two-ended PHASOR locator, positive sequence, one-cycle DFT of the LAST cycle of the window at both ends,
   nameplate Z1:  d = Re[(V1S - V1R + Z1 I1R) / (Z1 (I1S + I1R))]  (both currents into the line).
   DL, TG (tau windows), adapt TEST line faults (rule windows), CIGRE MV 14 two-terminal segments (rule windows).
B  my two-ended TD (aerial-mode R-L) locator, DL and TG:
   (i)  remote record shifted by k = +-1, 5, 10, 20 samples: rem_k[n] = rem[n - k] (k > 0: remote record late);
        the remote derivative is shifted with it;
   (ii) fractional delays +-1, 10, 50 us: whole remote record delayed by tau via FFT phase shift exp(-j w tau)
        (rfft, n = 4801), remote derivative recomputed (SG 11/2) from the shifted record;
   (iii) R1 and L1 both x {0.90, 0.95, 1.05, 1.10}; R1 only x {0.8, 1.2}.
D1 my two-ended TD locator on adapt TEST line faults (all; short circuits only).
C  is computed in the summary from my stored tokens (dtd of the oracle loop).
Output: results/validate_ha2_part8_perwindow.parquet (one row per window, all estimates unclipped).
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import mv as MVM  # noqa: E402
from ingrid import adapt_episodes, rule_windows  # noqa: E402
from mv import two_ended  # noqa: E402
from tokens import phasor  # noqa: E402

R = C.RESULTS
NW = int(os.environ.get("VAL_WORKERS", "6"))
A = C.A_OP
N = C.N_CYC
W0 = 2 * np.pi * C.F0
SHIFTS = [-20, -10, -5, -1, 1, 5, 10, 20]
FRAC_US = [-50, -10, -1, 1, 10, 50]
SCALE_RL = [0.90, 0.95, 1.05, 1.10]
SCALE_R = [0.8, 1.2]
OUT = R / "validate_ha2_part8_perwindow.parquet"


def sg(x: np.ndarray) -> np.ndarray:
    return savgol_filter(x, 11, 2, deriv=1, delta=1.0 / C.FS, axis=0)


def pos(p3: np.ndarray) -> complex:
    return (p3[0] + A * p3[1] + A ** 2 * p3[2]) / 3.0


def phasor_two_ended(S: np.ndarray, Rr: np.ndarray, s1: int, Z1: complex) -> float:
    pS, pR = phasor(S, s1 - N), phasor(Rr, s1 - N)
    I1S, V1S, I1R, V1R = pos(pS[0:3]), pos(pS[3:6]), pos(pR[0:3]), pos(pR[3:6])
    den = Z1 * (I1S + I1R)
    if abs(den) == 0:
        return np.nan
    return float(((V1S - V1R + Z1 * I1R) / den).real)


def frac_delay(x: np.ndarray, tau_s: float) -> np.ndarray:
    n = x.shape[0]
    f = np.fft.rfftfreq(n, d=1.0 / C.FS)
    X = np.fft.rfft(x, axis=0)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * tau_s)[:, None], n=n, axis=0)


def work(task: tuple) -> list[dict]:
    grid, r = task
    idx = int(r["sim_idx"])
    sens = grid in ("DL", "TG")
    if grid in ("DL", "TG"):
        S = np.load(C.CACHE / grid / f"ep{idx}.npy")
        Rr = np.load(C.CACHE / f"{grid}_R" / f"ep{idx}.npy")
        Z1, _ = C.line_z(grid, r["line"])
        nf = int(round((r["start"] - 1.0) * C.FS))
        ends = [nf + int(round(t / 1000.0 * C.FS)) for t in C.taus_for(r["etype"])]
    elif grid == "AD":
        S = np.load(C.CACHE / "AD" / f"ep{idx}.npy")
        Rr = np.load(C.CACHE / "AD_R" / f"ep{idx}.npy")
        Z1, _ = C.line_z("AD", r["line"])
        nf = math.floor((r["start"] - 1.0) * C.FS)
        ends = rule_windows(r["start"], r["etype"], r["dur"], len(S))
    else:  # MV, two-terminal segments only
        sig = np.load(MVM.CACHE / f"ep{idx}.npy")
        if np.isnan(sig[:, 0]).any() or np.isnan(sig[:, 6]).any():
            return []
        S, Rr = sig[:, 0:6], sig[:, 6:12]
        Z1, _ = MVM.line_z(r["line"])
        nf = math.floor((r["start"] - 1.0) * C.FS)
        D = math.floor(r["dur"] * C.FS) if "incipient" in r["etype"] else 289
        ends = [s1 for s1 in range(C.WIN, len(sig) + 1, 48) if s1 > nf and s1 - C.WIN < nf + D]
    R1, L1 = Z1.real, Z1.imag / W0
    dS, dR = sg(S[:, 0:3]), sg(Rr[:, 0:3])
    variants = {}
    if sens:
        for k in SHIFTS:
            variants[f"sh{k:+d}"] = (np.roll(Rr, k, axis=0), np.roll(dR, k, axis=0))
        for us in FRAC_US:
            Rf = frac_delay(Rr, us * 1e-6)
            variants[f"fr{us:+d}us"] = (Rf, sg(Rf[:, 0:3]))
    rows = []
    for s1 in ends:
        sl = slice(s1 - C.WIN, s1)
        if sens:
            assert s1 - C.WIN - 20 >= 0 and s1 + 20 <= len(S)
        row = {"grid": grid, "sim_idx": idx, "s1": s1, "nf": nf, "pft_ms": (s1 - nf) / 9.6, "ftype": C.ftype(r["etype"]),
               "line": r["line"], "res": r["res"], "y": r["loc"] / 100.0,
               "ph2e": phasor_two_ended(S, Rr, s1, Z1),
               "td2e": two_ended(S, Rr, dS, dR, R1, L1, sl)}
        if sens:
            for name, (Rv, dRv) in variants.items():
                row[f"td2e_{name}"] = two_ended(S, Rv, dS, dRv, R1, L1, sl)
            for s in SCALE_RL:
                row[f"td2e_RL{s:.2f}"] = two_ended(S, Rr, dS, dR, R1 * s, L1 * s, sl)
            for s in SCALE_R:
                row[f"td2e_R{s:.1f}"] = two_ended(S, Rr, dS, dR, R1 * s, L1, sl)
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


def main() -> None:
    if OUT.exists():
        print("exists", OUT)
        return
    with ProcessPoolExecutor(max_workers=NW) as ex:
        res = list(ex.map(work, tasks(), chunksize=4))
    df = pd.DataFrame([x for rr in res for x in rr])
    df.to_parquet(OUT, index=False)
    print(df.groupby("grid").size().to_dict(), flush=True)


if __name__ == "__main__":
    main()
