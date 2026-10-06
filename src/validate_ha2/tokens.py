"""Step 2: 59 local physics tokens + 18 time-domain tokens per official window, and the raw windows.

Outputs
  results/validate_ha2_tokens_{DL,TG}.parquet : metadata + 59 local + 18 TD tokens per window
  results/validate_ha2_windows_{DL,TG}.npz    : raw (n, 480, 6) float32 windows, aligned row-for-row
Only the local terminal's 6 channels and the faulted line's Z1/Z0 enter any token.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

A = C.A_OP
N = C.N_CYC
_W = np.exp(-2j * np.pi * np.arange(N) / N)  # exp(-j 2 pi n / N) for n mod N

LOOPS = ("ag", "bg", "cg", "ab", "bc", "ca")
PH = {"a": 0, "b": 1, "c": 2}
LOCAL_TOKENS = (
    [f"{t}_{lp}" for lp in LOOPS for t in ("zr", "zi", "absz", "react", "tak2", "takd")]
    + [f"tak0_{lp}" for lp in ("ag", "bg", "cg")]
    + ["i0r", "i2r", "v0r", "v2r", "cos_i2i1", "sin_i2i1", "cos_i0i1", "sin_i0i1"]
    + [f"{t}_{p}" for p in "abc" for t in ("onset", "drop", "share", "direl")]
)
TD_TOKENS = [f"{t}_{lp}" for lp in LOOPS for t in ("dtd", "drl", "tdres")]
assert len(LOCAL_TOKENS) == 59 and len(TD_TOKENS) == 18


def phasor(x: np.ndarray, start: int) -> np.ndarray:
    """(2/N) sum_{n=start}^{start+N-1} x[n] exp(-j 2 pi n / N), absolute n; per column."""
    n = np.arange(start, start + N)
    w = np.exp(-2j * np.pi * (n % N) / N)
    return (2.0 / N) * (w @ x[start:start + N])


def est(V: complex, I: complex, Pz: complex, Z1: complex) -> float:
    den = (Z1 * I * np.conj(Pz)).imag
    if abs(den) <= 1e-12:
        return 0.5
    return float(np.clip((V * np.conj(Pz)).imag / den, -1.0, 2.0))


def local_tokens(sig: np.ndarray, s1: int, Z1: complex, Z0: complex) -> dict[str, float]:
    L = phasor(sig, s1 - N)
    F = phasor(sig, s1 - C.WIN)
    LI, LV = L[0:3], L[3:6]
    FI, FV = F[0:3], F[3:6]
    I0 = LI.sum() / 3.0
    I1 = (LI[0] + A * LI[1] + A ** 2 * LI[2]) / 3.0
    I2 = (LI[0] + A ** 2 * LI[1] + A * LI[2]) / 3.0
    V0 = LV.sum() / 3.0
    V1 = (LV[0] + A * LV[1] + A ** 2 * LV[2]) / 3.0
    V2 = (LV[0] + A ** 2 * LV[1] + A * LV[2]) / 3.0
    dI = LI - FI
    k0 = (Z0 - Z1) / (3.0 * Z1)
    out: dict[str, float] = {}
    r_g = {"ag": 1.0 + 0j, "bg": A ** 2, "cg": A}
    r_p = {"ab": 1.0 - A ** 2, "bc": A ** 2 - A, "ca": A - 1.0}
    for lp in LOOPS:
        if lp.endswith("g"):
            p = PH[lp[0]]
            V = LV[p]
            I = LI[p] + k0 * 3.0 * I0
            pol2 = I2 / r_g[lp]
            dIl = dI[p] + k0 * dI.sum()
        else:
            p, q = PH[lp[0]], PH[lp[1]]
            V = LV[p] - LV[q]
            I = LI[p] - LI[q]
            pol2 = I2 / r_p[lp]
            dIl = dI[p] - dI[q]
        if abs(I) <= 1e-9:
            z = 0j
            react = 0.5
        else:
            zz = V / I
            z = zz / Z1
            react = float(np.clip(zz.imag / Z1.imag, -1.0, 2.0))
        out[f"zr_{lp}"] = float(np.clip(z.real, -5, 5))
        out[f"zi_{lp}"] = float(np.clip(z.imag, -5, 5))
        out[f"absz_{lp}"] = float(np.clip(abs(z), 0, 10))
        out[f"react_{lp}"] = react
        out[f"tak2_{lp}"] = est(V, I, pol2, Z1)
        out[f"takd_{lp}"] = est(V, I, dIl, Z1)
        if lp.endswith("g"):
            out[f"tak0_{lp}"] = est(V, I, 3.0 * I0, Z1)
    out["i0r"] = abs(I0) / (abs(I1) + 1e-9)
    out["i2r"] = abs(I2) / (abs(I1) + 1e-9)
    out["v0r"] = abs(V0) / (abs(V1) + 1e-9)
    out["v2r"] = abs(V2) / (abs(V1) + 1e-9)
    a21 = np.angle(I2 / I1)
    a01 = np.angle(I0 / I1)
    out["cos_i2i1"], out["sin_i2i1"] = float(np.cos(a21)), float(np.sin(a21))
    out["cos_i0i1"], out["sin_i0i1"] = float(np.cos(a01)), float(np.sin(a01))
    mx = np.abs(LI).max()
    for p, k in PH.items():
        out[f"onset_{p}"] = float(np.clip(abs(LI[k]) / (abs(FI[k]) + 1e-9), 0, 100))
        out[f"drop_{p}"] = float(np.clip(abs(LV[k]) / (abs(FV[k]) + 1e-9), 0, 5))
        out[f"share_{p}"] = float(abs(LI[k]) / (mx + 1e-9))
        out[f"direl_{p}"] = float(np.clip(abs(dI[k]) / (abs(FI[k]) + 1e-9), 0, 100))
    return out


def td_tokens(sig: np.ndarray, dsig: np.ndarray, s1: int, Z1: complex, Z0: complex) -> dict[str, float]:
    w0 = 2 * np.pi * C.F0
    R1, L1, R0, L0 = Z1.real, Z1.imag / w0, Z0.real, Z0.imag / w0
    i, di, v = sig[:, 0:3], dsig[:, 0:3], sig[:, 3:6]
    i0 = i.sum(axis=1) / 3.0
    di0 = di.sum(axis=1) / 3.0
    lo, hi = s1 - C.WIN + N, s1
    sl = slice(lo, hi)
    slm = slice(lo - N, hi - N)
    out: dict[str, float] = {}
    for lp in LOOPS:
        if lp.endswith("g"):
            p = PH[lp[0]]
            vv = v[:, p]
            u = R1 * i[:, p] + L1 * di[:, p] + (R0 - R1) * i0 + (L0 - L1) * di0
            il = i[:, p]
        else:
            p, q = PH[lp[0]], PH[lp[1]]
            vv = v[:, p] - v[:, q]
            u = R1 * (i[:, p] - i[:, q]) + L1 * (di[:, p] - di[:, q])
            il = i[:, p] - i[:, q]
        y = vv[sl]
        uu = u[sl]
        dil = il[sl] - il[slm]
        X = np.column_stack([uu, dil])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        out[f"dtd_{lp}"] = float(np.clip(beta[0], -1, 2))
        out[f"drl_{lp}"] = float(np.clip(np.dot(y, uu) / (np.dot(uu, uu) + 1e-12), -1, 2))
        out[f"tdres_{lp}"] = float(np.clip(np.linalg.norm(res) / (np.linalg.norm(y) + 1e-12), 0, 5))
    return out


def work(args: tuple) -> tuple[list[dict], np.ndarray]:
    grid, r = args
    sig = np.load(C.CACHE / grid / f"ep{r['sim_idx']}.npy")
    dsig = savgol_filter(sig, window_length=11, polyorder=2, deriv=1, delta=1.0 / C.FS, axis=0)
    Z1, Z0 = C.line_z(grid, r["line"])
    nf = int(round((r["start"] - 1.0) * C.FS))
    rows, wins = [], []
    for tau in C.taus_for(r["etype"]):
        s1 = nf + int(round(tau / 1000.0 * C.FS))
        assert s1 - C.WIN >= 0 and s1 <= len(sig)
        row = {"grid": grid, "sim_idx": r["sim_idx"], "tau": tau, "etype": r["etype"],
               "ftype": C.ftype(r["etype"]), "line": r["line"], "res": r["res"],
               "y": r["loc"] / 100.0, "oracle": C.oracle_loop(r["etype"], r["ph1"], r["ph2"]),
               "nf": nf, "s1": s1, "R1": Z1.real, "X1": Z1.imag, "R0": Z0.real, "X0": Z0.imag}
        row.update(local_tokens(sig, s1, Z1, Z0))
        row.update(td_tokens(sig, dsig, s1, Z1, Z0))
        rows.append(row)
        wins.append(sig[s1 - C.WIN:s1].astype(np.float32))
    return rows, np.stack(wins)


def main() -> None:
    for grid in ("DL", "TG"):
        ep = C.episodes(grid).sort_values("sim_idx").reset_index(drop=True)
        args = [(grid, r) for r in ep.to_dict("records")]
        with ProcessPoolExecutor(max_workers=12) as ex:
            res = list(ex.map(work, args, chunksize=8))
        rows = [x for rr, _ in res for x in rr]
        wins = np.concatenate([w for _, w in res], axis=0)
        df = pd.DataFrame(rows)
        assert len(df) == len(wins)
        df.to_parquet(C.RESULTS / f"validate_ha2_tokens_{grid}.parquet", index=False)
        np.savez(C.RESULTS / f"validate_ha2_windows_{grid}.npz", X=wins,
                 sim_idx=df["sim_idx"].to_numpy(), tau=df["tau"].to_numpy())
        print(grid, "episodes", len(ep), "windows", len(df), flush=True)
        tok = df[LOCAL_TOKENS + TD_TOKENS]
        print("  non-finite tokens:", int((~np.isfinite(tok.to_numpy())).sum()))


if __name__ == "__main__":
    main()
