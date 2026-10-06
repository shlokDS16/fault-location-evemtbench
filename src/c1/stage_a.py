"""C1 Stage A (research/C1_GATE.md): parameter-free two-ended positive-sequence fault locator on raw
EvEMTBench DoubleLine episodes. No learning; line impedance identified per episode from pre-fault phasors.

Conventions (fixed before looking at labels):
  - The terminals of line 'MainLnX-YZ' are the cubicles 'MainBusX_MainLnX-YZ' (S) and 'MainBusY_MainLnX-YZ' (R).
  - Currents are the cubicle currents (bus -> line assumed); location d is measured from S (bus X).
    The alternative orientation is reported as a check, never mixed in.
  - Phasors: one-cycle DFT (N = fs/f = 192). Pre-fault cycle ends one cycle before inception; post-fault
    cycle starts one cycle after inception (skips the first-cycle transient).
Output: results/c1_stage_a.csv (per episode) + summary printed.
"""
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRID = os.environ.get("EVEMT_GRID", "DoubleLine")
SNR = os.environ.get("EVEMT_SNR")  # optional test-time noise (dB, relative to each channel's pre-fault RMS)
TAG = ("" if GRID == "DoubleLine" else f"_{GRID}") + (f"_SNR{SNR}" if SNR else "")


def add_noise(arrs, sid):
    """AWGN per channel at EVEMT_SNR dB relative to the channel's first-90-ms RMS; seeded per episode."""
    if not SNR:
        return arrs
    rng = np.random.default_rng(int(sid) * 1000 + int(SNR))
    out = []
    for x in arrs:
        rms = np.sqrt(np.mean(x[: int(0.09 * FS)] ** 2)) + 1e-12
        out.append(x + rng.normal(0, rms / 10 ** (int(SNR) / 20), size=x.shape))
    return out
D = ROOT / "data" / "raw" / "evemt" / GRID
FS, F0, T0 = 9600.0, 50.0, 1.0
A = np.exp(2j * np.pi / 3)


def phasor(x, start, N):
    seg = x[start:start + N]
    n = np.arange(N)
    return (2.0 / N) * np.sum(seg * np.exp(-2j * np.pi * n / N))


def pos_seq(pa, pb, pc):
    return (pa + A * pb + A * A * pc) / 3.0


def terminal_cols(cols, bus, line):
    base = [c for c in cols if re.search(rf"pex_MainBus{bus}_{re.escape(line)}$", c)]
    assert len(base) == 1, (bus, line, base)
    b = base[0]
    return [b] + [f"{b}.{k}" for k in range(1, 6)]  # Isec A,B,C, Usec A,B,C


def locate(row, post_skip_cycles=1.0):
    line = row["events/event_target"]
    m = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line)
    bs, br = m.group(1), m.group(2)
    f = D / "data" / f"result{int(row['general/sim_idx'])}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs, cr = terminal_cols(hdr, bs, line), terminal_cols(hdr, br, line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    X = {k: d[k].to_numpy(float) for k in cs + cr}
    N = int(round(FS / F0))
    n0 = int(round((row["events/event_start"] - T0) * FS))
    pre = n0 - 2 * N
    post = n0 + int(round(post_skip_cycles * N))

    def seq(cols, start):
        I = pos_seq(*(phasor(X[c], start, N) for c in cols[:3]))
        V = pos_seq(*(phasor(X[c], start, N) for c in cols[3:]))
        return V, I

    VSp, ISp = seq(cs, pre); VRp, IRp = seq(cr, pre)
    VSf, ISf = seq(cs, post); VRf, IRf = seq(cr, post)
    Z = (VSp - VRp) / ((ISp - IRp) / 2.0)          # series impedance, secondary units (identified)
    dS = (VSf - VRf + Z * IRf) / (Z * (ISf + IRf))  # fraction from S
    return dict(sim_idx=int(row["general/sim_idx"]), line=line, etype=row["events/event_type"],
                R=row["events/event_flt_shc_resistance"], loc_true=row["events/event_flt_target_line_location"],
                loc_est=100 * float(np.real(dS)), loc_est_imag=100 * float(np.imag(dS)),
                Z_re=float(np.real(Z)), Z_im=float(np.imag(Z)), preload_ratio=float(abs(ISp + IRp) / (abs(ISp) + 1e-9)),
                Ipre=float(abs(ISp)), Ipost=float(abs(ISf)))


def _job(args):
    row, skip = args
    try:
        return locate(row, skip)
    except Exception as e:  # keep going; report failures
        return dict(sim_idx=int(row["general/sim_idx"]), error=repr(e)[:200])


def main(skip=1.0):
    s = pd.read_csv(D / "labels" / "settings_clean.csv")
    fl = s[s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()
           & s["events/event_type"].str.startswith("flt_")]
    rows = [(r, skip) for _, r in fl.iterrows()]
    with ProcessPoolExecutor(12) as ex:
        out = list(ex.map(_job, rows, chunksize=8))
    df = pd.DataFrame(out)
    tag = str(skip).replace(".", "p")
    df.to_csv(ROOT / "results" / f"c1_stage_a_skip{tag}{TAG}.csv", index=False)
    if "error" in df:
        print("errors:", df["error"].notna().sum())
        df = df[df["error"].isna()] if df["error"].notna().any() else df
    df["err"] = (df.loc_est.clip(0, 100) - df.loc_true).abs()
    df["err_rev"] = ((100 - df.loc_est).clip(0, 100) - df.loc_true).abs()
    print(f"n={len(df)}  MAE={df.err.mean():.2f}  median={df.err.median():.2f}  (reverse orientation MAE={df.err_rev.mean():.2f})")
    print(df.groupby("etype").err.agg(["count", "mean", "median"]).round(2).to_string())
    print(df.groupby("R").err.agg(["mean", "median"]).round(2).to_string())
    print(df.groupby("loc_true").err.agg(["mean", "median"]).round(2).to_string())
    print("preload ratio |IS+IR|/|IS| (pre-fault, should be ~0 for series model):", df.preload_ratio.describe().round(3).to_dict())


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
