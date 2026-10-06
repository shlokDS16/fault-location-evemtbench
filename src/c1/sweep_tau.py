"""C1 diagnostic: two-ended location error vs available post-fault duration tau (sub-cycle phasors).
Post-fault phasors: least-squares fit of a*cos(wt) + b*sin(wt) + c + e*t on samples in
[inception + 1 ms, inception + tau]. Pre-fault Z: one-cycle DFT before inception (relay history assumed).
Output: results/c1_sweep_tau.csv (per episode x tau) and a summary."""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import D, FS, F0, T0, ROOT, phasor, pos_seq, terminal_cols  # noqa: E402

TAUS_MS = (3, 5, 7.5, 10, 15, 20)
W = 2 * np.pi * F0


def ls_phasor(x, i0, i1):
    t = np.arange(i0, i1) / FS
    M = np.stack([np.cos(W * t), np.sin(W * t), np.ones_like(t), t - t[0]], 1)
    coef, *_ = np.linalg.lstsq(M, x[i0:i1], rcond=None)
    return coef[0] - 1j * coef[1]  # phasor in the absolute-time reference


def one(row):
    line = row["events/event_target"]
    m = re.match(r"MainLn(\d)-(\d)[AB]$", line)
    f = D / "data" / f"result{int(row['general/sim_idx'])}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    X = {k: d[k].to_numpy(float) for k in cs + cr}
    N = int(round(FS / F0))
    n0 = int(round((row["events/event_start"] - T0) * FS))

    def seq_dft(cols, start):
        return (pos_seq(*(phasor(X[c], start, N) for c in cols[3:])),
                pos_seq(*(phasor(X[c], start, N) for c in cols[:3])))

    VSp, ISp = seq_dft(cs, n0 - 2 * N)
    VRp, IRp = seq_dft(cr, n0 - 2 * N)
    Z = (VSp - VRp) / ((ISp - IRp) / 2.0)
    out = []
    i0 = n0 + int(round(0.001 * FS))
    for tau in TAUS_MS:
        i1 = n0 + int(round(tau / 1000 * FS))

        def seq_ls(cols):
            return (pos_seq(*(ls_phasor(X[c], i0, i1) for c in cols[3:])),
                    pos_seq(*(ls_phasor(X[c], i0, i1) for c in cols[:3])))
        VS, IS = seq_ls(cs)
        VR, IR = seq_ls(cr)
        dS = (VS - VR + Z * IR) / (Z * (IS + IR))
        out.append(dict(sim_idx=int(row["general/sim_idx"]), etype=row["events/event_type"], tau_ms=tau,
                        loc_true=row["events/event_flt_target_line_location"], loc_est=100 * float(np.real(dS))))
    return out


def main():
    s = pd.read_csv(D / "labels" / "settings_clean.csv")
    fl = s[s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()
           & s["events/event_type"].str.startswith("flt_")]
    with ProcessPoolExecutor(12) as ex:
        res = [r for chunk in ex.map(one, [r for _, r in fl.iterrows()], chunksize=8) for r in chunk]
    df = pd.DataFrame(res)
    df["err"] = (df.loc_est.clip(0, 100) - df.loc_true).abs()
    df.to_csv(ROOT / "results" / "c1_sweep_tau.csv", index=False)
    print(df.groupby("tau_ms").err.agg(["count", "mean", "median", lambda x: np.percentile(x, 90)]).round(2).to_string())
    print(df.pivot_table(index="etype", columns="tau_ms", values="err", aggfunc="mean").round(2).to_string())


if __name__ == "__main__":
    main()
