"""C1: time-domain aerial-mode two-ended locator on the OFFICIAL window set (16 windows per SHC episode,
tau = 5..80 ms; 10 per incipient episode, tau = 5..50 ms; matches the published n_test = 14,640).

Series R-L line model per aerial (Clarke alpha/beta) mode, identified positive-sequence R1, L1:
  v_S - v_R = R (d i_S - (1-d) i_R) + L (d i_S' - (1-d) i_R')
  => a_k = v_S - v_R + R i_R + L i_R'  =  d * b_k,   b_k = R (i_S + i_R) + L (i_S' + i_R')
  d_hat = sum_k a_k b_k / sum_k b_k^2  over ALL samples of the window and both aerial modes.
NO inception time is used (pre-fault samples have b_k ~ 0 and carry little weight). R1, L1 per line from
the median pre-fault identification (normal-operation calibration). Derivatives: central differences.
Also reported: the phasor locator (window-end cycle) for windows with >= 21 ms post-fault, which is
likewise inception-free.
"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import D, FS, F0, T0, ROOT, TAG, terminal_cols  # noqa: E402
from window_eval import zline, WLEN, STEP  # noqa: E402

W0 = 2 * np.pi * F0


def clarke(a, b, c):
    return (2 * a - b - c) / 3.0, (b - c) / np.sqrt(3.0)


def ddt(x):
    g = np.empty_like(x)
    g[1:-1] = (x[2:] - x[:-2]) * FS / 2
    g[0] = (x[1] - x[0]) * FS
    g[-1] = (x[-1] - x[-2]) * FS
    return g


def one(args):
    row, Zmap = args
    line = row["events/event_target"]
    m = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line)
    f = D / "data" / f"result{int(row['general/sim_idx'])}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    X = {k: d[k].to_numpy(float) for k in cs + cr}
    Z = Zmap[line]
    R, L = Z.real, Z.imag / W0
    modes = []
    for (iS, vS), (iR, vR) in [((clarke(*[X[c] for c in cs[:3]]), clarke(*[X[c] for c in cs[3:]])),
                                (clarke(*[X[c] for c in cr[:3]]), clarke(*[X[c] for c in cr[3:]])))]:
        for mi in range(2):
            a = vS[mi] - vR[mi] + R * iR[mi] + L * ddt(iR[mi])
            b = R * (iS[mi] + iR[mi]) + L * (ddt(iS[mi]) + ddt(iR[mi]))
            modes.append((a, b))
    nf = int(round((row["events/event_start"] - T0) * FS))
    inc = "incipient" in row["events/event_type"]
    taus = range(5, 51, 5) if inc else range(5, 81, 5)
    out = []
    for tau in taus:
        s1 = nf + int(round(tau / 1000 * FS))
        s0 = s1 - WLEN
        num = sum(float(np.dot(a[s0:s1], b[s0:s1])) for a, b in modes)
        den = sum(float(np.dot(b[s0:s1], b[s0:s1])) for a, b in modes) + 1e-12
        out.append(dict(sim_idx=int(row["general/sim_idx"]), etype=row["events/event_type"], tau_ms=tau,
                        loc_true=row["events/event_flt_target_line_location"], loc_td=100 * num / den))
    return out


def main():
    Zmap = zline()
    s = pd.read_csv(D / "labels" / "settings_clean.csv")
    fl = s[s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()
           & s["events/event_type"].str.startswith("flt_")]
    with ProcessPoolExecutor(12) as ex:
        res = [r for ch in ex.map(one, [(r, Zmap) for _, r in fl.iterrows()], chunksize=8) for r in ch]
    df = pd.DataFrame(res)
    df["err_td"] = (df.loc_td.clip(0, 100) - df.loc_true).abs()
    ph = pd.read_csv(ROOT / "results" / f"c1_window_eval{TAG}.csv")[["sim_idx", "tau_ms", "loc_est"]]
    ph["tau_ms"] = ph.tau_ms.round(1)
    df["tau_ms"] = df.tau_ms.astype(float).round(1)
    df = df.merge(ph, on=["sim_idx", "tau_ms"], how="left")
    df["err_ph"] = (df.loc_est.clip(0, 100) - df.loc_true).abs()
    # hybrid, inception-free: phasor if tau >= 21 ms else time-domain. In deployment tau is unknown;
    # this row is diagnostic only. The deployable rule is pure time-domain (err_td).
    df["err_hyb"] = np.where(df.tau_ms >= 21, df.err_ph, df.err_td)
    df.to_csv(ROOT / "results" / f"c1_td_official{TAG}.csv", index=False)
    print(f"official windows: {len(df)}")
    for c in ("err_td", "err_ph", "err_hyb"):
        print(f"{c}: MAE {df[c].mean():.3f}  median {df[c].median():.3f}")
    print(df.groupby("tau_ms")[["err_td", "err_ph"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
