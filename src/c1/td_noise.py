"""C1 robustness: time-domain aerial-mode locator under additive white Gaussian noise (official windows).
Noise per channel at SNR (dB) relative to that channel's pre-fault RMS (first 90 ms), seeded per episode.
Derivative: 'cd' central difference or 'sg' Savitzky-Golay (window 11 samples ~1.1 ms, order 2).
Usage: python src/c1/td_noise.py  -> results/c1_td_noise.csv + summary."""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import D, FS, F0, T0, ROOT, TAG, terminal_cols  # noqa: E402
from window_eval import zline, WLEN  # noqa: E402
from td_locator import clarke, ddt  # noqa: E402

W0 = 2 * np.pi * F0
SNRS = (None, 60, 40, 30)
DERIVS = ("cd", "sg")


def deriv(x, how):
    return ddt(x) if how == "cd" else savgol_filter(x, 11, 2, deriv=1, delta=1 / FS)


def one(args):
    row, Zmap = args
    line = row["events/event_target"]
    m = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line)
    sid = int(row["general/sim_idx"])
    f = D / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    X0 = {k: d[k].to_numpy(float) for k in cs + cr}
    Z = Zmap[line]
    R, L = Z.real, Z.imag / W0
    nf = int(round((row["events/event_start"] - T0) * FS))
    inc = "incipient" in row["events/event_type"]
    taus = range(5, 51, 5) if inc else range(5, 81, 5)
    out = []
    for snr in SNRS:
        rng = np.random.default_rng(sid * 1000 + (snr or 0))
        X = {}
        for k, x in X0.items():
            if snr is None:
                X[k] = x
            else:
                rms = np.sqrt(np.mean(x[: int(0.09 * FS)] ** 2)) + 1e-12
                X[k] = x + rng.normal(0, rms / 10 ** (snr / 20), size=x.shape)
        for how in DERIVS:
            iS, vS = clarke(*[X[c] for c in cs[:3]]), clarke(*[X[c] for c in cs[3:]])
            iR, vR = clarke(*[X[c] for c in cr[:3]]), clarke(*[X[c] for c in cr[3:]])
            modes = []
            for mi in range(2):
                a = vS[mi] - vR[mi] + R * iR[mi] + L * deriv(iR[mi], how)
                b = R * (iS[mi] + iR[mi]) + L * (deriv(iS[mi], how) + deriv(iR[mi], how))
                modes.append((a, b))
            for tau in taus:
                s1 = nf + int(round(tau / 1000 * FS)); s0 = s1 - WLEN
                num = sum(float(np.dot(a[s0:s1], b[s0:s1])) for a, b in modes)
                den = sum(float(np.dot(b[s0:s1], b[s0:s1])) for a, b in modes) + 1e-12
                out.append(dict(sim_idx=sid, snr=snr if snr else 999, deriv=how, tau_ms=tau,
                                loc_true=row["events/event_flt_target_line_location"], loc_est=100 * num / den))
    return out


def main():
    Zmap = zline()
    s = pd.read_csv(D / "labels" / "settings_clean.csv")
    fl = s[s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()
           & s["events/event_type"].str.startswith("flt_")]
    with ProcessPoolExecutor(12) as ex:
        res = [r for ch in ex.map(one, [(r, Zmap) for _, r in fl.iterrows()], chunksize=8) for r in ch]
    df = pd.DataFrame(res)
    df["err"] = (df["loc_est"].clip(0, 100) - df["loc_true"]).abs()
    df.to_csv(ROOT / "results" / f"c1_td_noise{TAG}.csv", index=False)
    print(df.pivot_table(index="snr", columns="deriv", values="err", aggfunc="mean").round(3).to_string())
    print(df[df.tau_ms == 5].pivot_table(index="snr", columns="deriv", values="err", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
