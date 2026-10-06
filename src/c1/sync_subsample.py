"""Design 24a (report-only): sub-sample synchronization error {1, 10, 50} us between the two ends (fractional delay of the
remote record by an FFT phase shift), two-ended TD and phasor locators, DL and TG.
Usage: python src/c1/sync_subsample.py  -> results/c1_review_sync_us.csv"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_pass  # noqa: E402
from review_checks import td_est, pos, jobs, mae, N, W0  # noqa: E402
from stage_a import terminal_cols  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
FS = 9600.0
DELAYS_US = (0, 1, 10, 50)


def delay(x, tau):
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * tau), n=len(x))


def one(args):
    row, LP, gdir = args
    line = row["events/event_target"]
    m = grid_pass.LINE_RE.match(line)
    f = gdir / "data" / f"result{int(row['general/sim_idx'])}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    XS, XR0 = [d[c].to_numpy(float) for c in cs], [d[c].to_numpy(float) for c in cr]
    nf, ends = grid_pass.window_ends(row, len(XS[0]))
    Z1 = LP[line][0]
    R1, L1 = Z1.real, Z1.imag / W0
    y = row["events/event_flt_target_line_location"] / 100.0
    rec = [dict(post_ms=(s1 - nf) / FS * 1000, y=y) for s1 in ends]
    for us in DELAYS_US:
        XR = [delay(x, us * 1e-6) for x in XR0] if us else XR0
        for r, v in zip(rec, td_est(XS, XR, R1, L1, ends)):
            r[f"td_{us}us"] = v
        for s1, r in zip(ends, rec):
            VS, IS, VR, IR = pos(XS[3:], s1 - N), pos(XS[:3], s1 - N), pos(XR[3:], s1 - N), pos(XR[:3], s1 - N)
            r[f"ph_{us}us"] = float(((VS - VR + Z1 * IR) / (Z1 * (IS + IR))).real)
    return rec


def main():
    rows = []
    for tag in ("DL", "TG"):
        with ProcessPoolExecutor(6) as ex:
            d = pd.DataFrame([r for rr in ex.map(one, jobs(tag), chunksize=4) for r in rr])
        r = dict(grid=tag, n=len(d))
        for c in d.columns:
            if c.endswith("us"):
                r[c] = mae(d[c], d.y.values)
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(R / "c1_review_sync_us.csv", index=False)


if __name__ == "__main__":
    main()
