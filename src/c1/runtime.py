"""Design section 25(e), report-only and indicative: computational cost per window on one CPU core (this laptop).
Records of the first 50 DoubleLine line-fault episodes are read into memory first (CSV reading is excluded); then the
two-ended TD estimate, the 77 local features, the classical one-ended estimates and H-A2 inference are timed.
Usage: python src/c1/runtime.py  -> results/c1_runtime.csv"""
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_pass  # noqa: E402
from classical_oe import episode  # noqa: E402
from local_features import oracle_loop  # noqa: E402
from review_checks import jobs, td_est, W0  # noqa: E402
from stage_a import terminal_cols  # noqa: E402
from tokens_core import ALLOW, episode_tokens  # noqa: E402
from ha2_eval import PARAMS  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
N_EP = 50


def load(js):
    eps = []
    for row, LP, gdir in js:
        line = row["events/event_target"]
        m = grid_pass.LINE_RE.match(line)
        f = gdir / "data" / f"result{int(row['general/sim_idx'])}.csv"
        hdr = pd.read_csv(f, nrows=0).columns.tolist()
        cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
        d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
        XS, XR = [d[c].to_numpy(float) for c in cs], [d[c].to_numpy(float) for c in cr]
        nf, ends = grid_pass.window_ends(row, len(XS[0]))
        eps.append((row, XS, XR, nf, ends, LP[line][0], LP[line][1]))
    return eps


def timed(fn, eps):
    nwin, t0 = 0, time.perf_counter()
    for e in eps:
        fn(e)
        nwin += len(e[4])
    return (time.perf_counter() - t0) / nwin * 1e3, nwin


def main():
    eps = load(jobs("DL")[:N_EP])
    rows = []
    for _ in range(2):  # first pass warms caches; the second is reported
        ms_two, n = timed(lambda e: td_est(e[1], e[2], e[5].real, e[5].imag / W0, e[4]), eps)
        ms_feat, _ = timed(lambda e: episode_tokens(e[1], e[5], e[6], e[4]), eps)
        ms_cl, _ = timed(lambda e: episode(e[1], e[2], e[5], e[6], e[3], e[4], oracle_loop(e[0]),
                                           e[0]["events/event_type"] == "flt_3ph_shc"), eps)
    tr = pd.read_parquet(R / "c1_grid_TG.parquet")
    te = pd.read_parquet(R / "c1_grid_DL.parquet")
    t0 = time.perf_counter()
    models = [HistGradientBoostingRegressor(**PARAMS, random_state=s).fit(tr[ALLOW], tr.y) for s in (0, 1, 2)]
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    np.mean([mdl.predict(te[ALLOW]) for mdl in models], 0)
    ms_inf = (time.perf_counter() - t0) / len(te) * 1e3
    cpu = platform.processor() or platform.machine()
    rows = [dict(item="two-ended TD estimate", ms_per_window=ms_two, windows=n),
            dict(item="77 local features", ms_per_window=ms_feat, windows=n),
            dict(item="classical one-ended (R, T, T2, MT, E)", ms_per_window=ms_cl, windows=n),
            dict(item="H-A2 inference (3 models, batch)", ms_per_window=ms_inf, windows=len(te)),
            dict(item="H-A2 training on TG (3 models), s", ms_per_window=t_train, windows=len(tr))]
    out = pd.DataFrame(rows).assign(cpu=cpu, threads=os.environ.get("OMP_NUM_THREADS"))
    out.to_csv(R / "c1_runtime.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
