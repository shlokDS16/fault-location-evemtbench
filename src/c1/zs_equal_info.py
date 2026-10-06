"""Equal-information zero-shot baselines on a new grid (design section 18): the benchmark MLP/GRU recipe
(equal_info.fit_predict, unchanged), -raw and +Z, trained on the pooled source grids' raw local windows
(results/c1_grid_<tag>_raw.npz), tested on the target grid. 3 seeds, resumable.
Usage: python src/c1/zs_equal_info.py <target_tag> <source_tag> [...] -> results/c1_zs_equal_info_<target>.csv
GPU job: run only when no other GPU job is running."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equal_info import fit_predict  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"


def main():
    tgt, srcs = sys.argv[1], sys.argv[2:]
    out = R / f"c1_zs_equal_info_{tgt}.csv"
    S = [np.load(R / f"c1_grid_{s}_raw.npz") for s in srcs]
    Xtr = np.concatenate([s["X"] for s in S])
    Ztr = np.concatenate([s["Z"] for s in S])
    ytr = np.concatenate([s["y"] for s in S]).astype(np.float32)
    gtr = np.concatenate([s["sim_idx"] + 10_000_000 * i for i, s in enumerate(S)])  # episode groups unique per grid
    T = np.load(R / f"c1_grid_{tgt}_raw.npz")
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.model, prev.variant, prev.seed))
    for model in ("MLP", "GRU"):
        for use_z in (False, True):
            var = "+Z" if use_z else "-raw"
            for seed in (0, 1, 2):
                if (model, var, seed) in done:
                    continue
                t0 = time.time()
                p, ep = fit_predict(model, use_z, Xtr, Ztr, ytr, gtr, T["X"], T["Z"], seed)
                err = np.abs(p - T["y"]) * 100
                pm = T["post_ms"]
                row = dict(sources="+".join(srcs), model=model, variant=var, seed=seed, mae=err.mean(), median=np.median(err),
                           post_le15=err[pm <= 15].mean(), post_ge21=err[pm >= 21].mean(), epochs=ep, sec=time.time() - t0)
                pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(row, flush=True)
    df = pd.read_csv(out)
    print(df.groupby(["model", "variant"])[["mae", "median", "post_le15", "post_ge21"]].agg(["mean", "std"]).round(2).to_string())


if __name__ == "__main__":
    main()
