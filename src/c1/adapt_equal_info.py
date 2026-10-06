"""Equal-information in-grid baselines (design section 17a): the benchmark MLP/GRU recipe (equal_info.fit_predict,
unchanged), -raw and +Z, trained on adapt_grid TRAIN line-fault windows, tested on adapt_grid TEST line-fault
windows and the benchmark family (14,640). 3 seeds. Resumable: one row per (model, variant, seed) appended to
results/c1_adapt_equal_info.csv. GPU job: run only when no other GPU job is running."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equal_info import fit_predict  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
OUT = R / "c1_adapt_equal_info.csv"


def main():
    a = np.load(R / "c1_adapt_raw.npz")
    te = a["is_test"] == 1
    tr = ~te
    b = np.load(R / "c1_raw.npz")
    tests = {"adapt_test": (a["X"][te], a["Z"][te], a["y"][te], a["post_ms"][te]),
             "benchmark": (b["X"], b["Z"], b["y"], b["tau"])}
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev.model, prev.variant, prev.seed))
    for model in ("MLP", "GRU"):
        for use_z in (False, True):
            var = "+Z" if use_z else "-raw"
            for seed in (0, 1, 2):
                if (model, var, seed) in done:
                    continue
                t0 = time.time()
                Xte = np.concatenate([tests[k][0] for k in tests])
                Zte = np.concatenate([tests[k][1] for k in tests])
                p, ep = fit_predict(model, use_z, a["X"][tr], a["Z"][tr], a["y"][tr].astype(np.float32), a["sim_idx"][tr],
                                    Xte, Zte, seed)
                row = dict(model=model, variant=var, seed=seed, epochs=ep, sec=time.time() - t0)
                i = 0
                for k, (X, _, y, pm) in tests.items():
                    err = np.abs(p[i:i + len(X)] - y) * 100
                    i += len(X)
                    row.update({f"{k}_mae": err.mean(), f"{k}_median": np.median(err), f"{k}_post_le15": err[pm <= 15].mean(),
                                f"{k}_post_ge21": err[pm >= 21].mean()})
                pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                print(row, flush=True)
    df = pd.read_csv(OUT)
    print(df.groupby(["model", "variant"]).agg(["mean", "std"]).filter(like="_mae").round(2).to_string())


if __name__ == "__main__":
    main()
