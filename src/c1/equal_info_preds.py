"""Paper plan step 2: re-run the zero-shot equal-information baselines (equal_info.py, unchanged recipe and seeds) and SAVE
per-window predictions for the time-since-inception figure. GPU training is not bit-deterministic, so the MAE of each
re-run is compared with the recorded row in results/c1_equal_info.csv and the difference is reported.
Usage: python src/c1/equal_info_preds.py  -> results/c1_equal_info_preds.parquet, c1_equal_info_preds_check.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equal_info import load, fit_predict, R  # noqa: E402


def main():
    data = {"DL": load(""), "TG": load("_TestGrid110kV")}
    rec = pd.read_csv(R / "c1_equal_info.csv")
    out, chk = [], []
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        Xtr, Ztr, ytr, gtr, _ = data[src]
        Xte, Zte, yte, ste, tte = data[tgt]
        base = pd.DataFrame(dict(direction=f"{src}_to_{tgt}", sim_idx=ste, tau=tte, y=yte))
        for model_name in ("MLP", "GRU"):
            for use_z in (False, True):
                var = "+Z" if use_z else "-raw"
                P = []
                for seed in (0, 1, 2):
                    p, ep = fit_predict(model_name, use_z, Xtr, Ztr, ytr, gtr, Xte, Zte, seed)
                    P.append(p)
                    old = rec[(rec.model == model_name) & (rec.variant == var) & (rec.direction == f"{src}_to_{tgt}")
                              & (rec.seed == seed)].mae.iloc[0]
                    new = float(np.mean(np.abs(p - yte)) * 100)
                    chk.append(dict(model=model_name, variant=var, direction=f"{src}_to_{tgt}", seed=seed,
                                    mae_recorded=old, mae_rerun=new, delta=new - old, epochs=ep))
                    print(chk[-1], flush=True)
                b = base.copy()
                b["method"] = f"{model_name}{var}"
                for s, p in enumerate(P):
                    b[f"pred_s{s}"] = p
                out.append(b)
    pd.concat(out).to_parquet(R / "c1_equal_info_preds.parquet", index=False)
    c = pd.DataFrame(chk)
    c.to_csv(R / "c1_equal_info_preds_check.csv", index=False)
    print(c.groupby(["model", "variant", "direction"])[["mae_recorded", "mae_rerun"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
