"""Paper Fig 4 data: zero-shot MAE vs time since inception (5-80 ms) on the official DL / TG windows for the two-ended
locator, classical one-ended (R, T, E; design 21/21b), H-A2 (3-seed mean prediction, as reported) and the
equal-information MLP / GRU (-raw, +Z; mean of the three single-seed MAEs, as reported in Tab. III).
Usage: python src/c1/fig4_data.py  -> results/c1_fig4_data.csv (+ printed table)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classical_report import zs_pred  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"


def main():
    eq = pd.read_parquet(R / "c1_equal_info_preds.parquet")
    rows = []
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        dirn = f"{src}_to_{tgt}"
        d = pd.read_parquet(R / f"c1_classical_{tgt}.parquet")
        d["k"] = d.post_ms.round(3)
        g = pd.read_parquet(R / f"c1_grid_{tgt}.parquet")[["sim_idx", "post_ms", "td2_est"]]
        d = d.merge(g.assign(k=g.post_ms.round(3)).drop(columns="post_ms"), on=["sim_idx", "k"], validate="one_to_one")
        h = zs_pred(src, tgt)
        d = d.merge(h.assign(k=h.post_ms.round(3)).drop(columns="post_ms"), on=["sim_idx", "k"], validate="one_to_one")
        e = eq[eq.direction == dirn]
        for m, gm in e.groupby("method"):
            err = np.mean([np.abs(gm[f"pred_s{s}"].values - gm.y.values) for s in range(3)], 0) * 100
            x = pd.DataFrame(dict(sim_idx=gm.sim_idx.values, k=gm.tau.astype(float).round(3).values, **{m: err}))
            d = d.merge(x, on=["sim_idx", "k"], how="left", validate="one_to_one")
            assert d[m].notna().all(), (m, d[m].isna().sum())
        d["t"] = d.post_ms.round().astype(int)
        cols = {"two-ended": "td2_est", "reactance": "R", "Takagi": "T", "Eriksson": "E", "H-A2": "pred"}
        for nm, c in cols.items():
            d[nm] = (d[c].fillna(0.5).clip(0, 1) - d.y).abs() * 100
        meths = list(cols) + sorted(e.method.unique())
        tab = d.groupby("t")[meths].mean()
        tab.loc["all"] = d[meths].mean()
        tab.insert(0, "n", list(d.groupby("t").size()) + [len(d)])
        tab.insert(0, "direction", dirn)
        rows.append(tab.reset_index())
        print(f"\n== {dirn}\n" + tab.drop(columns="direction").round(2).to_string())
    pd.concat(rows).to_csv(R / "c1_fig4_data.csv", index=False)


if __name__ == "__main__":
    main()
