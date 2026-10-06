"""Part 3 Part B: compare frozen in-grid neural numbers with results/c1_adapt_equal_info.csv."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

R = C.RESULTS
me = pd.read_csv(R / "validate_ha2_ingrid_neural_runs.csv")
le = pd.read_csv(R / "c1_adapt_equal_info.csv")
rows = []
for (m, v), g in me.groupby(["model", "variant"]):
    lv = "-raw" if v == "raw" else "+Z"
    lg = le[(le.model == m) & (le.variant == lv)]
    assert len(lg) == 3
    for t, lt in (("adapt_test", "adapt_test"), ("benchmark_DL", "benchmark")):
        gg = g[g.test == t]
        assert len(gg) == 3
        row = {"model": m, "variant": lv, "test": t}
        for k, lk in (("MAE", "mae"), ("median", "median"), ("MAE_pft_le15", "post_le15"), ("MAE_pft_ge21", "post_ge21")):
            row[f"{k}_mine"] = gg[k].mean()
            row[f"{k}_mine_sd"] = gg[k].std(ddof=1)
            row[f"{k}_lead"] = lg[f"{lt}_{lk}"].mean()
            row[f"{k}_lead_sd"] = lg[f"{lt}_{lk}"].std(ddof=1)
        d = row["MAE_mine"] - row["MAE_lead"]
        row["delta"] = d
        row["verdict"] = "REPRODUCED" if abs(d) <= 1.0 or abs(d) <= 2 * max(row["MAE_mine_sd"], row["MAE_lead_sd"]) else "NOT REPRODUCED"
        row["best_epochs_mine"] = "/".join(map(str, gg.sort_values("seed").best_epoch))
        row["epochs_lead"] = "/".join(map(str, lg.sort_values("seed").epochs))
        rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(R / "validate_ha2_ingrid_neural_partB_compare.csv", index=False)
with pd.option_context("display.width", 300, "display.max_columns", 40):
    print(out.round(3).to_string(index=False))
