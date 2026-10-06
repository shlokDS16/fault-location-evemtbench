"""Part 9 comparison (run AFTER the freeze): my frozen numbers against results/c1_revamp_stats.csv, plus a diagnostic
with the lead's 17.5 ms bin edges (le15 = post <= 17.5, ge20 / switch = post > 17.5) on my own per-window table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

R = C.RESULTS
mine = pd.read_csv(R / "validate_ha2_part9_summary.csv")
lead = pd.read_csv(R / "c1_revamp_stats.csv")
mine["stat"] = mine["stat"].str.replace("MAE", "mae").replace({"mean_m": "mae_m", "rel_red": "reduction"})
sel = mine["method"] == "two_zidRX"  # identified R1 AND X1 = the lead's c1_grid_MV_zid definition
mine.loc[mine["method"] == "two_zid", "method"] = "two_zid_R1only"
mine.loc[sel, "method"] = "two_zid"
m = mine.copy()
m.loc[m["method"] == "ha2_vs_eq", "value"] *= 100
m.loc[m["stat"].str.startswith("coverage"), "value"] *= 100
cmp = lead.merge(m, on=["set", "method", "stat"], how="left", suffixes=("_lead", "_mine"))
cmp["delta_mine_minus_lead"] = cmp["value_mine"] - cmp["value_lead"]

# diagnostic with the lead's edges on my per-window table
pw = pd.read_parquet(R / "validate_ha2_part9_perwindow.parquet")
rows = []
for g, d in pw.groupby("grid"):
    p = d["pft_ms"].to_numpy()
    for nm, k in (("le15", p <= 17.5), ("ge20", p > 17.5)):
        dk = k & d["E_def"].to_numpy()
        rows.append({"grid": g, "stat": f"E_defined mae_{nm} (lead edges)", "value": d["e_E"].to_numpy()[dk].mean()})
        rows.append({"grid": g, "stat": f"E_defined coverage_{nm} (lead edges)", "value": dk.sum() / k.sum() * 100})
    sw = np.where((p > 17.5) & d["E_def"].to_numpy(), d["e_E"], d["e_ha2"])
    rows.append({"grid": g, "stat": "switch mae (lead edges)", "value": sw.mean()})
    rows.append({"grid": g, "stat": "windows with 15 < post < 20 ms", "value": ((p > 15) & (p < 20)).sum()})
diag = pd.DataFrame(rows)
cmp.to_csv(R / "validate_ha2_part9cmp.csv", index=False)
diag.to_csv(R / "validate_ha2_part9cmp_diag.csv", index=False)
with pd.option_context("display.width", 200, "display.max_rows", 500):
    print(cmp.round(4).to_string())
    print(diag.round(4).to_string())
