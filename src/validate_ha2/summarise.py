"""Step 5: freeze the Part A numbers into results/validate_ha2_summary.csv (written before Part B)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

METRICS = ["MAE", "median", "MAE_tau_le15", "MAE_tau_ge21", "MAE_1phg_incipient",
           "MAE_1phg_incipient_w_arc", "MAE_1phg_shc", "MAE_1phg_shc_w_arc", "MAE_2ph_shc",
           "MAE_2phg_shc", "MAE_3ph_shc"]


def main() -> None:
    hp = pd.read_csv(C.RESULTS / "validate_ha2_summary_hgb_phys.csv")
    rows = []
    for r in hp.to_dict("records"):
        if r["seed"] != "avg":
            continue
        out = {"method": r["method"], "direction": r["direction"], "n_seeds": 3 if r["method"].startswith("HA") else 0}
        for m in METRICS:
            out[m] = r[m]
            out[m + "_sd"] = float("nan")
        rows.append(out)
    # per-seed spread of the single-seed HGB models (the reported number is the 3-seed ensemble)
    for (meth, d), g in hp[hp["seed"] != "avg"].groupby(["method", "direction"]):
        for out in rows:
            if out["method"] == meth and out["direction"] == d:
                out["single_seed_MAE_mean"] = g["MAE"].mean()
                out["single_seed_MAE_sd"] = g["MAE"].std(ddof=1)
    nn_ = pd.read_csv(C.RESULTS / "validate_ha2_neural_runs.csv")
    nn_ = nn_.drop_duplicates(["model", "variant", "direction", "seed"], keep="last")
    for (mdl, var, d), g in nn_.groupby(["model", "variant", "direction"]):
        assert len(g) == 3, (mdl, var, d, len(g))
        out = {"method": f"{mdl}_{'raw' if var == 'raw' else '+Z'}", "direction": d, "n_seeds": len(g)}
        for m in METRICS + ["MAE_unclipped"]:
            out[m] = g[m].mean()
            out[m + "_sd"] = g[m].std(ddof=1)
        out["best_epochs"] = "/".join(str(int(x)) for x in g.sort_values("seed")["best_epoch"])
        rows.append(out)
    s = pd.DataFrame(rows)
    gains = []
    for d, g in s.groupby("direction"):
        ei = g[g["method"].str.startswith(("MLP", "GRU"))]
        best = ei.loc[ei["MAE"].idxmin()]
        ha2 = float(g.loc[g["method"] == "HA2", "MAE"].iloc[0])
        gains.append({"direction": d, "best_equal_info": best["method"], "best_equal_info_MAE": best["MAE"],
                      "HA2_MAE": ha2, "relative_gain": (best["MAE"] - ha2) / best["MAE"]})
    s.to_csv(C.RESULTS / "validate_ha2_summary.csv", index=False)
    pd.DataFrame(gains).to_csv(C.RESULTS / "validate_ha2_gain.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(s[["method", "direction", "MAE", "MAE_sd", "median", "MAE_tau_le15", "MAE_tau_ge21"]].round(3).to_string(index=False))
        print(pd.DataFrame(gains).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
