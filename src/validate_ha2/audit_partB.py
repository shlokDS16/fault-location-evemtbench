"""Part B (run only after Part A was frozen): compare the lead's token tables, raw windows and results
with this validator's. Read-only on every results/c1_* file. Writes results/validate_ha2_partB_*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from tokens import LOCAL_TOKENS, TD_TOKENS  # noqa: E402

R = C.RESULTS
TAG = {"DL": "", "TG": "_TestGrid110kV"}
META = {"sim_idx", "line", "etype", "tau_ms", "R", "y", "oracle_loop"}  # lead's exclusion list


def lead_name(mine: str) -> str:
    """Map my token name to the lead's column name."""
    m = {"cos_i2i1": "i2c", "sin_i2i1": "i2s", "cos_i0i1": "i0c", "sin_i0i1": "i0s"}
    if "_" not in mine:
        return mine
    if mine in m:
        return m[mine]
    if mine.split("_")[0] in ("onset", "drop", "share", "direl"):
        kind, p = mine.split("_")
        return {"onset": f"i{p}_onset", "drop": f"v{p}_drop", "share": f"i{p}_share", "direl": f"di{p}_rel"}[kind]
    kind, lp = mine.split("_")
    return f"{lp}_{kind}"


def main() -> None:
    out_rows, col_rows = [], []
    for g in ("DL", "TG"):
        mine = pd.read_parquet(R / f"validate_ha2_tokens_{g}.parquet")
        lf = pd.read_parquet(R / f"c1_local_feats{TAG[g]}.parquet")
        td = pd.read_parquet(R / f"c1_td_tokens{TAG[g]}.parquet")
        feats = [c for c in lf.columns if c not in META and not c.startswith("phys_")]
        print(g, "lead local cols:", len(lf.columns), "-> features used by H-A:", len(feats))
        print("  all lead local columns:", list(lf.columns))
        print("  lead TD columns:", list(td.columns))
        merged_cols = [c for c in lf.merge(td, on=["sim_idx", "tau_ms"]).columns
                       if c not in META and not c.startswith("phys_")]
        print("  features used by H-A2:", len(merged_cols))
        mapping = {lead_name(c): c for c in LOCAL_TOKENS + TD_TOKENS}
        unmapped = [c for c in merged_cols if c not in mapping]
        missing = [c for c in mapping if c not in merged_cols]
        print("  lead features not in my token set:", unmapped, " my tokens missing in lead:", missing)
        lead = lf.merge(td, on=["sim_idx", "tau_ms"], validate="one_to_one")
        mm = mine.rename(columns={"tau": "tau_ms"})
        j = mm.merge(lead, on=["sim_idx", "tau_ms"], how="outer", suffixes=("_v", "_l"), indicator=True)
        cnt = j["_merge"].value_counts().to_dict()
        print("  window alignment:", cnt, "mine", len(mine), "lead", len(lead))
        j = j[j["_merge"] == "both"]
        out_rows.append({"grid": g, "n_mine": len(mine), "n_lead_local": len(lf), "n_lead_td": len(td),
                         "matched": len(j), "only_mine": cnt.get("left_only", 0), "only_lead": cnt.get("right_only", 0),
                         "y_maxdiff": float(np.abs(j["y_v"] - j["y_l"]).max()),
                         "etype_mismatch": int((j["etype_v"] != j["etype_l"]).sum()),
                         "line_mismatch": int((j["line_v"] != j["line_l"]).sum()),
                         "oracle_mismatch": int((j["oracle"] != j["oracle_loop"]).sum())})
        for c in LOCAL_TOKENS + TD_TOKENS:
            ln = lead_name(c)
            a = j[c + "_v"] if c + "_v" in j else j[c]
            b = j[ln + "_l"] if ln + "_l" in j else j[ln]
            d = np.abs(a.to_numpy() - b.to_numpy())
            col_rows.append({"grid": g, "token": c, "lead_col": ln, "max_abs_diff": float(d.max()),
                             "p99_abs_diff": float(np.quantile(d, 0.99)), "n_gt_1e-6": int((d > 1e-6).sum()),
                             "n_gt_1e-3": int((d > 1e-3).sum())})
        # raw windows
        raw = np.load(R / f"c1_raw{TAG[g]}.npz")
        w = np.load(R / f"validate_ha2_windows_{g}.npz")
        key_l = pd.DataFrame({"sim_idx": raw["sim_idx"], "tau": raw["tau"].astype(int), "il": np.arange(len(raw["sim_idx"]))})
        key_v = pd.DataFrame({"sim_idx": w["sim_idx"], "tau": w["tau"].astype(int), "iv": np.arange(len(w["sim_idx"]))})
        kk = key_v.merge(key_l, on=["sim_idx", "tau"], how="inner")
        Xl, Xv = raw["X"], w["X"]
        dmax = 0.0
        for s in range(0, len(kk), 4000):
            b = kk.iloc[s:s + 4000]
            dmax = max(dmax, float(np.abs(Xl[b["il"].to_numpy()] - Xv[b["iv"].to_numpy()]).max()))
        zl = raw["Z"].astype(np.float64)[kk["il"].to_numpy()]
        zv = mine[["R1", "X1", "R0", "X0"]].to_numpy()[kk["iv"].to_numpy()]
        yl = raw["y"][kk["il"].to_numpy()]
        yv = mine["y"].to_numpy()[kk["iv"].to_numpy()]
        out_rows[-1].update({"raw_n_lead": len(raw["sim_idx"]), "raw_matched": len(kk), "raw_window_maxabsdiff": dmax,
                             "raw_Z_maxreldiff": float(np.max(np.abs(zl - zv) / np.abs(zv))),
                             "raw_y_maxdiff": float(np.abs(yl - yv).max()), "raw_npz_keys": ",".join(raw.files)})
    pd.DataFrame(out_rows).to_csv(R / "validate_ha2_partB_alignment.csv", index=False)
    cd = pd.DataFrame(col_rows)
    cd.to_csv(R / "validate_ha2_partB_token_diff.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 300):
        print(pd.DataFrame(out_rows).T.to_string())
        print(cd.sort_values("max_abs_diff", ascending=False).head(25).to_string(index=False))


if __name__ == "__main__":
    main()
