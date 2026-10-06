"""Part 5: I0 guard re-check (claudedocs/validator_spec_guard.md), CPU only.

Rule, applied to COPIES of my own cached token tables (old files are not modified):
  if i0r = |I0| / (|I1| + 1e-9) < 1e-6  ->  cos_i0i1 = sin_i0i1 = 0 and tak0_ag = tak0_bg = tak0_cg = 0.5.
Nothing else changes. Same learners, settings, seeds and 77-token allow-list (same column order) as parts 1-4.

Outputs (results/validate_ha2_guard_*):
  tokens_{DL,TG,AD,MV}.parquet  guarded copies
  counts.csv                    guarded windows per grid / set and fault type
  summary.csv                   H-A / H-A2 per setting and test set, seeds 0-2 + 3-seed average ("avg")
  pred_<setting>_<method>_<testset>.csv  per-window 3-seed-average predictions
  oldnew.csv                    my pre-guard numbers (from my earlier frozen summaries) vs post-guard
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from physics_hgb import HGB_PARAMS, assert_allow_list  # noqa: E402
from tokens import LOCAL_TOKENS, TD_TOKENS  # noqa: E402

R = C.RESULTS
THR = 1e-6
GUARD_ZERO = ["cos_i0i1", "sin_i0i1"]
GUARD_HALF = ["tak0_ag", "tak0_bg", "tak0_cg"]
GUARDED = GUARD_ZERO + GUARD_HALF
METHODS = {"HA": list(LOCAL_TOKENS), "HA2": list(LOCAL_TOKENS) + list(TD_TOKENS)}


def guard(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    g = df.copy()
    m = (g["i0r"] < THR).to_numpy()
    g.loc[m, GUARD_ZERO] = 0.0
    g.loc[m, GUARD_HALF] = 0.5
    # only the 5 columns change, and only on guarded rows
    other = [c for c in df.columns if c not in GUARDED]
    assert df[other].equals(g[other])
    assert np.array_equal(df.loc[~m, GUARDED].to_numpy(), g.loc[~m, GUARDED].to_numpy())
    return g, m


def load_all() -> dict[str, pd.DataFrame]:
    src = {"DL": "validate_ha2_tokens_DL.parquet", "TG": "validate_ha2_tokens_TG.parquet",
           "AD": "validate_ha2_ingrid_tokens.parquet", "MV": "validate_ha2_mv_tokens.parquet"}
    out, counts = {}, []
    for k, f in src.items():
        df = pd.read_parquet(R / f)
        g, m = guard(df)
        g.to_parquet(R / f"validate_ha2_guard_tokens_{k}.parquet", index=False)
        g["_guarded"] = m
        if k == "AD":
            sets = {"adapt_train": g[g["split"] == "train"], "adapt_test": g[g["split"] == "test"]}
        else:
            sets = {"all": g}
        for sname, d in sets.items():
            for ft, gg in d.groupby("ftype"):
                counts.append({"grid": k, "set": sname, "ftype": ft, "n": len(gg), "n_guarded": int(gg["_guarded"].sum())})
            counts.append({"grid": k, "set": sname, "ftype": "ALL", "n": len(d), "n_guarded": int(d["_guarded"].sum())})
        out[k] = g.drop(columns="_guarded")
    pd.DataFrame(counts).to_csv(R / "validate_ha2_guard_counts.csv", index=False)
    return out


def summarise(te: pd.DataFrame, pred: np.ndarray, ycol: str, tcol: str) -> dict:
    err = np.abs(np.clip(pred, 0, 1) - te[ycol].to_numpy()) * 100
    t = te[tcol].to_numpy()
    ft = te["ftype"]
    sc = (ft.str.endswith("shc") | ft.str.endswith("shc_w_arc")).to_numpy()
    row = {"n": len(te), "MAE": err.mean(), "median": np.median(err),
           "MAE_post_le15": err[t <= 15].mean(), "MAE_post_ge21": err[t >= 21].mean(), "MAE_shortcircuit": err[sc].mean()}
    for f in sorted(ft.unique()):
        row[f"MAE_{f}"] = err[(ft == f).to_numpy()].mean()
    return row


def run(setting: str, tr: pd.DataFrame, tests: dict[str, tuple[pd.DataFrame, str, str]]) -> list[dict]:
    rows = []
    for meth, cols in METHODS.items():
        assert_allow_list(cols)
        preds = {k: [] for k in tests}
        for seed in (0, 1, 2):
            t0 = time.time()
            mdl = HistGradientBoostingRegressor(random_state=seed, **HGB_PARAMS).fit(
                tr[cols].to_numpy(np.float64), tr["y"].to_numpy())
            for k, (te, ycol, tcol) in tests.items():
                p = mdl.predict(te[cols].to_numpy(np.float64))
                preds[k].append(p)
                rows.append({"setting": setting, "testset": k, "method": meth, "seed": str(seed),
                             **summarise(te, p, ycol, tcol)})
            print(f"  {setting} {meth} seed {seed} ({time.time() - t0:.0f}s)", flush=True)
        for k, (te, ycol, tcol) in tests.items():
            p = np.clip(np.mean(preds[k], axis=0), 0, 1)
            rows.append({"setting": setting, "testset": k, "method": meth, "seed": "avg", **summarise(te, p, ycol, tcol)})
            keep = [c for c in ("sim_idx", "s1", "tau", "pft_ms", "ftype", "line") if c in te.columns]
            out = te[keep + [ycol]].copy()
            out["pred"] = p
            out["err"] = np.abs(p - out[ycol]) * 100
            out.to_csv(R / f"validate_ha2_guard_pred_{setting}_{meth}_{k}.csv", index=False)
            print(f"{setting} {meth} {k}: MAE {out['err'].mean():.3f}", flush=True)
    return rows


def old_numbers() -> pd.DataFrame:
    """My frozen pre-guard numbers from parts 1, 2 and 4 (3-seed averages)."""
    rows = []
    s = pd.read_csv(R / "validate_ha2_summary_hgb_phys.csv")
    for _, r in s[(s["seed"] == "avg") & s["method"].isin(["HA", "HA2"])].iterrows():
        rows.append({"setting": "zeroshot", "testset": r["direction"], "method": r["method"], "MAE": r["MAE"],
                     "median": r["median"], "MAE_post_le15": r["MAE_tau_le15"], "MAE_post_ge21": r["MAE_tau_ge21"]})
    s = pd.read_csv(R / "validate_ha2_ingrid_summary.csv")
    for _, r in s[(s["seed"].astype(str) == "avg") & s["method"].isin(["HA", "HA2"])].iterrows():
        rows.append({"setting": "ingrid", "testset": r["testset"], "method": r["method"], "MAE": r["MAE"],
                     "median": r["median"], "MAE_post_le15": r["MAE_pft_le15"], "MAE_post_ge21": r["MAE_pft_ge21"]})
    s = pd.read_csv(R / "validate_ha2_mv_summary.csv")
    for _, r in s[s["method"].isin(["HA", "HA2"])].iterrows():
        rows.append({"setting": "mv", "testset": "MV", "method": r["method"], "MAE": r["MAE"],
                     "median": r["median"], "MAE_post_le15": r["MAE_pft_le15"], "MAE_post_ge21": r["MAE_pft_ge21"]})
    return pd.DataFrame(rows)


def main() -> None:
    d = load_all()
    cnt = pd.read_csv(R / "validate_ha2_guard_counts.csv")
    print(cnt[cnt["ftype"] == "ALL"].to_string(index=False), flush=True)
    rows = []
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        rows += run("zeroshot", d[src], {f"{src}to{tgt}": (d[tgt], "y", "tau")})
    ad = d["AD"]
    tr = ad[ad["split"] == "train"].reset_index(drop=True)
    te_a = ad[ad["split"] == "test"].reset_index(drop=True)
    assert not set(tr["sim_idx"]) & set(te_a["sim_idx"])
    rows += run("ingrid", tr, {"adapt_test": (te_a, "y", "pft_ms"), "benchmark_DL": (d["DL"], "y", "tau")})
    pooled = pd.concat([d["DL"], d["TG"]], ignore_index=True)
    rows += run("mv", pooled, {"MV": (d["MV"], "y_local", "pft_ms")})
    s = pd.DataFrame(rows)
    s.to_csv(R / "validate_ha2_guard_summary.csv", index=False)
    new = s[s["seed"] == "avg"][["setting", "testset", "method", "MAE", "median", "MAE_post_le15", "MAE_post_ge21"]]
    on = old_numbers().merge(new, on=["setting", "testset", "method"], suffixes=("_old", "_new"))
    for c in ("MAE", "median", "MAE_post_le15", "MAE_post_ge21"):
        on[f"{c}_delta"] = on[f"{c}_new"] - on[f"{c}_old"]
    on.to_csv(R / "validate_ha2_guard_oldnew.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(on.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
