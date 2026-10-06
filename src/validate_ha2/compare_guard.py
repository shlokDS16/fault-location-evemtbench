"""Part 5, after freezing (results/validate_ha2_guard_frozen.sha256): compare with the lead's guarded tables and
results, and audit the patch. Read-only on every results/c1_* file. Writes results/validate_ha2_guard_cmp_*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from audit_partB import lead_name  # noqa: E402
from physics_hgb import HGB_PARAMS  # noqa: E402
from tokens import LOCAL_TOKENS, TD_TOKENS  # noqa: E402

R = C.RESULTS
TOK = LOCAL_TOKENS + TD_TOKENS
G5 = ["cos_i0i1", "sin_i0i1", "tak0_ag", "tak0_bg", "tak0_cg"]
LG5 = ["i0c", "i0s", "ag_tak0", "bg_tak0", "cg_tak0"]


def key(s: pd.Series) -> pd.Series:
    return (s * 9.6).round().astype(int)


def lead_tables(pre: bool) -> dict[str, pd.DataFrame]:
    sfx = "_preguard" if pre else ""
    out = {}
    for g, t in (("DL", ""), ("TG", "_TestGrid110kV")):
        lf = pd.read_parquet(R / f"c1_local_feats{t}{sfx}.parquet")
        td = pd.read_parquet(R / f"c1_td_tokens{t}.parquet")
        d = lf.merge(td, on=["sim_idx", "tau_ms"], validate="one_to_one")
        d["k"] = key(d["tau_ms"])
        out[g] = d
    a = pd.read_parquet(R / f"c1_adapt_tokens{sfx}.parquet")
    a["k"] = key(a["post_ms"])
    out["AD"] = a
    m = pd.read_parquet(R / f"c1_grid_MV{sfx}.parquet")
    m["k"] = key(m["post_ms"])
    out["MV"] = m
    return out


def my_tables(guarded: bool) -> dict[str, pd.DataFrame]:
    f = ({"DL": "validate_ha2_guard_tokens_DL", "TG": "validate_ha2_guard_tokens_TG",
          "AD": "validate_ha2_guard_tokens_AD", "MV": "validate_ha2_guard_tokens_MV"} if guarded else
         {"DL": "validate_ha2_tokens_DL", "TG": "validate_ha2_tokens_TG",
          "AD": "validate_ha2_ingrid_tokens", "MV": "validate_ha2_mv_tokens"})
    out = {}
    for g, n in f.items():
        d = pd.read_parquet(R / f"{n}.parquet")
        d["k"] = key(d["tau"]) if g in ("DL", "TG") else key(d["pft_ms"])
        out[g] = d
    return out


def token_agreement() -> pd.DataFrame:
    rows, masks = [], []
    for stage, guarded in (("preguard", False), ("guarded", True)):
        mine, lead = my_tables(guarded), lead_tables(not guarded)
        for g in ("DL", "TG", "AD", "MV"):
            j = mine[g].merge(lead[g], on=["sim_idx", "k"], how="inner", suffixes=("_v", "_l"), validate="one_to_one")
            assert len(j) == len(mine[g]) == len(lead[g]), (g, len(j), len(mine[g]), len(lead[g]))
            ft = j["ftype"] if "ftype" in j else j["ftype_v"]
            p23 = ft.isin(["2ph_shc", "3ph_shc"]).to_numpy()
            if guarded:
                mv, ml = (j["i0r_v"] < 1e-6).to_numpy(), (j["i0r_l"] < 1e-6).to_numpy()
                masks.append({"grid": g, "n": len(j), "guard_mine": int(mv.sum()), "guard_lead": int(ml.sum()),
                              "guard_mask_mismatch": int((mv != ml).sum()),
                              "i0r_maxabsdiff": float(np.abs(j["i0r_v"] - j["i0r_l"]).max())})
            for c in TOK:
                ln = lead_name(c)
                a = j[c + "_v"] if c + "_v" in j else j[c]
                b = j[ln + "_l"] if ln + "_l" in j else j[ln]
                d = np.abs(a.to_numpy() - b.to_numpy())
                rows.append({"stage": stage, "grid": g, "token": c, "max_abs_diff": float(d.max()),
                             "p99": float(np.quantile(d, .99)), "n_gt_1e-3": int((d > 1e-3).sum()),
                             "max_abs_diff_2ph3ph": float(d[p23].max()), "max_abs_diff_other": float(d[~p23].max())})
    pd.DataFrame(masks).to_csv(R / "validate_ha2_guard_cmp_masks.csv", index=False)
    t = pd.DataFrame(rows)
    t.to_csv(R / "validate_ha2_guard_cmp_token_diff.csv", index=False)
    return t


def patch_audit() -> pd.DataFrame:
    """Independent re-application of the rule to the lead's *_preguard tables; must equal the lead's patched tables."""
    rows = []
    names = ["c1_local_feats", "c1_local_feats_SNR30", "c1_local_feats_SNR40", "c1_local_feats_TestGrid110kV",
             "c1_local_feats_TestGrid110kV_SNR30", "c1_local_feats_TestGrid110kV_SNR40",
             "c1_grid_DL", "c1_grid_TG", "c1_grid_MV", "c1_grid_MV_zid", "c1_adapt_tokens"]
    for n in names:
        old = pd.read_parquet(R / f"{n}_preguard.parquet")
        cur = pd.read_parquet(R / f"{n}.parquet")
        exp = old.copy()
        z = (exp["i0r"] < 1e-6).to_numpy()
        exp.loc[z, ["i0c", "i0s"]] = 0.0
        exp.loc[z, ["ag_tak0", "bg_tak0", "cg_tak0"]] = 0.5
        changed_cols = [c for c in old.columns if not old[c].equals(cur[c])]
        rows.append({"table": n, "n": len(cur), "same_columns": list(old.columns) == list(cur.columns),
                     "guarded": int(z.sum()), "cur_equals_my_reapplication": bool(exp.equals(cur)),
                     "columns_changed_vs_preguard": ",".join(changed_cols)})
    p = pd.DataFrame(rows)
    p.to_csv(R / "validate_ha2_guard_cmp_patch_audit.csv", index=False)
    return p


def hgb_diag() -> pd.DataFrame:
    """Zero-shot H-A / H-A2: my HGB on (a) the lead's guarded tables in the lead's ALLOW order,
    (b) my guarded tokens in the lead's ALLOW order (separates token from column-order effects)."""
    sys.path.insert(0, str(C.ROOT / "src" / "c1"))
    lead = lead_tables(False)
    mine = my_tables(True)
    lead_local = [lead_name(c) for c in LOCAL_TOKENS]
    # the lead's ALLOW order (tokens_core) without importing the lead's code: reconstruct from its rule
    loops = ("ag", "bg", "cg", "ab", "bc", "ca")
    LT = [f"{lp}_{k}" for lp in loops for k in ("zr", "zi", "react", "tak2", "takd")
          + (("tak0",) if lp.endswith("g") else ()) + ("absz",)]
    LT += ["i0r", "i2r", "v0r", "v2r", "i2c", "i2s", "i0c", "i0s"]
    LT += [f"{k}{nm}_{s}" for nm in "abc" for k, s in (("i", "onset"), ("v", "drop"), ("i", "share"), ("di", "rel"))]
    TT = [f"{lp}_{k}" for lp in loops for k in ("dtd", "drl", "tdres")]
    assert sorted(LT) == sorted(lead_local) and len(LT) == 59
    inv = {lead_name(c): c for c in TOK}
    rows = []
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        for meth, cols in (("HA", LT), ("HA2", LT + TT)):
            for variant in ("lead_tables_lead_order", "my_tables_lead_order"):
                if variant.startswith("lead"):
                    tr = lead[src].sort_values(["sim_idx", "tau_ms"]).reset_index(drop=True)
                    te = lead[tgt].sort_values(["sim_idx", "tau_ms"]).reset_index(drop=True)
                    Xtr, Xte = tr[cols], te[cols]
                else:
                    tr, te = mine[src], mine[tgt]
                    Xtr = tr[[inv[c] for c in cols]].set_axis(cols, axis=1)
                    Xte = te[[inv[c] for c in cols]].set_axis(cols, axis=1)
                P, single = [], []
                for s in (0, 1, 2):
                    p = HistGradientBoostingRegressor(random_state=s, **HGB_PARAMS).fit(Xtr, tr["y"]).predict(Xte)
                    P.append(p)
                    single.append(np.abs(np.clip(p, 0, 1) - te["y"]).mean() * 100)
                mae = np.abs(np.clip(np.mean(P, 0), 0, 1) - te["y"]).mean() * 100
                rows.append({"direction": f"{src}to{tgt}", "method": meth, "variant": variant, "MAE": mae,
                             "single_seed_mean": np.mean(single), "single_seed_sd": np.std(single, ddof=1)})
                print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(R / "validate_ha2_guard_cmp_hgb_diag.csv", index=False)
    return d


def perwindow() -> pd.DataFrame:
    rows = []
    for k, lf in (("adapt_test", "c1_adapt_ingrid_pred_adapt_test"), ("benchmark_DL", "c1_adapt_ingrid_pred_benchmark")):
        for sfx in ("", "_preguard"):
            lp = pd.read_csv(R / f"{lf}{sfx}.csv")
            lp["k"] = key(lp["post_ms"])
            mf = f"validate_ha2_guard_pred_ingrid_HA2_{k}.csv" if not sfx else f"validate_ha2_ingrid_pred_HA2_{k}.csv"
            mp = pd.read_csv(R / mf)
            mp["k"] = key(mp["pft_ms"] if "pft_ms" in mp and mp["pft_ms"].notna().all() else mp["tau"])
            j = mp.merge(lp, on=["sim_idx", "k"], suffixes=("_v", "_l"), validate="one_to_one")
            d = np.abs(j["pred_v"] - j["pred_l"]) * 100
            rows.append({"set": f"ingrid_{k}", "stage": "guarded" if not sfx else "preguard", "n": len(j),
                         "mean_abs_diff_pp": d.mean(), "p95_pp": d.quantile(.95),
                         "r": np.corrcoef(j["pred_v"], j["pred_l"])[0, 1]})
    p = pd.DataFrame(rows)
    p.to_csv(R / "validate_ha2_guard_cmp_perwindow.csv", index=False)
    return p


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    p = patch_audit()
    print(p.to_string(index=False), flush=True)
    t = token_agreement()
    print(pd.read_csv(R / "validate_ha2_guard_cmp_masks.csv").to_string(index=False))
    g5 = t[t["token"].isin(G5)]
    print(g5.pivot_table(index=["grid", "token"], columns="stage",
                         values=["max_abs_diff", "max_abs_diff_2ph3ph", "p99"]).round(6).to_string())
    print(t.groupby(["stage", "grid"])[["max_abs_diff", "p99"]].max().to_string())
    print(t[t["stage"] == "guarded"].sort_values("max_abs_diff", ascending=False).head(8).to_string(index=False))
    print(perwindow().round(4).to_string(index=False), flush=True)
    print(hgb_diag().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
