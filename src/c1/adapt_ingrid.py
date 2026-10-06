"""Gate G2-A2 (design sections 17 / 17a): H-A2 (and H-A) trained on adapt_grid-DoubleLine TRAIN line-fault windows,
tested on (a) adapt_grid TEST line-fault windows and (b) the benchmark family (14,640 official windows).
Fixed HGB settings, seeds 0-2 averaged, explicit allow-list (tokens_core.ALLOW / LOCAL_TOKENS). No tuning.
Output: results/c1_adapt_ingrid.csv (summary) and results/c1_adapt_ingrid_pred_<test>.csv (per window)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokens_core import ALLOW, LOCAL_TOKENS  # noqa: E402
from ha2_eval import PARAMS, check_allow  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
PHYS = ("phys_react_oracle", "phys_tak2_oracle", "phys_react_minloop")


def bench():
    f = pd.read_parquet(R / "c1_local_feats.parquet").merge(pd.read_parquet(R / "c1_td_tokens.parquet"),
                                                            on=["sim_idx", "tau_ms"], validate="one_to_one")
    return f.rename(columns={"tau_ms": "post_ms"})


def summ(err, df):
    pm, shc = df.post_ms.values, ~df.etype.str.contains("incipient").values
    return dict(n=len(df), episodes=df.sim_idx.nunique(), mae=err.mean(), median=np.median(err),
                post_le15=err[pm <= 15].mean(), post_ge21=err[pm >= 21].mean(), mae_shc_only=err[shc].mean())


def main():
    ad = pd.read_parquet(R / "c1_adapt_tokens.parquet")
    tr = ad[ad.split == "train"].reset_index(drop=True)
    tests = {"adapt_test": ad[ad.split == "test"].reset_index(drop=True), "benchmark": bench()}
    assert not set(tr.sim_idx) & set(tests["adapt_test"].sim_idx)
    rows = []
    for tname, te in tests.items():
        for c in PHYS:
            err = (te[c].clip(0, 1) - te.y).abs().to_numpy() * 100
            rows.append(dict(method=c, test=tname, **summ(err, te)))
    for name, cols in (("H-A", LOCAL_TOKENS), ("H-A2", ALLOW)):
        check_allow(cols)
        models = [HistGradientBoostingRegressor(**PARAMS, random_state=s).fit(tr[cols], tr.y) for s in (0, 1, 2)]
        for tname, te in tests.items():
            P = np.stack([m.predict(te[cols]) for m in models])
            p = np.clip(P.mean(0), 0, 1)
            err = np.abs(p - te.y.values) * 100
            single = [(np.abs(np.clip(q, 0, 1) - te.y.values) * 100).mean() for q in P]
            rows.append(dict(method=name, test=tname, **summ(err, te), single_seed_mean=np.mean(single),
                             single_seed_sd=np.std(single, ddof=1)))
            if name == "H-A2":
                out = te[["sim_idx", "etype", "post_ms", "R", "y"]].copy()
                out["pred"], out["err"] = p, err
                out.to_csv(R / f"c1_adapt_ingrid_pred_{tname}.csv", index=False)
                print(out.groupby("etype").err.mean().round(2).to_string())
    res = pd.DataFrame(rows)
    res.to_csv(R / "c1_adapt_ingrid.csv", index=False)
    print(res.round(3).to_string())
    b = res[res.test == "benchmark"].set_index("method")
    best_phys = b.loc[list(PHYS), "mae"].min()
    m = b.loc["H-A2", "mae"]
    print(f"G2-A2: benchmark MAE {m:.3f} (bar 5.85) ; best physics {best_phys:.3f} -> ratio {m / best_phys:.3f} (bar 0.8)"
          f" -> {'PASS' if m <= 5.85 and m <= 0.8 * best_phys else 'FAIL'}")


if __name__ == "__main__":
    main()
