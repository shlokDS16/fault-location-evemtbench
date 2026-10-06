"""Zero-shot evaluation on a new grid (design section 18): H-A and H-A2 trained on the pooled source grids
(results/c1_grid_<tag>.parquet from grid_pass.py), tested on the target grid. Same fixed HGB settings and seeds,
explicit allow-list. Also: one-ended physics baselines and the two-ended TD locator on the target.
Usage: python src/c1/zs_eval.py <target_tag> <source_tag> [<source_tag> ...]  -> results/c1_zs_<target>.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokens_core import ALLOW, LOCAL_TOKENS  # noqa: E402
from ha2_eval import PARAMS, check_allow  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"


def summ(err, df):
    pm, shc = df.post_ms.values, ~df.etype.str.contains("incipient").values
    return dict(n=len(df), episodes=df.sim_idx.nunique(), mae=err.mean(), median=np.median(err),
                post_le15=err[pm <= 15].mean(), post_ge21=err[pm >= 21].mean(), mae_shc_only=err[shc].mean())


def main():
    tgt, srcs = sys.argv[1], sys.argv[2:]
    te = pd.read_parquet(R / f"c1_grid_{tgt}.parquet")
    tr = pd.concat([pd.read_parquet(R / f"c1_grid_{s}.parquet") for s in srcs], ignore_index=True)
    rows = []
    for c in ("phys_react_oracle", "phys_tak2_oracle", "phys_react_minloop", "td2_est"):
        sub = te[te[c].notna()]  # td2_est is undefined on lines measured at one end only (design 18a)
        err = (sub[c].clip(0, 1) - sub.y).abs().to_numpy() * 100
        rows.append(dict(method=c, **summ(err, sub)))
    for name, cols in (("H-A", LOCAL_TOKENS), ("H-A2", ALLOW)):
        check_allow(cols)
        P = np.stack([HistGradientBoostingRegressor(**PARAMS, random_state=s).fit(tr[cols], tr.y).predict(te[cols])
                      for s in (0, 1, 2)])
        p = np.clip(P.mean(0), 0, 1)
        err = np.abs(p - te.y.values) * 100
        single = [(np.abs(np.clip(q, 0, 1) - te.y.values) * 100).mean() for q in P]
        rows.append(dict(method=name, **summ(err, te), single_seed_mean=np.mean(single), single_seed_sd=np.std(single, ddof=1)))
        if name == "H-A2":
            out = te[["sim_idx", "line", "length_km", "etype", "post_ms", "R", "y"]].copy()
            out["pred"], out["err"], out["td2_err"] = p, err, (te.td2_est.clip(0, 1) - te.y).abs() * 100
            out.to_csv(R / f"c1_zs_{tgt}_pred.csv", index=False)
            print(out.groupby("etype")[["err", "td2_err"]].mean().round(2).to_string())
            print(out.groupby("line")[["length_km", "err", "td2_err"]].mean().round(2).to_string())
    # report-only ablations (design section 19, A3-MV): un-normalised tokens and HGB on raw windows (+-Z)
    from ablations import unnormalise, fit_eval
    r = fit_eval(unnormalise(tr)[ALLOW], tr.y.values, unnormalise(te)[ALLOW], te.y.values, te.post_ms.values)
    rows.append(dict(method="A3_unnormalised", n=len(te), **r))
    S = [np.load(R / f"c1_grid_{s}_raw.npz") for s in srcs]
    T = np.load(R / f"c1_grid_{tgt}_raw.npz")
    assert np.array_equal(T["y"], te.y.values)
    Xtr = np.concatenate([s["X"].reshape(len(s["X"]), -1) for s in S])
    Ztr = np.concatenate([s["Z"] for s in S])
    ytr = np.concatenate([s["y"] for s in S])
    Xte = T["X"].reshape(len(T["X"]), -1)
    for nm, a, b in (("A1_raw_window", Xtr, Xte), ("A1_raw_window+Z", np.hstack([Xtr, Ztr]), np.hstack([Xte, T["Z"]]))):
        rows.append(dict(method=nm, n=len(te), **fit_eval(a, ytr, b, te.y.values, te.post_ms.values)))
    res = pd.DataFrame(rows)
    res.insert(0, "sources", "+".join(srcs))
    res.to_csv(R / f"c1_zs_{tgt}.csv", index=False)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
