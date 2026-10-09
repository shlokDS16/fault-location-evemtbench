"""Design section 27 (report-only, second professor round): (a) CIs of the physical-input networks, (b) calibrated-ratio
baseline, (c) mid-line shortcut test for PFB, (d) feeder-trained PFB on CIGRE MV.
Usage: python src/c1/prof2.py -> results/c1_prof2.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revamp_stats import load_sets, err, key, boot_idx, ci  # noqa: E402
from tokens_core import ALLOW  # noqa: E402
from ha2_eval import PARAMS, check_allow  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
PQ = dict(engine="fastparquet")
rows = []


def add(part, st, method, e, ep, extra=None):
    lo, hi = ci(e, boot_idx(ep))
    rows.append(dict(part=part, set=st, method=method, n=len(e), mae=float(e.mean()), lo=lo, hi=hi, **(extra or {})))


def main():
    # (a) physical-input networks, best cell of s.26 per test set, seed-averaged per-window error
    p = pd.read_parquet(R / "c1_two_end_learn_preds.parquet", **PQ)
    p = p[p.input == "P"].assign(e=lambda d: err(d.pred, d.y))
    best = p.groupby(["model", "direction"]).e.mean()
    for d, st in (("TG_to_DL", "DL"), ("DL_to_TG", "TG"), ("DL+TG_to_MV", "MV")):
        m = best.xs(d, level="direction").idxmin()
        w = p[(p.model == m) & (p.direction == d)].groupby(["sim_idx", "tau"]).e.mean().reset_index()
        assert w.groupby("sim_idx").size().min() >= 1
        add("a", st, f"P-{m}", w.e.values, w.sim_idx.values)
    # (b) calibrated ratio a + b * td2_est, OLS on the training grid(s)
    g = {k: pd.read_parquet(R / f"c1_grid_{k}.parquet", columns=["sim_idx", "y", "td2_est"], **PQ).dropna()
         for k in ("DL", "TG", "MV")}
    for src, st in ((["TG"], "DL"), (["DL"], "TG"), (["DL", "TG"], "MV")):
        tr = pd.concat([g[s] for s in src])
        b, a = np.polyfit(tr.td2_est.values, tr.y.values, 1)
        te = g[st]
        add("b", st, "calibrated ratio", err(a + b * te.td2_est.values, te.y.values), te.sim_idx.values,
            dict(coef_a=a, coef_b=b))
        add("b", st, "TD Eq.(4)", err(te.td2_est.values, te.y.values), te.sim_idx.values)
    # (c) shortcut test on windows with true d = 0.5 (PFB cross-grid, Table III)
    sets = load_sets()
    for st in ("DL", "TG"):
        s = key(sets[st]).merge(key(pd.read_parquet(R / f"c1_classical_{st}.parquet", **PQ))[["sim_idx", "k", "etype"]],
                                on=["sim_idx", "k"], validate="one_to_one")
        half = np.abs(s.y - 0.5) < 1e-9
        hif = s.etype.str.contains("incipient|hif")
        for lab, msk in (("inc+HIF at d=0.5", half & hif), ("short circuit at d=0.5", half & ~hif)):
            q = s[msk]
            add("c", st, lab, q.ha2.values, q.sim_idx.values,
                dict(share_within_002=float((np.abs(np.clip(q.ha2_raw, 0, 1) - 0.5) <= 0.02).mean() * 100)))
    # (d) feeder-trained PFB on CIGRE MV, episode split
    mv = pd.read_parquet(R / "c1_grid_MV.parquet", columns=["sim_idx", "post_ms", "y"] + ALLOW, **PQ)
    check_allow(ALLOW)
    ep = np.unique(mv.sim_idx)
    test_ep = set(np.random.default_rng(0).permutation(ep)[: int(round(0.3 * len(ep)))])
    te_m = mv.sim_idx.isin(test_ep).values
    tr, te = mv[~te_m], mv[te_m]
    pred = np.mean([HistGradientBoostingRegressor(**PARAMS, random_state=s).fit(tr[ALLOW], tr.y).predict(te[ALLOW])
                    for s in (0, 1, 2)], 0)
    add("d", "MV", "PFB feeder-trained", err(pred, te.y.values), te.sim_idx.values,
        dict(n_train_ep=len(ep) - len(test_ep), n_test_ep=len(test_ep)))
    add("d", "MV", "constant (train median)", err(np.full(len(te), tr.y.median()), te.y.values), te.sim_idx.values)
    c = key(sets["MV"])[["sim_idx", "k", "E", "ha2"]]
    t = key(te[["sim_idx", "post_ms"]]).merge(c, on=["sim_idx", "k"], how="left", validate="one_to_one")
    assert t.E.notna().all()
    add("d", "MV", "Eriksson (same windows)", t.E.values, t.sim_idx.values)
    add("d", "MV", "PFB 110 kV-trained (same windows)", t.ha2.values, t.sim_idx.values)
    out = pd.DataFrame(rows)
    out.to_csv(R / "c1_prof2.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
