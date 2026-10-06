"""Design section 25 (report-only, S6): episode-bootstrap CIs, errors in metres and P95, Eriksson on defined windows,
the exploratory learned-then-Eriksson switch, from the existing per-window predictions.
Usage: python src/c1/revamp_stats.py  -> results/c1_revamp_stats.csv (+ printed checks against the paper macros)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classical_report import zs_pred  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
B, SEED = 2000, 0
LEN_DL = None


def err(p, y):
    """% of line length; clipped to [0, 1], undefined -> 0.5 (paper scoring)."""
    return np.abs(np.clip(np.nan_to_num(np.asarray(p, float), nan=0.5), 0, 1) - np.asarray(y, float)) * 100


def key(df):
    return df.assign(k=df.post_ms.round(3))


def boot_idx(ep):
    """Episode-bootstrap: list of index arrays (window rows) for B resamples of the episodes."""
    codes, uniq = pd.factorize(ep)
    rows = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([rows[i] for i in pick]))
    return out


def ci(e, idx):
    m = np.array([e[i].mean() for i in idx])
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def lengths():
    """Line lengths of DoubleLine (the adapt set is DoubleLine; DL and TG reuse line names with other lengths)."""
    g = pd.read_parquet(R / "c1_grid_DL.parquet", columns=["line", "length_km"])
    return g.drop_duplicates("line").set_index("line").length_km


def load_sets():
    """Per test set: one frame with sim_idx, post_ms, line, y and the per-window % errors of each method."""
    L = lengths()
    sets = {}
    eq = pd.read_parquet(R / "c1_equal_info_preds.parquet")
    for src, tgt, best in (("TG", "DL", "GRU+Z"), ("DL", "TG", "MLP+Z")):
        d = key(pd.read_parquet(R / f"c1_classical_{tgt}.parquet"))
        g = key(pd.read_parquet(R / f"c1_grid_{tgt}.parquet", columns=["sim_idx", "post_ms", "td2_est", "length_km"]))
        d = d.merge(g.drop(columns="post_ms"), on=["sim_idx", "k"], validate="one_to_one")
        h = key(zs_pred(src, tgt))
        d = d.merge(h.drop(columns="post_ms"), on=["sim_idx", "k"], validate="one_to_one")
        e = eq[(eq.direction == f"{src}_to_{tgt}") & (eq.method == best)]
        x = pd.DataFrame(dict(sim_idx=e.sim_idx.values, k=e.tau.astype(float).round(3).values,
                              eq=np.mean([err(e[f"pred_s{s}"], e.y) for s in range(3)], 0)))
        d = d.merge(x, on=["sim_idx", "k"], how="left", validate="one_to_one")
        assert d["eq"].notna().all()
        sets[tgt] = pd.DataFrame(dict(sim_idx=d.sim_idx, post_ms=d.post_ms, length_km=d.length_km,
                                      two=err(d.td2_est, d.y), E=err(d.E, d.y), E_def=d.E.notna(),
                                      ha2=err(d.pred, d.y), eq=d["eq"], E_raw=d.E, ha2_raw=d.pred, y=d.y))
    # CIGRE MV
    c = key(pd.read_parquet(R / "c1_classical_MV.parquet"))
    g = key(pd.read_parquet(R / "c1_grid_MV.parquet", columns=["sim_idx", "post_ms", "td2_est", "length_km"]))
    z = key(pd.read_parquet(R / "c1_grid_MV_zid.parquet", columns=["sim_idx", "post_ms", "td2_est"])).rename(
        columns={"td2_est": "td2_zid"})
    h = key(pd.read_csv(R / "c1_zs_MV_pred.csv"))[["sim_idx", "k", "pred"]]
    d = c.merge(g.drop(columns="post_ms"), on=["sim_idx", "k"], validate="one_to_one")
    d = d.merge(z.drop(columns="post_ms"), on=["sim_idx", "k"], validate="one_to_one")
    d = d.merge(h, on=["sim_idx", "k"], validate="one_to_one")
    both = d.td2_est.notna()
    sets["MV"] = pd.DataFrame(dict(sim_idx=d.sim_idx, post_ms=d.post_ms, length_km=d.length_km,
                                   two=np.where(both, err(d.td2_est, d.y), np.nan),
                                   two_zid=np.where(both, err(d.td2_zid, d.y), np.nan),
                                   E=err(d.E, d.y), E_def=d.E.notna(), ha2=err(d.pred, d.y),
                                   E_raw=d.E, ha2_raw=d.pred, y=d.y))
    # adapt test (in-grid H-A2)
    c = key(pd.read_parquet(R / "c1_classical_ADAPT.parquet"))
    t = pd.read_csv(R / "c1_adapt_td.csv")
    t = key(t[t.split == "test"])[["sim_idx", "k", "loc_est", "loc_true"]]
    h = key(pd.read_csv(R / "c1_adapt_ingrid_pred_adapt_test.csv"))[["sim_idx", "k", "pred"]]
    d = c.merge(t, on=["sim_idx", "k"], validate="one_to_one").merge(h, on=["sim_idx", "k"], validate="one_to_one")
    assert np.allclose(d.loc_true / 100, d.y)
    sets["ADAPT"] = pd.DataFrame(dict(sim_idx=d.sim_idx, post_ms=d.post_ms, length_km=d.line.map(L).values,
                                      two=err(d.loc_est / 100, d.y), E=err(d.E, d.y), E_def=d.E.notna(),
                                      ha2=err(d.pred, d.y), E_raw=d.E, ha2_raw=d.pred, y=d.y))
    assert sets["ADAPT"].length_km.notna().all()
    for k, n in (("DL", 14640), ("TG", 32940), ("MV", 56340), ("ADAPT", 6883)):
        assert len(sets[k]) == n, (k, len(sets[k]))
    return sets


def main():
    sets = load_sets()
    rows = []

    def add(s, meth, stat, val):
        rows.append(dict(set=s, method=meth, stat=stat, value=float(val)))
    for s, d in sets.items():
        meths = [m for m in ("two", "two_zid", "E", "ha2", "eq") if m in d]
        for m in meths:
            k = d[m].notna().values
            dd = d[k].reset_index(drop=True)
            e = dd[m].values
            idx = boot_idx(dd.sim_idx.values)
            lo, hi = ci(e, idx)
            add(s, m, "n", len(e))
            add(s, m, "mae", e.mean())
            add(s, m, "ci_lo", lo)
            add(s, m, "ci_hi", hi)
            add(s, m, "p95", np.percentile(e, 95))
            em = e / 100 * dd.length_km.values * 1000
            add(s, m, "mae_m", em.mean())
            add(s, m, "p95_m", np.percentile(em, 95))
        # (a) paired reduction of H-A2 against the best equal-information baseline
        if "eq" in d:
            idx = boot_idx(d.sim_idx.values)
            h, q = d.ha2.values, d["eq"].values
            red = np.array([1 - h[i].mean() / q[i].mean() for i in idx]) * 100
            add(s, "ha2_vs_eq", "reduction", (1 - h.mean() / q.mean()) * 100)
            add(s, "ha2_vs_eq", "ci_lo", np.percentile(red, 2.5))
            add(s, "ha2_vs_eq", "ci_hi", np.percentile(red, 97.5))
        # (c) Eriksson on defined windows only
        for nm, k in (("all", np.ones(len(d), bool)), ("le15", d.post_ms.values <= 15 + 1e-6),
                      ("ge20", d.post_ms.values >= 20 - 1e-6)):
            dk = d.E_def.values & k
            add(s, "E_defined", f"mae_{nm}", d.E.values[dk].mean())
            add(s, "E_defined", f"coverage_{nm}", dk.sum() / k.sum() * 100)
        # (d) exploratory switch: H-A2 before 20 ms, Eriksson where defined from 20 ms
        use_e = (d.post_ms.values >= 20 - 1e-6) & d.E_def.values
        sw = np.where(use_e, d.E.values, d.ha2.values)
        add(s, "switch", "mae", sw.mean())
        idx = boot_idx(d.sim_idx.values)
        lo, hi = ci(sw, idx)
        add(s, "switch", "ci_lo", lo)
        add(s, "switch", "ci_hi", hi)
    out = pd.DataFrame(rows)
    out.to_csv(R / "c1_revamp_stats.csv", index=False)
    piv = out.pivot_table(index=["set", "method"], columns="stat", values="value", sort=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
