"""Independent validation, part 11 (design s.27 (a)-(d)), implemented from the spec only.
Output: results/validate_part11.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src" / "c1"))
from revamp_stats import load_sets, err, key, boot_idx, ci  # noqa: E402
from tokens_core import ALLOW  # noqa: E402
from ha2_eval import PARAMS  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

PQ = dict(engine="fastparquet")
rows = []


def add(part, set_, method, n, mae, lo, hi, a=np.nan, b=np.nan, share=np.nan, ntr=np.nan, nte=np.nan):
    rows.append(dict(part=part, set=set_, method=method, n=n, mae=mae, lo=lo, hi=hi, coef_a=a, coef_b=b,
                     share_within_002=share, n_train_ep=ntr, n_test_ep=nte))


def bci(e, ep):
    idx = boot_idx(np.asarray(ep))
    return ci(np.asarray(e), idx)


sets = load_sets()
preds = pd.read_parquet(R / "c1_two_end_learn_preds.parquet", **PQ)
preds["k"] = preds.tau.round(3)
DIRS = {"DL": "TG_to_DL", "TG": "DL_to_TG", "MV": "DL+TG_to_MV"}
grids = {g: pd.read_parquet(R / f"c1_grid_{g}.parquet", **PQ) for g in ("DL", "TG", "MV")}
cl = {g: pd.read_parquet(R / f"c1_classical_{g}.parquet", **PQ) for g in ("DL", "TG", "MV")}

# ---- (a)
for s, dr in DIRS.items():
    d = sets[s]
    d = d.assign(k=d.post_ms.round(3))
    if s == "MV":
        d = d[d.two.notna()]
    p = preds[(preds.direction == dr) & (preds.input == "P")]
    best = None
    for m in ("MLP", "GRU"):
        pm = p[p.model == m]
        per = []
        for sd in (0, 1, 2):
            q = pm[pm.seed == sd][["sim_idx", "k", "y", "pred"]]
            j = d[["sim_idx", "k", "y"]].merge(q, on=["sim_idx", "k"], how="left", suffixes=("", "_p"), validate="one_to_one")
            assert j.pred.notna().all() and np.allclose(j.y, j.y_p)
            per.append(err(j.pred.values, j.y.values))
        e = np.mean(per, 0)
        if best is None or e.mean() < best[1].mean():
            best = (m, e)
    m, e = best
    lo, hi = bci(e, d.sim_idx.values)
    add("a", s, f"P-{m}", len(e), e.mean(), lo, hi)

# ---- (b)
TRAIN = {"DL": ["TG"], "TG": ["DL"], "MV": ["DL", "TG"]}
for s in ("DL", "TG", "MV"):
    tr = pd.concat([grids[g][["td2_est", "y"]] for g in TRAIN[s]]).dropna()
    b, a = np.polyfit(tr.td2_est.values, tr.y.values, 1)
    g = grids[s].reset_index(drop=True)
    d = sets[s].reset_index(drop=True)
    # sets[s] and grid share (sim_idx, post_ms); align via key
    gk = key(grids[s][["sim_idx", "post_ms", "td2_est"]])
    dk = d.assign(k=d.post_ms.round(3))[["sim_idx", "k", "y"]]
    j = dk.merge(gk.assign(k=gk.post_ms.round(3))[["sim_idx", "k", "td2_est"]], on=["sim_idx", "k"], validate="one_to_one")
    if s == "MV":
        j = j[j.td2_est.notna()]
    ep = j.sim_idx.values
    e1 = err(a + b * j.td2_est.values, j.y.values)
    e2 = err(j.td2_est.values, j.y.values)
    lo, hi = bci(e1, ep)
    add("b", s, "calibrated ratio", len(e1), e1.mean(), lo, hi, a=a, b=b)
    lo, hi = bci(e2, ep)
    add("b", s, "TD Eq.(4)", len(e2), e2.mean(), lo, hi)

# ---- (c)
for s in ("DL", "TG"):
    d = sets[s].reset_index(drop=True)
    et = grids[s].drop_duplicates("sim_idx").set_index("sim_idx").etype
    d["etype"] = d.sim_idx.map(et)
    d = d[np.abs(d.y - 0.5) < 1e-9]
    inc = d.etype.str.contains("incipient|hif")
    for nm, sub in (("inc+HIF at d=0.5", d[inc]), ("short circuit at d=0.5", d[~inc])):
        p = np.clip(np.nan_to_num(sub.ha2_raw.values.astype(float), nan=0.5), 0, 1)
        e = np.abs(p - sub.y.values) * 100
        lo, hi = bci(e, sub.sim_idx.values)
        add("c", s, nm, len(sub), e.mean(), lo, hi, share=(np.abs(p - 0.5) <= 0.02).mean() * 100)

# ---- (d)
mv = grids["MV"].reset_index(drop=True)
eps = np.sort(mv.sim_idx.unique())
perm = np.random.default_rng(0).permutation(eps)
nte = int(round(0.3 * len(eps)))
test_eps = set(perm[:nte])
te_m = mv.sim_idx.isin(test_eps).values
tr, te = mv[~te_m], mv[te_m]
P = []
for sd in (0, 1, 2):
    m = HistGradientBoostingRegressor(**PARAMS, random_state=sd).fit(tr[ALLOW], tr.y)
    P.append(m.predict(te[ALLOW]))
P = np.mean(P, 0)
e = err(P, te.y.values)
lo, hi = bci(e, te.sim_idx.values)
ntr_e, nte_e = tr.sim_idx.nunique(), te.sim_idx.nunique()
add("d", "MV", "PFB feeder-trained", len(te), e.mean(), lo, hi, ntr=ntr_e, nte=nte_e)
e = err(np.full(len(te), tr.y.median()), te.y.values)
lo, hi = bci(e, te.sim_idx.values)
add("d", "MV", "constant (train median)", len(te), e.mean(), lo, hi)
c = key(cl["MV"])
tk = key(te[["sim_idx", "post_ms", "y"]])
j = tk.merge(c[["sim_idx", "k", "E"]], on=["sim_idx", "k"], how="left", validate="one_to_one")
assert len(j) == len(te)
e = err(j.E.values, j.y.values)
lo, hi = bci(e, te.sim_idx.values)
add("d", "MV", "Eriksson (same windows)", len(te), e.mean(), lo, hi)
s = sets["MV"].assign(k=sets["MV"].post_ms.round(3))
j = tk.merge(s[["sim_idx", "k", "ha2"]], on=["sim_idx", "k"], how="left", validate="one_to_one")
e = j.ha2.values
lo, hi = bci(e, te.sim_idx.values)
add("d", "MV", "PFB 110 kV-trained (same windows)", len(te), e.mean(), lo, hi)

out = pd.DataFrame(rows)
out.to_csv(R / "validate_part11.csv", index=False)
st = pd.read_csv(R / "c1_prof2.csv")
m = out.merge(st, on=["part", "set", "method"], suffixes=("", "_s"), how="left")
for c_ in ("n", "mae", "lo", "hi", "coef_a", "coef_b", "share_within_002", "n_train_ep", "n_test_ep"):
    m["d_" + c_] = (m[c_] - m[c_ + "_s"]).abs()
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(m[["part", "set", "method", "n", "n_s", "mae", "mae_s", "d_mae", "d_lo", "d_hi", "d_share_within_002", "d_n_train_ep", "d_n_test_ep", "d_coef_a", "d_coef_b"]].round(4).to_string())
