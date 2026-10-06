"""Validator part 9 (claudedocs/validator_spec_part9.md; design 25 a-d). Report-only statistics from MY OWN per-window
predictions of earlier parts (no retraining):

  two-ended TD (nameplate)    results/validate_ha2_part8_perwindow.parquet  (td2e; DL, TG, AD = adapt test, MV 14 seg)
  two-ended TD identified Z1  results/validate_ha2_part9_mv_zid_perwindow.parquet (part9_mvzid.py, part-4 code)
  Eriksson (registered 21b)   results/validate_ha2_step3_classical_perwindow.parquet (E; NaN = no root in [-0.1, 1.1])
  H-A2 (I0 guard, 3 seeds)    results/validate_ha2_guard_pred_{zeroshot_HA2_TGtoDL, zeroshot_HA2_DLtoTG, mv_HA2_MV,
                              ingrid_HA2_adapt_test}.csv
  equal-information nets      results/validate_ha2_nnpred_{GRU_Z_TGtoDL, MLP_Z_DLtoTG}_s{0,1,2}.npy (raw; clipped here)

Scoring: estimate clipped to [0, 1], NaN -> 0.5, e = |d^ - d| x 100. Bootstrap: B = 2,000 resamples of episodes
(sim_idx) with replacement, percentile 95 % interval of the window-level MAE; my own RNG (default_rng(20261006)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import mv  # noqa: E402

R = C.RESULTS
B = 2000
SEED = 20261006
OUT = R / "validate_ha2_part9_summary.csv"
OUT_PW = R / "validate_ha2_part9_perwindow.parquet"


def err(est, y):
    d = np.clip(np.asarray(est, float), 0, 1)
    d = np.where(np.isnan(d), 0.5, d)
    return np.abs(d - np.asarray(y, float)) * 100


def length_km(grid: str, line: pd.Series) -> np.ndarray:
    if grid == "MV":
        return line.map(lambda s: mv.LEN[s[len("MainLn"):]]).to_numpy(float)
    return line.map(C.LINE_LEN[grid]).to_numpy(float)


def nn_err(tag: str, grid: str, keys: pd.DataFrame) -> np.ndarray:
    meta = pd.read_parquet(R / f"validate_ha2_tokens_{grid}.parquet", columns=["sim_idx", "s1", "tau", "y"])
    es = []
    for s in range(3):
        raw = np.load(R / f"validate_ha2_nnpred_{tag}_s{s}.npy")
        assert len(raw) == len(meta)
        es.append(err(raw, meta["y"]))
    meta["e_eq"] = np.mean(es, axis=0)
    m = keys[["sim_idx", "s1", "y_local"]].merge(meta, on=["sim_idx", "s1"], how="left", validate="one_to_one")
    assert m["e_eq"].notna().all() and np.allclose(m["y_local"], m["y"])
    return m["e_eq"].to_numpy()


def build() -> dict[str, pd.DataFrame]:
    cl = pd.read_parquet(R / "validate_ha2_step3_classical_perwindow.parquet")
    p8 = pd.read_parquet(R / "validate_ha2_part8_perwindow.parquet", columns=["grid", "sim_idx", "s1", "y", "td2e"])
    sets = {}
    hz = {"DL": "zeroshot_HA2_TGtoDL", "TG": "zeroshot_HA2_DLtoTG", "AD": "ingrid_HA2_adapt_test", "MV": "mv_HA2_MV"}
    for g in ("DL", "TG", "AD", "MV"):
        c = cl[cl["grid"] == g][["sim_idx", "s1", "pft_ms", "ftype", "line", "res", "y_local", "has_remote", "E"]]
        c = c.reset_index(drop=True)
        h = pd.read_csv(R / f"validate_ha2_guard_pred_{hz[g]}.csv")
        ycol = "y_local" if g == "MV" else "y"
        h = h[["sim_idx", "s1", ycol, "pred"]].rename(columns={ycol: "y_h", "pred": "ha2"})
        d = c.merge(h, on=["sim_idx", "s1"], how="left", validate="one_to_one")
        assert d["ha2"].notna().all() and np.allclose(d["y_local"], d["y_h"]), g
        t = p8[p8["grid"] == g][["sim_idx", "s1", "y", "td2e"]]
        d = d.merge(t, on=["sim_idx", "s1"], how="left", validate="one_to_one")
        if g != "MV":
            assert d["td2e"].notna().all() and np.allclose(d["y"], d["y_local"]), g
        else:
            assert d["td2e"].notna().sum() == 52584
            z = pd.read_parquet(R / "validate_ha2_part9_mv_zid_perwindow.parquet")
            d = d.merge(z, on=["sim_idx", "s1"], how="left", validate="one_to_one")
            ok = d["td2e"].notna()
            assert (ok == d["td_idR"].notna()).all()
            # part-8 td2e must equal the part-4 code's nameplate value
            assert np.nanmax(np.abs(d.loc[ok, "td2e"] - d.loc[ok, "td_np"])) < 1e-9
            # y from bus X (two-ended) is in p8; for 8-14 (no two-ended) take y = 1 - y_local is irrelevant
        d["L_km"] = length_km(g, d["line"])
        d["e_two"] = np.where(d["td2e"].notna(), err(d["td2e"].fillna(0.5), d["y"].fillna(0)), np.nan)
        if g == "MV":
            d["e_two_zid"] = np.where(d["td_idR"].notna(), err(d["td_idR"].fillna(0.5), d["y"].fillna(0)), np.nan)
            d["e_two_zidRX"] = np.where(d["td_idRX"].notna(), err(d["td_idRX"].fillna(0.5), d["y"].fillna(0)), np.nan)
        d["E_def"] = d["E"].notna()
        d["e_E"] = err(d["E"], d["y_local"])
        d["e_ha2"] = err(d["ha2"], d["y_local"])
        sw = np.where(d["pft_ms"] < 20, d["ha2"], np.where(d["E_def"], d["E"], d["ha2"]))
        d["e_switch"] = err(sw, d["y_local"])
        if g == "DL":
            d["e_eq"] = nn_err("GRU_Z_TGtoDL", "DL", d)
        if g == "TG":
            d["e_eq"] = nn_err("MLP_Z_DLtoTG", "TG", d)
        d["grid"] = g
        sets[g] = d
    return sets


class Boot:
    """Episode bootstrap on one fixed window set; the same resamples serve every method on that set (paired)."""

    def __init__(self, sim: np.ndarray, rng: np.random.Generator):
        self.u, self.inv = np.unique(sim, return_inverse=True)
        ne = len(self.u)
        idx = rng.integers(0, ne, size=(B, ne))
        self.M = np.stack([np.bincount(r, minlength=ne) for r in idx]).astype(float)
        self.n = np.bincount(self.inv, minlength=ne).astype(float)
        self.den = self.M @ self.n

    def maes(self, e: np.ndarray) -> np.ndarray:
        s = np.bincount(self.inv, weights=e, minlength=len(self.u))
        return (self.M @ s) / self.den


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main():
    sets = build()
    pw = pd.concat(sets.values(), ignore_index=True)
    pw.to_parquet(OUT_PW, index=False)
    rows = []
    rng = np.random.default_rng(SEED)
    name = {"DL": "DL", "TG": "TG", "AD": "ADAPT", "MV": "MV"}

    def add(s, m, st, v):
        rows.append({"set": name[s], "method": m, "stat": st, "value": float(v)})

    for g, d in sets.items():
        ones = Boot(d["sim_idx"].to_numpy(), rng)  # all windows of the set (one-ended rows)
        if g == "MV":
            m2 = d["e_two"].notna().to_numpy()
            two = Boot(d.loc[m2, "sim_idx"].to_numpy(), rng)
        else:
            m2 = np.ones(len(d), bool)
            two = ones
        meths = [("two", "e_two", two, m2), ("E", "e_E", ones, None), ("ha2", "e_ha2", ones, None),
                 ("switch", "e_switch", ones, None)]
        if g == "MV":
            meths += [("two_zid", "e_two_zid", two, m2), ("two_zidRX", "e_two_zidRX", two, m2)]
        if g in ("DL", "TG"):
            meths += [("eq", "e_eq", ones, None)]
        bs = {}
        for m, col, bt, mask in meths:
            sub = d if mask is None else d[mask]
            e = sub[col].to_numpy()
            assert np.isfinite(e).all(), (g, m)
            bs[m] = bt.maes(e)
            lo, hi = ci(bs[m])
            add(g, m, "n", len(e))
            add(g, m, "MAE", e.mean())
            add(g, m, "ci_lo", lo)
            add(g, m, "ci_hi", hi)
            if m in ("two", "two_zid", "two_zidRX", "E", "ha2"):
                em = e / 100 * sub["L_km"].to_numpy() * 1000
                add(g, m, "mean_m", em.mean())
                add(g, m, "p95", np.percentile(e, 95))
                add(g, m, "p95_m", np.percentile(em, 95))
        if g in ("DL", "TG"):
            rr = 1 - bs["ha2"] / bs["eq"]
            lo, hi = ci(rr)
            add(g, "ha2_vs_eq", "rel_red", 1 - d["e_ha2"].mean() / d["e_eq"].mean())
            add(g, "ha2_vs_eq", "ci_lo", lo)
            add(g, "ha2_vs_eq", "ci_hi", hi)
        # (c) Eriksson on defined windows only
        for lab, msk in (("all", np.ones(len(d), bool)), ("le15", (d["pft_ms"] <= 15).to_numpy()),
                         ("ge20", (d["pft_ms"] >= 20).to_numpy())):
            sd = d[msk]
            dd = sd[sd["E_def"]]
            add(g, "E_defined", f"MAE_{lab}", dd["e_E"].mean() if len(dd) else np.nan)
            add(g, "E_defined", f"coverage_{lab}", sd["E_def"].mean())
            add(g, "E_defined", f"n_{lab}", len(sd))
            add(g, "E_defined", f"ndef_{lab}", len(dd))
        if g == "MV":  # extra: Eriksson restricted to the 52,584 two-ended windows (8-14 has no remote end)
            sd = d[m2]
            add(g, "E_defined", "coverage_all_14seg", sd["E_def"].mean())
            add(g, "E_defined", "MAE_all_14seg", sd.loc[sd["E_def"], "e_E"].mean())
            add(g, "E", "MAE_14seg", sd["e_E"].mean())
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(out.pivot_table(index=["set", "method"], columns="stat", values="value", aggfunc="first").round(3))


if __name__ == "__main__":
    main()
