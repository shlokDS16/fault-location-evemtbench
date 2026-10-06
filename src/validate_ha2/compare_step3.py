"""Part 6, AFTER the freeze (results/validate_ha2_step3_frozen.sha256): compare with the lead's classical and chain
results and run the diagnostics that explain the differences. Read-only on every results/c1_* file.
Writes results/validate_ha2_step3cmp_*.csv."""
from __future__ import annotations

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import classical as K  # noqa: E402
import chain as CH  # noqa: E402
from audit_partB import lead_name  # noqa: E402
from tokens import LOCAL_TOKENS, LOOPS, TD_TOKENS  # noqa: E402

R = C.RESULTS
M = ["R", "T", "T2", "MT", "E"]
LEADTAG = {"DL": "DL", "TG": "TG", "AD": "ADAPT", "MV": "MV"}


def score(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(np.where(np.isfinite(v), np.clip(v, 0, 1), 0.5) - y) * 100


def lead_eriksson(V, I, dI, ZL, ZSA, ZSB):
    """The lead's fallbacks (src/c1/classical_oe.py), re-typed here for the diagnostic only."""
    K1 = 1 + ZSB / ZL + V / (ZL * I)
    K2 = V / (ZL * I) * (1 + ZSB / ZL)
    K3 = dI / (ZL * I) * (1 + (ZSA + ZSB) / ZL)
    if abs(K3.imag) < 1e-12:
        return np.nan
    p = K1.real - K3.real * K1.imag / K3.imag
    q = K2.real - K3.real * K2.imag / K3.imag
    disc = p * p - 4 * q
    if disc < 0:
        return float(np.clip(p / 2, -1, 2))
    r = [(p - math.sqrt(disc)) / 2, (p + math.sqrt(disc)) / 2]
    ok = [x for x in r if -0.1 <= x <= 1.1]
    return float(min(ok, key=lambda x: abs(x - 0.5))) if ok else float(min(r, key=lambda x: abs(x - 0.5)))


def _init_lead_e() -> None:
    K.eriksson = lead_eriksson


def classical_cmp() -> None:
    mine = pd.read_parquet(R / "validate_ha2_step3_classical_perwindow.parquet")
    mine["k"] = (mine["pft_ms"] * 9.6).round().astype(int)
    rows = []
    for g, tag in LEADTAG.items():
        lead = pd.read_parquet(R / f"c1_classical_{tag}.parquet")
        lead["k"] = (lead["post_ms"] * 9.6).round().astype(int)
        j = mine[mine["grid"] == g].merge(lead, on=["sim_idx", "k"], suffixes=("", "_l"), validate="1:1")
        assert len(j) == len(lead) == (mine["grid"] == g).sum(), g
        assert np.allclose(j["y_local"], j["y"])
        for m in M:
            a, b = score(j[m].to_numpy(), j["y"].to_numpy()), score(j[m + "_l"].to_numpy(), j["y"].to_numpy())
            d = np.abs(a - b)
            rows.append({"grid": g, "method": m, "n": len(j), "MAE_mine": a.mean(), "MAE_lead": b.mean(),
                         "delta": a.mean() - b.mean(), "nan_mine": int(j[m].isna().sum()), "nan_lead": int(j[m + "_l"].isna().sum()),
                         "windows_differ_gt_0.01pp": int((d > 0.01).sum()), "max_abs_diff_pp": float(d.max())})
    out = pd.DataFrame(rows)
    # diagnostic: my code with the lead's Eriksson fallbacks
    with ProcessPoolExecutor(max_workers=K.NW, initializer=_init_lead_e) as ex:
        res = list(ex.map(K.work, K.tasks(), chunksize=8))
    v = pd.DataFrame([x for rr in res for x in rr])
    e = []
    for g, d in v.groupby("grid"):
        e.append({"grid": g, "method": "E", "MAE_mine_with_lead_fallbacks": score(d["E"].to_numpy(), d["y_local"].to_numpy()).mean(),
                  "nan_with_lead_fallbacks": int(d["E"].isna().sum())})
        b = d[(d["res"] <= 1) & d["ftype"].str.contains("shc") & (d["pft_ms"] >= 25)]
        if g == "DL":
            e[-1]["DL_bolted_with_lead_fallbacks"] = score(b["E"].to_numpy(), b["y_local"].to_numpy()).mean()
    out = out.merge(pd.DataFrame(e), on=["grid", "method"], how="left")
    out.to_csv(R / "validate_ha2_step3cmp_classical.csv", index=False)
    rep = pd.read_csv(R / "c1_classical_report.csv")
    rep.to_csv(R / "validate_ha2_step3cmp_lead_classical_report_copy.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string(index=False))


def lead_chain(scen: str, g: str) -> pd.DataFrame:
    f = R / (f"c1_grid_{g}.parquet" if scen == "clean" else f"c1_chain_{scen}_{g}.parquet")
    d = pd.read_parquet(f)
    d["k"] = (d["post_ms"] * 9.6).round().astype(int)
    return d


def chain_cmp() -> None:
    names = {"clean": "clean", "AA": "AA", "CTm": "CT-mild", "CTs": "CT-severe", "CVT5": "CVT-5", "CVT15": "CVT-15",
             "FULL": "FULL"}
    rows = []
    for g in ("DL", "TG"):
        T = pd.read_parquet(R / f"validate_ha2_step3_chain_tokens_{g}.parquet")
        T["k"] = (T["s1"] - 960).astype(int)
        for s, ls in names.items():
            mine = T[T["scen"] == s]
            lead = lead_chain(ls, g)
            j = mine.merge(lead, on=["sim_idx", "k"], suffixes=("", "_l"), validate="1:1")
            assert len(j) == len(mine) == len(lead)
            td = np.abs(np.clip(j["two_ended"], 0, 1) - np.clip(j["td2_est"], 0, 1)) * 100
            tok = []
            for c in LOCAL_TOKENS + TD_TOKENS:
                ln = lead_name(c)
                b = j[ln + "_l"] if ln + "_l" in j else j[ln]
                tok.append(np.abs(j[c].to_numpy() - b.to_numpy()))
            tok = np.column_stack(tok)
            ii = j["oracle"].map({lp: k for k, lp in enumerate(LOOPS)}).to_numpy().astype(int)
            tak2_raw = j[[f"tak2_{lp}" for lp in LOOPS]].to_numpy()[np.arange(len(j)), ii]
            r = {"grid": g, "scen": s, "n": len(j), "two_ended_max_pp_diff": float(td.max()),
                 "tokens_max_abs_diff": float(tok.max()), "tokens_p99_abs_diff": float(np.quantile(tok, .99)),
                 "tak2_oracle_no3phfix_MAE_mine": float(np.abs(np.clip(tak2_raw, 0, 1) - j["y"]).mean() * 100)}
            if "sat" in lead:
                r["sat_agree"] = float((j["sat"].astype(int) == j["sat_l"].astype(int)).mean()) if "sat_l" in j else np.nan
                r["sat_mine"], r["sat_lead"] = float(j["sat"].mean()), float(j["sat_l"].mean())
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(R / "validate_ha2_step3cmp_chain_perwindow.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(4).to_string(index=False))


def _sat_variant(task: tuple) -> list[dict]:
    """Saturation flag variants for CT-severe on the local terminal: (RP = 1 | lead RP) x (i_s = actual i2 | i1/N)."""
    g, r, vs_eff = task
    loc = np.load(C.CACHE / g / f"ep{int(r['sim_idx'])}.npy")
    nf = int(round((r["start"] - 1.0) * C.FS))
    out = []
    res = {}
    for nm, vs in (("rp1", 400.0), ("rplead", vs_eff)):
        I2, ie, i2 = CH.ct_apply(loc[:, 0:3], dict(Vs=vs, Rb=2.0, rem=0.6))
        res[nm] = (ie, i2, loc[:, 0:3] / CH.NCT)
    for tau in C.taus_for(r["etype"]):
        s1 = nf + int(round(tau / 1000 * C.FS))
        w = slice(s1 - C.WIN, s1)
        row = {"grid": g, "sim_idx": int(r["sim_idx"]), "tau": tau}
        for nm, (ie, i2, iid) in res.items():
            row[f"{nm}_i2"] = bool((np.abs(ie[w]).max(0) > 0.05 * np.abs(i2[w]).max(0)).any())
            row[f"{nm}_ideal"] = bool((np.abs(ie[w]).max(0) > 0.05 * np.abs(iid[w]).max(0)).any())
        out.append(row)
    return out


def sat_diag() -> None:
    t = np.linspace(0, 2 * np.pi, 20001)
    rp = float(np.sqrt(np.mean(np.abs(np.sin(t)) ** (2 * 22.0))))
    vs_eff = 400.0 * rp ** (1 / 22.0)  # (10/RP)(K l)^S == 10 (K' l)^S with Vs' = Vs RP^(1/S)
    tasks = [(g, r, vs_eff) for g in ("DL", "TG") for r in C.episodes(g).to_dict("records")]
    with ProcessPoolExecutor(max_workers=CH.NW) as ex:
        res = list(ex.map(_sat_variant, tasks, chunksize=8))
    d = pd.DataFrame([x for rr in res for x in rr])
    s = d.groupby("grid")[[c for c in d.columns if c.startswith("rp")]].mean() * 100
    s["RP_lead"] = rp
    s.to_csv(R / "validate_ha2_step3cmp_sat_variants.csv")
    print(s.round(3).to_string())


if __name__ == "__main__":
    classical_cmp()
    chain_cmp()
    sat_diag()


def _sat_leadparam(task: tuple) -> list[dict]:
    """CT-severe with the lead's parameterisation (A = 10 / (RP lref^S), remanence 0.6 lref with lref from Vs = 400)."""
    g, r, vs_eff, rem_m = task
    loc = np.load(C.CACHE / g / f"ep{int(r['sim_idx'])}.npy")
    nf = int(round((r["start"] - 1.0) * C.FS))
    _, ie, i2 = CH.ct_apply(loc[:, 0:3], dict(Vs=vs_eff, Rb=2.0, rem=rem_m))
    iid = loc[:, 0:3] / CH.NCT
    out = []
    for tau in C.taus_for(r["etype"]):
        s1 = nf + int(round(tau / 1000 * C.FS))
        w = slice(s1 - C.WIN, s1)
        out.append({"grid": g, "sim_idx": int(r["sim_idx"]), "k": s1 - nf,
                    "sat_ideal": bool((np.abs(ie[w]).max(0) > 0.05 * np.abs(iid[w]).max(0)).any())})
    return out


def sat_leadparam() -> None:
    t = np.linspace(0, 2 * np.pi, 20001)
    rp = float(np.sqrt(np.mean(np.abs(np.sin(t)) ** 44.0)))
    vs_eff = 400.0 * rp ** (1 / 22.0)
    tasks = [(g, r, vs_eff, 0.6 * 400.0 / vs_eff) for g in ("DL", "TG") for r in C.episodes(g).to_dict("records")]
    with ProcessPoolExecutor(max_workers=CH.NW) as ex:
        d = pd.DataFrame([x for rr in ex.map(_sat_leadparam, tasks, chunksize=8) for x in rr])
    rows = []
    for g, dd in d.groupby("grid"):
        lead = lead_chain("CT-severe", g)
        j = dd.merge(lead[["sim_idx", "k", "sat"]], on=["sim_idx", "k"], validate="1:1")
        rows.append({"grid": g, "n": len(j), "sat_mine_leadparam_pct": 100 * j["sat_ideal"].mean(),
                     "sat_lead_pct": 100 * j["sat"].mean(), "agree": float((j["sat_ideal"] == (j["sat"] == 1)).mean())})
    s = pd.DataFrame(rows)
    s.to_csv(R / "validate_ha2_step3cmp_sat_leadparam.csv", index=False)
    print(s.round(4).to_string(index=False))
