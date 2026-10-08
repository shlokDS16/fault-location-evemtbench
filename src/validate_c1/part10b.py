"""Independent validation, part 10b (spec: claudedocs/validator_spec_part10b.md).

(a) re-score two-ended learner predictions + window-set / direction checks (+ d=0.5-excluded MAE)
(b) white-noise sweep (SNR 40, 30 dB) of the two-ended time-domain locator on DL and TG official windows
(c) d = 50 % exclusion tables (c1_prof_m6.csv, c1_prof_m6_extra.csv)

Run: python src/validate_c1/part10b.py [a|b|c ...]   (default: all). At most 3 worker processes.
Output: results/validate_part10b.csv (long format: part, key, validated, stored, abs_dev, tol, ok)
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
EV = ROOT / "data" / "raw" / "evemt"
sys.path.insert(0, str(ROOT / "src" / "c1"))
sys.path.insert(0, str(ROOT / "src" / "validate_c1"))

FS, F0, WIN = 9600.0, 50.0, 480
SEED = 20261008
OUT = RES / "validate_part10b.csv"
PQ = dict(engine="fastparquet")
ROWS: list[dict] = []


def rec(part, key, val, stored, tol):
    dev = abs(val - stored) if (pd.notna(val) and pd.notna(stored)) else np.nan
    ROWS.append(dict(part=part, key=key, validated=val, stored=stored, abs_dev=dev, tol=tol,
                     ok=bool(dev <= tol) if pd.notna(dev) else False))


def err_pp(pred, y):
    """common scoring: clip to [0,1], NaN -> 0.5, error in % of line length"""
    p = np.clip(np.where(np.isnan(pred), 0.5, pred), 0.0, 1.0)
    return np.abs(p - y) * 100.0


def rd(name):
    return pd.read_parquet(RES / name, **PQ)


# --------------------------------------------------------------------------------------- (a)
def part_a():
    pr = rd("c1_two_end_learn_preds.parquet")
    st = pd.read_csv(RES / "c1_two_end_learn.csv")
    pr["err"] = err_pp(pr.pred.to_numpy(float), pr.y.to_numpy(float))
    keys = ["input", "model", "direction", "seed"]
    g = pr.groupby(keys).agg(mae=("err", "mean"), n=("err", "size")).reset_index()
    m = st.merge(g, on=keys, suffixes=("_st", "_val"), validate="one_to_one")
    assert len(m) == len(st) == len(g), (len(m), len(st), len(g))
    for r in m.itertuples():
        rec("a", f"mae|{r.input}|{r.model}|{r.direction}|{r.seed}", r.mae_val, r.mae_st, 0.01)
        rec("a", f"n|{r.input}|{r.model}|{r.direction}|{r.seed}", r.n, r.n_test, 0)
    # window sets
    dl, tg, mv = rd("c1_grid_DL.parquet"), rd("c1_grid_TG.parquet"), rd("c1_grid_MV.parquet")
    mv2 = mv[mv.td2_est.notna()]
    official = {"TG_to_DL": (dl, 14640), "DL_to_TG": (tg, 32940), "DL+TG_to_MV": (mv2, 52584)}
    for d, (grid, n) in official.items():
        gw = set(zip(grid.sim_idx.astype(int), grid.post_ms.round(3)))
        assert len(gw) == n, (d, len(gw), n)
        sub = pr[pr.direction == d]
        for (inp, mod, seed), s in sub.groupby(["input", "model", "seed"]):
            sw = list(zip(s.sim_idx.astype(int), s.tau.round(3)))
            ok = (len(sw) == n and set(sw) == gw and len(set(sw)) == n)
            # y must equal the grid's label
            ysm = s.assign(k=s.tau.round(3)).merge(grid.assign(k=grid.post_ms.round(3))[["sim_idx", "k", "y"]],
                                                   on=["sim_idx", "k"], suffixes=("", "_g"))
            ok = ok and len(ysm) == n and np.allclose(ysm.y, ysm.y_g)
            rec("a", f"windows_exact|{d}|{inp}|{mod}|{seed}", float(ok), 1.0, 0)
        # train grid(s) are other files: test grid sim_idx set vs the other files' (different grid, different sims)
    # direction labels: test windows must not equal the *source* grid windows
    srcsets = {"TG_to_DL": [tg], "DL_to_TG": [dl], "DL+TG_to_MV": [dl, tg]}
    for d, srcs in srcsets.items():
        t = set(zip(official[d][0].sim_idx.astype(int), official[d][0].post_ms.round(3)))
        # test grid file differs from every training grid file (distinct generating grids/files)
        rec("a", f"test_grid_distinct_from_train|{d}", float(all(len(s) != len(official[d][0]) or
                                                                  not np.array_equal(s.line.values, official[d][0].line.values)
                                                                  for s in srcs)), 1.0, 0)
        # preds only contain test-grid sim_idx within the test grid range
        sub = pr[pr.direction == d]
        rec("a", f"pred_sim_idx_subset_of_test_grid|{d}",
            float(set(sub.sim_idx.astype(int)) <= set(official[d][0].sim_idx.astype(int))), 1.0, 0)
    # d = 0.5 excluded MAE (seed-averaged), report only
    ex = pr[(pr.y - 0.5).abs() >= 1e-9]
    per_seed = ex.groupby(keys).err.mean().reset_index()
    avg = per_seed.groupby(["input", "model", "direction"]).err.mean().reset_index()
    nexc = ex.groupby(["input", "model", "direction", "seed"]).size().groupby(["input", "model", "direction"]).first()
    allavg = pr.groupby(keys).err.mean().reset_index().groupby(["input", "model", "direction"]).err.mean()
    for r in avg.itertuples():
        n_ex = int(nexc[(r.input, r.model, r.direction)])
        ROWS.append(dict(part="a_excl", key=f"{r.input}|{r.model}|{r.direction}", validated=r.err,
                         stored=allavg[(r.input, r.model, r.direction)], abs_dev=np.nan, tol=np.nan, ok=True))
        ROWS[-1]["n_excl"] = n_ex
    print(avg.round(4).to_string(index=False))


# --------------------------------------------------------------------------------------- (b)
def line_params(grid):
    from read_graph import load
    G = load(str(EV / grid / "graphs" / "graph_benchmark.pickle"))
    out = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if str(k).startswith("MainLn"):
            p = d[f"{k}_param_dict"]
            out[str(k)] = (p["rline"] * p["length"], p["xline"] * p["length"] / (2 * np.pi * F0))
    return out


def clarke(x3):
    a, b, c = x3[:, 0], x3[:, 1], x3[:, 2]
    return (2 * a - b - c) / 3.0, (b - c) / np.sqrt(3.0)


def taus(etype):
    return range(5, (50 if "incipient" in etype else 80) + 1, 5)


def locate(arr, e0, e1, R, L):
    """arr (n,12): S[Ia Ib Ic Va Vb Vc], R[...]; returns d over [e0,e1) """
    dt = 1.0 / FS
    num = den = 0.0
    sl = slice(e0, e1)
    for k in (0, 1):
        iS = clarke(arr[:, 0:3])[k]; vS = clarke(arr[:, 3:6])[k]
        iR = clarke(arr[:, 6:9])[k]; vR = clarke(arr[:, 9:12])[k]
        diS = savgol_filter(iS, 11, 2, deriv=1, delta=dt)
        diR = savgol_filter(iR, 11, 2, deriv=1, delta=dt)
        u = (vS - vR + R * iR + L * diR)[sl]
        w = (R * (iS + iR) + L * (diS + diR))[sl]
        num += float(np.dot(u, w)); den += float(np.dot(w, w))
    return num / den


def work_b(args):
    grid, sim, etype, line, loc, start, R, L, snrs = args
    xy = line[len("MainLn"):].split("-")
    bs, br = f"MainBus{xy[0]}", f"MainBus{xy[1][0]}"  # buses are single digit in all grids used
    # terminals: cubicle column blocks named *pex_MainBusX_<line>
    f = EV / grid / "data" / f"result{sim}.csv"
    with open(f, newline="") as fh:
        rd_ = csv.reader(fh)
        names = next(rd_); units = next(rd_)
    def block(bus):
        idx = [j for j, n in enumerate(names) if n.endswith(f"pex_{bus}_{line}")]
        assert len(idx) == 6 and idx == list(range(idx[0], idx[0] + 6)), (f, bus, idx)
        for k, ch in enumerate(("Isec:A", "Isec:B", "Isec:C", "Usec:A", "Usec:B", "Usec:C")):
            assert ch in units[idx[k]]
        return idx
    cols = block(bs) + block(br)
    arr = pd.read_csv(f, skiprows=2, header=None, usecols=cols, dtype=np.float64, engine="c")
    arr = arr[cols].to_numpy()
    i0 = int(round((start - 1.0) * FS))
    out = []
    n90 = int(0.09 * FS)
    for snr in snrs:
        if snr >= 900:
            x = arr
        else:
            rng = np.random.default_rng([SEED, sim, snr])
            rms = np.sqrt(np.mean(arr[:n90] ** 2, axis=0))
            x = arr + rng.normal(0.0, 1.0, arr.shape) * (rms / 10 ** (snr / 20.0))
        for tau in taus(etype):
            e = i0 + int(round(tau * FS / 1000.0))
            d = locate(x, e - WIN, e, R, L)
            est = 100.0 * d
            est = 50.0 if np.isnan(est) else min(max(est, 0.0), 100.0)
            out.append((sim, snr, tau, loc, est))
    return out


def part_b():
    C = __import__("common")
    for grid, tag, stored_name in (("DoubleLine", "DL", "c1_td_noise.csv"),
                                   ("TestGrid110kV", "TG", "c1_td_noise_TestGrid110kV.csv")):
        lab = pd.read_csv(EV / grid / "labels" / "settings_clean.csv")
        lab = lab[C.line_fault_mask(lab)]
        lp = line_params(grid)
        jobs = [(grid, int(r[C.C_IDX]), str(r[C.C_TYPE]), str(r[C.C_TARGET]), float(r[C.C_LOC]),
                 float(r[C.C_START]), *lp[str(r[C.C_TARGET])], (999, 40, 30)) for _, r in lab.iterrows()]
        with ProcessPoolExecutor(3) as ex:
            res = [x for ch in ex.map(work_b, jobs, chunksize=16) for x in ch]
        d = pd.DataFrame(res, columns=["sim_idx", "snr", "tau_ms", "loc_true", "est"])
        d["err"] = (d.est - d.loc_true).abs()
        d.to_parquet(RES / f"validate_part10b_{tag}_td_noise.parquet", **PQ)
        st = pd.read_csv(RES / stored_name)
        st = st[st.deriv == "sg"]
        # window-set check
        for snr in (40, 30):
            a = d[d.snr == snr]; b = st[st.snr == snr]
            same = set(zip(a.sim_idx, a.tau_ms)) == set(zip(b.sim_idx, b.tau_ms)) and len(a) == len(b)
            rec("b", f"{tag}|windows_match|snr{snr}", float(same), 1.0, 0)
            rec("b", f"{tag}|n|snr{snr}", len(a), len(b), 0)
            rec("b", f"{tag}|MAE|snr{snr}", a.err.mean(), b.err.mean(), 0.05)
        # noise-free sanity (stored snr 999 sg)
        a = d[d.snr == 999]; b = st[st.snr == 999]
        rec("b", f"{tag}|MAE|noise_free_sanity", a.err.mean(), b.err.mean(), 0.05)
        print(tag, {s: round(d[d.snr == s].err.mean(), 4) for s in (999, 40, 30)},
              {s: round(st[st.snr == s].err.mean(), 4) for s in (999, 40, 30)})


# --------------------------------------------------------------------------------------- (c)
HALF = 1e-9


def mae(e):
    return float(np.mean(e)) if len(e) else np.nan


def join_k(a, b, cols):
    A = a.assign(k=a.post_ms.round(3)); B = b.assign(k=b.post_ms.round(3))[["sim_idx", "k"] + cols]
    o = A.merge(B, on=["sim_idx", "k"], how="inner", validate="one_to_one")
    assert len(o) == len(a), (len(o), len(a))
    return o


def part_c():
    import classical_report as CR
    m6 = pd.read_csv(RES / "c1_prof_m6.csv")
    ex6 = pd.read_csv(RES / "c1_prof_m6_extra.csv").set_index("set")
    cl = {g: rd(f"c1_classical_{g}.parquet") for g in ("DL", "TG", "MV", "ADAPT")}
    grid = {g: rd(f"c1_grid_{g}.parquet") for g in ("DL", "TG", "MV")}
    # PFB predictions
    pfb = {}
    pfb["DL"] = join_k(cl["DL"], CR.zs_pred("TG", "DL"), ["pred"])
    pfb["TG"] = join_k(cl["TG"], CR.zs_pred("DL", "TG"), ["pred"])
    pfb["MV"] = join_k(cl["MV"], pd.read_csv(RES / "c1_zs_MV_pred.csv"), ["pred"])
    ap = pd.read_csv(RES / "c1_adapt_ingrid_pred_adapt_test.csv")
    pfb["ADAPT"] = join_k(cl["ADAPT"], ap, ["pred"])
    mvz = pd.read_csv(RES / "c1_zs_MV_pred.csv")
    for g in ("DL", "TG", "MV", "ADAPT"):
        c = cl[g]
        y = c.y.to_numpy(float)
        ev = {}
        # TD two-ended (own window set)
        if g == "ADAPT":
            t = pd.read_csv(RES / "c1_adapt_td.csv"); t = t[t.split == "test"]
            td_y = (t.loc_true / 100).to_numpy(float); td_e = err_pp(t.loc_est.to_numpy(float) / 100, td_y)
        else:
            gr = grid[g]
            gr = gr[gr.td2_est.notna()] if g == "MV" else gr
            td_y = gr.y.to_numpy(float); td_e = err_pp(gr.td2_est.to_numpy(float), td_y)
            if g != "MV":
                assert np.allclose(grid[g].y.values, y)
        ev["TD two-ended"] = (td_e, td_y)
        for nm, col in (("reactance", "R"), ("Takagi", "T"), ("Eriksson", "E")):
            ev[nm] = (err_pp(c[col].to_numpy(float), y), y)
        p = pfb[g]
        ev["PFB"] = (err_pp(p.pred.to_numpy(float), p.y.to_numpy(float)), p.y.to_numpy(float))
        for nm, (e, yy) in ev.items():
            keep = np.abs(yy - 0.5) >= HALF
            s = m6[(m6.set == g) & (m6.method == nm)].iloc[0]
            rec("c", f"{g}|{nm}|n_all", len(e), s.n_all, 0)
            rec("c", f"{g}|{nm}|mae_all", mae(e), s.mae_all, 0.5 if (nm == "PFB" and g in ("DL", "TG")) else 0.01)
            rec("c", f"{g}|{nm}|n_excl", int(keep.sum()), s.n_excl, 0)
            rec("c", f"{g}|{nm}|mae_excl", mae(e[keep]), s.mae_excl, 0.5 if (nm == "PFB" and g in ("DL", "TG")) else 0.01)
        # extras
        half = np.abs(y - 0.5) < HALF
        sc_mask = ~c.etype.str.contains("incipient|hif", case=False).to_numpy()
        ex = ex6.loc[g]
        rec("c", f"{g}|extra|n", len(c), ex.n, 0)
        rec("c", f"{g}|extra|share_half", 100 * half.mean(), ex.share_half, 0.01)
        rec("c", f"{g}|extra|share_hifinc_at_half(share of inc/hif windows with y=0.5)",
            100 * half[~sc_mask].mean() if (~sc_mask).any() else 0.0, ex.share_hifinc_at_half, 0.01)
        eund = c["E"].isna().to_numpy()
        rec("c", f"{g}|extra|share_E_undef_and_half", 100 * (eund & half).mean(), ex.share_E_undef_and_half, 0.01)
        pe = ev["PFB"][0]
        tol = 0.5 if g in ("DL", "TG") else 0.01
        rec("c", f"{g}|extra|pfb_mae_inc_hif", mae(pe[~sc_mask]), ex.pfb_mae_inc_hif, tol)
        rec("c", f"{g}|extra|pfb_mae_sc", mae(pe[sc_mask]), ex.pfb_mae_sc, tol)
        rec("c", f"{g}|extra|const_half_mae_inc_hif", mae(np.abs(0.5 - y[~sc_mask]) * 100), ex.const_half_mae_inc_hif, 0.01)


def main():
    parts = sys.argv[1:] or ["a", "b", "c"]
    for p in parts:
        {"a": part_a, "b": part_b, "c": part_c}[p]()
    new = pd.DataFrame(ROWS)
    if OUT.exists() and sys.argv[1:]:
        old = pd.read_csv(OUT)
        old = old[~old.part.str.split("_").str[0].isin(parts)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(OUT, index=False)
    new["pp"] = new.part.str.split("_").str[0]
    chk = new[new.part != "a_excl"]
    print(chk.groupby("pp").agg(max_dev=("abs_dev", "max"), all_ok=("ok", "all"), n=("ok", "size")))
    print("seed", SEED)


if __name__ == "__main__":
    main()
