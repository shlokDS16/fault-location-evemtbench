"""Design section 23: one-ended conditioning analysis. Error of any one-ended locator = rho * kappa * sin(angle error),
rho = R_f / |Z_L|. Parts (a)-(d) on 1phg_shc windows (post >= 25 ms, both ends measured); part (e) MAE vs rho bins.
Usage: python src/c1/conditioning.py  -> results/c1_cond_windows.parquet, c1_cond_summary.csv, c1_cond_rho_bins.csv"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_pass  # noqa: E402
from classical_oe import jobs, ph, loop_quant, PRE_GAP  # noqa: E402
from local_features import N, oracle_loop  # noqa: E402
from stage_a import terminal_cols  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
FS = 9600.0
EPS = 0.05


def one(args):
    row, LP, gdir = args
    if row["events/event_type"] != "flt_1phg_shc":
        return []
    line = row["events/event_target"]
    m = grid_pass.LINE_RE.match(line)
    f = gdir / "data" / f"result{int(row['general/sim_idx'])}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    if not (grid_pass.has_term(hdr, m.group(1), line) and grid_pass.has_term(hdr, m.group(2), line)):
        return []
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    XS, XR = [d[c].to_numpy(float) for c in cs], [d[c].to_numpy(float) for c in cr]
    nf, ends = grid_pass.window_ends(row, len(XS[0]))
    Z1, Z0 = LP[line][0], LP[line][1]
    k0 = (Z0 - Z1) / (3 * Z1)
    ol = oracle_loop(row)
    p = "abc".index(ol[0])
    Spre = ph(XS, nf - PRE_GAP - N)
    y, Rf = row["events/event_flt_target_line_location"] / 100.0, row["events/event_flt_shc_resistance"]
    th = np.angle(Z1)
    out = []
    for s1 in ends:
        post = (s1 - nf) / FS * 1000
        if post < 25:
            continue
        S, Rr = ph(XS, s1 - N), ph(XR, s1 - N)
        V, I, dI = loop_quant(S, Spre, ol, k0)
        IF = S[p] + Rr[p]
        b = np.angle(IF / I)
        rec = dict(sim_idx=int(row["general/sim_idx"]), line=line, post_ms=post, R_f=Rf, y=y,
                   rho=Rf / abs(Z1), absZ=abs(Z1), ratio_IF_I=abs(IF / I), beta_deg=np.degrees(b))
        for nm, bh in (("R", 0.0), ("T", np.angle(dI / I))):
            dh = np.imag((V / I) * np.exp(-1j * bh)) / np.imag(Z1 * np.exp(-1j * bh))
            kap = abs(IF / I) / np.sin(th - bh)
            rec[f"e_obs_{nm}"] = dh - y
            rec[f"e_pred_{nm}"] = Rf / abs(Z1) * abs(IF / I) * np.sin(b - bh) / np.sin(th - bh)
            rec[f"kappa_{nm}"] = kap
            rec[f"angerr_{nm}_deg"] = np.degrees(np.angle(np.exp(1j * (b - bh))))
        rec["Dreq_deg"] = np.degrees(np.arcsin(min(1.0, EPS / max(rec["rho"] * abs(rec["kappa_T"]), 1e-12))))
        out.append(rec)
    return out


def rho_bins():
    """(e): MAE vs rho for all short-circuit windows with post >= 25 ms (two-ended, R, T, E, H-A2)."""
    import grid_pass as gp
    rows = []
    edges = [0, 0.3, 1, 3, 10, 30, 1e9]
    sets = {"DL": ("c1_classical_DL.parquet", "c1_grid_DL.parquet", "zsTG"),
            "TG": ("c1_classical_TG.parquet", "c1_grid_TG.parquet", "zsDL"),
            "MV": ("c1_classical_MV.parquet", "c1_grid_MV.parquet", "c1_zs_MV_pred.csv")}
    from classical_report import zs_pred, join
    for g, (cf, gf, hp) in sets.items():
        cl = pd.read_parquet(R / cf)
        gr = pd.read_parquet(R / gf)[["sim_idx", "post_ms", "td2_est", "Z1r", "Z1x"]]
        pr = zs_pred("TG", "DL") if hp == "zsTG" else zs_pred("DL", "TG") if hp == "zsDL" else pd.read_csv(R / hp)
        d = join(cl, pr)
        d = d.merge(gr.assign(k=gr.post_ms.round(3)).drop(columns="post_ms"), on=["sim_idx", "k"], validate="one_to_one")
        d = d[~d.etype.str.contains("incipient|hif") & (d.post_ms >= 25)]
        d["rho"] = d.R_f / np.hypot(d.Z1r, d.Z1x)
        d["bin"] = pd.cut(d.rho, edges)
        for m, col in (("two-ended", "td2_est"), ("R", "R"), ("T", "T"), ("E", "E"), ("H-A2", "pred")):
            k = d[col].notna() if m == "two-ended" else pd.Series(True, index=d.index)  # two-ended undefined on 8-14
            e = ((d.loc[k, col].fillna(0.5).clip(0, 1) - d.loc[k, "y"]).abs() * 100)
            for b, gg in e.groupby(d.loc[k, "bin"], observed=True):
                rows.append(dict(grid=g, method=m, rho_bin=str(b), n=len(gg), mae=gg.mean()))
    return pd.DataFrame(rows)


def main():
    allw = []
    for tag in ("DL", "TG", "ADAPT", "MV"):
        with ProcessPoolExecutor(6) as ex:
            w = pd.DataFrame([r for rr in ex.map(one, jobs(tag), chunksize=4) for r in rr])
        w.insert(0, "grid", tag)
        allw.append(w)
        print(tag, len(w), "windows", flush=True)
    W = pd.concat(allw, ignore_index=True)
    W.to_parquet(R / "c1_cond_windows.parquet", index=False)
    summ = []
    for g, d in W.groupby("grid", sort=False):
        r = dict(grid=g, n=len(d), episodes=d.sim_idx.nunique())
        for nm in ("R", "T"):
            dev = (d[f"e_obs_{nm}"] - d[f"e_pred_{nm}"]).abs() * 100
            r[f"ident_{nm}_med_pp"], r[f"ident_{nm}_p90_pp"] = dev.median(), dev.quantile(0.9)
            r[f"med_abs_eobs_{nm}"] = (d[f"e_obs_{nm}"].abs() * 100).median()
        r.update(rho_med=d.rho.median(), rho_p90=d.rho.quantile(0.9), kappaT_med=d.kappa_T.abs().median(),
                 angerr_T_med_deg=d.angerr_T_deg.abs().median(), angerr_T_p90_deg=d.angerr_T_deg.abs().quantile(0.9),
                 Dreq_med_deg=d.Dreq_deg.median(), share_Dreq_lt1deg=(d.Dreq_deg < 1).mean())
        summ.append(r)
    S = pd.DataFrame(summ)
    S.to_csv(R / "c1_cond_summary.csv", index=False)
    print(S.round(3).T.to_string())
    B = rho_bins()
    B.to_csv(R / "c1_cond_rho_bins.csv", index=False)
    print(B.pivot_table(index=["grid", "rho_bin"], columns="method", values="mae").round(1).to_string())


if __name__ == "__main__":
    main()
