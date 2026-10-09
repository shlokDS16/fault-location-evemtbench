"""Design section 28: physical-input ('P') two-ended networks through the measurement chain.
Distorted UW windows: the s.22 chain (meas_chain.Chain, applied to both terminal records, full record, before
windowing, currents in primary amperes, exactly as chain_pass.py -> grid_pass.one) followed by the same window
construction as src/c1/two_end_windows.py (Clarke, SG 11/2 derivative, nameplate R1/L1, Eq. (3) u/w).
Networks: MLP and GRU of two_end_learn.py, recipe unchanged, trained on CLEAN source-grid windows (c1_raw2*.npz,
input 'P' = UW / joint RMS of w), seeds 0-2; tested on the clean and the distorted target grid.
Self-checks (printed, also written to results/c1_two_end_chain_selfcheck.csv):
  (1) LS ratio sum(u*w)/sum(w*w) on the distorted UW vs the two_ended column of results/c1_chain_eval.csv;
  (2) clean network MAE vs results/c1_two_end_learn.csv (input P).
Outputs: results/c1_two_end_chain.csv, results/c1_two_end_chain_preds.parquet. Predictions parts: results/c1_two_end_chain_parts/ (distorted UW cached in memory only).
Usage: python src/c1/two_end_chain.py  (resumable)"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_pass import LINE_RE, ROOT, W0, der, has_term, line_params, window_ends  # noqa: E402
from meas_chain import Chain  # noqa: E402
from stage_a import terminal_cols  # noqa: E402
from td_locator import clarke  # noqa: E402
import two_end_learn as tel  # noqa: E402

R = ROOT / "results"
PARTS = R / "c1_two_end_chain_parts"
GRIDS = {"DL": "DoubleLine", "TG": "TestGrid110kV"}
RAW = {"DL": "", "TG": "_TestGrid110kV"}
MEM = {}
SCEN = ["CT-severe", "CVT-5", "CVT-15", "FULL"]
DIRS = (("TG", "DL"), ("DL", "TG"))


def one(args):
    """Same row filter / window construction as two_end_windows.one, with the chain applied to both ends first."""
    row, LP, gdir, scen = args
    line = row["events/event_target"]
    m = LINE_RE.match(line)
    sid = int(row["general/sim_idx"])
    f = gdir / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    flip = not has_term(hdr, m.group(1), line)
    if flip or not has_term(hdr, m.group(2), line):
        return None
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    chain = Chain(scen)
    XS = chain([d[c].to_numpy(float) for c in cs], sid, 0)
    XR = chain([d[c].to_numpy(float) for c in cr], sid, 1)
    nf, ends = window_ends(row, len(XS[0]))
    Z1, _, _ = LP[line]
    Rr, L = Z1.real, Z1.imag / W0
    iS, vS, iR, vR = clarke(*XS[:3]), clarke(*XS[3:]), clarke(*XR[:3]), clarke(*XR[3:])
    modes = [(vS[k] - vR[k] + Rr * iR[k] + L * der(iR[k]), Rr * (iS[k] + iR[k]) + L * (der(iS[k]) + der(iR[k])))
             for k in range(2)]
    UWfull = np.stack([modes[0][0], modes[1][0], modes[0][1], modes[1][1]], 1)
    return np.stack([UWfull[s1 - 480:s1].astype(np.float32) for s1 in ends])


def build(g, scen):
    if (g, scen) in MEM:  # in-memory cache only (the disk was full)
        return MEM[(g, scen)]
    gdir = ROOT / "data" / "raw" / "evemt" / GRIDS[g]
    LP = line_params(gdir)
    s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
    fl = s[s["events/event_type"].str.startswith("flt_") & s["events/event_target"].str.match(LINE_RE)
           & s["events/event_flt_target_line_location"].notna()]
    t0 = time.time()
    with ProcessPoolExecutor(4) as ex:
        res = [r for r in ex.map(one, [(r, LP, gdir, scen) for _, r in fl.iterrows()], chunksize=4) if r is not None]
    UW = np.concatenate(res)
    ref = np.load(R / f"c1_raw2{RAW[g]}.npz")
    assert len(UW) == len(ref["UW"]), (len(UW), len(ref["UW"]))
    MEM[(g, scen)] = UW
    print(f"built {g} {scen}: {UW.shape} in {time.time() - t0:.0f}s", flush=True)
    return UW


def ls(UW):
    A = UW.astype(np.float64)
    return (A[:, :, :2] * A[:, :, 2:]).sum((1, 2)) / ((A[:, :, 2:] ** 2).sum((1, 2)) + 1e-12)


def scaleP(UW):
    rms = np.sqrt((UW[:, :, 2:].astype(np.float64) ** 2).mean((1, 2)))
    return (UW / rms[:, None, None].astype(np.float32)).astype(np.float32)


def fit_predict_multi(model_name, Xtr, ytr, gtr, Xtes, seed, epochs=60, bs=256, patience=12):
    """two_end_learn.fit_predict (use_z=False) verbatim in training; predicts every array in Xtes with the same
    normalisation. Returns list of clipped predictions."""
    DEV = tel.DEV
    C = Xtr.shape[2]
    Ztr = np.zeros((len(Xtr), 4), np.float32)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eps = np.unique(gtr)
    val_eps = set(rng.choice(eps, max(1, len(eps) // 10), replace=False).tolist())
    vm = np.array([g in val_eps for g in gtr])
    cm, cs = Xtr[~vm].reshape(-1, C).mean(0), Xtr[~vm].reshape(-1, C).std(0) + 1e-6
    zm, zs = Ztr[~vm].mean(0), Ztr[~vm].std(0) + 1e-6
    ym, ys = ytr[~vm].mean(), ytr[~vm].std() + 1e-6

    def T(X, Z):
        return (torch.tensor((X - cm) / cs, dtype=torch.float32), torch.tensor((Z - zm) / zs, dtype=torch.float32))

    Xa, Za = T(Xtr, Ztr)
    ya = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    tr_idx, va_idx = np.where(~vm)[0], np.where(vm)[0]
    net = (tel.MLP(480 * C, False) if model_name == "MLP" else tel.GRU(C, False)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = lossf(net(Xa[b].to(DEV), Za[b].to(DEV)), ya[b].to(DEV))
            loss.backward()
            opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            vl = np.mean([lossf(net(Xa[va_idx[i:i + 1024]].to(DEV), Za[va_idx[i:i + 1024]].to(DEV)),
                                ya[va_idx[i:i + 1024]].to(DEV)).item() for i in range(0, len(va_idx), 1024)])
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    del Xa, Za
    outs = []
    for Xte in Xtes:
        Xb, Zb = T(Xte, np.zeros((len(Xte), 4), np.float32))
        with torch.no_grad():
            p = np.concatenate([net(Xb[i:i + 2048].to(DEV), Zb[i:i + 2048].to(DEV)).cpu().numpy()
                                for i in range(0, len(Xb), 2048)])
        outs.append(np.clip(p * ys + ym, 0, 1))
        del Xb
    return outs, ep + 1


def main():
    PARTS.mkdir(exist_ok=True)
    T0 = time.time()
    ev = pd.read_csv(R / "c1_chain_eval.csv")
    chk = []
    # ---- build distorted windows + self-check (1)
    for g in GRIDS:
        ref = np.load(R / f"c1_raw2{RAW[g]}.npz")
        y = ref["y"]
        direction = "TG_to_DL" if g == "DL" else "DL_to_TG"
        for sc in SCEN:
            UW = build(g, sc)
            mine = float(np.mean(np.abs(np.clip(ls(UW), 0, 1) - y)) * 100)
            theirs = float(ev[(ev.direction == direction) & (ev.scenario == sc)].two_ended.iloc[0])
            chk.append(dict(check="LS_vs_TD", grid=g, scenario=sc, mine=mine, reference=theirs, diff_pp=mine - theirs))
            print(f"selfcheck1 {g} {sc}: LS {mine:.4f} vs TD {theirs:.4f} (diff {mine - theirs:+.4f} pp)", flush=True)
    sys.stdout.flush()

    # ---- train + predict
    OUTCSV = R / "c1_two_end_chain.csv"
    done = set()
    if OUTCSV.exists():
        prev = pd.read_csv(OUTCSV)
        done = set(zip(prev.model, prev.direction, prev.seed))
    for src, tgt in DIRS:
        dname = f"{src}_to_{tgt}"
        S = np.load(R / f"c1_raw2{RAW[src]}.npz")
        Xtr, ytr, gtr = scaleP(S["UW"]), S["y"].astype(np.float32), S["sim_idx"]
        Tn = np.load(R / f"c1_raw2{RAW[tgt]}.npz")
        y = Tn["y"].astype(np.float32)
        tests = [("clean", scaleP(Tn["UW"]))] + [(sc, scaleP(build(tgt, sc))) for sc in SCEN]
        for model in ("MLP", "GRU"):
            for seed in (0, 1, 2):
                if (model, dname, seed) in done:
                    continue
                t0 = time.time()
                preds, ep = fit_predict_multi(model, Xtr, ytr, gtr, [t[1] for t in tests], seed)
                rows = []
                for (sc, _), p in zip(tests, preds):
                    np.save(PARTS / f"pred_{model}_{dname}_{sc}_s{seed}.npy", p.astype(np.float32))
                    rows.append(dict(model=model, direction=dname, scenario=sc, seed=seed,
                                     mae=float(np.mean(np.abs(p - y)) * 100), n=len(p)))
                pd.DataFrame(rows).to_csv(OUTCSV, mode="a", header=not OUTCSV.exists(), index=False)
                print(f"{model} {dname} s{seed} ep{ep} {time.time() - t0:.0f}s: "
                      + ", ".join(f"{r['scenario']} {r['mae']:.3f}" for r in rows), flush=True)
        del Xtr, tests

    # ---- self-check (2)
    df = pd.read_csv(OUTCSV)
    base = pd.read_csv(R / "c1_two_end_learn.csv")
    base = base[base.input == "P"]
    for (model, d, seed), r in df[df.scenario == "clean"].groupby(["model", "direction", "seed"]):
        b = base[(base.model == model) & (base.direction == d) & (base.seed == seed)].mae.iloc[0]
        chk.append(dict(check="clean_net_vs_s26", grid=d, scenario=f"{model}_s{seed}", mine=float(r.mae.iloc[0]),
                        reference=float(b), diff_pp=float(r.mae.iloc[0] - b)))
    c = pd.DataFrame(chk)
    c.to_csv(R / "c1_two_end_chain_selfcheck.csv", index=False)
    print(c.assign(ad=c.diff_pp.abs()).groupby("check").ad.max().rename("max |diff| pp").to_string(), flush=True)

    # ---- assemble predictions
    meta = {g: np.load(R / f"c1_raw2{RAW[g]}.npz") for g in GRIDS}
    parts = []
    for _, r in df.iterrows():
        m = meta[r.direction.split("_to_")[1]]
        p = np.load(PARTS / f"pred_{r.model}_{r.direction}_{r.scenario}_s{r.seed}.npy")
        parts.append(pd.DataFrame(dict(sim_idx=m["sim_idx"], tau=m["tau"], y=m["y"], pred=p, model=r.model,
                                       direction=r.direction, scenario=r.scenario, seed=int(r.seed))))
    pd.concat(parts, ignore_index=True).to_parquet(R / "c1_two_end_chain_preds.parquet", index=False, engine="fastparquet")
    print(df.groupby(["model", "direction", "scenario"]).mae.mean().unstack("scenario").round(3).to_string())
    print(f"total {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
