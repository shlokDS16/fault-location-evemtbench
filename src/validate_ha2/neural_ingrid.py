"""Part 3: in-grid equal-information baselines (same fixed recipe as neural.py: MLP / GRU, -raw / +Z, seeds 0-2).
Train: adapt TRAIN line-fault windows (local 480 x 6 + faulted-line R1, X1, R0, X0); 10 % of TRAIN episodes
(grouped by sim_idx) for validation / early stopping. Test: (a) adapt TEST line-fault windows, (b) benchmark DL.
Resumable: one row per (model, variant, seed, test) appended to results/validate_ha2_ingrid_neural_runs.csv."""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from neural import BATCH, EPOCHS, MLP, PATIENCE, GRUNet  # noqa: E402

DEV = torch.device("cuda")
R = C.RESULTS
RUNS = R / "validate_ha2_ingrid_neural_runs.csv"
Z_COLS = ["R1", "X1", "R0", "X0"]


def build_adapt() -> tuple[np.ndarray, pd.DataFrame]:
    path = R / "validate_ha2_ingrid_windows.npz"
    meta = pd.read_parquet(R / "validate_ha2_ingrid_tokens.parquet",
                           columns=["sim_idx", "split", "s1", "pft_ms", "ftype", "line", "y"])
    zmap = {ln: (z1.real, z1.imag, z0.real, z0.imag) for ln in C.LINE_LEN["AD"] for z1, z0 in [C.line_z("AD", ln)]}
    meta[Z_COLS] = np.array([zmap[ln] for ln in meta["line"]])
    if path.exists():
        d = np.load(path)
        assert (d["sim_idx"] == meta["sim_idx"].to_numpy()).all() and (d["s1"] == meta["s1"].to_numpy()).all()
        return d["X"], meta
    X = np.empty((len(meta), C.WIN, 6), np.float32)
    for sid, g in meta.groupby("sim_idx"):
        sig = np.load(C.CACHE / "AD" / f"ep{sid}.npy")
        for i, s1 in zip(g.index, g["s1"]):
            X[i] = sig[s1 - C.WIN:s1].astype(np.float32)
    np.savez(path, X=X, sim_idx=meta["sim_idx"].to_numpy(), s1=meta["s1"].to_numpy())
    return X, meta


def build_bench() -> tuple[np.ndarray, pd.DataFrame]:
    w = np.load(R / "validate_ha2_windows_DL.npz")
    meta = pd.read_parquet(R / "validate_ha2_tokens_DL.parquet", columns=["sim_idx", "tau", "ftype", "y"] + Z_COLS)
    assert (w["sim_idx"] == meta["sim_idx"].to_numpy()).all()
    meta["pft_ms"] = meta["tau"].astype(float)
    return w["X"], meta


def metrics(pred: np.ndarray, m: pd.DataFrame) -> dict:
    err = np.abs(np.clip(pred, 0, 1) - m["y"].to_numpy()) * 100
    pft = m["pft_ms"].to_numpy()
    sc = ~m["ftype"].str.contains("incipient").to_numpy()
    return {"MAE": err.mean(), "median": np.median(err), "MAE_pft_le15": err[pft <= 15].mean(),
            "MAE_pft_ge21": err[pft >= 21].mean(), "MAE_shortcircuit": err[sc].mean(),
            "MAE_unclipped": (np.abs(pred - m["y"].to_numpy()) * 100).mean()}


def run(model_name: str, variant: str, seed: int, Xs, ms, tests) -> list[dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    eps = np.unique(ms["sim_idx"].to_numpy())
    val_eps = set(rng.permutation(eps)[: int(math.ceil(0.10 * len(eps)))].tolist())
    is_val = ms["sim_idx"].isin(val_eps).to_numpy()
    tr_i, va_i = np.where(~is_val)[0], np.where(is_val)[0]
    mu = Xs[tr_i].reshape(-1, 6).mean(axis=0, dtype=np.float64)
    sd = Xs[tr_i].reshape(-1, 6).std(axis=0, dtype=np.float64)
    ys = ms["y"].to_numpy(np.float64)
    ymu, ysd = ys[tr_i].mean(), ys[tr_i].std()
    zs = ms[Z_COLS].to_numpy(np.float64)
    zmu, zsd = zs[tr_i].mean(axis=0), zs[tr_i].std(axis=0)
    zsd = np.where(zsd > 0, zsd, 1.0)

    def tx(X):
        return torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(DEV)

    def tz(Z):
        return torch.from_numpy(((Z - zmu) / zsd).astype(np.float32)).to(DEV) if variant == "Z" else None

    Xa, Za = tx(Xs), tz(zs)
    Ya = torch.from_numpy(((ys - ymu) / ysd).astype(np.float32)).to(DEV)
    n_z = 4 if variant == "Z" else 0
    model = (MLP(480 * 6 + n_z) if model_name == "MLP" else GRUNet(n_z)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss()
    tr_t, va_t = torch.from_numpy(tr_i).to(DEV), torch.from_numpy(va_i).to(DEV)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen).to(DEV)]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            lossf(model(Xa[b], None if Za is None else Za[b]), Ya[b]).backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = sum(float(lossf(model(Xa[b], None if Za is None else Za[b]), Ya[b])) * len(b)
                     for b in (va_t[i:i + 2048] for i in range(0, len(va_t), 2048))) / len(va_t)
        if vl < best:
            best, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    del Xa
    rows = []
    for tname, (Xt, mt) in tests.items():
        Xt_t, Zt = tx(Xt), tz(mt[Z_COLS].to_numpy(np.float64))
        with torch.no_grad():
            raw = torch.cat([model(Xt_t[i:i + 2048], None if Zt is None else Zt[i:i + 2048]).float().cpu()
                             for i in range(0, len(Xt_t), 2048)]).numpy() * ysd + ymu
        del Xt_t
        np.save(R / f"validate_ha2_ingrid_nnpred_{model_name}_{variant}_s{seed}_{tname}.npy", raw)
        rows.append({"model": model_name, "variant": variant, "seed": seed, "test": tname, "n": len(mt),
                     **metrics(raw, mt), "best_epoch": best_ep + 1, "epochs_run": ep + 1, "val_loss": best,
                     "n_val_episodes": len(val_eps), "secs": time.time() - t0})
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    torch.backends.cudnn.benchmark = True
    Xa, ma = build_adapt()
    tr = (ma["split"] == "train").to_numpy()
    te = (ma["split"] == "test").to_numpy()
    assert not set(ma.loc[tr, "sim_idx"]) & set(ma.loc[te, "sim_idx"])
    Xs, ms = Xa[tr], ma[tr].reset_index(drop=True)
    tests = {"adapt_test": (Xa[te], ma[te].reset_index(drop=True)), "benchmark_DL": build_bench()}
    print("train", len(ms), "episodes", ms["sim_idx"].nunique(), {k: len(v[1]) for k, v in tests.items()}, flush=True)
    done = set()
    if RUNS.exists():
        p = pd.read_csv(RUNS)
        done = {(r.model, r.variant, int(r.seed)) for r in p.itertuples()}
    for model_name in ("MLP", "GRU"):
        for variant in ("raw", "Z"):
            for seed in (0, 1, 2):
                if (model_name, variant, seed) in done:
                    continue
                rows = run(model_name, variant, seed, Xs, ms, tests)
                pd.DataFrame(rows).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
                print(model_name, variant, seed, [f"{r['test']} {r['MAE']:.3f}" for r in rows],
                      f"best_ep {rows[0]['best_epoch']} ({rows[0]['secs']:.0f}s)", flush=True)
    p = pd.read_csv(RUNS)
    s = p.groupby(["model", "variant", "test"])[["MAE", "median", "MAE_pft_le15", "MAE_pft_ge21", "MAE_shortcircuit"]].agg(["mean", "std"])
    s.to_csv(R / "validate_ha2_ingrid_neural_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s.round(3).to_string())


if __name__ == "__main__":
    main()
