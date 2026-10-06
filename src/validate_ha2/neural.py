"""Step 4: equal-information learned baselines (MLP / GRU, -raw / +Z), zero-shot, both directions.

Input: the raw 480 x 6 LOCAL window (float32); '+Z' adds (R1, X1, R0, X0) of the faulted line.
Resumable: one row per (model, variant, direction, seed) appended to
results/validate_ha2_neural_runs.csv; finished keys are skipped.
"""
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

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS = C.RESULTS / "validate_ha2_neural_runs.csv"
Z_COLS = ["R1", "X1", "R0", "X0"]  # line nameplate only (allowed for '+Z' normalisation input)
EPOCHS, PATIENCE, BATCH = 60, 12, 256


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        layers, d = [], d_in
        for h in (512, 256, 128):
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        h = x.flatten(1)
        if z is not None:
            h = torch.cat([h, z], dim=1)
        return self.net(h).squeeze(1)


class GRUNet(nn.Module):
    def __init__(self, n_z: int):
        super().__init__()
        self.gru = nn.GRU(input_size=6, hidden_size=128, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(128 + n_z, 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        _, hN = self.gru(x)
        h = self.drop(hN[-1])
        if z is not None:
            h = torch.cat([h, z], dim=1)
        return self.head(h).squeeze(1)


def load(grid: str) -> tuple[np.ndarray, pd.DataFrame]:
    w = np.load(C.RESULTS / f"validate_ha2_windows_{grid}.npz")
    meta = pd.read_parquet(C.RESULTS / f"validate_ha2_tokens_{grid}.parquet",
                           columns=["sim_idx", "tau", "ftype", "y"] + Z_COLS)
    assert (w["sim_idx"] == meta["sim_idx"].to_numpy()).all() and (w["tau"] == meta["tau"].to_numpy()).all()
    return w["X"], meta


def predict(model: nn.Module, X: torch.Tensor, Z: torch.Tensor | None) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            out.append(model(X[i:i + 2048], None if Z is None else Z[i:i + 2048]).float().cpu())
    return torch.cat(out).numpy()


def run(model_name: str, variant: str, src: str, tgt: str, seed: int, data: dict) -> dict:
    Xs, ms = data[src]
    Xt, mt = data[tgt]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    eps = np.unique(ms["sim_idx"].to_numpy())
    n_val = int(math.ceil(0.10 * len(eps)))
    val_eps = set(rng.permutation(eps)[:n_val].tolist())
    is_val = ms["sim_idx"].isin(val_eps).to_numpy()
    tr_i, va_i = np.where(~is_val)[0], np.where(is_val)[0]

    # standardisation with training-split statistics only
    mu = Xs[tr_i].reshape(-1, 6).mean(axis=0, dtype=np.float64)
    sd = Xs[tr_i].reshape(-1, 6).std(axis=0, dtype=np.float64)
    ys = ms["y"].to_numpy(np.float64)
    ymu, ysd = ys[tr_i].mean(), ys[tr_i].std()

    def tx(X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(DEV)

    Xtr_all, Xte = tx(Xs), tx(Xt)
    Ytr_all = torch.from_numpy(((ys - ymu) / ysd).astype(np.float32)).to(DEV)
    if variant == "Z":
        zs = ms[Z_COLS].to_numpy(np.float64)
        zmu, zsd = zs[tr_i].mean(axis=0), zs[tr_i].std(axis=0)
        zsd = np.where(zsd > 0, zsd, 1.0)
        Zs = torch.from_numpy(((zs - zmu) / zsd).astype(np.float32)).to(DEV)
        Zt = torch.from_numpy(((mt[Z_COLS].to_numpy(np.float64) - zmu) / zsd).astype(np.float32)).to(DEV)
        n_z = 4
    else:
        Zs = Zt = None
        n_z = 0
    model = (MLP(480 * 6 + n_z) if model_name == "MLP" else GRUNet(n_z)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss()
    tr_t = torch.from_numpy(tr_i).to(DEV)
    va_t = torch.from_numpy(va_i).to(DEV)
    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen).to(DEV)]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:  # BatchNorm needs > 1 sample
                continue
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr_all[b], None if Zs is None else Zs[b]), Ytr_all[b])
            loss.backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, len(va_t), 2048):
                b = va_t[i:i + 2048]
                vl += float(lossf(model(Xtr_all[b], None if Zs is None else Zs[b]), Ytr_all[b])) * len(b)
            vl /= len(va_t)
        if vl < best:
            best, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    raw = predict(model, Xte, Zt) * ysd + ymu
    y = mt["y"].to_numpy()
    pred = np.clip(raw, 0, 1)
    err = np.abs(pred - y) * 100
    tau = mt["tau"].to_numpy()
    tag = f"{model_name}_{variant}_{src}to{tgt}_s{seed}"
    np.save(C.RESULTS / f"validate_ha2_nnpred_{tag}.npy", raw.astype(np.float64))
    row = {"model": model_name, "variant": variant, "direction": f"{src}to{tgt}", "seed": seed,
           "MAE": err.mean(), "MAE_unclipped": (np.abs(raw - y) * 100).mean(), "median": np.median(err),
           "MAE_tau_le15": err[tau <= 15].mean(), "MAE_tau_ge21": err[tau >= 21].mean(),
           "best_epoch": best_ep + 1, "epochs_run": ep + 1, "val_loss": best, "secs": time.time() - t0}
    for ft in sorted(mt["ftype"].unique()):
        row[f"MAE_{ft}"] = err[(mt["ftype"] == ft).to_numpy()].mean()
    return row


def main() -> None:
    torch.backends.cudnn.benchmark = True
    print("device", DEV, flush=True)
    data = {g: load(g) for g in ("DL", "TG")}
    done = set()
    if RUNS.exists():
        prev = pd.read_csv(RUNS)
        done = {(r.model, r.variant, r.direction, int(r.seed)) for r in prev.itertuples()}
    for src, tgt in (("TG", "DL"), ("DL", "TG")):
        for model_name in ("MLP", "GRU"):
            for variant in ("raw", "Z"):
                for seed in (0, 1, 2):
                    key = (model_name, variant, f"{src}to{tgt}", seed)
                    if key in done:
                        continue
                    row = run(model_name, variant, src, tgt, seed, data)
                    pd.DataFrame([row]).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
                    print(f"{key}: MAE {row['MAE']:.3f} best_ep {row['best_epoch']} "
                          f"({row['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
