"""Equal-information learned baselines (EvEMTBench recipe re-implemented): MLP (512-256-128, BN, ReLU, dropout 0.3) and
GRU (hidden 128, 1 layer, dropout 0.3 head), AdamW lr 1e-3 wd 1e-4, cosine LR, 60 epochs, batch 256, early stopping
(patience 12) on an episode-grouped 10 % validation split, channel- and target-standardisation, MSE loss.
Variants: '-raw' (as published: waveforms only) and '+Z' (plus the faulted line's nameplate R1, X1, R0, X0 - the same
line information our physics method uses). Protocol: zero-shot, train on one grid -> test on the other.
Resumable: appends one row per (model, variant, direction, seed) to results/c1_equal_info.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

R = Path(__file__).resolve().parents[2] / "results"
OUT = R / "c1_equal_info.csv"
DEV = "cuda"


class MLP(nn.Module):
    def __init__(self, d_in, use_z):
        super().__init__()
        layers, prev = [], d_in + (4 if use_z else 0)
        for h in (512, 256, 128):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        self.net = nn.Sequential(*layers, nn.Linear(prev, 1))
        self.use_z = use_z

    def forward(self, x, z):
        x = x.flatten(1)
        return self.net(torch.cat([x, z], 1) if self.use_z else x).squeeze(1)


class GRU(nn.Module):
    def __init__(self, use_z):
        super().__init__()
        self.gru = nn.GRU(6, 128, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 + (4 if use_z else 0), 1))
        self.use_z = use_z

    def forward(self, x, z):
        h = self.gru(x)[0][:, -1]
        return self.head(torch.cat([h, z], 1) if self.use_z else h).squeeze(1)


def load(tag):
    d = np.load(R / f"c1_raw{tag}.npz")
    return d["X"], d["Z"], d["y"].astype(np.float32), d["sim_idx"], d["tau"]


def fit_predict(model_name, use_z, Xtr, Ztr, ytr, gtr, Xte, Zte, seed, epochs=60, bs=256, patience=12):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    eps = np.unique(gtr)
    val_eps = set(rng.choice(eps, max(1, len(eps) // 10), replace=False).tolist())
    vm = np.array([g in val_eps for g in gtr])
    cm, cs = Xtr[~vm].reshape(-1, 6).mean(0), Xtr[~vm].reshape(-1, 6).std(0) + 1e-6
    zm, zs = Ztr[~vm].mean(0), Ztr[~vm].std(0) + 1e-6
    ym, ys = ytr[~vm].mean(), ytr[~vm].std() + 1e-6

    def T(X, Z):
        return (torch.tensor((X - cm) / cs, dtype=torch.float32), torch.tensor((Z - zm) / zs, dtype=torch.float32))

    Xa, Za = T(Xtr, Ztr)
    ya = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    tr_idx, va_idx = np.where(~vm)[0], np.where(vm)[0]
    net = (MLP(480 * 6, use_z) if model_name == "MLP" else GRU(use_z)).to(DEV)
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
    Xb, Zb = T(Xte, Zte)
    with torch.no_grad():
        p = np.concatenate([net(Xb[i:i + 2048].to(DEV), Zb[i:i + 2048].to(DEV)).cpu().numpy() for i in range(0, len(Xb), 2048)])
    return np.clip(p * ys + ym, 0, 1), ep + 1


def main():
    data = {"DL": load(""), "TG": load("_TestGrid110kV")}
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev.model, prev.variant, prev.direction, prev.seed))
    for model_name in ("MLP", "GRU"):
        for use_z in (False, True):
            var = "+Z" if use_z else "-raw"
            for src, tgt in (("TG", "DL"), ("DL", "TG")):
                for seed in (0, 1, 2):
                    key = (model_name, var, f"{src}_to_{tgt}", seed)
                    if key in done:
                        continue
                    Xtr, Ztr, ytr, gtr, _ = data[src]
                    Xte, Zte, yte, _, tte = data[tgt]
                    t0 = time.time()
                    p, ep = fit_predict(model_name, use_z, Xtr, Ztr, ytr, gtr, Xte, Zte, seed)
                    err = np.abs(p - yte) * 100
                    row = dict(model=model_name, variant=var, direction=f"{src}_to_{tgt}", seed=seed, mae=err.mean(),
                               median=np.median(err), tau_le15=err[tte <= 15].mean(), tau_ge21=err[tte >= 21].mean(),
                               epochs=ep, sec=time.time() - t0)
                    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                    print(row, flush=True)
    df = pd.read_csv(OUT)
    print(df.groupby(["model", "variant", "direction"])[["mae", "median", "tau_le15", "tau_ge21"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
