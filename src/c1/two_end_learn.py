"""Two-ended learners at equal information (design section 26a / professor comment M1).
Inputs: 'P' = UW (u_alpha, u_beta, w_alpha, w_beta of Eq. (3)), each window divided by the joint RMS of its
(w_alpha, w_beta), no Z;  'Z' = X12 (S and R terminals, Ia Ib Ic Va Vb Vc) plus the faulted line's R1, X1, R0, X0.
Models: MLP and GRU of src/c1/equal_info.py, recipe unchanged (AdamW 1e-3, wd 1e-4, cosine, 60 epochs, batch 256,
patience 12, episode-grouped 10 % validation, channel/target standardisation, MSE); only the input width is generalised
(fit_predict copied with C channels instead of 6). Seeds 0, 1, 2. Directions TG->DL, DL->TG, DL+TG->MV.
Scoring: predictions clipped to [0, 1], e = |d^ - d| x 100 (MAE over the test windows).
Resumable. Outputs: results/c1_two_end_learn.csv (one row per input/model/direction/seed), per-window predictions
results/c1_two_end_learn_preds.parquet (assembled from results/c1_two_end_learn_parts/*.npy), log by the caller.
Needs results/c1_raw2.npz, c1_raw2_TestGrid110kV.npz, c1_raw2_CigreMVGrid.npz (src/c1/two_end_windows.py). GPU job."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

R = Path(__file__).resolve().parents[2] / "results"
OUT = R / "c1_two_end_learn.csv"
PARTS = R / "c1_two_end_learn_parts"
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
    def __init__(self, c, use_z):
        super().__init__()
        self.gru = nn.GRU(c, 128, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 + (4 if use_z else 0), 1))
        self.use_z = use_z

    def forward(self, x, z):
        h = self.gru(x)[0][:, -1]
        return self.head(torch.cat([h, z], 1) if self.use_z else h).squeeze(1)


def fit_predict(model_name, use_z, Xtr, Ztr, ytr, gtr, Xte, Zte, seed, epochs=60, bs=256, patience=12):
    """equal_info.fit_predict with the channel count generalised (identical otherwise)."""
    C = Xtr.shape[2]
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
    net = (MLP(480 * C, use_z) if model_name == "MLP" else GRU(C, use_z)).to(DEV)
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
    Xb, Zb = T(Xte, Zte)
    with torch.no_grad():
        p = np.concatenate([net(Xb[i:i + 2048].to(DEV), Zb[i:i + 2048].to(DEV)).cpu().numpy() for i in range(0, len(Xb), 2048)])
    return np.clip(p * ys + ym, 0, 1), ep + 1


def load(tag, inp):
    d = np.load(R / f"c1_raw2{tag}.npz")
    if inp == "P":
        UW = d["UW"]
        rms = np.sqrt((UW[:, :, 2:].astype(np.float64) ** 2).mean((1, 2)))
        X = (UW / rms[:, None, None].astype(np.float32)).astype(np.float32)
        Z = np.zeros((len(X), 4), np.float32)
    else:
        X, Z = d["X12"], d["Z"]
    return X, Z, d["y"].astype(np.float32), d["sim_idx"], d["tau"]


DIRS = (("TG", "DL", ["TG"], "DL"), ("DL", "TG", ["DL"], "TG"), ("DL+TG", "MV", ["DL", "TG"], "MV"))
TAGS = {"DL": "", "TG": "_TestGrid110kV", "MV": "_CigreMVGrid"}


def main():
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev.input, prev.model, prev.direction, prev.seed))
    PARTS.mkdir(exist_ok=True)
    for inp in ("P", "Z"):
        use_z = inp == "Z"
        data = {}
        for src, tgt, srcs, tg in DIRS:
            dname = f"{src}_to_{tgt}"
            if all((inp, m, dname, s) in done for m in ("MLP", "GRU") for s in (0, 1, 2)):
                continue
            S = [data.setdefault(k, load(TAGS[k], inp)) for k in srcs]
            T = data.setdefault(tg, load(TAGS[tg], inp))
            Xtr = np.concatenate([s[0] for s in S])
            Ztr = np.concatenate([s[1] for s in S])
            ytr = np.concatenate([s[2] for s in S])
            gtr = np.concatenate([s[3] + 10_000_000 * i for i, s in enumerate(S)])
            for model in ("MLP", "GRU"):
                for seed in (0, 1, 2):
                    if (inp, model, dname, seed) in done:
                        continue
                    t0 = time.time()
                    p, ep = fit_predict(model, use_z, Xtr, Ztr, ytr, gtr, T[0], T[1], seed)
                    sec = time.time() - t0
                    mae = float((np.abs(p - T[2]) * 100).mean())
                    np.save(PARTS / f"{inp}_{model}_{dname}_s{seed}.npy", p.astype(np.float32))
                    row = dict(input=inp, model=model, direction=dname, seed=seed, mae=mae, n_test=len(p), train_s=sec)
                    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                    print({**row, "epochs": ep}, flush=True)
            del Xtr, Ztr
        del data
    # assemble per-window predictions
    df = pd.read_csv(OUT)
    meta = {}
    for k in ("DL", "TG", "MV"):
        d = np.load(R / f"c1_raw2{TAGS[k]}.npz")
        meta[k] = (d["sim_idx"], d["tau"], d["y"])
    parts = []
    for _, r in df.iterrows():
        tg = r.direction.split("_to_")[1]
        sid, tau, y = meta[tg]
        p = np.load(PARTS / f"{r.input}_{r.model}_{r.direction}_s{r.seed}.npy")
        parts.append(pd.DataFrame(dict(sim_idx=sid, tau=tau, y=y, pred=p, input=r.input, model=r.model, direction=r.direction,
                                       seed=int(r.seed))))
    pd.concat(parts, ignore_index=True).to_parquet(R / "c1_two_end_learn_preds.parquet", index=False, engine="fastparquet")
    print(df.groupby(["input", "model", "direction"]).mae.agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
