"""Design section 24 (report-only): (a) two-ended PHASOR locator (C37.114 synchronized form); (b) Suonan-Qi-type one-ended
TD baseline = oracle-loop 'dtd' token; (c) two-ended TD sensitivity to synchronization and line-parameter errors;
(d) inception-free cost (windows with vs without pre-fault samples).
Usage: python src/c1/review_checks.py  -> results/c1_review_checks.csv, c1_review_phasor_bytime.csv"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_pass  # noqa: E402
from local_features import phasor_abs, N, WLEN  # noqa: E402
from stage_a import terminal_cols, A  # noqa: E402
from td_locator import clarke  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
FS, W0 = 9600.0, 2 * np.pi * 50.0
SHIFTS = (-20, -10, -5, -1, 1, 5, 10, 20)
SCALES = (("RL", 0.90), ("RL", 0.95), ("RL", 1.05), ("RL", 1.10), ("R", 0.8), ("R", 1.2))


def der(x):
    return savgol_filter(x, 11, 2, deriv=1, delta=1 / FS)


def td_est(XS, XR, R1, L1, ends):
    iS, vS, iR, vR = clarke(*XS[:3]), clarke(*XS[3:]), clarke(*XR[:3]), clarke(*XR[3:])
    modes = [(vS[k] - vR[k] + R1 * iR[k] + L1 * der(iR[k]), R1 * (iS[k] + iR[k]) + L1 * (der(iS[k]) + der(iR[k])))
             for k in range(2)]
    out = []
    for s1 in ends:
        s0 = s1 - WLEN
        num = sum(float(np.dot(a[s0:s1], b[s0:s1])) for a, b in modes)
        den = sum(float(np.dot(b[s0:s1], b[s0:s1])) for a, b in modes) + 1e-12
        out.append(num / den)
    return out


def pos(x3, s):
    a, b, c = (phasor_abs(x, s) for x in x3)
    return (a + A * b + A * A * c) / 3


def one(args):
    row, LP, gdir = args
    line = row["events/event_target"]
    m = grid_pass.LINE_RE.match(line)
    sid = int(row["general/sim_idx"])
    f = gdir / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    if not (grid_pass.has_term(hdr, m.group(1), line) and grid_pass.has_term(hdr, m.group(2), line)):
        return []
    cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    XS, XR = [d[c].to_numpy(float) for c in cs], [d[c].to_numpy(float) for c in cr]
    nf, ends = grid_pass.window_ends(row, len(XS[0]))
    Z1 = LP[line][0]
    R1, L1 = Z1.real, Z1.imag / W0
    y = row["events/event_flt_target_line_location"] / 100.0
    rec = [dict(sim_idx=sid, line=line, etype=row["events/event_type"], post_ms=(s1 - nf) / FS * 1000, y=y) for s1 in ends]
    for r, v in zip(rec, td_est(XS, XR, R1, L1, ends)):
        r["td"] = v
    for s1, r in zip(ends, rec):  # (a) phasor two-ended, last cycle of the window
        VS, IS, VR, IR = pos(XS[3:], s1 - N), pos(XS[:3], s1 - N), pos(XR[3:], s1 - N), pos(XR[:3], s1 - N)
        den = Z1 * (IS + IR)
        r["ph2"] = float(((VS - VR + Z1 * IR) / den).real) if abs(den) > 1e-9 else np.nan
    if gdir.name in ("DoubleLine", "TestGrid110kV"):  # (c) sensitivity
        for sh in SHIFTS:
            XRs = [np.roll(x, sh) for x in XR]
            for r, v in zip(rec, td_est(XS, XRs, R1, L1, ends)):
                r[f"sync{sh:+d}"] = v
        for what, k in SCALES:
            for r, v in zip(rec, td_est(XS, XR, R1 * k, L1 * (k if what == "RL" else 1.0), ends)):
                r[f"{what}x{k:.2f}"] = v
    return rec


def jobs(tag):
    if tag == "ADAPT":
        import adapt_tokens as at
        gdir, LP = at.AD, {k: (v[0], v[1]) for k, v in at.line_params().items()}
        ids = {int(l.strip().split(":")[2]) for l in open(at.SPLITS / "test.txt") if l.strip()}
        s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
        s = s[s["general/sim_idx"].isin(ids)]
    else:
        gdir = ROOT / "data" / "raw" / "evemt" / {"DL": "DoubleLine", "TG": "TestGrid110kV", "MV": "CigreMVGrid"}[tag]
        LP = grid_pass.line_params(gdir)
        s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
    fl = s[s["events/event_type"].str.startswith("flt_") & s["events/event_target"].str.match(grid_pass.LINE_RE)
           & s["events/event_flt_target_line_location"].notna()]
    return [(r, LP, gdir) for _, r in fl.iterrows()]


def mae(p, y):
    return float(np.mean(np.abs(np.clip(np.nan_to_num(p, nan=0.5), 0, 1) - y)) * 100)


def main():
    rows, bytime = [], []
    for tag in ("DL", "TG", "ADAPT", "MV"):
        with ProcessPoolExecutor(6) as ex:
            d = pd.DataFrame([r for rr in ex.map(one, jobs(tag), chunksize=4) for r in rr])
        if tag in ("DL", "TG", "MV"):  # reproduction check of the stored two-ended estimates
            g = pd.read_parquet(R / f"c1_grid_{tag}.parquet", columns=["sim_idx", "post_ms", "td2_est"]).dropna()
            chk = d.assign(k=d.post_ms.round(3)).merge(g.assign(k=g.post_ms.round(3)).drop(columns="post_ms"),
                                                       on=["sim_idx", "k"])
            assert len(chk) == len(d), (tag, len(chk), len(d))
            dev = float(np.max(np.abs(chk.td - chk.td2_est)))
            assert dev < 1e-9, (tag, dev)
        y, sc = d.y.values, ~d.etype.str.contains("incipient|hif").values
        r = dict(grid=tag, n=len(d), td=mae(d.td, y), td_sc=mae(d.td[sc], y[sc]), ph2=mae(d.ph2, y), ph2_sc=mae(d.ph2[sc], y[sc]),
                 td_with_prefault=mae(d.td[d.post_ms <= 50], y[d.post_ms <= 50]),
                 td_post_only=mae(d.td[d.post_ms > 52.5], y[d.post_ms > 52.5]))
        for c in d.columns:
            if c.startswith("sync") or "x" in c and c[:2] in ("RL", "Rx"):
                r[c] = mae(d[c], y)
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        d["tbin"] = pd.cut(d.post_ms, [0, 17.5, 32.5, 52.5, 1e9], labels=["<=15", "20-30", "35-50", ">=55"])
        for b, g in d.groupby("tbin", observed=True):
            bytime.append(dict(grid=tag, tbin=b, n=len(g), td=mae(g.td, g.y), ph2=mae(g.ph2, g.y)))
    pd.DataFrame(rows).to_csv(R / "c1_review_checks.csv", index=False)
    pd.DataFrame(bytime).to_csv(R / "c1_review_phasor_bytime.csv", index=False)
    print(pd.DataFrame(bytime).pivot(index="tbin", columns="grid", values=["td", "ph2"]).round(2).to_string())
    # (b) Suonan-Qi-type one-ended TD = oracle-loop dtd token (no raw read)
    sq = []
    for tag, f in (("DL", "c1_grid_DL"), ("TG", "c1_grid_TG"), ("MV", "c1_grid_MV"), ("ADAPT", "c1_adapt_tokens")):
        t = pd.read_parquet(R / f"{f}.parquet")
        if tag == "ADAPT":
            t = t[t.split == "test"]
        est = np.array([t[f"{ol}_dtd"].iloc[i] for i, ol in enumerate(t.oracle_loop.values)])
        sc = ~t.etype.str.contains("incipient|hif").values
        sq.append(dict(grid=tag, n=len(t), suonan_td_oracle=mae(est, t.y.values), suonan_sc=mae(est[sc], t.y.values[sc]),
                       le15=mae(est[t.post_ms.values <= 17.5], t.y.values[t.post_ms.values <= 17.5]),
                       ge55=mae(est[t.post_ms.values > 52.5], t.y.values[t.post_ms.values > 52.5])))
    sq = pd.DataFrame(sq)
    sq.to_csv(R / "c1_review_suonan.csv", index=False)
    print(sq.round(2).to_string())


if __name__ == "__main__":
    main()
