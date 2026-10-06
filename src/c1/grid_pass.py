"""One pass over the LINE-fault episodes of any EvEMTBench grid folder (design sections 17a / 18):
windows by the reconstructed benchmark rule, H-A2 tokens (explicit ALLOW list, local terminal S = first bus of
'MainLnX-Y'), one-ended physics baselines, raw local windows (equal-information baselines) and the two-ended
time-domain aerial-mode locator estimate (SG derivative, nameplate R1/L1). No learning here.
Usage: python src/c1/grid_pass.py <grid_dir relative to data/raw/evemt> <tag>
  -> results/c1_grid_<tag>.parquet, results/c1_grid_<tag>_raw.npz"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokens_core import episode_tokens, ALLOW, FS  # noqa: E402
from local_features import LOOPS, WLEN, oracle_loop  # noqa: E402
from stage_a import terminal_cols  # noqa: E402
from td_locator import clarke  # noqa: E402
from read_graph import load as load_graph  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
W0 = 2 * np.pi * 50.0
D_SC = 289
LINE_RE = re.compile(r"MainLn(\d+)-(\d+)[AB]?$")


def line_params(gdir):
    gs = sorted((gdir / "graphs").glob("*.pickle"))
    assert len(gs) == 1, gs
    G = load_graph(gs[0])
    out = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if str(k).startswith("MainLn"):
            p = d[f"{k}_param_dict"]
            L = p["length"]
            out[k] = (L * complex(p["rline"], p["xline"]), L * complex(p["rline0"], p["xline0"]), L)
    return out


def window_ends(row, n_samples):
    nf = int(np.floor((row["events/event_start"] - 1.0) * FS))
    inc = "incipient" in row["events/event_type"]
    D = int(np.floor(row["events/event_iflt_duration"] * FS)) if inc else D_SC
    ends = 480 + 48 * np.arange((n_samples - 480) // 48 + 1)
    return nf, [int(e) for e in ends if e > nf and e - 480 < nf + D]


def der(x):
    return savgol_filter(x, 11, 2, deriv=1, delta=1 / FS)


def has_term(hdr, bus, line):
    return any(h.endswith(f"pex_MainBus{bus}_{line}") for h in hdr)


def identify_z(gdir, s, LP, per_line=10):
    """Design 18a(1) variant (ii): Z1 identified two-endedly from PRE-FAULT normal operation (median over up to
    `per_line` episodes per line, no labels); Z0 = nameplate with R0 scaled by the identified/nameplate R1 ratio.
    Lines measured at one end only keep their nameplate values."""
    from stage_a import phasor, pos_seq
    out = dict(LP)
    for line, g in s.groupby("events/event_target"):
        m = LINE_RE.match(line)
        zs = []
        for _, row in g.head(per_line).iterrows():
            f = gdir / "data" / f"result{int(row['general/sim_idx'])}.csv"
            hdr = pd.read_csv(f, nrows=0).columns.tolist()
            if not (has_term(hdr, m.group(1), line) and has_term(hdr, m.group(2), line)):
                break
            cs, cr = terminal_cols(hdr, m.group(1), line), terminal_cols(hdr, m.group(2), line)
            d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
            pre = int(round((row["events/event_start"] - 1.0) * FS)) - 2 * 192
            V = lambda cols: pos_seq(*(phasor(d[c].to_numpy(float), pre, 192) for c in cols))  # noqa: E731
            zs.append((V(cs[3:]) - V(cr[3:])) / ((V(cs[:3]) - V(cr[:3])) / 2))
        if zs:
            Z1np, Z0np, length = LP[line]
            Z1 = complex(np.median(np.real(zs)), np.median(np.imag(zs)))
            out[line] = (Z1, complex(Z0np.real * Z1.real / Z1np.real, Z0np.imag), length)
            print(f"{line}: Z1 nameplate {Z1np:.4f} identified {Z1:.4f}")
    return out


def one(args):
    row, LP, gdir, *opt = args
    chain = opt[0] if opt else None  # optional measurement-chain model (design 22): chain(X6, sid, end) -> X6
    line = row["events/event_target"]
    m = LINE_RE.match(line)
    sid = int(row["general/sim_idx"])
    f = gdir / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    # local terminal S = the first bus of 'MainLnX-Y'; if it is not measured (CIGRE MainLn8-14), the other end is
    # local, the location is mapped to that end (y_local = 1 - y, design 18a(2)) and the two-ended locator is undefined
    flip = not has_term(hdr, m.group(1), line)
    two = not flip and has_term(hdr, m.group(2), line)
    cs = terminal_cols(hdr, m.group(2) if flip else m.group(1), line)
    cr = terminal_cols(hdr, m.group(2), line) if two else []
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    XS = [d[c].to_numpy(float) for c in cs]
    if chain is not None:
        XS = chain(XS, sid, 0)
    nf, ends = window_ends(row, len(XS[0]))
    Z1, Z0, length = LP[line]
    toks = episode_tokens(XS, Z1, Z0, ends)
    # two-ended aerial-mode R-L locator (as td_noise.py / adapt_td.py)
    R, L = Z1.real, Z1.imag / W0
    if two:
        XR = [d[c].to_numpy(float) for c in cr]
        if chain is not None:
            XR = chain(XR, sid, 1)
        iS, vS, iR, vR = clarke(*XS[:3]), clarke(*XS[3:]), clarke(*XR[:3]), clarke(*XR[3:])
        modes = [(vS[k] - vR[k] + R * iR[k] + L * der(iR[k]), R * (iS[k] + iR[k]) + L * (der(iS[k]) + der(iR[k])))
                 for k in range(2)]
    y_label = row["events/event_flt_target_line_location"] / 100.0
    ol = oracle_loop(row)
    Xa = np.stack(XS, 1).astype(np.float32)
    recs, wins = [], []
    for s1, t in zip(ends, toks):
        s0 = s1 - WLEN
        if two:
            num = sum(float(np.dot(a[s0:s1], b[s0:s1])) for a, b in modes)
            den = sum(float(np.dot(b[s0:s1], b[s0:s1])) for a, b in modes) + 1e-12
        minloop = min(LOOPS, key=lambda l: t[f"{l}_absz"])
        rec = dict(sim_idx=sid, line=line, length_km=length, etype=row["events/event_type"], post_ms=(s1 - nf) / FS * 1000,
                   R=row["events/event_flt_shc_resistance"], y=(1 - y_label) if flip else y_label, y_label=y_label,
                   flip=int(flip), oracle_loop=ol, phys_react_oracle=t[f"{ol}_react"], phys_tak2_oracle=t[f"{ol}_tak2"],
                   phys_react_minloop=t[f"{minloop}_react"], td2_est=num / den if two else np.nan,
                   Z1r=Z1.real, Z1x=Z1.imag, Z0r=Z0.real, Z0x=Z0.imag)
        rec.update({k: t[k] for k in ALLOW})
        recs.append(rec)
        wins.append(Xa[s0:s1])
    return recs, wins


def main():
    gdir = ROOT / "data" / "raw" / "evemt" / sys.argv[1]
    tag = sys.argv[2]
    LP = line_params(gdir)
    s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
    fl = s[s["events/event_type"].str.startswith("flt_") & s["events/event_target"].str.match(LINE_RE)
           & s["events/event_flt_target_line_location"].notna()]
    missing = set(fl["events/event_target"]) - set(LP)
    assert not missing, missing
    if len(sys.argv) > 3 and sys.argv[3] == "zid":
        LP = identify_z(gdir, fl, LP)
    print(f"{len(fl)} line-fault episodes on {fl['events/event_target'].nunique()} lines")
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(one, [(r, LP, gdir) for _, r in fl.iterrows()], chunksize=4))
    df = pd.DataFrame([r for rr, _ in res for r in rr])
    W = np.stack([w for _, ww in res for w in ww])
    df.to_parquet(ROOT / "results" / f"c1_grid_{tag}.parquet", index=False)
    np.savez(ROOT / "results" / f"c1_grid_{tag}_raw.npz", X=W, sim_idx=df.sim_idx.values, post_ms=df.post_ms.values,
             y=df.y.values, Z=df[["Z1r", "Z1x", "Z0r", "Z0x"]].to_numpy(np.float32))
    df["td2_err"] = (df.td2_est.clip(0, 1) - df.y).abs() * 100
    print(f"windows {len(df)}; two-ended TD MAE {df.td2_err.mean():.3f} (median {df.td2_err.median():.3f})")
    print(df.groupby("etype").td2_err.mean().round(3).to_string())


if __name__ == "__main__":
    main()
