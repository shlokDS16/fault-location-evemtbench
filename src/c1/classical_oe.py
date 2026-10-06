"""Design section 21: classical one-ended fault locators (reactance, Takagi, Takagi-I2, modified Takagi, Eriksson) on
the official windows of a grid, oracle loop, WITH pre-fault memory (cycle ending 1 ms before inception).
Source impedances for MT / Eriksson: local from local superimposed quantities, remote from the remote end's
superimposed quantities (oracle settings knowledge; disclosed). No learning, no label used except the oracle loop.
Usage: python src/c1/classical_oe.py <DL|TG|MV|ADAPT>  -> results/c1_classical_<tag>.parquet"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_features import N, LOOPS, ROT_GROUND, ROT_PHASE, phasor_abs, oracle_loop  # noqa: E402
from stage_a import terminal_cols, A  # noqa: E402
import grid_pass  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FS = 9600.0
GRIDS = {"DL": "DoubleLine", "TG": "TestGrid110kV", "MV": "CigreMVGrid"}
PRE_GAP = 10  # samples (~1 ms) between the end of the pre-fault cycle and inception
METHODS = ["R", "T", "T2", "MT", "E"]


def seq(a, b, c):
    return (a + b + c) / 3, (a + A * b + A * A * c) / 3, (a + A * A * b + A * c) / 3


def ph(X, start):
    """Phasors of the 6 channels (Ia Ib Ic Va Vb Vc) for the cycle starting at `start`."""
    return np.array([phasor_abs(x, start) for x in X])


def takagi(V, I, Z1, P):
    den = np.imag(Z1 * I * np.conj(P))
    return float(np.imag(V * np.conj(P)) / den) if abs(den) > 1e-12 else np.nan


def eriksson(V, I, dI, ZL, ZSA, ZSB):
    K1 = 1 + ZSB / ZL + V / (ZL * I)
    K2 = V / (ZL * I) * (1 + ZSB / ZL)
    K3 = dI / (ZL * I) * (1 + (ZSA + ZSB) / ZL)
    if abs(K3.imag) < 1e-12:
        return np.nan
    p = K1.real - K3.real * K1.imag / K3.imag
    q = K2.real - K3.real * K2.imag / K3.imag
    disc = p * p - 4 * q
    if disc < 0:  # design 21b: no real root -> undefined (scored 0.5)
        return np.nan
    r = [(p - np.sqrt(disc)) / 2, (p + np.sqrt(disc)) / 2]
    ok = [x for x in r if -0.1 <= x <= 1.1]
    return float(min(ok, key=lambda x: abs(x - 0.5))) if ok else np.nan


def loop_quant(P, Ppre, lp, k0):
    """Loop voltage, loop current, loop superimposed current from post (P) and pre-fault (Ppre) phasors."""
    I0, dI0 = P[:3].sum() / 3, (P[:3] - Ppre[:3]).sum() / 3
    pi = {"a": 0, "b": 1, "c": 2}
    if lp.endswith("g"):
        p = pi[lp[0]]
        return P[3 + p], P[p] + k0 * 3 * I0, (P[p] - Ppre[p]) + k0 * 3 * dI0
    p, q = pi[lp[0]], pi[lp[1]]
    return P[3 + p] - P[3 + q], P[p] - P[q], (P[p] - Ppre[p]) - (P[q] - Ppre[q])


def episode(XS, XR, Z1, Z0, nf, ends, ol, three_ph=False):
    k0 = (Z0 - Z1) / (3 * Z1)
    pre = nf - PRE_GAP - N
    PSpre = ph(XS, pre)
    PRpre = ph(XR, pre) if XR is not None else None
    out = []
    for s1 in ends:
        P = ph(XS, s1 - N)
        V, I, dI = loop_quant(P, PSpre, ol, k0)
        I0, I1, I2 = seq(*P[:3])
        V0, V1, _ = seq(*P[3:])
        d0I, d1I, _ = seq(*(P[:3] - PSpre[:3]))
        d0V, d1V, _ = seq(*(P[3:] - PSpre[3:]))
        rot = ROT_GROUND[ol] if ol.endswith("g") else ROT_PHASE[ol]
        rec = dict(R=float(np.imag(V / I) / np.imag(Z1)) if abs(I) > 1e-9 else np.nan,
                   T=takagi(V, I, Z1, dI), T2=takagi(V, I, Z1, I2 / rot), MT=np.nan, E=np.nan)
        if XR is not None:
            PR = ph(XR, s1 - N)
            e0I, e1I, _ = seq(*(PR[:3] - PRpre[:3]))
            e0V, e1V, _ = seq(*(PR[3:] - PRpre[3:]))
            ZSA = -d1V / d1I if abs(d1I) > 1e-9 else np.nan
            ZSB = -e1V / e1I if abs(e1I) > 1e-9 else np.nan
            if np.isfinite(ZSA) and np.isfinite(ZSB):
                rec["E"] = eriksson(V, I, dI, Z1, ZSA, ZSB)
            if ol.endswith("g") and abs(d0I) > 1e-9 and abs(e0I) > 1e-9:
                Z0SA, Z0SB = -d0V / d0I, -e0V / e0I
                d = 0.5
                for _ in range(3):
                    D0 = (Z0SB + (1 - d) * Z0) / (Z0SA + Z0 + Z0SB)
                    d = takagi(V, I, Z1, 3 * I0 * np.exp(-1j * np.angle(D0)))
                    d = 0.5 if not np.isfinite(d) else float(np.clip(d, -1, 2))
                rec["MT"] = d
        if not ol.endswith("g"):
            rec["MT"] = rec["T"]
        if three_ph:  # design 21a: I2 ~ 0 for 3ph faults -> Takagi-I2 falls back to reactance (as in G1 one_ended.py)
            rec["T2"] = rec["R"]
        out.append({k: (float(np.clip(v, -1, 2)) if np.isfinite(v) else np.nan) for k, v in rec.items()})
    return out


def one(args):
    row, LP, gdir = args
    line = row["events/event_target"]
    m = grid_pass.LINE_RE.match(line)
    sid = int(row["general/sim_idx"])
    f = gdir / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    flip = not grid_pass.has_term(hdr, m.group(1), line)
    two = not flip and grid_pass.has_term(hdr, m.group(2), line)
    cs = terminal_cols(hdr, m.group(2) if flip else m.group(1), line)
    cr = terminal_cols(hdr, m.group(2), line) if two else []
    d = pd.read_csv(f, skiprows=[1], usecols=cs + cr)
    XS = [d[c].to_numpy(float) for c in cs]
    XR = [d[c].to_numpy(float) for c in cr] if two else None
    nf, ends = grid_pass.window_ends(row, len(XS[0]))
    Z1, Z0 = LP[line][0], LP[line][1]
    ol = oracle_loop(row)
    y = row["events/event_flt_target_line_location"] / 100.0
    recs = []
    for s1, r in zip(ends, episode(XS, XR, Z1, Z0, nf, ends, ol, row["events/event_type"] == "flt_3ph_shc")):
        recs.append(dict(sim_idx=sid, line=line, etype=row["events/event_type"], post_ms=(s1 - nf) / FS * 1000,
                         R_f=row["events/event_flt_shc_resistance"], y=(1 - y) if flip else y, flip=int(flip),
                         two=int(two), **r))
    return recs


def jobs(tag):
    if tag == "ADAPT":
        import adapt_tokens as at
        gdir = at.AD
        LP = at.line_params()
        ids = {int(l.strip().split(":")[2]) for l in open(at.SPLITS / "test.txt") if l.strip()}
        s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
        s = s[s["general/sim_idx"].isin(ids)]
    else:
        gdir = ROOT / "data" / "raw" / "evemt" / GRIDS[tag]
        LP = grid_pass.line_params(gdir)
        s = pd.read_csv(gdir / "labels" / "settings_clean.csv")
    fl = s[s["events/event_type"].str.startswith("flt_") & s["events/event_target"].str.match(grid_pass.LINE_RE)
           & s["events/event_flt_target_line_location"].notna()]
    return [(r, LP, gdir) for _, r in fl.iterrows()]


def main():
    tag = sys.argv[1]
    js = jobs(tag)
    if len(sys.argv) > 2:  # quick check on a subset
        js = js[: int(sys.argv[2])]
    with ProcessPoolExecutor(12) as ex:
        res = [r for rr in ex.map(one, js, chunksize=4) for r in rr]
    df = pd.DataFrame(res)
    df.to_parquet(ROOT / "results" / f"c1_classical_{tag}.parquet", index=False)
    print(f"{tag}: {len(df)} windows, {df.sim_idx.nunique()} episodes")
    bolted = df[~df.etype.str.contains("incipient|hif") & (df.R_f <= 1) & (df.post_ms >= 25)]
    print("bolted SC post>=25 ms MAE %:", {m: round(float((bolted[m].clip(0, 1) - bolted.y).abs().mean() * 100), 2)
                                           for m in METHODS}, "n", len(bolted))
    print("all windows MAE %:", {m: round(float((df[m].clip(0, 1) - df.y).abs().mean() * 100), 2) for m in METHODS})


if __name__ == "__main__":
    main()
