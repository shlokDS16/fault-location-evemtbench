"""C1 gate H-A (claudedocs/c1_hybrid_design.md section 9): local-view (terminal S only) physics tokens and
physics-only one-ended baselines on the OFFICIAL windows of one grid.

Phasors: one-cycle DFT with an ABSOLUTE-TIME reference (exponent uses the global sample index), so phasors
from different cycles share one reference and superimposed quantities (last - first cycle) are valid.
Loops: ground a/b/c with k0 compensation (I_l = I_p + k0 * 3I0, k0 = (Z0 - Z1) / (3 Z1), nameplate Z0/Z1),
and phase ab/bc/ca. Estimates are fractions of line length (d in [0, 1] ideally).
Usage: EVEMT_GRID=<grid> python src/c1/local_features.py -> results/c1_local_feats<TAG>.parquet
"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import D, FS, F0, T0, ROOT, TAG, terminal_cols, A, add_noise  # noqa: E402
from read_graph import load as load_graph  # noqa: E402

N = int(round(FS / F0))
WLEN = 480
LOOPS = ["ag", "bg", "cg", "ab", "bc", "ca"]
ROT_PHASE = {"ab": 1 - A * A, "bc": A * A - A, "ca": A - 1}
ROT_GROUND = {"ag": 1.0, "bg": A * A, "cg": A}


def line_params():
    G = load_graph(D / "graphs" / "graph_benchmark.pickle")
    out = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if str(k).startswith("MainLn"):
            p = d[f"{k}_param_dict"]
            L = p["length"]
            out[k] = (L * complex(p["rline"], p["xline"]), L * complex(p["rline0"], p["xline0"]))
    return out


def phasor_abs(x, start):
    n = np.arange(start, start + N)
    return (2.0 / N) * np.sum(x[start:start + N] * np.exp(-2j * np.pi * n / N))


def est(V, I, Z1, P):
    den = np.imag(Z1 * I * np.conj(P))
    return float(np.clip(np.imag(V * np.conj(P)) / den, -1, 2)) if abs(den) > 1e-12 else 0.5


def window_feats(X, s1, Z1, Z0):
    k0 = (Z0 - Z1) / (3 * Z1)
    s0 = s1 - WLEN
    L = [phasor_abs(X[c], s1 - N) for c in range(6)]   # last cycle: Ia Ib Ic Va Vb Vc
    Fp = [phasor_abs(X[c], s0) for c in range(6)]      # first cycle of the window
    Ia, Ib, Ic, Va, Vb, Vc = L
    I0, I1, I2 = (Ia + Ib + Ic) / 3, (Ia + A * Ib + A * A * Ic) / 3, (Ia + A * A * Ib + A * Ic) / 3
    V0, V1, V2 = (Va + Vb + Vc) / 3, (Va + A * Vb + A * A * Vc) / 3, (Va + A * A * Vb + A * Vc) / 3
    dI = [L[c] - Fp[c] for c in range(3)]
    ph = {"a": 0, "b": 1, "c": 2}
    f = {}
    loopVI = {}
    for lp in LOOPS:
        if lp.endswith("g"):
            p = ph[lp[0]]
            V, I = L[3 + p], L[p] + k0 * 3 * I0
            pol2 = I2 / ROT_GROUND[lp]
            dIl = dI[p] + k0 * (dI[0] + dI[1] + dI[2])
        else:
            p, q = ph[lp[0]], ph[lp[1]]
            V, I = L[3 + p] - L[3 + q], L[p] - L[q]
            pol2 = I2 / ROT_PHASE[lp]
            dIl = dI[p] - dI[q]
        loopVI[lp] = (V, I)
        z = (V / I) / Z1 if abs(I) > 1e-9 else 0j
        f[f"{lp}_zr"], f[f"{lp}_zi"] = float(np.clip(z.real, -5, 5)), float(np.clip(z.imag, -5, 5))
        f[f"{lp}_react"] = float(np.clip(np.imag(V / I) / np.imag(Z1), -1, 2)) if abs(I) > 1e-9 else 0.5
        f[f"{lp}_tak2"] = est(V, I, Z1, pol2)
        f[f"{lp}_takd"] = est(V, I, Z1, dIl)
        if lp.endswith("g"):
            f[f"{lp}_tak0"] = est(V, I, Z1, 3 * I0)
        f[f"{lp}_absz"] = float(np.clip(abs(z), 0, 10))
    m1 = abs(I1) + 1e-9
    f.update(i0r=abs(I0) / m1, i2r=abs(I2) / m1, v0r=abs(V0) / (abs(V1) + 1e-9), v2r=abs(V2) / (abs(V1) + 1e-9),
             i2c=np.cos(np.angle(I2 / I1)), i2s=np.sin(np.angle(I2 / I1)),
             i0c=np.cos(np.angle(I0 / I1)), i0s=np.sin(np.angle(I0 / I1)))
    imax = max(abs(L[c]) for c in range(3)) + 1e-9
    for p, nm in enumerate("abc"):
        f[f"i{nm}_onset"] = float(np.clip(abs(L[p]) / (abs(Fp[p]) + 1e-9), 0, 100))
        f[f"v{nm}_drop"] = float(np.clip(abs(L[3 + p]) / (abs(Fp[3 + p]) + 1e-9), 0, 5))
        f[f"i{nm}_share"] = abs(L[p]) / imax
        f[f"di{nm}_rel"] = float(np.clip(abs(dI[p]) / (abs(Fp[p]) + 1e-9), 0, 100))
    return guard_i0(f), loopVI


I0_ZERO = 1e-6  # design section 20


def guard_i0(f):
    """I0 at rounding level (2ph / 3ph faults): its angle and the 3I0-polarised Takagi estimates are float noise.
    Works on a dict (one window) or a DataFrame (stored tables); reads only i0r, which it does not change."""
    z = f["i0r"] < I0_ZERO
    if isinstance(f, dict):
        if z:
            f.update(i0c=0.0, i0s=0.0, ag_tak0=0.5, bg_tak0=0.5, cg_tak0=0.5)
        return f
    f.loc[z, ["i0c", "i0s"]] = 0.0
    f.loc[z, ["ag_tak0", "bg_tak0", "cg_tak0"]] = 0.5
    return f


def oracle_loop(row):
    et = row["events/event_type"]
    if et.startswith("flt_1phg"):
        return str(row["events/event_phase_select_1ph"]).lower()[0] + "g"
    if et in ("flt_2ph_shc", "flt_2phg_shc"):
        return str(row["events/event_phase_select_2ph"]).lower()
    return "ab"  # 3ph


def one(args):
    row, LP = args
    line = row["events/event_target"]
    m = re.match(r"MainLn(\d+)-(\d+)[AB]?$", line)
    sid = int(row["general/sim_idx"])
    f = D / "data" / f"result{sid}.csv"
    hdr = pd.read_csv(f, nrows=0).columns.tolist()
    cs = terminal_cols(hdr, m.group(1), line)
    d = pd.read_csv(f, skiprows=[1], usecols=cs)
    X = add_noise([d[c].to_numpy(float) for c in cs], sid)
    Z1, Z0 = LP[line]
    nf = int(round((row["events/event_start"] - T0) * FS))
    inc = "incipient" in row["events/event_type"]
    ol = oracle_loop(row)
    out = []
    for tau in (range(5, 51, 5) if inc else range(5, 81, 5)):
        s1 = nf + int(round(tau / 1000 * FS))
        feats, loopVI = window_feats(X, s1, Z1, Z0)
        minloop = min(LOOPS, key=lambda l: feats[f"{l}_absz"])
        rec = dict(sim_idx=sid, line=line, etype=row["events/event_type"], tau_ms=tau,
                   R=row["events/event_flt_shc_resistance"], y=row["events/event_flt_target_line_location"] / 100.0,
                   oracle_loop=ol,
                   phys_react_oracle=feats[f"{ol}_react"], phys_tak2_oracle=feats[f"{ol}_tak2"],
                   phys_takd_oracle=feats[f"{ol}_takd"], phys_react_minloop=feats[f"{minloop}_react"])
        rec.update(feats)
        out.append(rec)
    return out


def main():
    LP = line_params()
    s = pd.read_csv(D / "labels" / "settings_clean.csv")
    fl = s[s["events/event_target"].str.startswith("MainLn") & s["events/event_flt_target_line_location"].notna()
           & s["events/event_type"].str.startswith("flt_")]
    with ProcessPoolExecutor(12) as ex:
        res = [r for ch in ex.map(one, [(r, LP) for _, r in fl.iterrows()], chunksize=8) for r in ch]
    df = pd.DataFrame(res)
    out = ROOT / "results" / f"c1_local_feats{TAG}.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)} windows, {df.shape[1]} columns -> {out}")
    for c in ("phys_react_oracle", "phys_tak2_oracle", "phys_takd_oracle", "phys_react_minloop"):
        e = (df[c].clip(0, 1) - df.y).abs() * 100
        print(f"{c:22s} MAE {e.mean():.2f} %  (tau>=21ms: {e[df.tau_ms >= 21].mean():.2f} %)")


if __name__ == "__main__":
    main()
