"""Design section 22: measurement-chain models applied to primary-quantity EMT records (Ia Ib Ic Va Vb Vc).
AA : 2nd-order Butterworth low-pass, fc = 2 kHz, causal.
CT : IEEE PSRC 'CT SAT' model (G. Swift, theory for the C37.110 CT saturation calculator):
     i_e = sgn(l) A |l|^S, A = 10 / (RP (sqrt2 Vs / w)^S), RP = rms/peak of sin^S;
     dl/dt = Rt i2 (Lb = 0), i2 = i1 / N - i_e; backward Euler + Newton per sample; remanence = rem * sqrt2 Vs / w.
CVT: second-order equivalent H(s) = (2/tau) s / (s^2 + (2/tau) s + w0^2) (unity gain, zero phase at 50 Hz).
Every filter starts from steady state: the first pre-fault cycle is repeated for 0.2 s before the record, then dropped.
Usage: python src/c1/meas_chain.py  -> prints the CVT residual-voltage table and a CT sanity case."""
import numpy as np
from numba import njit
from scipy.signal import bilinear, butter, lfilter

FS, F0 = 9600.0, 50.0
W0 = 2 * np.pi * F0
NC = int(FS / F0)
PAD = int(0.2 * FS) // NC * NC
CT_MILD = dict(ratio=240.0, Vs=800.0, S=22.0, Rw=0.6, Rb=1.0, rem=0.0)
CT_SEVERE = dict(ratio=240.0, Vs=400.0, S=22.0, Rw=0.6, Rb=2.0, rem=0.6)


def pad(x):
    return np.concatenate([np.tile(x[:NC], PAD // NC), x])


def aa(x):
    b, a = butter(2, 2000.0, fs=FS)
    return lfilter(b, a, pad(x))[PAD:]


def cvt(x, tau):
    b, a = bilinear([2 / tau, 0.0], [1.0, 2 / tau, W0 ** 2], fs=FS)
    return lfilter(b, a, pad(x))[PAD:]


def rp(S):
    t = np.linspace(0, 2 * np.pi, 20001)
    return float(np.sqrt(np.mean(np.abs(np.sin(t)) ** (2 * S))))


@njit(cache=True)
def _ct(is_, A, S, Rt, dt, lam0):
    n = is_.shape[0]
    lam = lam0
    ie_out = np.empty(n)
    for k in range(n):
        lp = lam
        x = lam
        for _ in range(50):
            ax = abs(x)
            ie = A * ax ** S * (1.0 if x >= 0 else -1.0)
            g = x - lp - Rt * dt * (is_[k] - ie)
            gp = 1.0 + Rt * dt * A * S * ax ** (S - 1.0)
            step = g / gp
            x -= step
            if abs(step) < 1e-12 * (1.0 + abs(x)):
                break
        lam = x
        ie_out[k] = A * abs(lam) ** S * (1.0 if lam >= 0 else -1.0)
    return ie_out


def ct(i1, ratio, Vs, S, Rw, Rb, rem):
    """Measured primary-referred current (i2 * ratio) and the excitation current (secondary amperes)."""
    lref = np.sqrt(2) * Vs / W0
    A = 10.0 / (rp(S) * lref ** S)
    is_ = pad(i1) / ratio
    ie = _ct(is_, A, S, Rw + Rb, 1 / FS, rem * lref)
    return ((is_ - ie) * ratio)[PAD:], ie[PAD:], is_[PAD:]


class Chain:
    """chain(X6, sid, end) -> distorted X6; scenario in {AA, CT-mild, CT-severe, CVT-5, CVT-15, FULL}."""

    def __init__(self, scen):
        self.scen = scen

    def __call__(self, X, sid, end):
        I, V = [np.asarray(x, float) for x in X[:3]], [np.asarray(x, float) for x in X[3:]]
        s = self.scen
        if s in ("CT-mild", "CT-severe", "FULL"):
            p = CT_MILD if s == "CT-mild" else CT_SEVERE
            I = [ct(i, **p)[0] for i in I]
        if s in ("CVT-5", "CVT-15", "FULL"):
            tau = 0.005 if s == "CVT-5" else 0.015
            V = [cvt(v, tau) for v in V]
        if s in ("AA", "FULL"):
            I, V = [aa(i) for i in I], [aa(v) for v in V]
        return I + V


def cvt_residual():
    """Residual output after a bolted terminal collapse at voltage peak / zero, in % of the pre-fault peak."""
    t = np.arange(int(0.4 * FS)) / FS
    rows = []
    for tau in (0.005, 0.015):
        for name, phi in (("peak", np.pi / 2), ("zero", 0.0)):
            v = np.sin(W0 * t + phi - W0 * 0.2)  # angle at t = 0.2 s is phi
            v[t >= 0.2] = 0.0
            y = cvt(v, tau)
            r = {ms: 100 * np.abs(y[(t >= 0.2 + ms / 1000) & (t < 0.2 + ms / 1000 + 1 / F0)]).max() for ms in (10, 20, 40)}
            rows.append((tau * 1000, name, r[10], r[20], r[40]))
    return rows


if __name__ == "__main__":
    print("CVT residual (% of pre-fault peak, max over the following cycle) at 10 / 20 / 40 ms after collapse:")
    for r in cvt_residual():
        print("tau %4.0f ms, collapse at %s: %5.1f / %5.1f / %5.1f" % r)
    # CT sanity: 110 kV-like fault, 11 kA rms symmetrical, full offset, X/R = 12
    t = np.arange(int(0.3 * FS)) / FS
    tf = 0.1
    Tl = 12 / W0
    i1 = np.where(t < tf, 400 * np.sqrt(2) * np.sin(W0 * t),
                  11000 * np.sqrt(2) * (np.sin(W0 * (t - tf) - np.pi / 2) + np.exp(-(t - tf) / Tl)))
    for nm, p in (("CT-mild", CT_MILD), ("CT-severe", CT_SEVERE)):
        i2, ie, is_ = ct(i1, **p)
        post = t >= tf
        k = np.argmax(np.abs(ie[post]) > 0.05 * np.abs(is_[post]).max())
        print(f"{nm}: time to saturation {t[post][k] * 1000 - tf * 1000:.1f} ms, max |ie|/max|is| "
              f"{np.abs(ie[post]).max() / np.abs(is_[post]).max():.2f}, pre-fault error "
              f"{np.abs(i2[~post] - i1[~post]).max() / 400 / np.sqrt(2) * 100:.3f} %")
