"""Sec. 5.8 (sec:sensing), limit of detection of a Lorentzian resonance sensor.

Tutorial claims (tutorial.tex line numbers):
  l.2468  [eq:res_amplitude]  A(lambda) = | Q_R^-1 / (2i (lambda - lambda0)/lambda0 + Q_R^-1 + Q_NR^-1) |^2
  l.2466  detection threshold  A(lambda0) - A(lambda') = 3 sigma
  l.2477  [eq:lod]  LOD = lambda0 sqrt(3 sigma) / (S Q_R A(lambda0))                (as printed)
  l.2491-2495  Q_R A(lambda0) is maximized at Q_R = Q_NR ("critical coupling"), where
               Q_R A(lambda0) = Q_R/4.

Derivation: with gamma = Q_R^-1 + Q_NR^-1 and delta = Delta lambda/lambda0,
  A(delta) = A0 gamma^2/(4 delta^2 + gamma^2),  A0 = (Q_tot/Q_R)^2,
  A0 - A(delta) = 3 sigma  =>  delta = gamma sqrt(3 sigma) / (2 sqrt(A0 - 3 sigma))  (exact)
                            ~= gamma sqrt(3 sigma)/(2 sqrt(A0)) = sqrt(3 sigma)/(2 Q_R A0),
using sqrt(A0) = Q_tot/Q_R.  Hence LOD = lambda0 sqrt(3 sigma)/(2 S Q_R A0): the printed
[eq:lod] is 2x too large (for sigma << A0).

Checks:
  A. Root-find A(lambda0) - A(lambda0 + Delta lambda) = 3 sigma directly on [eq:res_amplitude]
     for several (Q_R, Q_NR, sigma), convert to LOD = Delta lambda/S, and compare with the
     printed formula and the corrected one.
  B. Q_R A(lambda0) vs Q_R at fixed Q_NR: maximum at Q_R = Q_NR with value Q_NR/4.

Run:  python sensing_lod.py   (< 1 s)
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar


def amplitude(lam, lam0, QR, QNR):
    d = (lam - lam0) / lam0
    return np.abs((1 / QR) / (2j * d + 1 / QR + 1 / QNR)) ** 2


def lod_numeric(lam0, S, QR, QNR, sigma):
    A0 = amplitude(lam0, lam0, QR, QNR)
    f = lambda dl: A0 - amplitude(lam0 + dl, lam0, QR, QNR) - 3 * sigma
    dl = brentq(f, 0.0, lam0 * 10 / min(QR, QNR) * 10)
    return dl / S, A0


def lod_printed(lam0, S, QR, A0, sigma):
    return lam0 * np.sqrt(3 * sigma) / (S * QR * A0)


def lod_corrected(lam0, S, QR, A0, sigma):
    return lam0 * np.sqrt(3 * sigma) / (2 * S * QR * A0)


CASES = [  # lam0 (nm), S (nm/RIU), QR, QNR, sigma
    (1550.0, 100.0, 1e4, 1e4, 1e-3),
    (1550.0, 100.0, 1e4, 1e5, 1e-3),
    (800.0, 300.0, 200.0, 50.0, 1e-4),
    (633.0, 50.0, 1e3, 3e3, 1e-2),
    (1000.0, 1000.0, 1e5, 1e4, 1e-5),
]


def main():
    print("=" * 96)
    print("Sec. 5.8  limit of detection for [eq:res_amplitude] with threshold A(l0) - A(l') = 3 sigma")
    print("=" * 96)
    print(f"{'l0':>6} {'S':>6} {'Q_R':>8} {'Q_NR':>8} {'sigma':>7} {'A0':>7} | {'LOD numeric':>12}"
          f" {'printed':>11} {'corrected':>11} {'printed/num':>11}")
    for lam0, S, QR, QNR, sig in CASES:
        L, A0 = lod_numeric(lam0, S, QR, QNR, sig)
        p = lod_printed(lam0, S, QR, A0, sig)
        c = lod_corrected(lam0, S, QR, A0, sig)
        print(f"{lam0:6.0f} {S:6.0f} {QR:8.0e} {QNR:8.0e} {sig:7.0e} {A0:7.4f} | {L:12.4e} {p:11.4e} {c:11.4e} {p/L:11.4f}")
    print("-> the printed [eq:lod] is 2x the directly computed LOD (corrected form agrees to O(sigma/A0)).")

    print("\nQ_R A(lambda0) vs Q_R at fixed Q_NR = 1e4:")
    QNR = 1e4
    g = lambda lq: -10 ** lq * amplitude(1.0, 1.0, 10 ** lq, QNR)
    r = minimize_scalar(g, bounds=(2, 6), method="bounded", options=dict(xatol=1e-10))
    print(f"   argmax Q_R = {10**r.x:.6e} (Q_NR = {QNR:.0e});  max Q_R A0 = {-r.fun:.6f}  (Q_NR/4 = {QNR/4:.1f})")


if __name__ == "__main__":
    main()
