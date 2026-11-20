"""Ratio of the embedded error estimate to the true local error along variable-step runs
(revision R1, item R1-Q4).

For every accepted step of a run with the controller of adaptive.py, the true local error is
|| y_{n+1} - u(t_n + h) ||, where u solves the problem from (t_n, y_n) and is computed by DOP853 of scipy
with rtol = 1e-14 (raised by scipy to 2.2e-14), atol = 1e-16; the estimate is || y_{n+1} - yhat_{n+1} ||.  Reported: the median of the
ratio estimate / true over the accepted steps whose true local error exceeds 1e-14 (to stay above the
accuracy of the reference), for each pair, problem and tolerance.

The pairs of orders five and six are taken from order56/code/base (CRK6(4), DP6(5)), those of orders
seven and eight from this folder (CRK7(5), Vern7, CRK8(6), Vern8); each group runs in its own process.
Usage: python est_ratio.py [56|78|table]   writes ../data/est_ratio_<group>.json
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BASE56 = os.path.join(HERE, "..", "..", "order56", "code", "base", "code")
PROBS = ["VanderPol", "Rossler", "Lorenz5", "Kepler", "Brusselator"]
TOLS = [1e-4, 1e-6, 1e-8, 1e-10]
GROUPS = {"56": (BASE56, ["CRK6(4)", "DP6(5)"]), "78": (HERE, ["CRK7(5)", "Vern7", "CRK8(6)", "Vern8"])}


def run_record(pair, f, y0, T, tol, max_steps=200_000):
    """adaptive.solve with the same controller, recording the ratio at every accepted step."""
    A, b, bh, q, proj, fsal = pair["A"], pair["b"], pair["bh"], pair["q"], pair["projected"], pair["fsal"]
    s = len(b); c = A.sum(1); p = pair["p"]
    y = np.array(y0, float); t = 0.0
    sc = lambda u, v: tol + tol * np.maximum(np.abs(u), np.abs(v))
    f0 = np.real(f(t, y)) * 1.0
    d0 = np.sqrt(np.mean((y / sc(y, y)) ** 2)); d1 = np.sqrt(np.mean((f0 / sc(y, y)) ** 2))
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    f1 = np.real(f(t + h0, y + h0 * f0))
    d2 = np.sqrt(np.mean(((f1 - f0) / sc(y, y)) ** 2)) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1.0 / (p + 1))
    h = min(100 * h0, h1, T)
    k1 = f0; facmax = 5.0; n = 0
    dtype = complex if proj else float
    ratios, hs = [], []
    freal = lambda tt, yy: np.real(f(tt, yy))
    while t < T * (1 - 1e-14) and n < max_steps:
        n += 1
        if t + h > T:
            h = T - t
        K = np.zeros((s, len(y)), dtype)
        K[0] = k1
        for i in range(1, s):
            K[i] = f(t + c[i] * h, y + h * (A[i, :i] @ K[:i]))
        ynew = y + h * (b @ K); yhat = y + h * (bh @ K)
        if proj:
            ynew = ynew.real; yhat = yhat.real
        err = np.sqrt(np.mean(((ynew - yhat) / sc(y, ynew)) ** 2))
        if err <= 1.0:
            u = solve_ivp(freal, (t, t + h), y, method="DOP853", rtol=1e-14, atol=1e-16).y[:, -1]
            true = np.linalg.norm(ynew - u)
            if true > 1e-14:
                ratios.append(np.linalg.norm(ynew - yhat) / true); hs.append(h)
            t += h; y = ynew
            k1 = K[-1].real if fsal else np.real(f(t, y)) * 1.0
            h *= min(facmax, max(0.2, 0.9 * err ** (-1.0 / (q + 1)) if err > 0 else facmax)); facmax = 5.0
        else:
            h *= max(0.2, 0.9 * err ** (-1.0 / (q + 1))); facmax = 1.0
    return ratios, hs


def load56(names):
    """The relevant part of order56 adaptive.load_pairs (CRK6-opt with its 4th-order weights, DP6(5))."""
    from tableaux import load_refined
    TT = load_refined()
    emb = json.load(open(os.path.join(BASE56, "..", "data", "embedded.json")))
    arr = lambda T: np.array([[complex(T.A[i][j]) for j in range(T.s)] for i in range(T.s)])

    def vec(v, s):
        out = np.zeros(s, complex); out[:len(v)] = [complex(x) for x in v]; return out
    T = TT["CRK6-opt"]
    pairs = {"CRK6(4)": dict(A=arr(T), b=vec(T.levels[0][2], T.s),
                             bh=vec([complex(float(a), float(b)) for a, b in emb["CRK6(4)"]["bh"]], T.s),
                             p=T.levels[0][1], q=emb["CRK6(4)"]["q"], projected=True, fsal=False)}
    T = TT["DP6(5)"]
    A, b, bh = arr(T).real, vec(T.levels[0][2], T.s).real, vec(T.levels[1][2], T.s).real
    fsal = bool(np.allclose(A[-1, :-1], b[:-1], atol=1e-15) and abs(b[-1]) < 1e-15)
    pairs["DP6(5)"] = dict(A=A, b=b, bh=bh, p=T.levels[0][1], q=T.levels[1][1], projected=False, fsal=fsal)
    return pairs


def run(group):
    path, names = GROUPS[group]
    sys.path.insert(0, path)
    import adaptive as AD
    from problems import P
    if group == "56":
        pairs = load56(names)
    else:
        AD._NAMES = names
        pairs = AD.load_pairs(names)
    res = {}
    for name in names:
        for prob in PROBS:
            pr = P[prob]
            for tol in TOLS:
                r, hs = run_record(pairs[name], pr["f_np"], [float(v) for v in pr["y0"]], float(pr["T"]), tol)
                med = float(np.median(r)) if r else None
                res.setdefault(name, {}).setdefault(prob, {})[f"{tol:g}"] = dict(
                    median=med, q10=float(np.quantile(r, 0.1)) if r else None,
                    q90=float(np.quantile(r, 0.9)) if r else None, n=len(r),
                    hmed=float(np.median(hs)) if hs else None)
                print(f"{name:8s} {prob:12s} tol {tol:.0e}  median ratio {med if med is None else round(med, 2)}"
                      f"  ({len(r)} steps)", flush=True)
    json.dump(res, open(os.path.join(DATA, f"est_ratio_{group}.json"), "w"), indent=1)


def table():
    res = {}
    for g in GROUPS:
        res.update(json.load(open(os.path.join(DATA, f"est_ratio_{g}.json"))))
    for name, d in res.items():
        row = []
        for tol in TOLS:
            v = [d[p][f"{tol:g}"]["median"] for p in PROBS if d[p][f"{tol:g}"]["median"]]
            row.append(float(np.exp(np.mean(np.log(v)))) if v else float("nan"))
        print(f"{name:8s} " + "  ".join(f"{x:8.2f}" for x in row))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "78"
    if what == "table":
        table()
    else:
        run(what)
