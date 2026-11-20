"""Real extrapolation methods of orders seven and eight against CRK7 and CRK8 at equal latency
(revision R1, items R1-Q6 and R3-Q5; order56/code/r1/code/gbs_r1.py does the same at orders five and six).

GBS(2,4,6,8): extrapolated Gragg midpoint method with the step numbers 2, 4, 6, 8 and no smoothing, a real
17-stage RK tableau of order eight; the extrapolation of 2, 4, 6 is its embedded formula of order six, so
it is a pair 8(6), like CRK8(6).  On three processors ({8}, {6, 2}, {4}) a step takes 8 sequential
evaluations.  EX7: extrapolated explicit Euler method with the step numbers 1, ..., 7, a real 22-stage
tableau of order seven with the embedded EX6 (order six); on four processors ({7}, {6, 1}, {5, 2}, {4, 3}) a
step takes 7 sequential evaluations.  Each update adds one product and one addition, so a step takes
depth * l_R.  A step of a projected method with s stages, the first m = 3 of them real, takes
T_C = D_R + (m - 1) l_R + (L_m + L_a + D_C) + (s - m - 1)(L_m + 2 L_a + D_C) + L_m + 2 L_a
(latency model of Section 8.1 of the paper; D_R, D_C from order56/code/r1/data/latency_r1.json).

Variable steps with the controller of adaptive.py on the ten problems.  Writes ../data/adaptive_gbs78.json
and ../data/gbs78.json (evaluations at 1e-8 and latency ratios complex/real at nu = 1/4 and nu = 1).
Usage: python gbs78.py [run|table|all]
"""
import json
import os
import sys
from fractions import Fraction as F
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FN = os.path.join(DATA, "adaptive_gbs78.json")
LAT = os.path.join(HERE, "..", "..", "order56", "code", "r1", "data", "latency_r1.json")
POLY = ["VanderPol", "Rossler", "Bouali", "Lorenz5", "RigidBody", "Brusselator"]   # polynomial problems


def _vec(out, w, s):
    return [sum((w[n] * out[n].get(k, 0) for n in w), F(0)) for k in range(s)]


def _matrix(stages):
    s = len(stages)
    A = [[F(0)] * s for _ in range(s)]
    for i, st in enumerate(stages):
        for k, v in st.items():
            A[i][k] = v
    return A


def gbs_tableau(seq):
    stages, out = [dict()], {}
    for n in seq:
        d = F(1, n)
        z_prev, z = {}, {0: d}
        for _ in range(1, n):
            stages.append(dict(z))
            ki = len(stages) - 1
            z_new = dict(z_prev)
            z_new[ki] = z_new.get(ki, 0) + 2 * d
            z_prev, z = z, z_new
        out[n] = z

    def weights(ns):
        w = {}
        for nj in ns:
            wj = F(1)
            for ni in ns:
                if ni != nj:
                    wj *= F(nj * nj, nj * nj - ni * ni)
            w[nj] = wj
        return w

    s = len(stages)
    return _matrix(stages), _vec(out, weights(seq), s), _vec(out, weights(seq[:-1]), s)


def euler_tableau(seq):
    stages, out = [dict()], {}
    for n in seq:
        d = F(1, n)
        z = {0: d}
        for _ in range(1, n):
            stages.append(dict(z))
            z = dict(z)
            ki = len(stages) - 1
            z[ki] = z.get(ki, 0) + d
        out[n] = z

    def weights(ns):
        w = {}
        for nj in ns:
            wj = F(1)
            for ni in ns:
                if ni != nj:
                    wj *= F(nj, nj - ni)
            w[nj] = wj
        return w

    s = len(stages)
    return _matrix(stages), _vec(out, weights(seq), s), _vec(out, weights(seq[:-1]), s)


# name: (tableau, order p, embedded order q, depth, processors, complex pairs, s, real stages m)
METHODS = {
    "GBS(2,4,6,8)": (lambda: gbs_tableau((2, 4, 6, 8)), 8, 6, 8, 3, ("CRK8(6)", "CRK8-s(6)"), 9, 3),
    "EX7": (lambda: euler_tableau(tuple(range(1, 8))), 7, 6, 7, 4, ("CRK7(5)", "CRK7-s(5)"), 7, 3),
}


def pair(name):
    fun, p, q = METHODS[name][:3]
    A, b, bh = fun()
    return dict(A=np.array([[float(x) for x in r] for r in A]), b=np.array([float(x) for x in b]),
                bh=np.array([float(x) for x in bh]), p=p, q=q, projected=False, fsal=False)


_P = _PROB = _REF = None


def _init():
    global _P, _PROB, _REF
    sys.path.insert(0, HERE)
    from problems import P, references
    _PROB = P
    _REF = {k: np.array([float(v) for v in vals]) for k, vals in references().items()}
    _P = {n: pair(n) for n in METHODS}


def task(args):
    import adaptive as AD
    name, prob = args
    pr = _PROB[prob]
    rows = []
    for tol in AD.TOLS:
        out = AD.solve(_P[name], pr["f_np"], [float(v) for v in pr["y0"]], float(pr["T"]), tol)
        if out is None:
            rows.append(dict(tol=tol, err=None))
            continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=float(np.linalg.norm(y - _REF[prob])), nfev=nf, nacc=na, nrej=nr))
    return name, prob, rows


def check():
    """Orders of the tableaux (rooted-tree residuals) and the depth on the stated processors."""
    sys.path.insert(0, HERE)
    import mpmath as mp
    import crk
    mp.mp.dps = 40
    for name, (fun, p, q, depth, nproc, *_ ) in METHODS.items():
        A, b, bh = fun()
        cv = lambda v: [mp.mpf(x.numerator) / x.denominator for x in v]
        T = crk.Tableau(name, [cv(r) for r in A], [(len(b), p, cv(b)), (len(b), q, cv(bh))], projected=False)
        res = T.residuals()
        print(name, "stages", len(b), "max residual order", p, mp.nstr(max(abs(v) for v in res[0]), 3),
              " embedded order", q, mp.nstr(max(abs(v) for v in res[1]), 3), " depth", depth, "on", nproc)


def run(nproc=6):
    sys.path.insert(0, HERE)
    from problems import ADAPTIVE
    res = {}
    with Pool(nproc, initializer=_init) as pool:
        for name, prob, rows in pool.imap_unordered(task, [(n, p) for n in METHODS for p in ADAPTIVE]):
            res.setdefault(name, {})[prob] = rows
            ok = [r for r in rows if r["err"] is not None]
            print(f"{name:13s} {prob:13s} last err={ok[-1]['err']:.2e} nfev={ok[-1]['nfev']}", flush=True)
    json.dump(res, open(FN, "w"), indent=1)
    print("wrote", FN)


def table(eps=1e-8):
    sys.path.insert(0, HERE)
    from figures_cost import adapt_curve, cost_at
    g = json.load(open(FN))
    ad = json.load(open(os.path.join(DATA, "adaptive.json")))
    lat = json.load(open(LAT))
    cost = lambda rows: cost_at(np.array([r["nfev"] for r in rows if r["err"]], float),
                                np.array([r["err"] for r in rows if r["err"]]), eps)
    num = {}
    for name, (fun, p, q, depth, nproc, cms, s, m) in METHODS.items():
        evals = len(fun()[1])
        for prob in POLY:
            d = lat[prob]
            cg = cost(g[name][prob])
            cc, cm = min((cost_at(*adapt_curve(ad, c, prob)[:2], eps), c) for c in cms)
            r = {}
            for nu in (0.25, 1.0):
                DR = d["path_real"][0] + d["path_real"][1] * nu          # L_m = 1, L_a = nu
                DC = d["path_cplx"][0] + d["path_cplx"][1] * nu
                lR = DR + 1 + nu
                TC = DR + (m - 1) * lR + (1 + nu + DC) + (s - m - 1) * (1 + 2 * nu + DC) + 1 + 2 * nu
                klat = TC / (s * lR)
                r[nu] = (cc * klat) / (cg * depth / evals)
            num.setdefault(name, {})[prob] = dict(real=cg, cplx=cc, cplx_method=cm, ratio_nu025=r[0.25],
                                                  ratio_nu1=r[1.0])
            print(f"{prob:12s} {name:13s} {cg:7.0f}  {cm:10s} {cc:7.0f}  latency ratio nu=1/4 {r[0.25]:.2f}"
                  f"  nu=1 {r[1.0]:.2f}")
    json.dump(num, open(os.path.join(DATA, "gbs78.json"), "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("check", "all"):
        check()
    if what in ("run", "all"):
        run()
    if what in ("table", "all"):
        table()
