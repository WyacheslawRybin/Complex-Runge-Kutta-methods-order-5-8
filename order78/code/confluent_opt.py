"""
Local minimization of the ninth-order error constant E_9 of the confluent scheme L
(Section 3.4) on its solution set, under max|coef| <= 5, in double precision.
Start: ../data/confluent_L.json. Method: projected gradient on the tangent space of the
200 order conditions (rank 62 of 68 unknowns), Gauss-Newton return to the set.
Writes ../data/confluent_L_opt.json (17 digits); refine it with confluent_mp.py --from-opt.
    python confluent_opt.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crk
from confluent_mp import NAMES, STAGES

DATA = os.path.join(HERE, "..", "data")
TR8 = list(crk.trees_upto(8))
TR9 = list(crk.TREES[9])
G8 = np.array([1.0 / crk.gamma(t) for t in TR8])
G9 = np.array([1.0 / crk.gamma(t) for t in TR9])
S9 = np.array([float(crk.sigma(t)) for t in TR9])
BOUND = 5.0

def unpack(x):
    v = {k: complex(x[2 * n], x[2 * n + 1]) for n, k in enumerate(NAMES)}
    v["β_1"], v["β_G"] = complex(x[-2]), complex(x[-1])
    return v

def bphi_all(x, trees):
    v = unpack(x)
    tau, ttau = (), ((),)
    cache = {}
    def Phi(t):
        if t in cache:
            return cache[t]
        out = {1: 1.0 if t == tau else 0.0, "G": 1.0 if t == ttau else 0.0}
        for i in STAGES:
            prod = 1.0 + 0j
            for u in t:
                pu = Phi(u)
                prod *= (v[f"γ_{i}"] * pu[1] + v[f"α_{i}"] * pu["G"]
                         + sum(v[f"a_{i}{j}"] * pu[j] for j in range(3, i)))
            out[i] = prod
        cache[t] = out
        return out
    res = []
    for t in trees:
        ph = Phi(t)
        res.append((v["β_1"] * ph[1] + v["β_G"] * ph["G"] + sum(v[f"b_{i}"] * ph[i] for i in STAGES)).real)
    return np.array(res)

def resid(x):
    return bphi_all(x, TR8) - G8

def E9(x):
    return float(np.linalg.norm((bphi_all(x, TR9) - G9) / S9))

def maxcoef(x):
    return max(abs(c) for c in unpack(x).values())

def jac(F, x, d=1e-7):
    f0 = F(x)
    J = np.empty((len(f0), len(x)))
    for k in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[k] += d; xm[k] -= d
        J[:, k] = (F(xp) - F(xm)) / (2 * d)
    return J

def restore(x, iters=12):
    for _ in range(iters):
        r = resid(x)
        if np.max(np.abs(r)) < 1e-14:
            break
        J = jac(resid, x)
        x = x - np.linalg.lstsq(J, r, rcond=1e-9)[0]
    return x, float(np.max(np.abs(resid(x))))

def objective(x):
    return E9(x) * (1.0 + 10.0 * max(0.0, maxcoef(x) - BOUND))

if __name__ == "__main__":
    d0 = json.load(open(os.path.join(DATA, "confluent_L.json"), encoding="utf-8"))
    x = []
    for k in NAMES + ["β_1", "β_G"]:
        re_s, im_s = d0["coef"][k]
        x += [float(re_s), float(im_s)] if k not in ("β_1", "β_G") else [float(re_s)]
    x = np.array(x)
    x, r = restore(x)
    f = objective(x)
    print(f"start: E9 = {E9(x):.4e}, max|coef| = {maxcoef(x):.3f}, resid {r:.1e}", flush=True)
    step = 0.05
    for it in range(400):
        J = jac(resid, x)
        U, S, Vt = np.linalg.svd(J)
        rank = int(np.sum(S > 1e-9 * S[0]))
        N = Vt[rank:].T
        g = jac(lambda y: np.array([objective(y)]), x)[0]
        dg = -N @ (N.T @ g)
        if np.linalg.norm(dg) < 1e-14:
            break
        dg /= np.linalg.norm(dg)
        ok = False
        while step > 1e-6:
            xn, rn = restore(x + step * dg)
            fn = objective(xn)
            if rn < 1e-12 and fn < f:
                x, f, ok = xn, fn, True
                step *= 1.5
                break
            step /= 2
        if not ok:
            break
        if it % 10 == 0:
            print(f"it {it}: E9 = {E9(x):.5e}, max|coef| = {maxcoef(x):.3f}, rank {rank}, step {step:.2e}", flush=True)
    print(f"final: E9 = {E9(x):.5e}, max|coef| = {maxcoef(x):.4f}, resid {np.max(np.abs(resid(x))):.1e}")
    v = unpack(x)
    out = dict(E9=E9(x), maxcoef=maxcoef(x), coef={k: [repr(v[k].real), repr(v[k].imag)] for k in NAMES + ["β_1", "β_G"]})
    json.dump(out, open(os.path.join(DATA, "confluent_L_opt.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
