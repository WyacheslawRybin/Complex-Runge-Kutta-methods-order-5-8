"""
Singular-value gaps at the refined CRK7 and CRK8 tableaux in multiple precision
(answer to the question whether the ranks of Table A2 and the span of Proposition 3 are certified).
1) Span of Phi_u, rho(u) <= p-1, over R: the 2s x N real matrix [Re Phi; Im Phi], expected rank 2s-m.
2) Jacobian of the order conditions of the propagated level (central differences), expected
   rank 48 (CRK7) and 79 (CRK8).
Prints the singular values around the gap relative to the largest one and writes ../data/sv_gaps.json.
    python sv_gaps.py [dps]
"""
import json, os, sys
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crk
from tableaux import load_refined

mp.mp.dps = int(sys.argv[1]) if len(sys.argv) > 1 else 90
NAMES = ["CRK7-opt", "CRK7-s", "CRK8-opt", "CRK8-s"]
EXPECT = {"CRK7": (11, 48), "CRK8": (15, 79)}

def svals(M):
    sv = mp.svd_r(M, compute_uv=False)
    return sorted([sv[i] for i in range(len(sv))], reverse=True)

def around(sv, r):
    return {f"sigma_{k + 1}": float(sv[k] / sv[0]) for k in range(r - 2, min(len(sv), r + 2))}

TT = load_refined()
out = {}
for name in NAMES:
    T = TT[name].level(0) if hasattr(TT[name], "level") else TT[name]
    s, p = T.s, T.p
    rs, rj = EXPECT[name[:4]]
    # 1) span of Phi_u
    ph = T.phi(p - 1)
    trees = list(crk.trees_upto(p - 1))
    M = mp.matrix(2 * s, len(trees))
    for k, t in enumerate(trees):
        for i in range(s):
            M[i, k] = mp.re(ph[t][i])
            M[s + i, k] = mp.im(ph[t][i])
    svs = svals(M)
    # 2) Jacobian
    x = T.get_x()
    F = lambda xx: [v for v in T.set_x(xx).residuals()[0]]
    h = mp.mpf(10) ** -(mp.mp.dps // 3)
    cols = []
    for k in range(len(x)):
        xp, xm = x[:], x[:]
        xp[k] += h; xm[k] -= h
        cols.append([(a - b) / (2 * h) for a, b in zip(F(xp), F(xm))])
    T.set_x(x)
    J = mp.matrix(len(cols[0]), len(x))
    for k, col in enumerate(cols):
        for i, v in enumerate(col):
            J[i, k] = v
    svj = svals(J)
    res = max(abs(v) for v in T.residuals()[0])
    out[name] = dict(s=s, p=p, unknowns=len(x), conditions=J.rows, resid=float(res),
                     span_rank=rs, span=around(svs, rs), jac_rank=rj, jac=around(svj, rj))
    print(name, f"s={s} unknowns={len(x)} conditions={J.rows} resid={mp.nstr(res, 3)}")
    print("  span:", {k: f"{v:.2e}" for k, v in out[name]["span"].items()})
    print("  jac: ", {k: f"{v:.2e}" for k, v in out[name]["jac"].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "data", "sv_gaps.json"), "w"), indent=1)
