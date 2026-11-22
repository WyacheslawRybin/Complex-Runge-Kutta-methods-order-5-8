"""
Break-even ratio kappa* over the accuracies 1e-4 ... 1e-12 with step-size control.
For each order and problem, kappa* = (evaluations of the best real pair) / (evaluations of the best
projected pair) at the accuracy eps, both counted at kappa = 1 (lower-envelope interpolation of
make_tables.cost_at). A projected pair is cheaper at a cost ratio kappa exactly when kappa < kappa*,
The groups are those of Tables 13 and 14 (geometric means 1/1.97, 1.00, 1/0.78, 1/0.81 at 1e-8),
so the crossover accuracy for kappa = 1, 2, 4 is where kappa* crosses that value.
Writes ../data/kappa_cross.json and prints the geometric means over the problems.
    python kappa_cross.py
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tables import cost_at, adapt_curve

D56 = os.path.join(HERE, "..", "..", "order56", "code", "r1", "data", "merged", "adaptive.json")
D78 = os.path.join(HERE, "..", "data", "adaptive.json")
GROUPS = {
    5: (D56, ["CRK5(4)"], ["DP5", "CK5", "Tsit5(4)", "BS5(4)"]),
    6: (D56, ["CRK6(4)"], ["DP6(5)", "Verner6(5)", "Vern6(5)", "RK6-opt(4)"]),
    7: (D78, ["CRK7(5)"], ["Verner7(6)", "Vern7"]),
    8: (D78, ["CRK8(6)", "CRK8-s(6)"], ["DP8(7)", "Verner8(7)", "DOP853", "Vern8"]),
}
PROBS = ["exponential", "nonautonomous", "VanderPol", "Rossler", "Bouali", "Lorenz5", "Kepler",
         "RigidBody", "Brusselator", "Arenstorf"]
EPS = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

out = {}
for order, (fn, cplx, real) in GROUPS.items():
    ad = json.load(open(fn))
    out[order] = {}
    for eps in EPS:
        ks = {}
        for p in PROBS:
            c = min(cost_at(*adapt_curve(ad, m, p)[:2], eps) for m in cplx)
            r = min(cost_at(*adapt_curve(ad, m, p)[:2], eps) for m in real)
            if math.isfinite(c) and math.isfinite(r):
                ks[p] = float(r / c)
        g = float(np.exp(np.mean(np.log(list(ks.values()))))) if ks else None
        out[order][f"{eps:.0e}"] = dict(per_problem=ks, geomean=g, n=len(ks),
                                        wins={k: int(sum(v > k for v in ks.values())) for k in (1, 2, 4)})
    print(f"order {order}: kappa* (geometric mean, problems) and wins at kappa = 1/2/4:")
    for eps in EPS:
        o = out[order][f"{eps:.0e}"]
        print(f"   {eps:.0e}: {o['geomean']:.2f} (n={o['n']}, range {min(o['per_problem'].values()):.2f}-"
              f"{max(o['per_problem'].values()):.2f}), wins {o['wins'][1]}/{o['wins'][2]}/{o['wins'][4]}")
json.dump(out, open(os.path.join(HERE, "..", "data", "kappa_cross.json"), "w"), indent=1)
