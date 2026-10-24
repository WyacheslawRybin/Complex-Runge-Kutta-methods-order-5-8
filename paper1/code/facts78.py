"""Numbers quoted in the text for orders seven and eight that change when CCK7(6) is removed.  Read-only on
the data of the order 7-8 paper; prints the numbers and writes ../data/facts78.json.

usage: python facts78.py      (from this folder, PYTHONIOENCODING=utf-8)
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OLD = os.path.abspath(os.path.join(ROOT, "..", "order78", "code"))
sys.path.insert(0, OLD)

import make_tables as MT  # noqa: E402

OUT = {}
CCK = ("CCK7(6)", "CCK7-casc")
MT.ORDER7[:] = [m for m in MT.ORDER7 if m not in CCK]


def say(k, v):
    OUT[k] = v
    print(f"{k}: {v}")


def main():
    S = MT.measured()
    n_ok = n01 = 0
    dev = []
    for m in S:
        p = MT.ORDER_OF(m)
        for q, v in S[m].items():
            if v["cls"] == "deg":
                continue
            n_ok += 1
            d = abs(v["order"] - p)
            n01 += d <= 0.1
            if d > 0.3:
                dev.append((m, q, round(d, 3)))
    say("order_counts(nondeg, within0.1)", (n_ok, n01))
    say("order_dev_gt_0.3", dev)
    ad = json.load(open(os.path.join(MT.DATA, "adaptive.json")))
    real7, real8 = ["Verner7(6)", "Vern7"], ["DP8(7)", "Verner8(7)", "DOP853", "Vern8"]
    c8 = ["CRK8(6)", "CRK8-s(6)"]
    for eps in (1e-6, 1e-8, 1e-10):
        r7, x = {}, {}
        for q in MT.ADAPT_PROBS:
            c = {m: MT.cost_at(*MT.adapt_curve(ad, m, q)[:2], eps) for m in ["CRK7(5)"] + real7 + c8 + real8}
            b7r = min(c[m] for m in real7)
            r7[q] = c["CRK7(5)"] / b7r if math.isfinite(b7r) else None
            b7 = min(c[m] for m in ["CRK7(5)"] + real7)
            b8 = min(c[m] for m in c8 + real8)
            x[q] = b8 / b7 if math.isfinite(b7) and math.isfinite(b8) else None
        say(f"CRK7(5)_over_best_real7_{eps:g}", {k: (None if v is None else round(v, 3)) for k, v in r7.items()})
        say(f"best8_over_best7_{eps:g}", {k: (None if v is None else round(v, 3)) for k, v in x.items()})
    # summary: relative variable-step costs (kappa = 1), their kappa = 2 values and the break-even kappa
    B = MT.bench()
    MT.table_summary(S, B) if False else None
    num = json.load(open(os.path.join(ROOT, "data", "numbers78.json")))
    sm = num["summary"]
    for p, pairs in (("7", {"CRK7-opt": 7}), ("8", {"CRK8-opt": 9, "CRK8-s": 9})):
        for m, s in pairs.items():
            v = float(sm[p]["ad1"]["rel"][m])
            k2 = v * (2 * s - 3) / s
            kb = (s / v - 3) / (s - 3)
            say(f"summary_{m}_ad1", dict(rel=round(v, 4), kappa2=round(k2, 4), breakeven=round(kb, 3)))
            f = float(sm[p]["fix1"]["rel"][m])
            say(f"summary_{m}_fix1", dict(rel=round(f, 4), breakeven=round((s / f - 3) / (s - 3), 3)))
    json.dump(OUT, open(os.path.join(ROOT, "data", "facts78.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
