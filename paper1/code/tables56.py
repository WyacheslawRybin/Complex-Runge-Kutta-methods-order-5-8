"""Tables of orders five and six for the CRK 5-8 paper: the comparisons of the order 5-6 paper without the
complex Cash--Karp pairs (CCK6(5), CCK6(5)-s, CCK6-cascade).

The code and the data of the order 5-6 paper are used read-only: r1patch.build() is replaced by the path of
the merged data that it wrote, so nothing is written into ../order56.  The tables go to
../paper/tables/56_*.tex and the derived numbers to ../data/numbers56.json.

usage: python tables56.py      (from this folder, PYTHONIOENCODING=utf-8)
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OLD = os.path.abspath(os.path.join(ROOT, "..", "order56", "code", "r1", "code"))
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "paper", "figures")
DATA = os.path.join(ROOT, "data")
for d in (TAB, FIG, DATA):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, OLD)

import r1common as R  # noqa: E402

R.TAB, R.FIG = TAB, FIG
import r1patch  # noqa: E402

r1patch.build = lambda: r1patch.MERGED
MT = r1patch.apply()
import figures_cost as FC  # noqa: E402

CCK = ("CCK6(5)", "CCK6(5)-s", "CCK6-casc", "CCK6-cascade")
# tableaux studied in the order 5-6 paper but not reported in this one
DROP = ("CRK5-1", "CRK5-2", "CRK6-0", "CRK6-lam")
for lst in (FC.FIXED5, FC.FIXED6, FC.FIG5, FC.FIG6, FC.ADAPT5, FC.ADAPT6, FC.ADAPT, FC.FIXED):
    lst[:] = [m for m in lst if m not in CCK + DROP]

import tables_cpu_r1 as TC  # noqa: E402
from figures_cost import adapt_curve, cost_at  # noqa: E402


def rename(name):
    """tables_cpu_r1 writes bench/cost_ratio/summary.tex into R.TAB; give them the prefix 56_."""
    src = os.path.join(TAB, name + ".tex")
    dst = os.path.join(TAB, "56_" + name + ".tex")
    os.replace(src, dst)
    return open(dst, encoding="utf-8").read()


def kappa_cpu(B):
    """A complex stage of CRK6-opt over a real stage of RK6-opt (the order 5-6 paper used CCK6(5))."""
    return {p: (B[(p, "CRK6-opt")]["ns"] / 6) / (B[(p, "RK6-opt")]["ns"] / 7)
            for p in TC.BENCH_PROBS if (p, "CRK6-opt") in B}


def table_bench(B, kap):
    methods = [m for m in TC.BENCH_METHODS if m not in CCK]
    lines = ["problem & " + " & ".join(MT.rot(MT.mname(m)) for m in methods) + r" & $\kappa_{\mathrm{CPU}}$\\",
             r"\midrule"]
    for p in kap:
        name = {"Lorenz": "Lorenz", "Bruss2D": "BRUSS-2D"}.get(p, MT.PNAME.get(p, p))
        cells = ["---" if (p, m) not in B else f"${B[(p, m)]['ns']:.0f}$" for m in methods]
        lines.append(name + " & " + " & ".join(cells) + f" & ${kap[p]:.2f}$" + r"\\")
    open(os.path.join(TAB, "56_bench.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def latency(eps=1e-8):
    """Cost ratio CRK6(4) / best real order-six pair at eps for the polynomial problems, scaled by kappa_lat."""
    L = json.load(open(os.path.join(R.DATA, "latency_r1.json")))
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    real = [m for m in FC.ADAPT6 if not m.startswith("CRK")]
    out = {}
    for p in ("VanderPol", "Rossler", "Bouali", "Lorenz5", "RigidBody", "Brusselator"):
        c = cost_at(*adapt_curve(ad, "CRK6(4)", p)[:2], eps)
        r = min(cost_at(*adapt_curve(ad, m, p)[:2], eps) for m in real)
        out[p] = c / r
    return out, L


def adaptive_tables():
    """The variable-step tables 10^-6, 10^-8, 10^-10 without the CCK columns, as in tables_r1."""
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    res = {}
    for eps in (1e-6, 1e-8, 1e-10):
        lines = []
        for p in FC.PREF_ADAPT:
            row = []
            for grp in (FC.ADAPT5, FC.ADAPT6):
                v = [cost_at(*adapt_curve(ad, m, p)[:2], eps) for m in grp]
                b = min(v)
                row += ["---" if not math.isfinite(x) else (MT.best(f"{x:.0f}") if x <= b * (1 + 5e-3) else f"{x:.0f}")
                        for x in v]
                res[f"{p}|{eps:g}"] = {m: (x if math.isfinite(x) else None) for m, x in zip(grp, v)}
            lines.append(MT.PNAME[p] + " & " + " & ".join(row) + r"\\")
        head = "problem & " + " & ".join(MT.rot(MT.mname(m)) for m in FC.ADAPT5 + FC.ADAPT6) + r"\\"
        open(os.path.join(TAB, f"56_adaptive_{-int(round(math.log10(eps)))}.tex"), "w", encoding="utf-8").write(
            head + "\n\\midrule\n" + "\n".join(lines) + "\n")
    return res


def measured_tables():
    """measured5/6 and kappa_star of tables_r1 without the CCK columns."""
    import tables_r1 as TR
    TR.M5[:] = [m for m in TR.M5 if not m.startswith("CCK") and m not in DROP]
    TR.M6[:] = [m for m in TR.M6 if not m.startswith("CCK") and m not in DROP]
    TR.KAPPA_COLS[:] = [(c, r) for c, r in TR.KAPPA_COLS if not c.startswith("CCK")]
    TR.write = lambda name, lines: open(os.path.join(TAB, "56_" + name), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    try:
        TR.table_measured()
    except KeyError:
        # its break-even part still loops over the CCK tableaux; measured5/6 are already written, and the
        # break-even table is the published one without the CCK6(5) columns (the other values do not change)
        pass


def main():
    measured_tables()
    B = TC.bench(os.path.join(R.DATA, "bench.csv"))
    kap = kappa_cpu(B)
    table_bench(B, kap)
    TC.table_cost_ratio(B)
    TC.table_summary(B)
    for n in ("cost_ratio", "summary"):
        print("==", n)
        print(rename(n))
    ad = adaptive_tables()
    lat, L = latency()
    num = dict(TC.NUM, kappa_cpu=kap, latency_cost_ratio=lat, adaptive=ad,
               adapt5=FC.ADAPT5, adapt6=FC.ADAPT6, fixed6=FC.FIXED6)
    json.dump(num, open(os.path.join(DATA, "numbers56.json"), "w"), indent=1,
              default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else str(o))
    print("kappa_CPU (CRK6-opt / RK6-opt):", {k: round(v, 2) for k, v in kap.items()})
    print("latency cost ratios (CRK6(4) / best real, 1e-8):", {k: round(v, 3) for k, v in lat.items()})
    print("latency json keys:", list(L)[:20])
    print("ADAPT6:", FC.ADAPT6, " FIXED6:", FC.FIXED6)


if __name__ == "__main__":
    main()
