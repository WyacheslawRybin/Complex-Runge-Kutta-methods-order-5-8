"""Tables of orders seven and eight for the CRK 5-8 paper: the comparisons of the order 7-8 paper without the
complex Cash--Karp pair CCK7(6).

The code and the data of the order 7-8 paper are used read-only (make_tables reads ../data and writes only
through its write(), which is redirected here).  The tables go to ../paper/tables/78_*.tex and the derived
numbers to ../data/numbers78.json.

usage: python tables78.py      (from this folder, PYTHONIOENCODING=utf-8)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OLD = os.path.abspath(os.path.join(ROOT, "..", "order78", "code"))
TAB = os.path.join(ROOT, "paper", "tables")
DATA = os.path.join(ROOT, "data")
for d in (TAB, DATA):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, OLD)

import make_tables as MT  # noqa: E402

CCK = ("CCK7(6)", "CCK7-casc")


def write(name, lines):
    open(os.path.join(TAB, "78_" + name + ".tex"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


MT.write = write
MT.TAB = TAB
for lst in (MT.ORDER7, MT.ADAPT7, MT.BENCH_METHODS, MT.FIXEDPOINT7):
    lst[:] = [m for m in lst if m not in CCK]
MT.KAPPA_COLS[:] = [(c, r) for c, r in MT.KAPPA_COLS if c not in CCK]
MT.STAB_ROWS7[:] = [row for row in MT.STAB_ROWS7 if row[0] not in CCK]
MT.PAIRS7[:] = [row for row in MT.PAIRS7 if row[0] not in CCK]
for G in (MT.CR_FIXED, MT.CR_ADAPT):
    G[7] = ([m for m in G[7][0] if m not in CCK], G[7][1])
for k in CCK:
    MT.PAIR_OF.pop(k, None)


def main():
    S = MT.measured()
    MT.table_measured(S); MT.table_kappa(S); MT.table_stability(); MT.table_adaptive(); MT.table_pairs()
    B = MT.bench()
    MT.table_bench(B); MT.table_cost_ratio(B); MT.table_summary(S, B)
    MT.table_fixedpoint(); MT.table_l96(B)
    json.dump(MT.NUM, open(os.path.join(DATA, "numbers78.json"), "w"), indent=1, default=str)
    for n in ("cost_ratio", "summary", "adaptive_8", "l96", "bench", "kappa_star"):
        print("==", n)
        print(open(os.path.join(TAB, "78_" + n + ".tex"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
