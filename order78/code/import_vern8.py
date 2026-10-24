"""Vern8: Verner's "most efficient" 13-stage 8(7) pair (Numer. Algorithms 53 (2010) 383-396,
RKV87.IIa.Efficient.000000282866.081208), exact rational coefficients as in the generic
Vern8Tableau(::Type{T}, ::Type{T2}) of OrdinaryDiffEq.jl (lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl).

Usage (from code/):  python import_vern8.py PATH/verner_tableaus.jl
Writes ../data/vern8_source.jl (the parsed block, for provenance) and ../data/vern8_tableau.json
in the layout of rk_tableaux.json (A_exact, b_exact, bhat_exact as rational strings) and checks the
order conditions of both formulas in 60-digit arithmetic."""
import json, os, re, sys
from fractions import Fraction
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

src = open(sys.argv[1], encoding="utf-8").read()
i0 = src.index("function Vern8Tableau(::Type{T}, ::Type{T2}) where {T, T2}")
i1 = src.index("extra = Vern8ExtraStages(T, T2)", i0)
block = src[i0:i1]
open(os.path.join(DATA, "vern8_source.jl"), "w", encoding="utf-8", newline="\n").write(
    "# excerpt of OrdinaryDiffEq.jl lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl (generic Vern8Tableau)\n" + block)

ALLOWED = re.compile(r"^[0-9()+\-/ ]+$")
vals = {}
for m in re.finditer(r"^\s*(\w+)\s*=\s*convert\(\s*T2?\s*,", block, re.M):
    name, k, depth = m.group(1), m.end(), 1
    while depth:                                  # scan to the parenthesis that closes convert(
        depth += {"(": 1, ")": -1}.get(block[k], 0); k += 1
    expr = block[m.end():k - 1]
    expr = re.sub(r"\s+", " ", expr).replace("BigInt", "").replace("//", "/")
    if not ALLOWED.match(expr):
        sys.exit(f"unexpected expression for {name}: {expr[:80]}")
    # evaluate p/q with Fractions only (no eval of arbitrary code)
    toks = re.findall(r"\d+|[()+\-/]", expr)
    def parse(pos):                       # expr := term ('/' term)*, term := ['-'] (int | '(' expr ')')
        def term(p):
            neg = False
            while toks[p] in "+-":
                neg ^= toks[p] == "-"; p += 1
            if toks[p] == "(":
                v, p = parse(p + 1); assert toks[p] == ")"; p += 1
            else:
                v, p = Fraction(int(toks[p])), p + 1
            return (-v if neg else v), p
        v, pos = term(pos)
        while pos < len(toks) and toks[pos] == "/":
            w, pos = term(pos + 1); v /= w
        return v, pos
    v, pos = parse(0)
    assert pos == len(toks), name
    vals[name] = v

s = 13
A = [[Fraction(0)] * s for _ in range(s)]
for k, v in vals.items():
    m = re.fullmatch(r"a(\d\d)(\d\d)", k)
    if m:
        A[int(m.group(1)) - 1][int(m.group(2)) - 1] = v
b = [vals.get(f"b{i}", Fraction(0)) for i in range(1, s + 1)]
bt = [vals.get(f"btilde{i}", Fraction(0)) for i in range(1, s + 1)]
bhat = [x - y for x, y in zip(b, bt)]            # btilde = b - bhat (bhat13 = -btilde13, b13 = 0)
for i in range(1, 12):                            # c_i = sum_j a_ij must match the listed abscissae
    if f"c{i + 1}" in vals:
        assert sum(A[i]) == vals[f"c{i + 1}"], i
assert sum(A[11]) == 1 and sum(A[12]) == 1

d = {"name": "Verner 2010 most efficient 8(7) pair (Vern8)", "order": 8, "embedded_order": 7, "stages": s,
     "reference": "J.H. Verner, Numerically optimal Runge-Kutta pairs with interpolants, Numer. Algorithms 53 "
                  "(2010) 383-396; RKV87.IIa.Efficient.000000282866.081208; rational coefficients from "
                  "OrdinaryDiffEq.jl (Vern8Tableau, generic method)",
     "A_exact": [[str(A[i][j]) for j in range(i)] for i in range(s)],
     "b_exact": [str(v) for v in b], "bhat_exact": [str(v) for v in bhat],
     "c_exact": [str(sum(A[i])) for i in range(s)]}
json.dump({"Vern8": d}, open(os.path.join(DATA, "vern8_tableau.json"), "w"), indent=1)

# order check in 60 digits
sys.path.insert(0, HERE)
from crk import Tableau
mp.mp.dps = 60
F = lambda q: mp.mpf(q.numerator) / q.denominator
Am = [[mp.mpc(F(A[i][j])) for j in range(s)] for i in range(s)]
T = Tableau("Vern8", Am, [(s, 8, [mp.mpc(F(v)) for v in b]), (s, 7, [mp.mpc(F(v)) for v in bhat])], projected=False)
for L, lev in enumerate(T.residuals()):
    print(f"level {L}: order {T.levels[L][1]}, max residual {mp.nstr(max(abs(v) for v in lev), 3)}")
print("E9 =", mp.nstr(T.E(0), 6), " E8(bhat) =", mp.nstr(T.E(1), 6), " max|coef| =", mp.nstr(T.maxcoef(), 6),
      " stages used by b:", max(i + 1 for i, v in enumerate(b) if v != 0))
