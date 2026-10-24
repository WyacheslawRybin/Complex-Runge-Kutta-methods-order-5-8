"""Order conditions after rounding the coefficients to a fixed-point format (referee comment m8).

Every real and imaginary part of A and of the weights of all levels is rounded to F fractional bits,
x -> round(x * 2^F) / 2^F, with the inert Im b_1 = Im b_2 = Im b_3 = 0 of tableaux.json.  For each format
the script reports the largest residual of the (projected) order conditions of each level, the relative
change of E_{p+1} of the propagated level, and the number of integer bits ceil(log2(max|coef| + 2^-F))
needed for the largest real or imaginary part (sign bit not counted).

Usage (from code/):  python fixedpoint.py [F ...]          (default F = 16 24 32)
Writes ../data/fixedpoint.json."""
import json, math, os, sys
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crk import Tableau
from tableaux import load_refined, real_methods

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
NAMES = ["CRK7-opt", "CRK7-s", "CCK7(6)", "CRK8-opt", "CRK8-s",
         "Vern7", "Verner7(6)", "DOP853", "DP8(7)", "Verner8(7)", "Vern8"]

def rnd(x, F):
    s = mp.mpf(2) ** F
    return mp.nint(x * s) / s

def rounded(T, F):
    r = lambda z: mp.mpc(rnd(mp.re(z), F), rnd(mp.im(z), F))
    A = [[r(v) for v in row] for row in T.A]
    levels = [(m, p, [r(v) for v in b]) for m, p, b in T.levels]
    return Tableau(T.name, A, levels, T.projected, T.cost)

def int_bits(T):
    parts = [abs(mp.re(v)) for row in T.A for v in row] + [abs(mp.im(v)) for row in T.A for v in row]
    parts += [abs(mp.re(v)) for _, _, b in T.levels for v in b] + [abs(mp.im(v)) for _, _, b in T.levels for v in b]
    m = max(parts)
    return max(0, math.ceil(math.log2(float(m)))) if m >= 1 else 0, float(m)

if __name__ == "__main__":
    mp.mp.dps = 50
    Fs = [int(a) for a in sys.argv[1:]] or [16, 24, 32]
    TT = load_refined(); TT.update(real_methods())
    out = {}
    for name in NAMES:
        T = TT[name]
        E0 = T.E(0)
        ib, m = int_bits(T)
        out[name] = dict(int_bits=ib, max_part=m, E=float(E0), F={})
        line = f"{name:11s} int.bits {ib}  "
        for F in Fs:
            R = rounded(T, F)
            res = [float(max(abs(v) for v in lev)) for lev in R.residuals()]
            dE = float(abs(R.E(0) - E0) / E0)
            out[name]["F"][str(F)] = dict(resid=res, dE=dE)
            line += f"| F={F}: resid {', '.join(f'{v:.1e}' for v in res)}  dE {dE:.1e} "
        print(line, flush=True)
    json.dump(out, open(os.path.join(DATA, "fixedpoint.json"), "w"), indent=1)
