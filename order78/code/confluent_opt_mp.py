"""Newton refinement of ../data/confluent_L_opt.json in 90-digit arithmetic (as confluent_mp.py);
writes ../data/confluent_L_opt_mp.json (45 digits).   python confluent_opt_mp.py"""
import json, os, sys
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import crk
from confluent_mp import NAMES, pack, unpack, refine, residual, weights, bphi
mp.mp.dps = 90
d = json.load(open(os.path.join(HERE, "..", "data", "confluent_L_opt.json"), encoding="utf-8"))
v = {k: mp.mpc(mp.mpf(a), mp.mpf(b)) for k, (a, b) in d["coef"].items()}
x, rmax, S = refine(pack(v), iters=10)
r = max(abs(t) for t in residual(x))
rank = sum(1 for s in S if s > mp.mpf(10) ** -20 * S[0])
v = unpack(x); ph = weights(v, 9)
e9 = [(mp.re(bphi(v, ph[t])) - mp.mpf(1) / crk.gamma(t)) / crk.sigma(t) for t in crk.TREES[9]]
E9 = mp.sqrt(mp.fsum(e ** 2 for e in e9)); M = max(abs(v[k]) for k in v)
lam = 2.041 / float(E9) ** 0.125 - 7
print(f"L_opt: max|r| = {mp.nstr(r, 3)}, rank {rank} of {len(x)}, E_9 = {mp.nstr(E9, 5)}, max|coef| = {mp.nstr(M, 5)}, "
      f"lambda* = {lam:.3f}, eff(lambda=1) = {8 * float(E9) ** 0.125:.4f}, beta_G = {mp.nstr(mp.re(v['β_G']), 4)}")
json.dump(dict(resid=float(r), rank=rank, E9=float(E9), maxcoef=float(M), lambda_star=lam,
               coef={k: [mp.nstr(mp.re(v[k]), 45), mp.nstr(mp.im(v[k]), 45)] for k in NAMES + ["β_1", "β_G"]}),
          open(os.path.join(HERE, "..", "data", "confluent_L_opt_mp.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
