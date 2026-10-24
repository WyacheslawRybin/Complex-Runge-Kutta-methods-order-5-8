"""Large test problem for referee comment M3: the Lorenz-96 system with d = 1000,
    x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F,   F = 8,   indices mod d,
    x_i(0) = F + sin(i), i = 1..d,   t in [0, 1].
The right-hand side is a quadratic polynomial (holomorphic, no branch cuts), chaotic and nonstiff.

  python lorenz96.py ref            reference solution by a Taylor-series integrator in fixed-point integer
                                    arithmetic (256 fractional bits), two step sizes for a self-check
                                    -> ../data/l96_reference.json
  python lorenz96.py adaptive [NPROC] [PAIR,...]
                                    variable-step runs of the pairs of adaptive.py (same controller, 21
                                    tolerances 1e-3..1e-13), error in the RMS norm -> ../data/adaptive_l96.json
  python lorenz96.py summary        evaluations needed for the RMS accuracies 1e-6, 1e-8, 1e-10"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
D, FORCE, TEND = 1000, 8, 1.0

def y0_float():
    return FORCE + np.sin(np.arange(1, D + 1, dtype=float))

def f_np(t, y):
    return (np.roll(y, -1) - np.roll(y, 2)) * np.roll(y, 1) - y + FORCE

# ---------------------------------------------------------------- reference (fixed-point Taylor series)
SHIFT = 256
ONE = 1 << SHIFT

def _y0_fixed():
    import mpmath as mp
    with mp.workdps(90):
        return np.array([int(mp.nint((FORCE + mp.sin(i)) * mp.mpf(2) ** SHIFT)) for i in range(1, D + 1)], dtype=object)

def taylor_step(x, h_num, h_den, nterms):
    """One Taylor step of length h = h_num/h_den; x: object array of fixed-point ints."""
    X = [x]
    for k in range(nterms):
        acc = np.zeros(D, dtype=object)
        for j in range(k + 1):
            a = X[j]; b = X[k - j]
            acc = acc + (np.roll(a, -1) - np.roll(a, 2)) * np.roll(b, 1)
        acc = acc >> SHIFT                       # product of two fixed-point numbers
        acc = acc - X[k]
        if k == 0:
            acc = acc + FORCE * ONE
        X.append(acc // (k + 1))
    # Horner: sum_k X_k h^k
    y = X[-1]
    for k in range(nterms - 1, -1, -1):
        y = (y * h_num) // h_den + X[k]
    return y

def reference(n_steps, nterms):
    x = _y0_fixed()
    t0 = time.time()
    for n in range(n_steps):
        x = taylor_step(x, 1, n_steps, nterms)
    print(f"  {n_steps} steps, {nterms} terms: {time.time() - t0:.0f}s", flush=True)
    return x

def cmd_ref():
    import mpmath as mp
    x1 = reference(200, 40)
    x2 = reference(400, 48)
    diff = max(abs(int(a) - int(b)) for a, b in zip(x1, x2)) / ONE
    print(f"self-consistency max|x(h) - x(h/2)| = {diff:.2e}", flush=True)
    with mp.workdps(60):
        vals = [mp.nstr(mp.mpf(int(v)) / mp.mpf(2) ** SHIFT, 40) for v in x2]
    json.dump(dict(d=D, F=FORCE, T=TEND, y0="F + sin(i)", check=f"{diff:.2e}", ref=vals),
              open(os.path.join(DATA, "l96_reference.json"), "w"), indent=0)

# ---------------------------------------------------------------- variable step
_P = _REF = None
def _init(names):
    global _P, _REF
    sys.path.insert(0, HERE)
    import adaptive
    _P = adaptive.load_pairs(names)
    _REF = np.array([float(v) for v in json.load(open(os.path.join(DATA, "l96_reference.json")))["ref"]])

def task(name):
    import adaptive
    rows = []
    for tol in adaptive.TOLS:
        t0 = time.time()
        out = adaptive.solve(_P[name], f_np, y0_float(), TEND, tol)
        if out is None:
            rows.append(dict(tol=tol, err=None)); continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=float(np.sqrt(np.mean((y - _REF) ** 2))), nfev=nf, nacc=na, nrej=nr,
                         wall=time.time() - t0))
    return name, rows

def cmd_adaptive(nproc, names):
    sys.path.insert(0, HERE)
    import adaptive
    names = names or [n for n, _, _ in adaptive.PAIRS]
    fn = os.path.join(DATA, "adaptive_l96.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    with Pool(nproc, initializer=_init, initargs=(names,)) as pool:
        for name, rows in pool.imap_unordered(task, names):
            res[name] = rows
            ok = [r for r in rows if r["err"] is not None]
            print(f"{name:11s} tol 1e-13: err={ok[-1]['err']:.2e} nfev={ok[-1]['nfev']}" if ok else f"{name} FAILED",
                  flush=True)
            json.dump(res, open(fn, "w"), indent=1)

def cost_at(cost, err, eps):
    """As make_tables.cost_at: log-log interpolation along the lower envelope."""
    o = np.argsort(cost)
    c, e = cost[o], np.minimum.accumulate(err[o])
    if eps < e[-1]:
        return np.inf
    if eps >= e[0]:
        return c[0]
    k = int(np.argmax(e <= eps))
    y0, y1 = np.log(e[k - 1]), np.log(e[k])
    if y1 == y0:
        return c[k]
    x0, x1 = np.log(c[k - 1]), np.log(c[k])
    return float(np.exp(x0 + (np.log(eps) - y0) * (x1 - x0) / (y1 - y0)))

def costs(eps_list=(1e-6, 1e-8, 1e-10)):
    res = json.load(open(os.path.join(DATA, "adaptive_l96.json")))
    out = {}
    for name, rows in res.items():
        ok = [r for r in rows if r["err"] is not None and r["err"] > 0]
        n = np.array([r["nfev"] for r in ok], float); e = np.array([r["err"] for r in ok])
        out[name] = {f"{eps:.0e}": cost_at(n, e, eps) for eps in eps_list}
    return out

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "ref":
        cmd_ref()
    elif cmd == "adaptive":
        cmd_adaptive(int(sys.argv[2]) if len(sys.argv) > 2 else 11, sys.argv[3].split(",") if len(sys.argv) > 3 else None)
    else:
        for name, c in costs().items():
            print(f"{name:11s} " + "  ".join(f"{k}: {v:8.0f}" for k, v in c.items()))
