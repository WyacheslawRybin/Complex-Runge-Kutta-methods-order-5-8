"""Sensitivity of the variable-step comparison to the step-size controller (revision R1, item R2-Q4).

The runs of adaptive.py use the elementary controller h_new = h min(5, max(0.2, 0.9 err^(-1/k))), k = q + 1.
Here every pair is rerun with the PI controller of Gustafsson (HNW I, Sect. II.4 and IV.2):
h_new = h min(5, max(0.2, 0.9 err_n^(-alpha) err_(n-1)^beta)), alpha = 0.7/k, beta = 0.4/k, with the
elementary formula after a rejection; everything else as in adaptive.py.  Then the cost of the best complex
pair is divided by that of the best real pair of the same order at the accuracies 1e-6, 1e-8, 1e-10, for
both controllers.

Usage: python pi_check.py [78|table]   writes ../data/pi_check_<group>.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BASE56 = os.path.join(HERE, "..", "..", "order56", "code", "base", "code")
GROUPS = {
    "78": (HERE, {7: (["CRK7(5)", "CRK7-s(5)"], ["Verner7(6)", "Vern7"]),
                  8: (["CRK8(6)", "CRK8-s(6)"], ["DP8(7)", "Verner8(7)", "DOP853", "Vern8"])}),
}
EPS = (1e-6, 1e-8, 1e-10)


def solve_pi(pair, f, y0, T, tol, max_steps=2_000_000):
    A, b, bh, q, proj, fsal = pair["A"], pair["b"], pair["bh"], pair["q"], pair["projected"], pair["fsal"]
    s = len(b); c = A.sum(1); p = pair["p"]; k = q + 1
    al, be = 0.7 / k, 0.4 / k
    y = np.array(y0, float); t = 0.0
    sc = lambda u, v: tol + tol * np.maximum(np.abs(u), np.abs(v))
    nf = 0
    f0 = np.real(f(t, y)) * 1.0; nf += 1
    d0 = np.sqrt(np.mean((y / sc(y, y)) ** 2)); d1 = np.sqrt(np.mean((f0 / sc(y, y)) ** 2))
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    f1 = np.real(f(t + h0, y + h0 * f0)); nf += 1
    d2 = np.sqrt(np.mean(((f1 - f0) / sc(y, y)) ** 2)) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1.0 / (p + 1))
    h = min(100 * h0, h1, T)
    k1 = f0; nacc = nrej = 0; facmax = 5.0; errold = 1.0
    dtype = complex if proj else float
    while t < T * (1 - 1e-14):
        if t + h > T:
            h = T - t
        K = np.zeros((s, len(y)), dtype)
        K[0] = k1
        for i in range(1, s):
            K[i] = f(t + c[i] * h, y + h * (A[i, :i] @ K[:i])); nf += 1
        ynew = y + h * (b @ K); yhat = y + h * (bh @ K)
        if proj:
            ynew = ynew.real; yhat = yhat.real
        if pair.get("dop853"):
            sk = sc(y, ynew)
            e5 = np.mean(((ynew - yhat) / sk) ** 2)
            e3 = np.mean((h * ((b - pair["bh3"]) @ K) / sk) ** 2)
            err = e5 / np.sqrt(e5 + 0.01 * e3) if e5 > 0 else 0.0
        else:
            err = np.sqrt(np.mean(((ynew - yhat) / sc(y, ynew)) ** 2))
        if not np.isfinite(err):
            return None
        if err <= 1.0:
            t += h; y = ynew; nacc += 1
            k1 = K[-1].real if fsal else np.real(f(t, y)) * 1.0
            if not fsal:
                nf += 1
            e = max(err, 1e-10)
            h *= min(facmax, max(0.2, 0.9 * e ** (-al) * errold ** be))
            errold = e; facmax = 5.0
        else:
            nrej += 1
            h *= max(0.2, 0.9 * err ** (-1.0 / k)); facmax = 1.0
        if nacc + nrej > max_steps or h < 1e-14 * T:
            return None
    if not fsal:
        nf -= 1
    return y, nf, nacc, nrej


_G = None


def _init(group):
    global _G
    path = GROUPS[group][0]
    sys.path.insert(0, path)
    import adaptive as AD
    from problems import P, references
    pairs = AD.load_pairs()
    ref = {k: np.array([float(v) for v in vals]) for k, vals in references().items()}
    _G = (AD, P, pairs, ref)


def task(args):
    name, prob = args
    AD, P, pairs, ref = _G
    pr = P[prob]
    rows = []
    for tol in AD.TOLS:
        out = solve_pi(pairs[name], pr["f_np"], [float(v) for v in pr["y0"]], float(pr["T"]), tol)
        if out is None:
            rows.append(dict(tol=tol, err=None)); continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=float(np.linalg.norm(y - ref[prob])), nfev=nf, nacc=na, nrej=nr))
    return name, prob, rows


def run(group, nproc=6):
    path, orders = GROUPS[group]
    sys.path.insert(0, path)
    from problems import ADAPTIVE
    names = [n for c, r in orders.values() for n in c + r]
    res = {}
    with Pool(nproc, initializer=_init, initargs=(group,)) as pool:
        for name, prob, rows in pool.imap_unordered(task, [(n, p) for n in names for p in ADAPTIVE]):
            res.setdefault(name, {})[prob] = rows
            print(name, prob, flush=True)
    json.dump(res, open(os.path.join(DATA, f"pi_check_{group}.json"), "w"), indent=1)


def _cost(rows, eps):
    from figures_cost import cost_at
    ok = [r for r in rows if r["err"] is not None and r["err"] > 0]
    return cost_at(np.array([r["nfev"] for r in ok], float), np.array([r["err"] for r in ok]), eps)


def table():
    sys.path.insert(0, HERE)
    out = {}
    for group, (path, orders) in GROUPS.items():
        pi = json.load(open(os.path.join(DATA, f"pi_check_{group}.json")))
        el_file = (os.path.join(HERE, "..", "..", "order56", "code", "base", "data", "adaptive.json") if group == "56"
                   else os.path.join(DATA, "adaptive.json"))
        el = json.load(open(el_file))
        for order, (cplx, real) in sorted(orders.items()):
            for eps in EPS:
                line = []
                for lab, src in (("elementary", el), ("PI", pi)):
                    probs = [p for p in pi[cplx[0]]]
                    rat = []
                    for p in probs:
                        cc = min(_cost(src[m][p], eps) for m in cplx if m in src and p in src[m])
                        cr = min(_cost(src[m][p], eps) for m in real if m in src and p in src[m])
                        if np.isfinite(cc) and np.isfinite(cr):
                            rat.append(cc / cr)
                    gm = float(np.exp(np.mean(np.log(rat))))
                    wins = int(sum(r < 1 for r in rat))
                    out.setdefault(f"{order}|{eps:g}", {})[lab] = dict(geomean=gm, wins=wins, n=len(rat))
                    line.append(f"{lab}: gm {gm:.2f} wins {wins}/{len(rat)}")
                print(f"order {order} eps {eps:g}  " + "   ".join(line))
    json.dump(out, open(os.path.join(DATA, "pi_check.json"), "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "table"
    if what == "table":
        table()
    else:
        run(what)
