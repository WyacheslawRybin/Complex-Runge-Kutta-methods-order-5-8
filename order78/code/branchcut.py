"""Distance of the complex stage values to the branch cut of (q1^2 + q2^2)^{3/2} (review item C20).

The Kepler and Arenstorf right-hand sides contain (x^2 + y^2)^{3/2} (Arenstorf: about the two
centers), evaluated with numpy's principal branch.  For a stage value Q = X + iH (X, H real
2-vectors) and a center c, r^2 = |X - c|^2 - |H|^2 + 2i (X - c).H.  The branch cut is
r^2 in (-inf, 0]; its distance from r^2 is

    d = |r^2|           if Re r^2 >= 0,
    d = |Im r^2|        if Re r^2 <  0,

and |arg r^2| < pi means that the cut is not crossed.  Along the segment X + i tau H, 0 <= tau <= 1,
Im r^2 = 2 tau (X - c).H keeps its sign, so the segment stays in one closed half-plane and
reaches the cut only if r^2 itself lies on it: |arg r^2| < pi at every stage therefore implies that
the principal branch is the analytic continuation of the real right-hand side.  As in the order 5-6 check (singular_r1.py
of RBKB26) we also record rho = |H| / |X - c|; rho < 1 implies Re r^2 > 0, i.e. |arg r^2| < pi/2.

Runs (all stages i >= 2; the first stage is real):
  * variable step: the pairs of adaptive.py with the same controller, starting step and
    tolerances 1e-3 ... 1e-13, all attempted steps including rejected ones (also recorded for the accepted steps alone);
  * fixed step: the methods of measure.py on Kepler with the coarsest step counts of the
    fixed-step experiments (N = 50, 100, 200, 400) and on Arenstorf (not part of the fixed-step
    experiments) with N = 2000, 4000, 8000, 16000.
For every run: min d, min d / |r^2|_real (relative to the squared distance of the real point,
i.e. of the real part of the stage, to the center), max |arg r^2|, max rho.  Real baseline: the
minimum of |X - c|^2 over the same stages.  Double precision.

Usage (from code/):  python branchcut.py      -> ../data/branchcut.json and a printed summary
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
import adaptive as AD
from problems import P, references

MU = 0.012277471
CENTERS = {"Kepler": [(0.0, 0.0)], "Arenstorf": [(-MU, 0.0), (1 - MU, 0.0)]}
PAIRS = ["CRK7(5)", "CRK7-s(5)", "CCK7(6)", "CRK8(6)", "CRK8-s(6)"]
FIXED = ["CRK7-opt", "CRK7-s", "CCK7(6)", "CRK8-opt", "CRK8-s"]
NFIX = {"Kepler": [50, 100, 200, 400], "Arenstorf": [2000, 4000, 8000, 16000]}


class Track:
    def __init__(self):
        self.d = np.inf; self.drel = np.inf; self.arg = 0.0; self.rho = 0.0; self.r2real = np.inf; self.n = 0

    def add(self, Y, centers):
        X, H = Y[:2].real, Y[:2].imag
        for c in centers:
            Xc = X - np.array(c)
            z = Xc @ Xc - H @ H + 2j * (Xc @ H)
            d = abs(z) if z.real >= 0 else abs(z.imag)
            rr = Xc @ Xc
            self.d = min(self.d, d); self.drel = min(self.drel, d / rr); self.r2real = min(self.r2real, rr)
            self.arg = max(self.arg, abs(np.angle(z))); self.rho = max(self.rho, np.sqrt(H @ H / rr))
        self.n += 1

    def merge(self, o):
        self.d = min(self.d, o.d); self.drel = min(self.drel, o.drel); self.r2real = min(self.r2real, o.r2real)
        self.arg = max(self.arg, o.arg); self.rho = max(self.rho, o.rho); self.n += o.n

    def as_dict(self):
        return dict(min_dist=float(self.d), min_dist_rel=float(self.drel), max_abs_arg=float(self.arg),
                    max_rho=float(self.rho), min_r2_real=float(self.r2real), stages=self.n)


def solve_track(pair, f, y0, T, tol, centers, max_steps=2_000_000):
    """adaptive.solve (projected pairs, non-FSAL) with the stage values recorded."""
    A, b, bh, q = pair["A"], pair["b"], pair["bh"], pair["q"]
    assert pair["projected"] and not pair["fsal"]
    s = len(b); c = A.sum(1); p = pair["p"]
    y = np.array(y0, float); t = 0.0
    sc = lambda u, v: tol + tol * np.maximum(np.abs(u), np.abs(v))
    f0 = np.real(f(t, y)) * 1.0
    d0 = np.sqrt(np.mean((y / sc(y, y)) ** 2)); d1 = np.sqrt(np.mean((f0 / sc(y, y)) ** 2))
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    f1 = np.real(f(t + h0, y + h0 * f0))
    d2 = np.sqrt(np.mean(((f1 - f0) / sc(y, y)) ** 2)) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1.0 / (p + 1))
    h = min(100 * h0, h1, T)
    k1 = f0; nacc = nrej = 0; facmax = 5.0
    tr = Track(); tra = Track()
    while t < T * (1 - 1e-14):
        if t + h > T:
            h = T - t
        K = np.zeros((s, len(y)), complex)
        K[0] = k1; st = Track()
        for i in range(1, s):
            Yi = y + h * (A[i, :i] @ K[:i])
            st.add(Yi, centers)
            K[i] = f(t + c[i] * h, Yi)
        tr.merge(st)
        ynew = (y + h * (b @ K)).real; yhat = (y + h * (bh @ K)).real
        err = np.sqrt(np.mean(((ynew - yhat) / sc(y, ynew)) ** 2))
        if not np.isfinite(err):
            return None
        if err <= 1.0:
            tra.merge(st)
            t += h; y = ynew; nacc += 1
            k1 = np.real(f(t, y)) * 1.0
            h *= min(facmax, max(0.2, 0.9 * err ** (-1.0 / (q + 1)) if err > 0 else facmax)); facmax = 5.0
        else:
            nrej += 1
            h *= max(0.2, 0.9 * err ** (-1.0 / (q + 1))); facmax = 1.0
        if nacc + nrej > max_steps or h < 1e-14 * T:
            return None
    return y, tr, nacc, nrej, tra


def fixed_track(A, b, f, y0, T, N, centers):
    s = len(b); c = A.sum(1); h = T / N
    y = np.array(y0, float); t = 0.0; tr = Track()
    for n in range(N):
        K = np.zeros((s, len(y)), complex)
        K[0] = f(t, y)
        for i in range(1, s):
            Yi = y + h * (A[i, :i] @ K[:i])
            tr.add(Yi, centers)
            K[i] = f(t + c[i] * h, Yi)
        y = (y + h * (b @ K)).real; t += h
    return y, tr


def main():
    import mpmath as mp
    from tableaux import load_refined
    mp.mp.dps = 40
    REF = {k: np.array([float(v) for v in vals]) for k, vals in references().items()}
    pairs = AD.load_pairs(PAIRS)
    TT = load_refined()
    out = {"adaptive": {}, "fixed": {}}
    np.seterr(all="ignore")
    for prob in ("Kepler", "Arenstorf"):
        pr = P[prob]; T = float(pr["T"]); y0 = [float(v) for v in pr["y0"]]; cen = CENTERS[prob]
        for name in PAIRS:
            rows = []; tot = Track(); tota = Track()
            for tol in AD.TOLS:
                r = solve_track(pairs[name], pr["f_np"], y0, T, tol, cen)
                if r is None:
                    rows.append(dict(tol=tol, failed=True)); continue
                y, tr, na, nr, tra = r
                tot.merge(tr); tota.merge(tra)
                rows.append(dict(tol=tol, err=float(np.linalg.norm(y - REF[prob])), nacc=na, nrej=nr, **tr.as_dict(),
                                 accepted=tra.as_dict()))
            out["adaptive"].setdefault(prob, {})[name] = dict(all=tot.as_dict(), accepted=tota.as_dict(), rows=rows)
            nf = sum(1 for x in rows if x.get("failed"))
            print(f"adapt {prob:9s} {name:10s} min d={tot.d:.3e} min d/|X-c|^2={tot.drel:.3f} "
                  f"max|arg|={tot.arg:.4f} (pi-max={np.pi - tot.arg:.3f}) max rho={tot.rho:.3f} "
                  f"min|X-c|^2={tot.r2real:.3e} failed={nf}; accepted: max|arg|={tota.arg:.4f} max rho={tota.rho:.3f}", flush=True)
        for name in FIXED:
            Tb = TT[name]; s = Tb.s
            A = np.array([[complex(Tb.A[i][j]) for j in range(s)] for i in range(s)])
            b = np.array([complex(v) for v in Tb.levels[0][2]])
            tot = Track(); rows = []
            for N in NFIX[prob]:
                y, tr = fixed_track(A, b, pr["f_np"], y0, T, N, cen)
                tot.merge(tr)
                rows.append(dict(N=N, err=float(np.linalg.norm(y - REF[prob])), **tr.as_dict()))
            out["fixed"].setdefault(prob, {})[name] = dict(all=tot.as_dict(), rows=rows)
            print(f"fixed {prob:9s} {name:10s} min d={tot.d:.3e} min d/|X-c|^2={tot.drel:.3f} "
                  f"max|arg|={tot.arg:.4f} (pi-max={np.pi - tot.arg:.3f}) max rho={tot.rho:.3f} "
                  f"min|X-c|^2={tot.r2real:.3e} errs={['%.1e' % x['err'] for x in rows]}", flush=True)
    json.dump(out, open(os.path.join(DATA, "branchcut.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
