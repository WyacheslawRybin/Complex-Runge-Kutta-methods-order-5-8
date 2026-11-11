"""Numbers quoted in the text for orders five and six that change when the complex Cash--Karp pairs are
removed (the order 5-6 paper quoted them with CCK6 included).  Read-only on the data of the order 5-6 paper;
prints the numbers and writes ../data/facts56.json.

usage: python facts56.py      (from this folder, PYTHONIOENCODING=utf-8)
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OLD = os.path.abspath(os.path.join(ROOT, "..", "order56", "code", "r1", "code"))
sys.path.insert(0, OLD)

import r1common as R  # noqa: E402
import r1patch  # noqa: E402

r1patch.build = lambda: r1patch.MERGED
MT = r1patch.apply()
import figures_cost as FC  # noqa: E402
from figures_cost import adapt_curve, cost_at, fixed_curve  # noqa: E402
from results import classify  # noqa: E402

OUT = {}
MERGED = r1patch.MERGED
meas = json.load(open(os.path.join(MERGED, "measured.json")))
ad = json.load(open(os.path.join(MERGED, "adaptive.json")))
tab = json.load(open(os.path.join(MERGED, "tableaux.json")))
EV = FC.EVALS
CRK5 = ["CRK5-opt", "CRK5-s", "GJM5"]
CRK6 = ["CRK6-opt", "CRK6-s"]
REAL5 = ["DP5", "CK5", "Tsit5(4)", "BS5(4)"]
REAL6 = ["Butcher6", "RK6-opt", "DP6(5)", "Verner6(5)", "Vern6(5)"]
CCK_LEVELS = ["CCK6(5)/5", "CCK6-casc/5", "CCK6(5)", "CCK6(5)-s", "CCK6-casc"]
NONLIN = [p for p in MT.PROBS if p not in ("Dahlquist", "pendulum")]


def say(key, val):
    OUT[key] = val
    print(f"{key}: {val}")


def order_counts(methods):
    n_all = n_ok = n01 = 0
    worst = 0.0
    for m in methods:
        p = MT.ORDER_OF(m.split("/")[0]) if "/" not in m else 5
        for q in MT.PROBS:
            if q not in meas.get(m, {}):
                continue
            v = classify(meas[m][q]["rows"], p)
            n_all += 1
            if v["cls"] == "deg":
                continue
            n_ok += 1
            d = abs(v["order"] - p)
            n01 += d <= 0.1
            worst = max(worst, d)
    return n_all, n_ok, n01


def S(m, q):
    p = 6 if m in CRK6 + REAL6 else 5
    return classify(meas[m][q]["rows"], p)


def main():
    # empirical orders (Section "Order"): check against the published 266 of 294 with CCK, then without
    print("measured.json methods:", sorted(meas))
    with_cck = CRK5 + REAL5 + CRK6 + REAL6 + [m for m in CCK_LEVELS if m in meas]
    say("order_counts_with_cck(all, nondeg, within0.1)", order_counts(with_cck))
    say("order_counts_crk(all, nondeg, within0.1)", order_counts(CRK5 + REAL5 + CRK6 + REAL6))
    devs = []
    for m in CRK5 + REAL5 + CRK6 + REAL6:
        p = 6 if m in CRK6 + REAL6 else 5
        for q in MT.PROBS:
            v = classify(meas[m][q]["rows"], p)
            if v["cls"] != "deg" and abs(v["order"] - p) > 0.1:
                devs.append((m, q, round(v["order"] - p, 3)))
    say("order_dev_gt_0.1", devs)

    # E7 ranking of the CRK6 tableaux on the eleven nonlinear problems (pairs of tableaux x problems)
    for fam, p in ((CRK5, 5), (CRK6, 6)):
        agree = tot = 0
        for a, b in itertools.combinations(fam, 2):
            for q in NONLIN:
                va, vb = S(a, q), S(b, q)
                if va["cls"] == "deg" or vb["cls"] == "deg":
                    continue
                tot += 1
                agree += (tab[a]["E"][0] < tab[b]["E"][0]) == (va["C"] < vb["C"])
        say(f"E_rank_agreement_{fam[0][:4]}", (agree, tot, round(agree / tot, 3)))
    wins = {}
    for q in NONLIN:
        vals = {m: S(m, q)["C"] for m in CRK6 if S(m, q)["cls"] != "deg"}
        w = min(vals, key=vals.get)
        wins[w] = wins.get(w, 0) + 1
    say("most_accurate_CRK6_tableau_counts", wins)
    # Vern6(5) against the best CRK6 tableau
    rat = {}
    for q in NONLIN:
        vv = S("Vern6(5)", q)
        if q == "nonautonomous":
            continue
        best = min(S(m, q)["C"] for m in CRK6 if S(m, q)["cls"] != "deg")
        rat[q] = best / vv["C"]
    say("Vern6_vs_best_CRK6_ratio_range", (round(min(rat.values()), 1), round(max(rat.values()), 1)))

    # fixed step at 1e-8: CRK6-opt against Vern6(5) on the ten nonlinear problems other than y' = -y^2
    fx = {}
    for q in [q for q in NONLIN if q != "quadratic"]:
        c = cost_at(EV["CRK6-opt"] * fixed_curve(meas, "CRK6-opt", q)[0], fixed_curve(meas, "CRK6-opt", q)[1], 1e-8)
        v = cost_at(EV["Vern6(5)"] * fixed_curve(meas, "Vern6(5)", q)[0], fixed_curve(meas, "Vern6(5)", q)[1], 1e-8)
        fx[q] = c / v
    say("fixed_1e-8_CRK6opt_over_Vern6", {k: round(v, 3) for k, v in fx.items()})

    # performance profile at omega = 1.25 (within 25% of the cheapest pair of the group)
    for eps in (1e-6, 1e-8, 1e-10):
        for grp in (FC.ADAPT5, [m for m in FC.ADAPT6 if not m.startswith("CCK")]):
            cnt = {m: 0 for m in grp}
            for q in FC.PREF_ADAPT:
                c = {m: cost_at(*adapt_curve(ad, m, q)[:2], eps) for m in grp}
                b = min(c.values())
                for m in grp:
                    cnt[m] += c[m] <= 1.25 * b
            say(f"profile_1.25_{eps:g}_{grp[0]}", cnt)
    # variable step: CRK6(4) against the best real order-six pair, and best order-6 vs best 5(4) pair
    A6 = [m for m in FC.ADAPT6 if not m.startswith("CCK")]
    for eps in (1e-6, 1e-8, 1e-10):
        r6, cr = {}, {}
        for q in FC.PREF_ADAPT:
            c = {m: cost_at(*adapt_curve(ad, m, q)[:2], eps) for m in FC.ADAPT5 + A6}
            br = min(c[m] for m in A6 if m != "CRK6(4)")
            r6[q] = c["CRK6(4)"] / br
            b5, b6 = min(c[m] for m in FC.ADAPT5), min(c[m] for m in A6)
            cr[q] = b6 / b5 if math.isfinite(b5) and math.isfinite(b6) else None
        say(f"CRK6(4)_over_best_real_{eps:g}", {k: round(v, 3) for k, v in r6.items()})
        say(f"best6_over_best5_{eps:g}", {k: (None if v is None else round(v, 3)) for k, v in cr.items()})

    # CPU: fixed-step break-even (time ratio x constant ratio) with the complex methods CRK5-opt, CRK6-opt only
    import tables_cpu_r1 as TC
    TC.COLS[:] = [(c, r) for c, r in TC.COLS if not c.startswith("CCK")]
    B = TC.bench(os.path.join(R.DATA, "bench.csv"))
    wins, allv = TC.cpu_breakeven(B)
    say("cpu_fixed_wins", [(c, r, q, round(v, 3)) for c, r, q, v in wins])
    say("cpu_fixed_cases", len(allv))
    # break-even ratios of Table kappastar exceeded by kappa_CPU? (CRK6-opt stage / RK6-opt stage)
    kap = {p: (B[(p, "CRK6-opt")]["ns"] / 6) / (B[(p, "RK6-opt")]["ns"] / 7) for p in TC.BENCH_PROBS
           if (p, "CRK6-opt") in B}
    say("kappa_cpu_CRK6opt", {k: round(v, 2) for k, v in kap.items()})
    # share of f in a step of CRK6-opt (time of six complex evaluations, the first one real)
    share = {}
    for p in TC.BENCH_PROBS:
        rows = [l.strip().split(",") for l in open(os.path.join(R.DATA, "bench.csv")) if l.startswith(p + ",CRK6-opt,")]
        if rows:
            r = rows[0]
            ns, fr, fc = float(r[4]), float(r[5]), float(r[6])
            share[p] = round((fr + 5 * fc) / ns, 3)
    say("f_share_of_CRK6opt_step", share)
    comb = {}
    for p in TC.BENCH_PROBS:
        g = lambda m: [l.strip().split(",") for l in open(os.path.join(R.DATA, "bench.csv")) if l.startswith(p + "," + m + ",")]
        a, b = g("CRK6-opt"), g("RK6-opt")
        if a and b:
            a, b = a[0], b[0]
            ca = (float(a[4]) - float(a[5]) - 5 * float(a[6])) / 6
            cb = (float(b[4]) - 7 * float(b[5])) / 7
            comb[p] = round(ca / cb, 2)
            share.setdefault(p, None)
    say("combination_cost_ratio_per_stage", comb)
    say("fc_over_fr", {p: round(float(r[6]) / float(r[5]), 2) for p in TC.BENCH_PROBS
                       for r in [l.strip().split(",") for l in open(os.path.join(R.DATA, "bench.csv"))
                                 if l.startswith(p + ",CRK6-opt,")]})

    # BRUSS-2D on a CPU: best real over best complex method (time = evaluations x time per evaluation)
    bc = json.load(open(os.path.join(R.DATA, "bruss2d_costs.json")))
    t_eval = lambda m: B[("Bruss2D", m)]["ns"] / B[("Bruss2D", m)]["evals"]
    for order, cg, rg in ((5, ["CRK5-opt"], REAL5), (6, ["CRK6-opt"], REAL6)):
        for mode in ("fixed", "adaptive"):
            rr = []
            for k in range(3):
                tc = min(bc[m][mode][k] * t_eval(m) for m in cg if bc[m][mode][k])
                tr = min(bc[m][mode][k] * t_eval(m) for m in rg if m in bc and bc[m][mode][k]
                         and (mode == "fixed" or m != "Butcher6"))
                rr.append(round(tc / tr, 2))
            say(f"bruss2d_cpu_complex_over_real_o{order}_{mode}", rr)

    # robustness of the variable-step comparison without the CCK pairs (shifted tolerance grid)
    rob = json.load(open(os.path.join(R.DATA, "robust_r1.json")))
    off = rob["offset_runs"]
    pairs = [m for m in off if not m.startswith("CCK")]
    for eps in (1e-6, 1e-8, 1e-10):
        d = []
        for q in FC.PREF_ADAPT:
            for m in pairs:
                if q not in off[m]:
                    continue
                ok = lambda rows: [r for r in rows if r["err"] is not None and r["err"] > 0]
                a = cost_at(*adapt_curve(ad, m, q)[:2], eps)
                ro = ok(off[m][q])
                b = cost_at(np.array([r["nfev"] for r in ro], float), np.array([r["err"] for r in ro]), eps)
                if math.isfinite(a) and math.isfinite(b):
                    d.append(abs(a - b) / min(a, b))
        d = np.array(d)
        say(f"robust_{eps:g}", dict(median=round(float(np.median(d)), 4), q90=round(float(np.quantile(d, 0.9)), 4),
                                     n=int(d.size), pairs=pairs))

    # latency (Table latency): CRK6(4) / best real, kappa_lat, nu*
    L = json.load(open(os.path.join(R.DATA, "latency_r1.json")))
    import latency_r1 as LR
    lat = {}
    for q in ("VanderPol", "Rossler", "Bouali", "Lorenz5", "RigidBody", "Brusselator"):
        c = cost_at(*adapt_curve(ad, "CRK6(4)", q)[:2], 1e-8)
        r = min(cost_at(*adapt_curve(ad, m, q)[:2], 1e-8) for m in A6 if m != "CRK6(4)")
        v = c / r
        k1, k4 = L[q]["kappa_lat"]["6|0.25"], L[q]["kappa_lat"]["6|1"]
        nus = LR.nu_star(q, v)
        lat[q] = (round(v, 3), round(v * k1, 3), round(v * k4, 3), None if nus is None else round(nus, 3))
    bc6 = [bc[m]["adaptive"][1] for m in ("RK6-opt", "DP6(5)", "Verner6(5)", "Vern6(5)")]
    v = bc["CRK6-opt"]["adaptive"][1] / min(bc6)
    k1, k4 = L["Bruss2D"]["kappa_lat"]["6|0.25"], L["Bruss2D"]["kappa_lat"]["6|1"]
    lat["Bruss2D"] = (round(v, 3), round(v * k1, 3), round(v * k4, 3), None)
    say("latency_rows(kappa1, nu=1/4, nu=1, nu*)", lat)
    json.dump(OUT, open(os.path.join(ROOT, "data", "facts56.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
