"""Figures of orders seven and eight for the CRK 5-8 paper, without the complex Cash--Karp pair CCK7(6).

Uses the code and data of the order 7-8 paper read-only and writes ../paper/figures/fig78_pareto,
fig78_measured, fig78_wp_adaptive and fig78_stability (.pdf/.png).  The panels of the front and constant
figures are labelled (c) and (d), below panels (a) and (b) of figs56.py.

usage: python figs78.py      (from this folder)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OLD = os.path.abspath(os.path.join(ROOT, "..", "order78", "code"))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, OLD)

import figures as F  # noqa: E402
import figures_cost as FC  # noqa: E402
import fig_measured as FM  # noqa: E402
import make_tables as MT  # noqa: E402
import matplotlib  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, NullFormatter  # noqa: E402

plt = F.plt
CCK = ("CCK7(6)", "CCK7-casc")
for lst in (MT.ORDER7, FC.ADAPT7, FC.FIG7):
    lst[:] = [m for m in lst if m not in CCK]


def save(fig, name):
    import names
    names.rename_figure(fig)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


F.save = FC.save = save
F.FIG = FIG


def fig_pareto():
    TI = F.tableau_EM()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.3))
    fkw = dict(color="k", lw=1.1, drawstyle="steps-post", label=r"$E_{\min}(B)$")
    panels = [(ax[0], "CRK7", ["Emin_77_5.json", "Emin_77_5_polish.json"],
               ["CRK7-opt", "CRK7-s", "Butcher7", "Verner7(6)", "Vern7", "Fehlberg7"], "(c) CRK7, $E_8$"),
              (ax[1], "CRK8", ["Emin_98_5.json", "Emin_98_5_polish.json"],
               ["CRK8-opt", "CRK8-s", "CooperVerner8", "DP8(7)", "DOP853", "Fehlberg8", "Verner8(7)", "Vern8"],
               "(d) CRK8, $E_9$")]
    for a, fam, files, marks, title in panels:
        E, M = [], []
        for fn in files:
            for r in json.load(open(os.path.join(F.DATA, fn)))["results"]:
                if r.get("E") and r.get("resid", 0) < 1e-10:
                    E.append(r["E"]); M.append(r.get("maxcoef_n", r["maxcoef"]))
        a.axvline(5, color="#777777", lw=0.6, ls=":", label="$B=5$")
        a.scatter(M, E, s=4, color="#bbbbbb", lw=0, label="local minima, $B=5$")
        a.plot(*F.front_E(fam, files), **fkw)
        for k in marks:
            if k in TI:
                a.plot(TI[k][1], TI[k][0], F.ST[k][1], color=F.ST[k][0], ms=5, mec="k", mew=0.4, label=F.lbl(k))
        a.set_title(title)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"$\max|$coefficient$|$")
        a.set_xlim(0.3, 300)
        a.xaxis.set_minor_formatter(NullFormatter())
        a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100, 200]))
        a.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.0, handletextpad=0.3,
                 columnspacing=0.8, handlelength=1.6)
    ax[0].set_ylabel(r"$E_{p+1}=\|e\|_2$")
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig78_pareto")


def fig_measured():
    S = MT.measured()
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.6))
    FM.panel(ax[0], S, MT.ORDER7, "CRK7-opt", "(c) order seven: constant relative to CRK7-opt")
    FM.panel(ax[1], S, MT.ORDER8, "CRK8-opt", "(d) order eight: constant relative to CRK8-opt")
    fig.tight_layout()
    save(fig, "fig78_measured")


def fig_stability():
    """F.fig_stability without CCK7(6): the stability data are filtered before the plot."""
    load = json.load
    json.load = lambda f: {k: v for k, v in load(f).items() if k not in CCK}
    try:
        F.fig_stability()
    finally:
        json.load = load
    for ext in ("pdf", "png"):
        os.replace(os.path.join(FIG, "fig_stability." + ext), os.path.join(FIG, "fig78_stability." + ext))


def fig_wp_adaptive():
    FC.fig_adaptive()
    for ext in ("pdf", "png"):
        os.replace(os.path.join(FIG, "fig_wp_adaptive." + ext), os.path.join(FIG, "fig78_wp_adaptive." + ext))
        os.remove(os.path.join(FIG, "fig_tolerance." + ext))


if __name__ == "__main__":
    fig_pareto()
    fig_measured()
    fig_stability()
    fig_wp_adaptive()
