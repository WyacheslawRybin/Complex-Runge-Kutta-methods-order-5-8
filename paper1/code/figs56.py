"""Figures of orders five and six for the CRK 5-8 paper, without the complex Cash--Karp pairs.

Uses the code and data of the order 5-6 paper read-only (as tables56.py) and writes
../paper/figures/fig56_pareto, fig56_measured and fig56_wp_adaptive (.pdf/.png): panels (a) and (b) of the
combined figures of the paper; the panels of orders seven and eight come from figs78.py.

usage: python figs56.py      (from this folder)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OLD = os.path.abspath(os.path.join(ROOT, "..", "order56", "code", "r1", "code"))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, OLD)

import r1common as R  # noqa: E402

R.FIG = FIG
import r1patch  # noqa: E402

r1patch.build = lambda: r1patch.MERGED
MT = r1patch.apply()
import figures as F  # noqa: E402
import figures_cost as FC  # noqa: E402
import figures_r1 as FR  # noqa: E402  (replaces F.front_E by the version with the B = 100 minima)
import fig_measured_r1 as FM  # noqa: E402
import matplotlib  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, NullFormatter  # noqa: E402

F.FIG = FIG
F.OLD5 = os.path.join(ROOT, "..", "order56", "old", "crk5_paper")
F.OLD6 = os.path.join(ROOT, "..", "order56", "old", "crk6_figures")
plt = F.plt
CCK = ("CCK6(5)", "CCK6(5)-s", "CCK6-casc", "CCK6(5)/5")
for lst in (FC.FIXED6, FC.FIG6, FC.ADAPT6, FC.ADAPT, FC.FIXED, FM.M5, FM.M6):
    lst[:] = [m for m in lst if m not in CCK]


def save(fig, name):
    import names
    names.rename_figure(fig)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


F.save = save


def fig_pareto():
    """Panels (a) CRK5 and (b) CRK6 of the front figure (base figures.fig_pareto without the CCK6 panel)."""
    tab = json.load(open(os.path.join(F.DATA, "tableaux.json")))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.3))
    fkw = dict(color="k", lw=1.1, drawstyle="steps-post", label=r"$E_{\min}(B)$")
    cl = F.read_csv(os.path.join(F.OLD5, "fig3_pareto_cloud.csv"))
    ax[0].scatter(cl["max_abs_coef"], cl["C2"], s=3, color="#bbbbbb", lw=0, label="random walks")
    ax[0].plot(*F.front_E("CRK5"), **fkw)
    for k in ["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5"]:
        ax[0].plot(tab[k]["maxcoef"], tab[k]["E"][0], F.ST[k][1], color=F.ST[k][0], ms=5, mec="k", mew=0.4,
                   label=F.lbl(k))
    ax[0].set_ylabel(r"$E_{p+1}=\|e\|_2$")
    ax[0].set_title("(a) CRK5, $E_6$")
    ax[0].set_ylim(1.2e-3, 4e-2)
    c6 = F.read_csv(os.path.join(F.OLD6, "fig3_pareto_cloud.csv"))
    dims = 42 - np.array(c6["rank_J"])
    for dval, col in [(11, "#1a8a4a"), (8, "#e0a020"), (6, "#7799cc"), (5, "#cc6677")]:
        msk = dims == dval
        if msk.any():
            ax[1].scatter(np.array(c6["max_abs_coef"])[msk], np.array(c6["C2"])[msk], s=7, color=col, lw=0,
                          alpha=0.8, label=f"dim {dval - 1}")
    ax[1].plot(*F.front_E("CRK6"), **fkw)
    for k in ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam", "RK6-opt", "Butcher6"]:
        ax[1].plot(tab[k]["maxcoef"], tab[k]["E"][0], F.ST[k][1], color=F.ST[k][0], ms=5, mec="k", mew=0.4,
                   label=F.lbl(k))
    ax[1].set_title("(b) CRK6, $E_7$")
    ax[1].set_ylim(2.0e-4, 5e-2)
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"$\max|$coefficient$|$")
        a.set_xlim(0.3, 300)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.0, handletextpad=0.3,
                 columnspacing=0.8, handlelength=1.6)
        a.xaxis.set_minor_formatter(NullFormatter())
        a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100, 200]))
        a.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig56_pareto")


def fig_measured():
    meas = json.load(open(os.path.join(r1patch.MERGED, "measured.json")))
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.6))
    FM.panel(ax[0], meas, FM.M5, "CRK5-opt", "(a) order five: constant relative to CRK5-opt")
    FM.panel(ax[1], meas, FM.M6, "CRK6-opt", "(b) order six: constant relative to CRK6-opt")
    for a in ax:
        a.set_ylabel(r"$C/C_{\mathrm{ref}}$")
    fig.tight_layout()
    save(fig, "fig56_measured")


def fig_wp_adaptive():
    FR.fig_adaptive()
    for ext in ("pdf", "png"):
        os.replace(os.path.join(FIG, "fig_wp_adaptive." + ext), os.path.join(FIG, "fig56_wp_adaptive." + ext))
        os.remove(os.path.join(FIG, "fig_tolerance." + ext))
    print("wrote fig56_wp_adaptive")


if __name__ == "__main__":
    fig_pareto()
    fig_measured()
    fig_wp_adaptive()
