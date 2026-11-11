"""Figures of orders seven and eight for the CRK 5-8 paper, without the complex Cash--Karp pair CCK7(6).

Uses the code and data of the order 7-8 paper read-only and writes ../paper/figures/fig78_wp_adaptive
and fig78_stability (.pdf/.png).

usage: python figs78.py      (from this folder)
"""
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OLD = os.path.abspath(os.path.join(ROOT, "..", "order78", "code"))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, OLD)

import figures as F  # noqa: E402
import figures_cost as FC  # noqa: E402
import make_tables as MT  # noqa: E402

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
    fig_stability()
    fig_wp_adaptive()
