"""Figures of orders five and six for the CRK 5-8 paper, without the complex Cash--Karp pairs.

Uses the code and data of the order 5-6 paper read-only (as tables56.py) and writes
../paper/figures/fig56_wp_adaptive (.pdf/.png), the order five and six half of the work-precision figure;
the other half comes from figs78.py.

usage: python figs56.py      (from this folder)
"""
import os
import sys


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
import figures_r1 as FR  # noqa: E402

F.FIG = FIG
plt = F.plt
CCK = ("CCK6(5)", "CCK6(5)-s", "CCK6-casc", "CCK6(5)/5")
for lst in (FC.FIXED6, FC.FIG6, FC.ADAPT6, FC.ADAPT, FC.FIXED):
    lst[:] = [m for m in lst if m not in CCK]


def save(fig, name):
    import names
    names.rename_figure(fig)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


F.save = save


def fig_wp_adaptive():
    FR.fig_adaptive()
    for ext in ("pdf", "png"):
        os.replace(os.path.join(FIG, "fig_wp_adaptive." + ext), os.path.join(FIG, "fig56_wp_adaptive." + ext))
        os.remove(os.path.join(FIG, "fig_tolerance." + ext))
    print("wrote fig56_wp_adaptive")


if __name__ == "__main__":
    fig_wp_adaptive()
