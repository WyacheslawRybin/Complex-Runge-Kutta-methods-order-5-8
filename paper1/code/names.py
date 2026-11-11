"""One name per method in the CRK 5-8 paper.

A tableau that carries an embedded formula is named, like the real pairs DP5(4) or Vern6(5), by its name
followed by the order of the embedded formula, both as a pair and with a fixed step (then its propagated
formula is used): CRK5-opt(4), CRK6-opt(4), CRK7-opt(5), CRK7-s(5), CRK8-opt(6), CRK8-s(6), RK6-opt(4).
Verner's 2010 pairs are Vern6(5), Vern7(6) and Vern8(7), his 1978 pairs Verner78-6(5), Verner78-7(6) and
Verner78-8(7) (a reviewer could not tell "Verner6(5)" from "Vern6(5)").  rename() maps the names used by the code and the
older texts (CRK6(4), CRK6-opt, Vern7, ...) to these; it is applied to the assembled tex and to every text
of the figures.
"""
import re

RULES = [
    (r"\bCRK5\(4\)", "CRK5-opt(4)"), (r"\bCRK6\(4\)", "CRK6-opt(4)"),
    (r"\bCRK7\(5\)", "CRK7-opt(5)"), (r"\bCRK8\(6\)", "CRK8-opt(6)"),
    (r"\bCRK5-opt(?!\()", "CRK5-opt(4)"), (r"\bCRK6-opt(?!\()", "CRK6-opt(4)"),
    (r"\bCRK7-opt(?!\()", "CRK7-opt(5)"), (r"\bCRK8-opt(?!\()", "CRK8-opt(6)"),
    (r"\bCRK7-s(?![\w(])", "CRK7-s(5)"), (r"\bCRK8-s(?![\w(])", "CRK8-s(6)"),
    (r"(?<![\w-])RK6-opt(?!\()", "RK6-opt(4)"),
    (r"\bVerner([678])\((\d)\)", r"Verner78-\1(\2)"),
    (r"\bVern7(?![\w(])", "Vern7(6)"), (r"\bVern8(?![\w(])", "Vern8(7)"),
]
# "X / X(q)" rows of the tables name the same method twice after the renaming
DOUBLE = re.compile(r"(\b[A-Z][\w-]*\(\d\)) / \1")


def rename(s):
    for a, b in RULES:
        s = re.sub(a, b, s)
    return DOUBLE.sub(r"\1", s)


def rename_figure(fig):
    """Apply rename() to every text object (titles, labels, tick labels, legends) of a matplotlib figure."""
    import matplotlib.text
    for t in fig.findobj(matplotlib.text.Text):
        old = t.get_text()
        new = rename(old)
        if new != old:
            t.set_text(new)


if __name__ == "__main__":
    for s in ["CRK6(4) and CRK6-opt", "RK6-opt / RK6-opt(4)", "CRK6-opt / CRK6(4)", "Vern7 and Vern8 and Verner7(6)",
              "CRK7-s, CRK7-s(5), CRK8-s", r"\rot{CRK8-s(6)}", "GJM5 CRK5-s"]:
        print(f"{s!r:40} -> {rename(s)!r}")
