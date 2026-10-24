"""Checks for stack_compare.py (the Stack & compare tab).

    python tests_stack_compare.py
"""
import sys
import numpy as np
import stack_compare as sc

FAILED = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        FAILED.append(name)


notes = b"# exported\n# by the AFM\nTime (s),Force (nN)\n0,1\n0.1,2\n0.2,5\n0.3,3\n"
f = sc.read_table(notes, "a.csv")
check("notes above the header are skipped", list(f.columns) == ["Time (s)", "Force (nN)"] and len(f) == 4)
f2 = sc.read_table(b"0\t1\n1\t4\n2\t9\n", "b.txt")
check("tab separated, no header", f2.shape == (3, 2))
f3 = sc.read_table(b"t;v\n0,5;1\n1,5;2\n", "c.csv")
check("decimal commas", float(f3["t"].iloc[1]) == 1.5)
x, y = f["Time (s)"].to_numpy(), f["Force (nN)"].to_numpy()
check("zero at the peak", sc.zero_point(x, y, "At the highest point") == 0.2)
check("zero at a crossing, interpolated",
      abs(sc.zero_point(x, y, "Where it first crosses a level", level=3.5) - 0.15) < 1e-9)
xz, yz = sc.prepare(f, "Time (s)", "Force (nN)", x_factor=1000, zero=100)
check("x factor then zero", np.allclose(xz, [-100, 0, 100, 200]))
idx = sc.thin_index(np.sin(np.arange(1_000_000) / 50.0), limit=1000)
check("thinning keeps at most the limit", len(idx) <= 1000)
fig = sc.stacked_figure([dict(x=x, y=y, label="a", ylabel="F", colour="red"),
                         dict(x=x, y=-y, label="b", ylabel="F", colour="blue", neg_up=True)],
                        linked=False)
second = "yaxis" + fig.data[1].yaxis[1:]
check("one panel each, the second flipped",
      len(fig.data) == 2 and fig.layout[second].autorange == "reversed")
fig = sc.stacked_figure([dict(x=x, y=y, label="a", ylabel="F", colour="red", panel=1),
                         dict(x=x, y=y * 100, label="b", ylabel="h", colour="blue", panel=1,
                              right=True, y_range=(0, 600)),
                         dict(x=x, y=y, label="c", ylabel="L", colour="green", panel=2,
                              x_range=(0.1, 0.2))], linked=False)
check("two overlaid in panel 1, the second on its own right axis",
      fig.data[0].xaxis == fig.data[1].xaxis and fig.data[1].yaxis != fig.data[0].yaxis
      and fig.layout["yaxis" + fig.data[1].yaxis[1:]].side == "right")
check("each axis takes its own range",
      tuple(fig.layout["yaxis" + fig.data[1].yaxis[1:]].range) == (0, 600)
      and tuple(fig.layout["xaxis" + fig.data[2].xaxis[1:]].range) == (0.1, 0.2))
rng = np.random.default_rng(0)
xc = np.linspace(0, 10, 2001)
yc = 0.2 * xc + rng.normal(0, 0.05, xc.size)
yc[xc > 4] += 3 * (xc[xc > 4] - 4) ** 1.5
contact, coeffs, sd = sc.baseline_departure(xc, yc, 0, 3, k=3, order=1)
check("the contact point is where the signal leaves its baseline (true 4)",
      contact is not None and abs(contact - 4.0) < 0.1, str(contact))
check("descriptive statistics for the balloon",
      dict(sc.describe([1, 2, 3, 4]))["n"] == "4")
print("all passing" if not FAILED else f"{len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
