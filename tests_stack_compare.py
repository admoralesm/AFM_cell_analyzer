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

# --------------------------------------------------- lines drawn by hand
bar = sc.hand_line_traces({"x": [1.0, 3.0], "y": [5.0, 5.0], "dash": "solid",
                           "colour": "#7B1FA2", "width": 5, "caps": True,
                           "text": "Approach"}, "#000", 1.8, room=0.5)
check("a bar is its line with upright ends, its label over its middle and "
      "room kept above it",
      len(bar) == 3 and bar[0].line.width == 5
      and bar[0].marker.symbol == "line-ns" and "lines" in bar[0].mode
      and bar[1].text[0] == "<b>Approach</b>" and bar[1].x[0] == 2.0
      and bar[1].textfont.color == "#7B1FA2" and bar[2].y[0] == 5.5,
      str([t.name for t in bar]))
plain = sc.hand_line_traces({"x": [0, 1], "y": [0, 1], "dash": "dash"},
                            "#123456", 2.0)
check("a plain line drawn by hand is one trace in the file's colour",
      len(plain) == 1 and plain[0].mode == "lines"
      and plain[0].line.color == "#123456" and plain[0].line.width == 2.0)

# ---------------------------------------------------------------- series
def trace(label, x0, x1, panel, zero=0.0, **extra):
    x = np.linspace(x0, x1, 101)
    t = {"x": x, "y": np.sin(x), "raw_x": x + zero, "zero": zero,
         "label": label, "ylabel": "F", "colour": "#111", "slot": panel - 1,
         "file": label, "panel": panel, "xcol": "t"}
    t.update(extra)
    return t

a = trace("a", 0.0, 10.0, 1)
b = trace("b", -2.0, 3.0, 2, zero=5.0,
          baseline_line=(np.linspace(-2, 3, 101), np.zeros(101)),
          peak_marks=[(np.array([-1.0, 2.0]), np.array([1.0, 1.0]),
                       "triangle-up", "peak", 1)],
          sat_line={"x": np.array([0.0, 1.0]), "y": np.array([2.0, 2.0]),
                    "flat": (0.0, 1.0), "peak": None},
          extend={"after": True, "level": 0.0})
c = trace("c", 0.0, 4.0, 3, x_range=(1.0, 2.0))
laid = sc.series_layout([a, b, c], gap=1.0)
check("the files follow each other in place order, the gap between them",
      np.isclose(laid[0]["x"].min(), 0) and np.isclose(laid[0]["x"].max(), 10)
      and np.isclose(laid[1]["x"].min(), 11)
      and np.isclose(laid[1]["x"].max(), 16)
      and np.isclose(laid[2]["series"]["start"], 17)
      and np.isclose(laid[2]["series"]["end"], 18),
      str([(t["x"].min(), t["x"].max(), t["series"]) for t in laid]))
check("x + zero is still each file's own x (so clicks keep their meaning)",
      all(np.allclose(t["x"] + t["zero"], o["raw_x"])
          for t, o in zip(laid, (a, b, c))))
check("its baseline, peaks and flat part move with it",
      np.isclose(laid[1]["baseline_line"][0][0], 11)
      and np.allclose(laid[1]["peak_marks"][0][0], [12, 15])
      and np.allclose(laid[1]["sat_line"]["x"], [13, 14])
      and laid[1]["sat_line"]["flat"] == (13.0, 14.0))
check("its own zero sits at its offset", np.isclose(
      laid[1]["series"]["offset"], 13.0), str(laid[1]["series"]))
short = sc.series_layout([trace("long", 0, 6, 1),
                          trace("brief", 0, 2, 1, extend={"after": True,
                                                         "level": 0.0}),
                          trace("next", 0, 3, 2)])
fig_e = sc.stacked_figure(short, overlay=True, zero_line=False)
carried = [t for t in fig_e.data if t.name == "brief carried on"]
check("a level carried on stops at the end of its own place",
      len(carried) == 1 and np.allclose(carried[0].x, [2, 6]),
      str([list(t.x) for t in carried]))
cy = laid[2]["y"]
check("only the part in its x range is drawn",
      np.isnan(cy[laid[2]["x"] < 17 - 1e-9]).all()
      and np.isfinite(cy[(laid[2]["x"] > 17) & (laid[2]["x"] < 18)]).all())
check("the input traces are not changed",
      np.isclose(b["x"].min(), -2) and b["zero"] == 5.0
      and "series" not in b)
two = sc.series_layout([trace("p", 0, 2, 1), trace("q", 0, 5, 1),
                        trace("r", 0, 1, 2)])
check("files with the same place overlay, the next after the longer",
      np.isclose(two[0]["x"].min(), 0) and np.isclose(two[1]["x"].min(), 0)
      and np.isclose(two[2]["x"].min(), 5), str([t["x"].min() for t in two]))
kept = sc.series_layout([trace("k", -3, 2, 1), trace("m", 0, 1, 2)],
                        start=sc.SERIES_STARTS[1])
check("or the first file keeps its own x",
      np.isclose(kept[0]["x"].min(), -3) and np.isclose(kept[1]["x"].min(), 2))
zl = sc.series_layout([trace("z1", 0, 4, 1), trace("z2", -2, 3, 2),
                       trace("z3", 1, 4, 3), trace("z4", -5, -1, 4)],
                      from_zero=True)
check("from its zero: a file starts in the series at its zero",
      np.isclose(zl[1]["series"]["start"], 4)
      and np.isclose(zl[1]["series"]["offset"], 4)
      and np.isclose(zl[1]["series"]["end"], 7),
      str(zl[1]["series"]))
check("what comes before its zero is left out",
      np.isnan(zl[1]["y"][zl[1]["x"] < 4 - 1e-9]).all()
      and np.isfinite(zl[1]["y"][zl[1]["x"] > 4 + 1e-9]).all())
check("a zero before its first point leaves an empty stretch before it",
      np.isclose(zl[2]["series"]["start"], 7)
      and np.isclose(np.nanmin(zl[2]["x"]), 8))
check("a zero past its last point leaves the file whole",
      np.isclose(zl[3]["series"]["start"], 11)
      and np.isfinite(zl[3]["y"]).all()
      and np.isclose(zl[3]["series"]["end"], 15), str(zl[3]["series"]))
fig = sc.stacked_figure(laid, overlay=True, zero_line=False)
sc.series_decorate(fig, laid, separators=True, names=True, shade=True,
                   zeros=True)
lines_v = [sh for sh in fig.layout.shapes if sh.type == "line"]
rects = [sh for sh in fig.layout.shapes if sh.type == "rect"]
names = [an.text for an in fig.layout.annotations if "span" in (an.text or "")]
check("dividers between the files, a name over each, every other shaded",
      len(names) == 3 and len(rects) == 1
      and any(np.isclose(sh.x0, 10.5) for sh in lines_v)
      and any(np.isclose(sh.x0, 16.5) for sh in lines_v),
      str([(sh.type, sh.x0) for sh in fig.layout.shapes]))
check("and each file's zero marked where it falls",
      any(np.isclose(sh.x0, 13.0) for sh in lines_v))
table = sc.series_table(laid)
check("the table has x along the series and each file's own x",
      list(table.columns[:3]) == ["a | x (series)", "a | x (own)", "a | F"]
      and np.isclose(table["b | x (own)"].min(), -2)
      and np.isclose(table["b | x (series)"].min(), 11))

print("peak statistics: median and range")
# three peaks, 1, 2 and 4 high, at 10, 20 and 35 s: intervals 10 and 15
xp = np.arange(0, 45, 0.05)
yp = sum(h * np.exp(-((xp - c) / 0.3) ** 2 / 2)
         for h, c in ((1.0, 10.0), (2.0, 20.0), (4.0, 35.0)))
found = sc.peak_analysis(xp, yp, np.zeros_like(xp), 0.5, "up", 1.0,
                         unit="µN", x_unit="s")
said = dict(found["stats"])
check("the amplitudes' median and range",
      said.get("  median (up)") == "2 µN"
      and said.get("  range (up)") == "1 to 4 µN (spread 3)", said)
check("the intervals' median and range",
      said.get("  median interval") == "12.5 s"
      and said.get("  interval range") == "10 to 15 s", said)

print("all passing" if not FAILED else f"{len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
