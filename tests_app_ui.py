"""
Drive the app the way a person does: through the widgets.

Setting session_state directly is not a test of a Streamlit app. Streamlit
rejects a write to a widget key after that widget exists, so a bug of that
kind only shows up when a button is actually clicked. Every case here clicks.
"""
import html.parser
import pathlib
import sys
import numpy as np
import pandas as pd

# Everything here is found relative to this file, never by absolute path.
# A hard-coded "/root/..." works on exactly one machine, and when this file
# is copied somewhere else it fails with a traceback that names a directory
# the reader has never heard of.
HERE = pathlib.Path(__file__).resolve().parent
APP = str(HERE / "app.py")

sys.path.insert(0, str(HERE))
from streamlit.testing.v1 import AppTest  # noqa: E402
from lulevich_model import LulevichModel  # noqa: E402


def _refuse_to_be_the_app():
    """Say so plainly if this file is deployed as the Streamlit app.

    Four files travel together and they are easy to mix up when saving
    them. Deployed under the name app.py, this one used to fail somewhere
    deep in the standard library with a traceback that named neither file,
    which is a long way to go to be told the wrong file was copied.
    """
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # pragma: no cover - streamlit not installed
        return
    if get_script_run_ctx() is None:
        return
    st.set_page_config(page_title="AFM Cell Analyzer", layout="wide")
    st.error(
        "**This is `tests_app_ui.py`, the test suite, not the app.**\n\n"
        "It has been deployed under the app's name. Copy the real `app.py` "
        "over it, keep `lulevich_model.py`, `plot_utils.py` and this file "
        "beside it under their own names, and reboot.\n\n"
        "Nothing is broken: the four files simply got mixed up on the way "
        "into the repository."
    )
    st.stop()


_refuse_to_be_the_app()

# Label -> the argument the model takes, mirroring app.py.
MEMBRANE_MODE = {"holds what it reached": "freeze", "keeps stiffening": "continue"}
MEMBRANE_MODE_ALL = dict(MEMBRANE_MODE, **{"starts stretching at ε₁": "late"})
CYTO_MODE = {"at ε₁": "break", "from the very start": "zero"}


def synthetic(membrane="freeze", cyto_start="break", En_kPa=3.0, n=260, noise=0.01):
    """A curve built from a known composition, so the answer is known."""
    eps = np.linspace(0.001, 0.60, n)
    model = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
    m, c, nu = model.composition_terms(eps, 0.15, 0.40, membrane, cyto_start)
    force = m * 0.6e6 + c * 1.2e3 + nu * En_kPa * 1e3
    rng = np.random.default_rng(0)
    force = force * (1.0 + noise * rng.standard_normal(n))
    return eps, force


def load(app, eps, force):
    app.session_state["data"] = {
        "epsilon": eps, "force_N": force, "source": "synthetic.csv", "n_dropped": 0,
    }


# The C2C12 page opens on the four-regime fit. Everything in this file
# before it was written against the spring-network models, so unless a case
# asks for the four regimes it is run on those.
PIECEWISE_MODE = "4-regime piecewise (C2C12)"
ADVANCED_MODE = "Spring-network models (advanced)"


def start(**state):
    state.setdefault("c2c12_fit_mode", ADVANCED_MODE)
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    eps, force = synthetic()
    load(app, eps, force)
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    # Choosing a cell type applies that type's defaults, which is right in
    # the app and wrong in a test that wanted to set one of those defaults by
    # hand. Anything the caller asked for that the preset then overwrote is
    # put back and the app run once more.
    def current(key):
        try:
            return app.session_state[key]
        except Exception:
            return None

    changed = {
        key: value for key, value in state.items()
        if key != "cell_type" and current(key) != value
    }
    if changed:
        for key, value in changed.items():
            app.session_state[key] = value
        app.run()
    return app


def state(app, key, default=None):
    """One session value, or a default. AppTest's session state has no get."""
    try:
        return app.session_state[key]
    except (KeyError, AttributeError):
        return default


def widget_by_label(app, kind, label):
    for w in getattr(app, kind):
        if label.lower() in (w.label or "").lower():
            return w
    return None


def button_by_label(app, label):
    for b in app.button:
        if label.lower() in (b.label or "").lower():
            return b
    return None


class _FlatTableReader(html.parser.HTMLParser):
    """Pull the app's HTML result tables back into DataFrames."""

    def __init__(self):
        super().__init__()
        self.tables, self._rows, self._cells, self._text = [], None, None, None

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "")
        if tag == "table" and "flat-table" in classes:
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._cells = []
        elif tag in ("td", "th") and self._cells is not None:
            self._text = []

    def handle_data(self, data):
        if self._text is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._text is not None:
            self._cells.append("".join(self._text).strip())
            self._text = None
        elif tag == "tr" and self._cells is not None:
            self._rows.append(self._cells)
            self._cells = None
        elif tag == "table" and self._rows is not None:
            if len(self._rows) > 1:
                self.tables.append(
                    pd.DataFrame(self._rows[1:], columns=self._rows[0])
                )
            self._rows = None


def bare(name):
    """A component name with any leading emoji taken off, for matching."""
    head, _, rest = str(name or "").partition(" ")
    return rest.strip() if rest and not head[:1].isalnum() else str(name or "")


def flat_tables(app):
    """Every result table on the page, as DataFrames of strings.

    The app draws these as plain HTML so that no column can be clipped off
    the right-hand edge, which means AppTest sees markdown rather than a
    dataframe element. Parsing it back is how a test still reads them.
    """
    reader = _FlatTableReader()
    for element in app.get("markdown"):
        text = str(element.value)
        if "flat-table" in text:
            reader.feed(text)
    return reader.tables


def table_with(app, *columns):
    """The one result table carrying all of these column names, or None."""
    for frame in flat_tables(app):
        if all(c in frame.columns for c in columns):
            return frame
    return None


SOURCE = pathlib.Path(APP).read_text()

def app_module_terms():
    """The materials the current cell type is fitted with."""
    import app as app_module
    return app_module.terms_for("Myoblast (C2C12)")


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def no_exception(app, name):
    if app.exception:
        print(f"  FAIL {name}: {app.exception[0].value}")
        FAILURES.append(name)
        return False
    return True


# ---------------------------------------------------------------- cases ---

def four_regime_curve(n=900, noise_N=0.2e-9):
    """A curve built from the four-regime laws, with the spec's boundaries."""
    x = np.linspace(0.0, 91.2, n)
    k, c0 = 2e-10, 1e-10
    ks, kc, kn, knc, kcore = 1e-13, 2e-11, 5e-12, 2e-10, 2e-9
    f5 = k * 5 + c0
    f40 = f5 + ks * 35 ** 3 + kc * 35 ** 1.5
    f60 = f40 + kn * 20 ** 3 + knc * 20 ** 1.5
    d = lambda a: np.clip(x - a, 0.0, None)  # noqa: E731
    # The membrane acts throughout: its cube law keeps rising after 40 %.
    shell = ks * d(5) ** 3
    f60 = f60 + ks * (55 ** 3 - 35 ** 3)
    force = np.where(
        x < 5, k * x + c0,
        np.where(x < 40, f5 + shell + kc * d(5) ** 1.5,
                 np.where(x < 60, f40 + kn * d(40) ** 3 + knc * d(40) ** 1.5
                          + shell - ks * 35 ** 3,
                          f60 + kcore * d(60) ** 1.5 + shell - ks * 55 ** 3)))
    rng = np.random.default_rng(1)
    return x / 100.0, force + noise_N * rng.standard_normal(n)


def case_c2c12_opens_on_the_four_regime_fit():
    print("a C2C12 curve is fitted with the four regimes as soon as it loads")
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    eps, force = four_regime_curve()
    load(app, eps, force)
    app.run()
    if not no_exception(app, "the four-regime page"):
        return
    check("the page opens on the four-regime fit",
          state(app, "c2c12_fit_mode") == PIECEWISE_MODE,
          str(state(app, "c2c12_fit_mode")))
    fit = (state(app, "results") or {}).get("fit") or {}
    check("it has fitted without anything being pressed",
          fit.get("coupling") == "piecewise", str(fit.get("coupling")))
    check("at the spec's boundaries",
          [state(app, k) for k in ("pw_b1", "pw_b2", "pw_b3", "pw_end")]
          == [5.0, 40.0, 60.0, 91.2],
          str([state(app, k) for k in ("pw_b1", "pw_b2", "pw_b3", "pw_end")]))
    labels = [m.label for m in app.metric]
    for symbol in ("E_shell", "E_cyto", "E_ne", "E_nc", "E_core", "E_align"):
        check(f"{symbol} is on the page", any(symbol in l for l in labels),
              str(labels))
    check("the fit follows the curve", fit.get("r_squared", 0) > 0.999,
          str(fit.get("r_squared")))

    app.number_input(key="pw_b2").set_value(35.0).run()
    if no_exception(app, "moving a boundary"):
        moved = state(app, "results")["fit"]["piecewise"]["boundaries_pct"]
        check("moving a boundary refits at once", moved[2] == 35.0, str(moved))

    app.button(key="pw_reset").click().run()
    if no_exception(app, "putting the spec back"):
        check("the reset puts the spec's boundaries back",
              state(app, "pw_b2") == 40.0, str(state(app, "pw_b2")))

    check("the membrane acts throughout by default",
          state(app, "pw_membrane_throughout") is True
          and state(app, "results")["fit"]["piecewise"]["membrane_throughout"])

    app.button(key="pw_find").click().run()
    if no_exception(app, "finding the boundaries"):
        found = state(app, "pw_boundary_search") or {}
        check("the boundary search ran", found.get("success") is True,
              str(found.get("error")))
        used = [state(app, k) for k in ("pw_b1", "pw_b2", "pw_b3")]
        fitted = state(app, "results")["fit"]["piecewise"]
        check("and the fit on the page is at the boundaries it left",
              [round(v, 2) for v in fitted["boundaries_pct"][1:4]]
              == [round(v, 2) for v in used],
              f"{fitted['boundaries_pct']} vs {used}")
        check("which says where they came from",
              "search" in fitted["boundary_source"]
              or "specification" in fitted["boundary_source"],
              fitted["boundary_source"])

    app.radio(key="c2c12_fit_mode").set_value(ADVANCED_MODE).run()
    if no_exception(app, "switching to the spring-network models"):
        check("and the spring-network fit is one click away",
              button_by_label(app, "Fit this cell") is not None,
              str([b.label for b in app.button][:12]))


def case_loads_clean():
    print("app loads with a curve, no exception")
    app = start()
    no_exception(app, "clean load")
    check("no old hybrid wording anywhere",
          not any("parallel below" in str(r.value) for r in app.radio))


def case_model_names():
    print("every model choice runs")
    app = start()
    radio = widget_by_label(app, "radio", "how the cell is modelled")
    check("model radio present", radio is not None)
    if radio is None:
        return
    check("no confusing name",
          not any("below" in o and "above" in o for o in radio.options),
          str(radio.options))
    for option in radio.options:
        if option.startswith("Compare"):
            continue  # slow, covered separately
        app2 = start()
        widget_by_label(app2, "radio", "how the cell is modelled").set_value(option).run()
        no_exception(app2, f"model {option!r}")


def case_composition_radios():
    print("composition radios drive the fit")
    seen = set()
    for membrane in ("holds what it reached", "keeps stiffening"):
        for cyto in ("at ε₁", "from the very start"):
            app = start()
            widget_by_label(app, "radio", "the membrane").set_value(membrane).run()
            widget_by_label(app, "radio", "cytoskeleton starts").set_value(cyto).run()
            if not no_exception(app, f"composition {membrane}/{cyto}"):
                continue
            state = app.session_state
            seen.add((state["membrane_after_break"], state["cyto_starts_at"]))
    check("all four combinations reachable", len(seen) == 4, str(seen))


def case_highlight():
    print("highlight selector")
    app = start()
    radio = widget_by_label(app, "radio", "highlight on the plot")
    check("highlight radio present", radio is not None)
    if radio is None:
        return
    for option in radio.options:
        app2 = start()
        widget_by_label(app2, "radio", "highlight on the plot").set_value(option).run()
        no_exception(app2, f"highlight {option!r}")


def case_search_applies_in_one_press():
    print("one press searches and applies the winner")
    app = start()
    button = button_by_label(app, "Find the best combination and fit it")
    check("search button present", button is not None)
    if button is None:
        return
    before = (
        app.session_state["segment_break_1"], app.session_state["segment_break_2"],
        app.session_state["membrane_after_break"], app.session_state["cyto_starts_at"],
    )
    button.click().run()
    if not no_exception(app, "combination search"):
        return
    search = app.session_state["composition_search"]
    check("search succeeded", bool(search and search.get("success")),
          str(search.get("error") if search else None))
    if not (search and search.get("success")):
        return
    import lulevich_model as lm
    check("every composition is ranked",
          len({(r["membrane"], r["cyto_start"]) for r in search["candidates"]})
          == len(lm.COMPOSITIONS),
          f"{len({(r['membrane'], r['cyto_start']) for r in search['candidates']})} "
          f"of {len(lm.COMPOSITIONS)}")

    truth = ("freeze", "break")
    best = search["best"]
    picked = (best["membrane"], best["cyto_start"])
    check("the true composition is the one picked", picked == truth,
          f"picked {picked}")
    check("ε₁ recovered", abs(best["break_1"] - 0.15) < 0.02,
          f"{best['break_1']:.3f}")
    check("ε₂ recovered", abs(best["break_2"] - 0.40) < 0.02,
          f"{best['break_2']:.3f}")

    # The winner should already be in the widgets: no second click needed.
    after = (
        app.session_state["segment_break_1"], app.session_state["segment_break_2"],
        app.session_state["membrane_after_break"], app.session_state["cyto_starts_at"],
    )
    check("the winner was applied without a second click", after != before,
          f"{before} -> {after}")
    check("applied ε₁ matches the winner",
          abs(after[0] - best["break_1"]) < 0.002, f"{after[0]} vs {best['break_1']}")
    check("applied composition matches the winner",
          MEMBRANE_MODE[after[2]] == best["membrane"]
          and CYTO_MODE[after[3]] == best["cyto_start"])

    override = button_by_label(app, "Use this one instead")
    check("override button present", override is not None)
    if override is not None:
        override.click().run()
        no_exception(app, "override the pick")
    print(f"       picked {best['label']!r}, ε₁={after[0]}, ε₂={after[1]}")


def case_search_beats_the_old_grid():
    print("the refined search finds the breakpoints the coarse grid missed")
    for membrane, cyto in (
        ("freeze", "break"), ("freeze", "zero"), ("continue", "break"),
    ):
        eps, force = synthetic(membrane, cyto)
        model = LulevichModel(force, eps, cell_height=8.0e-6)

        coarse = model._best_breakpoints(
            0.0, 0.60, membrane, cyto, True, "uniform", 12, rounds=0
        )
        refined = model._best_breakpoints(
            0.0, 0.60, membrane, cyto, True, "uniform", 12, rounds=2
        )
        check(f"refinement does not make {membrane}/{cyto} worse",
              refined["ss_res"] <= coarse["ss_res"] * (1 + 1e-9),
              f"{refined['ss_res']:.4g} vs {coarse['ss_res']:.4g}")
        check(f"ε₁ within 0.02 of truth for {membrane}/{cyto}",
              abs(refined["break_1"] - 0.15) < 0.02, f"{refined['break_1']:.4f}")


def case_search_flags_what_it_cannot_see():
    print("the search says when a term or a boundary does nothing")
    eps, force = synthetic("continue", "zero")
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    found = model.search_compositions(0.0, 0.60)
    check("search succeeded", found.get("success"))
    if not found.get("success"):
        return
    row = next(
        r for r in found["candidates"]
        if r["membrane"] == "continue" and r["cyto_start"] == "zero" and r["use_nucleus"]
    )
    check("ε₁ flagged as unused when nothing depends on it",
          "ε₁" in row.get("idle_breaks", []), str(row.get("idle_breaks")))

    # A curve with no nucleus should be answered without one.
    eps, force = synthetic("freeze", "zero", En_kPa=0.0)
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    found = model.search_compositions(0.0, 0.60)
    check("a curve with no nucleus is answered without one",
          found["best"]["use_nucleus"] is False
          or found["best"]["En_kPa"] <= 0,
          f"En={found['best']['En_kPa']:.4g}, "
          f"use_nucleus={found['best']['use_nucleus']}")


def case_bare_plot():
    print("the plot is data and fit, and nothing else unless it is sent")
    app = start()
    # The bare switch is gone: a plot without its data is not a plot of
    # anything, so data and fit are not optional, and everything else
    # arrives by being sent rather than by being left switched on.
    check("there is no bare-plot switch to hunt for",
          not any("data and fit only" in (box.label or "").lower()
                  for box in app.checkbox),
          str([box.label for box in app.checkbox][:10]))
    check("element curves off by default",
          app.session_state["show_components"] is False)
    check("and only the boundaries are on it until something is sent",
          [row["kind"] for row in (state(app, "plot_layers") or [])]
          == ["boundaries"], str(state(app, "plot_layers")))

    import app as app_module

    everything_on = {name: True for name in app_module.PLOT_EXTRAS}
    normal = app_module.plot_flags({**everything_on, "bare_plot": False})
    check("individual boxes respected when not bare", all(normal.values()),
          str(normal))

    stripped = app_module.plot_flags({**everything_on, "bare_plot": True})
    for flag, drawn in stripped.items():
        check(f"{flag} forced off by the bare switch", drawn is False)

    partial = app_module.plot_flags(
        {**everything_on, "show_legend": False, "bare_plot": False}
    )
    check("one box off leaves the others on",
          partial["show_legend"] is False and partial["show_components"] is True)

    # And the app itself must still run with the switch on.
    check("app runs with the bare switch on", not app.exception)


def case_buttons_do_not_break_widgets():
    print("every button clicks cleanly, for every cell type")
    # Both cell types, because they build different models out of different
    # numbers of terms, and a button that only ever ran under the default
    # one is a button that has only been half tested. "Find the segments"
    # passed a four-term list into the three-term segmented machinery and
    # raised a KeyError on the fourth, for every cardiomyocyte, and this
    # case did not catch it because it only ever ran as a myoblast.
    for cell_type in ("Myoblast (C2C12)", "Cardiomyocyte"):
        app = start(cell_type=cell_type)
        labels = [b.label for b in app.button]
        for label in labels:
            if any(word in label.lower()
                   for word in ("box", "database", "send", "upload")):
                continue
            app2 = start(cell_type=cell_type)
            target = button_by_label(app2, label)
            if target is None:
                continue
            target.click().run()
            no_exception(app2, f"{cell_type}: button {label!r}")


def case_preset_round_trip():
    print("save a preset, then apply it back")
    app = start()
    widget_by_label(app, "radio", "the membrane").set_value("keeps stiffening").run()
    name_box = None
    for box in app.text_input:
        if "preset" in (box.key or ""):
            name_box = box
    check("preset name box present", name_box is not None)
    if name_box is None:
        return
    name_box.set_value("test preset").run()
    save = button_by_label(app, "Save current")
    check("save button present", save is not None)
    if save is None:
        return
    save.click().run()
    if not no_exception(app, "save preset"):
        return
    stored = app.session_state["range_presets"].get("test preset")
    check("preset stored", stored is not None)
    if stored is None:
        return
    check("preset carries the model name",
          stored["coupling"].startswith("Segmented"), str(stored.get("coupling")))
    check("preset carries the composition",
          stored["membrane_after_break"] == "keeps stiffening",
          str(stored.get("membrane_after_break")))
    check("preset carries the boundaries",
          "segment_break_1" in stored and "segment_break_2" in stored)

    # Now switch away and apply the preset back.
    widget_by_label(app, "radio", "the membrane").set_value(
        "holds what it reached"
    ).run()
    apply = button_by_label(app, "Apply this preset")
    if apply is not None:
        apply.click().run()
        if no_exception(app, "apply preset"):
            check("preset restored the composition",
                  app.session_state["membrane_after_break"] == "keeps stiffening",
                  str(app.session_state["membrane_after_break"]))


def case_companion_file_guard():
    print("mismatched companion files are named, not crashed on")
    import app as app_module

    check("no stale files in a matched tree", app_module.STALE_FILES == [],
          str(app_module.STALE_FILES))

    def old_signature(a, b, title=None):
        return (a, b, title)

    filtered = app_module.figure_kwargs(
        old_signature, title="keep", highlight_window=(0.1, 0.2), fit_window=(0, 1)
    )
    check("unknown keywords dropped", filtered == {"title": "keep"}, str(filtered))

    def new_signature(a, title=None, highlight_window=None):
        return None

    kept = app_module.figure_kwargs(
        new_signature, title="keep", highlight_window=(0.1, 0.2)
    )
    check("known keywords kept", set(kept) == {"title", "highlight_window"}, str(kept))

    def takes_anything(a, **kwargs):
        return None

    passthrough = app_module.figure_kwargs(takes_anything, anything=1, else_=2)
    check("**kwargs passes everything through", len(passthrough) == 2)

    check("nothing this app needs is imported with a bare from-import",
          "\nfrom plot_utils import" not in SOURCE
          and "\nfrom lulevich_model import" not in SOURCE)
    check("and every name it needs is pulled by hand instead",
          SOURCE.count("_pull(") >= 4, str(SOURCE.count("_pull(")))


def case_a_half_updated_deploy_says_so():
    print("a companion file that is behind names itself instead of crashing")
    import shutil
    import tempfile

    here = HERE
    # A deploy that picked up app.py but left an older plot_utils.py behind.
    # This used to die on the import line, and Streamlit Cloud shows that as
    # a truncated traceback with the reason cut out: the one thing the person
    # needed to know was the one thing they could not see.
    with tempfile.TemporaryDirectory() as folder:
        stale = pathlib.Path(folder)
        for name in ("app.py", "lulevich_model.py", "baseline_correction.py",
                     "video_analysis.py", "video_processor.py",
                     "igor_parser.py", "google_sheets_manager.py",
                     "google_drive.py", "onedrive_store.py"):
            source = here / name
            if source.exists():
                shutil.copy(source, stale / name)
        for name in ("reference_WT_cardiomyocyte.csv",
                     "reference_WT_vcm_four.csv"):
            if (here / name).exists():
                shutil.copy(here / name, stale / name)

        text = (here / "plot_utils.py").read_text()
        cut = text.index("def balloon_figure(")
        resume = text.index("def exponent_profile_figure(")
        older = text[:cut] + text[resume:]
        older = older[:older.index("def ordering_figure(")]
        (stale / "plot_utils.py").write_text(older)

        sys.path.insert(0, str(stale))
        for name in ("app", "plot_utils", "lulevich_model"):
            sys.modules.pop(name, None)
        try:
            app = AppTest.from_file(str(stale / "app.py"), default_timeout=300)
            app.run()
            check("a half-updated deploy does not raise",
                  not app.exception,
                  str(app.exception[0].value) if app.exception else "")
            shown = " ".join(str(e.value) for e in app.get("error"))
            check("it says the deploy is half updated",
                  "half updated" in shown, shown[:120])
            check("it names the file that is behind",
                  "plot_utils.py" in shown, shown[:200])
            check("and the piece that is missing",
                  "balloon_figure" in shown, shown[:200])
            check("it does not carry on into a broken page",
                  not app.get("tabs"), str(len(app.get("tabs"))))
        finally:
            sys.path.remove(str(stale))
            for name in ("app", "plot_utils", "lulevich_model"):
                sys.modules.pop(name, None)

    # And an optional piece missing switches that feature off rather than
    # taking the app with it.
    import app as app_module
    check("the app is back to the matched tree",
          app_module.STALE_FILES == [], str(app_module.STALE_FILES))
    check("the ordering explorer is available here",
          app_module.HAS_ORDERINGS and app_module.HAS_ORDER_PLOTS)


def case_fit_survives_a_rerun():
    print("the fit survives a rerun with live refitting off")
    app = start()
    # Fit once, then take live refitting away, as uploading a video does.
    check("a fit exists to begin with",
          app.session_state["_last_fit"] is not None)
    app.session_state["live_fit"] = False
    app.run()
    if not no_exception(app, "rerun with live refitting off"):
        return
    check("the fit is still there after a rerun",
          app.session_state["_last_fit"] is not None)

    # The database section must still be on the page, which is what actually
    # vanished before: it lived inside the fit-succeeded branch.
    uploader_labels = [u.label for u in app.get("file_uploader")]
    check("the video uploader is still on the page",
          any("compression video" in (l or "").lower() for l in uploader_labels),
          str(uploader_labels))
    check("the send button is still on the page",
          button_by_label(app, "Send to OneDrive") is not None)


def case_database_section_without_a_fit():
    print("the database section is reachable before any fit")
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.session_state["c2c12_fit_mode"] = ADVANCED_MODE
    eps, force = synthetic()
    load(app, eps, force)
    app.session_state["live_fit"] = False
    app.run()
    if not no_exception(app, "no fit yet"):
        return
    check("no fit has been made", app.session_state["_last_fit"] is None)
    uploader_labels = [u.label for u in app.get("file_uploader")]
    check("the video uploader is reachable with no fit",
          any("compression video" in (l or "").lower() for l in uploader_labels),
          str(uploader_labels))
    send = button_by_label(app, "Send to OneDrive")
    check("the send button is shown with no fit", send is not None)
    if send is not None:
        check("and it is disabled until there is a fit", send.disabled is True)


class FakeStore:
    """Stands in for Box so the send path can be exercised for real."""

    last = {}

    def save_cell(self, record, curve_csv=None, thumbnail_png=None,
                  video_bytes=None, video_name=None):
        FakeStore.last = {
            "record": record, "curve": curve_csv, "thumb": thumbnail_png,
            "video": video_bytes, "video_name": video_name,
        }
        return {"cell_id": record["cell_id"], "video_url": ""}

    def load_index(self):
        return pd.DataFrame(columns=["cell_id", "date", "Em_MPa", "Ec_kPa"])

    def check(self):
        return {"ok": True, "detail": "fake store"}

    def auth_method(self):
        return "fake"


def case_send_without_a_video():
    print("a cell can be sent with no video at all")
    FakeStore.last = {}
    app = start(onedrive_store=FakeStore(), cell_name="cell-01")
    check("no video is loaded", not app.session_state["video_path"])

    send = button_by_label(app, "Send to OneDrive")
    check("send button present", send is not None)
    if send is None:
        return
    check("send button is enabled with a fit and no video", send.disabled is False)

    send.click().run()
    if not no_exception(app, "send with no video"):
        return
    check("no error was shown", not app.error, str([e.value for e in app.error]))
    check("a success message was shown",
          any("cell-01" in (m.value or "") for m in app.success),
          str([m.value for m in app.success]))

    saved = FakeStore.last
    check("the cell reached the store", bool(saved))
    check("no video bytes were sent", saved.get("video") is None)
    check("no morphology frame was sent", saved.get("thumb") is None)
    check("the curve went anyway", bool(saved.get("curve")))
    check("the moduli went anyway",
          "Em_MPa" in (saved.get("record") or {})
          and "Ec_kPa" in (saved.get("record") or {}))
    check("the fitted column is in the curve csv",
          "fit_N" in (saved.get("curve") or ""))


def case_clear_cell_wins_over_dark_debris():
    print("the detector picks the clear cell, not a dark blob of the same shape")
    try:
        import cv2
        import video_analysis as va
    except Exception as exc:
        print(f"  skip (no OpenCV: {exc})")
        return

    def scene():
        """A dark blob that is bigger and more central than the real cell."""
        f = np.full((300, 420, 3), 150, np.uint8)
        cv2.ellipse(f, (210, 175), (52, 48), 0, 0, 360, (95, 95, 95), -1)
        cv2.ellipse(f, (90, 175), (40, 37), 0, 0, 360, (205, 205, 205), -1)
        cv2.rectangle(f, (0, 40), (420, 70), (18, 18, 18), -1)   # cantilever
        rng = np.random.default_rng(0)
        return np.clip(f.astype(int) + rng.normal(0, 4, f.shape), 0, 255).astype(np.uint8)

    frame = scene()

    on_shape = va.detect_cell(frame, appearance="either", min_area_frac=0.005)
    check("without a hint the dark blob wins, which is the reported bug",
          on_shape.get("found") and on_shape["center"][0] > 150,
          str(on_shape.get("center")))

    clear = va.detect_cell(frame, appearance="clear", min_area_frac=0.005)
    check("told the cell is clear, it picks the clear one",
          clear.get("found") and clear["center"][0] < 150,
          str(clear.get("center")))

    dark = va.detect_cell(frame, appearance="dark", min_area_frac=0.005)
    check("told the cell is dark, it picks the dark one",
          dark.get("found") and dark["center"][0] > 150,
          str(dark.get("center")))

    # The stored crop must actually contain the cell, not background.
    crop = va.crop(frame, clear)
    check("the crop is centred on the clear cell",
          crop is not None and float(crop.mean()) > float(frame.mean()),
          f"crop mean {float(crop.mean()):.0f} vs frame {float(frame.mean()):.0f}")


def case_all_three_moduli_always_reported():
    print("all three moduli appear even when a term is not in the model")
    # Switched off after the curve has been read, because the app picks the
    # materials for you when a curve loads and would simply turn them back
    # on. What is being tested is the reporting, not the picking.
    app = start()
    if not no_exception(app, "loaded"):
        return
    # Unticked through the widgets, the way a person does it, and refitted
    # with them off: the tiles report the fit rather than the checkboxes, so
    # a tile that changed the moment a box was unticked would be claiming a
    # fit that had not happened.
    import app as app_module
    names = app_module.components_for("Myoblast (C2C12)")
    # One widget interaction per run: AppTest applies the last one only, so
    # unticking three boxes in one pass silently leaves two ticked.
    for term in ("interior", "nucleus", "nucleus_shell"):
        box = next(
            (c for c in app.checkbox
             if (c.label or "").startswith(names[term][0])), None
        )
        if box is not None:
            box.uncheck().run()
    fit_button = button_by_label(app, "Fit curve")
    if fit_button is not None:
        fit_button.click().run()
    if not no_exception(app, "membrane only"):
        return
    import app as app_module
    rows = app_module.result_rows(state(app, "_last_fit"))
    labels = [row[1] for row in rows]
    for wanted in ("membrane", "cytoskeleton", "nucleus"):
        check(f"{wanted} has a row with only the membrane selected",
              any(wanted in (l or "").lower() for l in labels), str(labels))
    off = [row for row in rows
           if row[0].startswith("modulus_")
           and row[0] != "modulus_membrane"]
    check("the switched-off moduli are all listed", len(off) >= 2, str(labels))
    for row in off:
        check(f"{row[1]} says it was not in the model",
              "not in this model" in row[2], str(row[2]))


def case_load_share_table():
    print("the range-by-range table says who carries the load")
    # A full-control diagnostic: in guided mode the page answers the same
    # question in words, and one page carrying both is the clutter this was
    # trimmed to remove.
    app = start(ui_mode="Full control · every setting")
    if not no_exception(app, "load share table"):
        return
    text = " ".join(str(m.value) for m in app.get("markdown"))
    check("the table has a heading", "share the load" in text, text[:120])


def case_download_when_box_is_absent():
    print("a cell can be saved with no Box account")
    app = start(cell_name="cell-01")
    drive = button_by_label(app, "Send to OneDrive")
    check("the OneDrive button is present", drive is not None)
    if drive is not None:
        check("and disabled without a connection", drive.disabled is True)
    check("Box is gone from the app",
          button_by_label(app, "Send to Box") is None
          and not any("Box" in (b.label or "") for b in app.button),
          str([b.label for b in app.button]))

    downloads = [d for d in app.get("download_button")
                 if "download this cell" in (d.label or "").lower()]
    check("a download is offered instead", len(downloads) == 1, str(len(downloads)))

    import zipfile, io as _io, json as _json
    import app as app_module
    fit = app.session_state["_last_fit"]
    eps, force = synthetic()
    # Build the bundle through the app so session state is live.
    check("there is a fit to package", fit is not None)


class FakeWorksheet:
    """A worksheet that records what would be written to the real sheet."""

    def __init__(self, header):
        self.header = list(header)
        self.rows = []
        self.updates = []
        self.all_values = [list(header)]
        self.written_rows = []
        self.cleared = False

    def row_values(self, index):
        return list(self.header) if index == 1 else []

    def get_all_values(self):
        return [list(row) for row in self.all_values]

    def clear(self):
        self.cleared = True

    def update(self, values=None, range_name=None, **kwargs):
        self.header = list(values[0])
        self.updates.append(list(values[0]))
        if len(values) > 1:
            self.written_rows = [list(r) for r in values[1:]]

    def append_row(self, row, **kwargs):
        self.rows.append(list(row))

    def insert_row(self, row, index):
        self.header = list(row)


def case_sheet_row_matches_the_header():
    print("the sheet row is built from the sheet's own header")
    try:
        from google_sheets_manager import GoogleSheetsManager
    except Exception as exc:
        print(f"  skip (gspread missing: {exc})")
        return

    # The user's real sheet as it stood before this change.
    existing = [
        "Cell ID", "Date Analyzed", "Cell Height (μm)",
        "Cantilever Constant (pN/nm)", "Young's Modulus (Em, MPa)",
        "Young's Modulus (Ei, kPa)", "Video Link", "Force Curve Created",
        "Fit Quality (R²)", "Notes", "Analysis Status", "Timestamp",
    ]
    manager = GoogleSheetsManager.__new__(GoogleSheetsManager)
    sheet = FakeWorksheet(existing)
    manager.worksheet = sheet

    ok, message = manager.append_cell_data({
        "cell_id": "cell-01", "Em": 0.605, "Ei": 1.196, "En": 3.027,
        "Em_range": "0.000 to 0.151", "En_range": "0.403 to 0.600",
        "break_1": 0.151, "break_2": 0.403, "fit_quality": 0.9997,
        "cell_height": 8.0, "spring_constant": 0.08, "notes": "test",
    })
    check("the write succeeded", ok, message)

    check("Date Analyzed was renamed, not duplicated",
          "Experiment Date" in sheet.header
          and "Date Analyzed" not in sheet.header, str(sheet.header))
    check("Cantilever Constant became Spring Constant",
          "Spring Constant, K (N/m)" in sheet.header
          and "Cantilever Constant (pN/nm)" not in sheet.header, str(sheet.header))
    check("renamed columns kept their position",
          sheet.header[1] == "Experiment Date"
          and sheet.header[3] == "Spring Constant, K (N/m)", str(sheet.header[:5]))
    for wanted in ("Young's Modulus (En, kPa)", "Em range (ε)", "Ei range (ε)",
                   "En range (ε)", "Height (um)", "Video Comment"):
        check(f"{wanted} column present", wanted in sheet.header, str(sheet.header))
    # The working belongs on the second tab now. An existing sheet keeps
    # whatever columns it already had, in place, but the app stops adding
    # these to the summary tab.
    for moved in ("ε₁ membrane hands over", "RMSE (N)", "Points fitted"):
        check(f"{moved} is not added to the summary tab",
              moved not in sheet.header, str(sheet.header))

    check("exactly one row was written", len(sheet.rows) == 1)
    written = dict(zip(sheet.header, sheet.rows[0]))
    check("cell id in the right column", written["Cell ID"] == "cell-01")
    check("Em in the Em column",
          abs(float(written["Young's Modulus (Em, MPa)"]) - 0.605) < 1e-6)
    check("En in the nucleus column",
          abs(float(written["Young's Modulus (En, kPa)"]) - 3.027) < 1e-6)
    check("the Em range travelled with it",
          written["Em range (ε)"] == "0.000 to 0.151", str(written["Em range (ε)"]))
    check("spring constant in N/m, not converted",
          abs(float(written["Spring Constant, K (N/m)"]) - 0.08) < 1e-9)
    check("row length matches the header", len(sheet.rows[0]) == len(sheet.header))

    # A reordered header must still be written correctly.
    manager2 = GoogleSheetsManager.__new__(GoogleSheetsManager)
    sheet2 = FakeWorksheet(list(reversed(existing)))
    manager2.worksheet = sheet2
    manager2.append_cell_data({"cell_id": "cell-02", "Em": 1.5})
    written2 = dict(zip(sheet2.header, sheet2.rows[0]))
    check("reordered header still gets the right cell id",
          written2["Cell ID"] == "cell-02")
    check("reordered header still gets Em in the Em column",
          abs(float(written2["Young's Modulus (Em, MPa)"]) - 1.5) < 1e-6)


def case_sheet_reorder_keeps_the_data():
    print("reordering the columns does not lose a row")
    try:
        from google_sheets_manager import GoogleSheetsManager
    except Exception as exc:
        print(f"  skip (gspread missing: {exc})")
        return

    header = ["Cell ID", "Date Analyzed", "Video Link", "Notes", "Custom of mine"]
    rows = [
        ["cell-01", "2026-01-01", "http://v/1", "first", "keep me"],
        ["cell-02", "2026-01-02", "http://v/2", "second", "keep me too"],
    ]
    manager = GoogleSheetsManager.__new__(GoogleSheetsManager)
    sheet = FakeWorksheet(header)
    sheet.all_values = [header] + rows
    manager.worksheet = sheet

    ok, message = manager.reorder_columns()
    check("the reorder succeeded", ok, message)
    new_header = sheet.header
    # No second tab could be made here (this manager has no spreadsheet), so
    # every column stays on the one tab rather than being stripped off it
    # and written nowhere. That is the whole point: the working moves only
    # once there is somewhere for it to move to.
    check("the summary columns come first",
          new_header[:2] == ["Experiment Date", "Cell ID"],
          str(new_header[:3]))
    check("and the working is still on this tab, not dropped",
          "Video Link" in new_header and "Notes" in new_header,
          str(new_header))
    check("with the message saying so",
          "still on this one" in message, message)
    check("a column the app does not know about was kept",
          "Custom of mine" in new_header)
    check("the header was renamed here too",
          "Experiment Date" in new_header and "Date Analyzed" not in new_header)

    written = [dict(zip(new_header, row)) for row in sheet.written_rows]
    check("both rows survived", len(written) == 2, str(len(written)))
    if len(written) == 2:
        check("row values followed their column",
              written[0]["Cell ID"] == "cell-01"
              and written[0]["Notes"] == "first"
              and written[0]["Video Link"] == "http://v/1",
              str(written[0]))
        check("the date moved to the renamed column",
              written[1]["Experiment Date"] == "2026-01-02", str(written[1]))
        check("the unknown column kept its value",
              written[1]["Custom of mine"] == "keep me too")


def case_fit_line_and_heights_toggle():
    print("the fitted curve and the component heights each have a switch")
    import plot_utils

    eps, force = synthetic()
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.60, 0.15, 0.40)
    mb, cb, nb = model.composition_terms(eps, 0.15, 0.40)
    fitted = mb * fit["Em"] + cb * fit["Ei"] + nb * fit["En"]

    def build(**flags):
        style = plot_utils.PlotStyle(force_unit="N", show_components=True, **flags)
        return plot_utils.force_curve_figure(
            eps, force, style, fit_force_N=fitted,
            membrane_N=mb * fit["Em"], interior_N=cb * fit["Ei"],
            nucleus_N=nb * fit["En"],
        )

    on = build()
    check("the model line is drawn by default",
          any(t.name == "Model" for t in on.data))
    check("no height labels by default", len(on.layout.annotations) == 0)

    off = build(show_fit_line=False)
    check("switching the fit off removes the model line",
          not any(t.name == "Model" for t in off.data))
    check("the data is still there",
          any(t.name == "Experimental data" for t in off.data))

    labelled = build(show_component_heights=True)
    texts = [a.text for a in labelled.layout.annotations]
    check("a height label per component plus the total",
          len(texts) == 4, str(texts))
    check("the labels carry the force unit",
          all("N" in (t or "") for t in texts), str(texts))


def case_plot_options_are_under_the_plot():
    print("the plot switches are next to the plot, not buried in the sidebar")
    app = start(cell_name="cell-01")
    if not no_exception(app, "plot options"):
        return
    labels = [c.label for c in app.checkbox]
    for wanted in ("Element curves", "Shaded segment bands", "Legend",
                   "Note saying which range was fitted"):
        check(f"“{wanted}” is on the page", wanted in labels, str(labels))

    text = " ".join(
        str(m.value) for m in list(app.get("markdown")) + list(app.get("caption"))
    )
    check("the sidebar points at the new place",
          "Plot options" in text or "sent there from the page" in text,
          "no pointer found")

    # Ticking one must actually change the figure.
    switch = widget_by_label(app, "checkbox", "Shaded segment bands")
    check("the switch is reachable", switch is not None)
    if switch is not None:
        switch.set_value(True).run()
        no_exception(app, "turning the bands on")
        check("the switch stuck", state(app, "show_fit_window") is True)


def case_save_the_plot():
    print("the plot can be saved with a sensible name")
    import app as app_module

    eps, force = synthetic()
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.60, 0.15, 0.40)

    app = start(cell_name="Cell 04 / trial 2")
    downloads = [d for d in app.get("download_button")
                 if "save" in (d.label or "").lower()]
    check("a save button is offered", len(downloads) >= 1,
          str([d.label for d in app.get("download_button")]))

    import datetime as _dt
    name = app_module.suggested_plot_name(
        "Cell 04 / trial 2", fit, _dt.date(2026, 3, 4), ".png"
    )
    check("the name carries the cell", name.startswith("Cell_04___trial_2"), name)
    check("the name carries the date", "2026-03-04" in name, name)
    check("the name carries the moduli", "Em" in name and "Ec" in name, name)
    check("the name has no path separators", "/" not in name and "\\" not in name, name)
    check("the extension is kept", name.endswith(".png"), name)
    blank = app_module.suggested_plot_name("", fit, _dt.date(2026, 3, 4), ".html")
    check("an unnamed cell still gets a usable name",
          blank.startswith("cell_") and blank.endswith(".html"), blank)


def case_fit_maths_box():
    print("the working behind the fit is shown")
    app = start(cell_name="cell-01")
    if not no_exception(app, "maths box"):
        return
    blocks = [str(b.value) for b in app.get("latex")]
    check("equations are rendered", len(blocks) >= 3, str(len(blocks)))
    joined = " ".join(blocks)
    check("the model equation is there", "A_m E_m" in joined, joined[:120])
    check("the least-squares statement is there",
          "arg\\min" in joined or "argmin" in joined, joined[-160:])
    code = " ".join(str(c.value) for c in app.get("code"))
    check("the prefactors are printed with values", "Am =" in code, code[:160])
    check("the fitted moduli are printed", "MPa" in code and "kPa" in code)


def case_guided_mode_is_the_default():
    print("guided mode: one button, an answer in words")
    app = start(cell_name="cell-01")
    if not no_exception(app, "guided load"):
        return
    check("there is no mode to choose any more",
          not any("How much to show" in (r.label or "")
                  for r in app.get("radio")),
          str([r.label for r in app.get("radio")]))

    work = button_by_label(app, "Fit this cell")
    check("the one-press button is there", work is not None)

    text = " ".join(str(m.value) for m in app.get("markdown"))
    check("the results are headed as results",
          "Fitting results" in text, text[:200])
    # The retelling of the fit is gone. It restated the boundaries drawn on
    # the curve and the two numbers printed beside R², and ended on a
    # sentence about chi-squared that said nothing the numbers did not.
    for gone in ("What this cell did as it was squashed",
                 "Does the model match the measurement"):
        check(f"“{gone[:32]}…” is not on the page", gone not in text)
    check("no bare jargon in the headline",
          "coupling" not in text.lower(), "the word coupling leaked out")

    # The stiffness table is gone: every modulus is in the metrics row and
    # again in the equation's coefficients, and a third copy in words was a
    # third place for them to disagree.
    parts = [f for f in flat_tables(app) if "Part of the cell" in f.columns]
    check("the stiffness table is gone", not parts, str(len(parts)))
    check("but every modulus is still on the page",
          any("Eₘ" in (m.label or "") for m in app.get("metric")),
          str([m.label for m in app.get("metric")][:6]))

    if work is not None:
        was = float(state(app, "segment_break_1"))
        work.click().run()
        if no_exception(app, "fitting"):
            fit = state(app, "_last_fit")
            check("the fit is made", bool(fit and fit.get("success")))
            check("at the boundary that was on the page",
                  float(state(app, "segment_break_1")) == was,
                  f"{was} -> {state(app, 'segment_break_1')}")


def case_full_control_shows_everything():
    print("full control puts the settings back on the page")
    app = start(cell_name="cell-01", ui_mode="Full control · every setting")
    if not no_exception(app, "full control"):
        return
    # There is one page now, so "full control" is the page: the button is
    # on it, and so are the choices it needs.
    check("the fit button is on it", button_by_label(app, "Fit this cell")
          is not None)
    check("and so is the range",
          any("1 · Relative deformation range" in str(m.value)
              for m in app.get("markdown")))
    check("the expert search button is still there",
          button_by_label(app, "Find the best combination and fit it") is not None)
    check("the retelling of the fit is gone from here too",
          not any("What this cell did as it was squashed" in str(m.value)
                  for m in app.get("markdown")))


def case_plain_language_helpers():
    print("the everyday wording is honest about scale")
    import app as app_module

    check("a jelly-soft modulus reads soft",
          "jelly" in app_module.stiffness_in_words(1.2e3),
          app_module.stiffness_in_words(1.2e3))
    check("a rubbery modulus reads firm",
          "rubber" in app_module.stiffness_in_words(6e5),
          app_module.stiffness_in_words(6e5))
    check("zero is called unmeasurable",
          "not measurable" in app_module.stiffness_in_words(0.0))
    check("a NaN is called unmeasurable",
          "not measurable" in app_module.stiffness_in_words(float("nan")))
    check("the comparisons increase with stiffness",
          len({app_module.stiffness_in_words(v)
               for v in (5e2, 5e3, 5e4, 5e5)}) == 4)

    check("a good fit is described as close",
          "exactly" in app_module.quality_in_words(0.999))
    check("a bad fit is called out",
          "wrong" in app_module.quality_in_words(0.5),
          app_module.quality_in_words(0.5))


def case_curve_saved_as_a_tab():
    print("the force curve can be stored in the spreadsheet")
    try:
        from google_sheets_manager import GoogleSheetsManager
    except Exception as exc:
        print(f"  skip (gspread missing: {exc})")
        return

    class FakeSpreadsheet:
        url = "https://docs.google.com/spreadsheets/d/abc/edit"

        def __init__(self):
            self.tabs = {}

        def worksheet(self, title):
            import gspread
            if title not in self.tabs:
                raise gspread.WorksheetNotFound(title)
            return self.tabs[title]

        def add_worksheet(self, title, rows, cols):
            sheet = FakeWorksheet([])
            sheet.id = 7
            self.tabs[title] = sheet
            return sheet

        def worksheets(self):
            return list(self.tabs.values())

    manager = GoogleSheetsManager.__new__(GoogleSheetsManager)
    book = FakeSpreadsheet()
    manager.spreadsheet = book

    eps, force = synthetic()
    fitted = force * 0.99
    fitted[:5] = np.nan          # outside the fitted range

    ok, message, url = manager.save_curve("Cell 04 / trial 2", eps, force, fitted)
    check("the curve was saved", ok, message)
    check("the tab is named after the cell",
          "curve_Cell 04 _ trial 2" in book.tabs, str(list(book.tabs)))
    check("the url points at the tab", "#gid=" in url, url)

    sheet = list(book.tabs.values())[0]
    written = sheet.written_rows if sheet.written_rows else []
    header = sheet.header
    check("the header names the columns",
          header == ["relative_deformation", "force_N", "fit_N"], str(header))
    check("every point was written", len(written) == len(eps), str(len(written)))
    check("NaN outside the fit is written as blank",
          written[0][2] == "", repr(written[0][2]))
    check("a fitted point carries its value",
          isinstance(written[-1][2], float), repr(written[-1][2]))

    # Saving the same cell twice replaces the tab rather than adding another.
    manager.save_curve("Cell 04 / trial 2", eps, force, fitted)
    check("refitting replaces the tab, not duplicates it", len(book.tabs) == 1,
          str(list(book.tabs)))
    check("the tab was cleared before rewriting", sheet.cleared is True)


def case_axis_ranges():
    print("the axes can be pinned, and deformation defaults to 0-1")
    import plot_utils
    import app as app_module

    check("0 to 1 is the default deformation axis",
          app_module.DEFAULTS["x_axis_mode"].startswith("0 to 1"),
          app_module.DEFAULTS["x_axis_mode"])

    eps, force = synthetic()
    style = plot_utils.PlotStyle(force_unit="N", x_range=(0.0, 1.0))
    fig = plot_utils.force_curve_figure(eps, force, style)
    check("the deformation axis is pinned to 0-1",
          list(fig.layout.xaxis.range) == [0.0, 1.0], str(fig.layout.xaxis.range))
    check("autorange is off so it stays pinned",
          fig.layout.xaxis.autorange is False, str(fig.layout.xaxis.autorange))

    style = plot_utils.PlotStyle(force_unit="N", y_range=(0.0, 5.0))
    fig = plot_utils.force_curve_figure(eps, force, style)
    check("the force axis can be pinned too",
          list(fig.layout.yaxis.range) == [0.0, 5.0], str(fig.layout.yaxis.range))

    loose = plot_utils.force_curve_figure(eps, force, plot_utils.PlotStyle(force_unit="N"))
    check("no range means plotly decides", loose.layout.xaxis.range is None)

    # A reversed or nonsense range must be ignored, not applied.
    bad = plot_utils.force_curve_figure(
        eps, force, plot_utils.PlotStyle(force_unit="N", x_range=(1.0, 0.0))
    )
    check("a reversed range is ignored", bad.layout.xaxis.range is None,
          str(bad.layout.xaxis.range))

    # Log axes take decades, so a linear pair must not be applied there.
    logged = plot_utils.force_curve_figure(
        eps, force,
        plot_utils.PlotStyle(force_unit="N", log_scale=True, x_range=(0.0, 1.0)),
    )
    check("ranges are not applied on log axes",
          logged.layout.xaxis.range is None, str(logged.layout.xaxis.range))


def case_nucleus_spring_is_shorter():
    print("a component met late gets a shorter spring")
    import plot_utils

    style = plot_utils.PlotStyle(force_unit="N")

    def springs(cyto_start):
        fig = plot_utils.cell_schematic(
            style, epsilon=0.5, break_1=0.15, break_2=0.40,
            membrane_mode="freeze", cyto_start=cyto_start,
            Em_MPa=0.6, Ei_kPa=1.2, En_kPa=3.0,
        )
        # Springs are line traces; measure how far each spans vertically.
        spans = []
        for trace in fig.data:
            ys = [v for v in (trace.y or []) if v is not None]
            if len(ys) > 4:
                spans.append(max(ys) - min(ys))
        return sorted(spans)

    late = springs("break")
    early = springs("zero")
    check("some springs were drawn", len(late) >= 2, str(late))
    if len(late) >= 2 and len(early) >= 2:
        check("the late component's spring is shorter than the earliest one",
              min(late) < max(late) * 0.95, str(late))
        check("starting at zero makes the interior spring full length",
              max(early) >= max(late) * 0.99, f"{early} vs {late}")


def case_component_names_follow_the_cell_type():
    print("a cardiomyocyte is not described as a myoblast")
    import app as app_module

    myoblast = app_module.components_for("Myoblast (C2C12)")
    cardio = app_module.components_for("Cardiomyocyte")
    check("the myoblast interior is the cytoskeleton",
          bare(myoblast["interior"][0]) == "Cytoskeleton", str(myoblast["interior"]))
    # One interior, and its name says what is in it. The cytoskeleton and
    # the myofibrils are not two springs: they are one incompressible
    # material, because nothing in the curve separates them.
    check("the cardiomyocyte interior is the non-sarcomeric cytoskeleton",
          "non-sarcomeric" in cardio["interior"][0].lower(),
          str(cardio["interior"]))
    check("with the sarcolemma outside it and myofibrils below",
          "sarcolemma" in cardio["membrane"][0].lower()
          and "myofibril" in cardio["nucleus"][0].lower(),
          str([cardio["membrane"][0], cardio["nucleus"][0]]))
    check("the deep slot is myofibrils, never a nucleus",
          "myofibril" in cardio["nucleus"][0].lower(), str(cardio["nucleus"]))
    check("and nothing in its component set says nucleus",
          not any("nucleus" in v[0].lower() for v in cardio.values()),
          str([v[0] for v in cardio.values()]))
    # Only the slots this cell type actually has. The rest carry neutral
    # fallback names so that nothing reaching for one gets a KeyError, and
    # those are never shown.
    check("every material it has carries an emoji",
          all(bare(cardio[term][0]) != cardio[term][0]
              for term in app_module.terms_for("Cardiomyocyte")),
          str([cardio[t2][0] for t2 in app_module.terms_for("Cardiomyocyte")]))
    check("the membrane is the shell resisting being stretched",
          "stretch" in cardio["membrane"][1], str(cardio["membrane"]))
    # No in-plane spring anywhere at the moment: the plain cardiomyocyte is
    # being settled first.
    check("no cell type is offered a tension spring",
          not any("tension" in app_module.terms_for(name)
                  for name in app_module.CELL_TYPES),
          str({n: app_module.terms_for(n) for n in app_module.CELL_TYPES}))
    check("an unknown cell type still gets names",
          app_module.components_for("Something else")["interior"][0])

    app = start(cell_name="cell-01", cell_type="Cardiomyocyte")
    if not no_exception(app, "cardiomyocyte names"):
        return
    # The components are named on their own ticks now, not in a table.
    listed = [str(c.label or "") for c in app.checkbox]
    check("the components are named on the page", listed)
    check("it lists the sarcolemma and both cytoskeletons",
          any("sarcolemma" in v.lower() for v in listed)
          and any("non-sarcomeric" in v.lower() for v in listed)
          and any("myofibril" in v.lower() for v in listed),
          str(listed))
    check("and never calls anything a nucleus",
          not any("Nucleus" in v for v in listed), str(listed))
    check("the shell is offered once, as the sarcolemma",
          sum("Sarcolemma" in v for v in listed) == 1, str(listed))
    check("the fit succeeds for a cardiomyocyte",
          app.session_state["_last_fit"] is not None
          and app.session_state["_last_fit"].get("success"))
    fit = app.session_state["_last_fit"]
    if fit:
        check("and it follows the data",
              fit["r_squared"] > 0.9, f"R2={fit['r_squared']:.4f}")


def case_cardiomyocyte_model_is_flagged_provisional():
    print("the cardiomyocyte model says it is provisional")
    app = start(cell_name="cell-01",
                model_kind="Cardiomyocyte (Morales Maldonado)")
    if not no_exception(app, "cardiomyocyte model"):
        return
    warnings = " ".join(str(w.value) for w in app.warning)
    check("a provisional warning is shown", "provisional" in warnings.lower(),
          warnings[:120])
    check("it says the equations were not available",
          "equations" in warnings.lower(), warnings[:200])
    check("it asks for the paper's equation",
          "send me" in warnings.lower(), warnings[:200])


def case_a_zero_modulus_is_a_measurement():
    print("a modulus that came back at zero says so, and says what it means")
    app = start(cell_name="cell-01")
    if not no_exception(app, "a zero modulus"):
        return
    # The range-by-range table that used to explain "not reached" is gone.
    # What has to survive it is the point that table was making: a term the
    # fit gave nothing to is a measurement, not a missing number.
    app.session_state["use_nucleus_shell"] = False
    app.run()
    if not no_exception(app, "a component left out"):
        return
    import app as app_module
    rows = app_module.result_rows(state(app, "_last_fit"))
    check("a component left out of the model is named as left out",
          any("not in this model" in row[2] for row in rows),
          str([row[2] for row in rows]))
    check("rather than reading as a measured zero",
          any(row[2][0].isdigit() for row in rows if row[0].startswith("modulus_")),
          str([row[2] for row in rows]))


def case_fit_statistics():
    print("chi squared catches what R-squared misses")
    from lulevich_model import LulevichModel as LM, noise_sigma, fit_statistics

    eps = np.linspace(0.001, 0.60, 260)
    g = LM(np.zeros_like(eps), eps, cell_height=8.0e-6)
    m, c, nu = g.composition_terms(eps, 0.15, 0.40, "freeze", "break")
    # A deep element that really carries something. Its prefactor uses the
    # nucleus radius squared (Lulevich eq 6), so at 3 kPa its force is a
    # rounding error and "dropping a real term" would not be dropping one.
    clean = m * 0.6e6 + c * 1.2e3 + nu * 15e3

    force = clean + 5e-11 * np.random.default_rng(0).standard_normal(260)
    model = LM(force, eps, cell_height=8.0e-6)
    right = model.fit_composition(0.0, 0.60, 0.15, 0.40)
    check("the right model gives chi2/dof near 1",
          0.2 < right["chi_squared_reduced"] < 3.0,
          f"{right['chi_squared_reduced']:.2f}")

    wrong = model.fit_composition(0.0, 0.60, 0.15, 0.40, use_nucleus=False)
    check("dropping a real term is caught by chi2",
          wrong["chi_squared_reduced"] > 10 * right["chi_squared_reduced"],
          f"{wrong['chi_squared_reduced']:.1f} vs {right['chi_squared_reduced']:.2f}")
    check("R-squared alone would not have caught it",
          wrong["r_squared"] > 0.97, f"R2={wrong['r_squared']:.4f}")

    check("adjusted R-squared is reported",
          np.isfinite(right["adj_r_squared"]))
    check("adjusted is never above plain",
          right["adj_r_squared"] <= right["r_squared"] + 1e-12)
    check("degrees of freedom account for the parameters",
          right["dof"] == right["n_points"] - right["n_params"],
          f"{right['dof']} vs {right['n_points']}-{right['n_params']}")

    # Noise that grows with force must not be read as the quiet end.
    sigma, typical = noise_sigma(eps, clean * (1 + 0.02 *
                                np.random.default_rng(0).standard_normal(260)))
    check("the noise estimate is per point", np.size(sigma) == 260, str(np.size(sigma)))
    check("and it grows where the force grows",
          np.median(sigma[-50:]) > np.median(sigma[:50]) * 3,
          f"{np.median(sigma[:50]):.2e} -> {np.median(sigma[-50:]):.2e}")
    check("a typical value is reported", np.isfinite(typical))

    # A curve too short to estimate noise must not crash.
    stats = fit_statistics(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 2,
                           epsilon=np.array([0.1, 0.2]))
    check("two points do not raise", np.isnan(stats["chi_squared"]))


def case_chi_squared_reaches_the_page():
    print("the fit statistics are shown and stored")
    app = start(cell_name="cell-01")
    if not no_exception(app, "chi squared on the page"):
        return
    import app as app_module
    rows = app_module.result_rows(state(app, "_last_fit"))
    check("a chi-squared row is shown",
          any("χ²" in row[3] for row in rows), str([row[3] for row in rows]))
    captions = " ".join(str(c.value) for c in app.get("caption"))
    check("it explains what about 1 means",
          "as close to the points as the scatter" in captions, "no explanation")
    check("it warns that R² can look fine",
          "R² can still look excellent" in captions, "no warning")

    try:
        from google_sheets_manager import GoogleSheetsManager
    except Exception:
        return
    names = [name for _, name in GoogleSheetsManager.COLUMNS]
    for wanted in ("Chi squared", "Chi squared / dof", "Adjusted R²",
                   "Measured noise σ (N)"):
        check(f"the sheet records {wanted}", wanted in names, str(names))


def case_search_stays_fast():
    print("the search does not crawl on a realistic curve")
    import time
    from lulevich_model import LulevichModel as LM

    # A real .ibw curve has thousands of points, not a few hundred. The
    # search fits hundreds of candidates, so anything expensive per fit is
    # multiplied by that. This is the guard on that multiplication.
    for n, budget in ((1000, 6.0), (3000, 12.0)):
        eps = np.linspace(0.001, 0.60, n)
        g = LM(np.zeros_like(eps), eps, cell_height=8.0e-6)
        m, c, nu = g.composition_terms(eps, 0.15, 0.40, "freeze", "break")
        force = (m * 0.6e6 + c * 1.2e3 + nu * 3e3) * (
            1 + 0.01 * np.random.default_rng(0).standard_normal(n)
        )
        model = LM(force, eps, cell_height=8.0e-6)

        start_time = time.time()
        found = model.search_compositions(0.0, 0.60)
        elapsed = time.time() - start_time
        check(f"the search finishes on {n} points", found.get("success"))
        check(f"and takes under {budget:.0f}s at {n} points ({elapsed:.1f}s)",
              elapsed < budget, f"{elapsed:.1f}s")
        if found.get("success"):
            best = found["best"]
            check(f"and is still right at {n} points",
                  (best["membrane"], best["cyto_start"]) == ("freeze", "break"),
                  f"{best['membrane']}/{best['cyto_start']}")

    # The winner must still come back with its statistics, even though the
    # candidates were fitted without them.
    check("the winner carries chi squared",
          np.isfinite(found["best"].get("chi_squared_reduced", np.nan))
          or np.isfinite(
              model.fit_composition(
                  0.0, 0.60, found["best"]["break_1"], found["best"]["break_2"]
              )["chi_squared_reduced"]
          ))


def case_png_is_not_rendered_every_run():
    print("the plot is not re-rendered to PNG on every rerun")
    import plot_utils
    import app as app_module

    calls = []
    original = plot_utils.go.Figure.to_image

    def counting(self, *args, **kwargs):
        calls.append(1)
        raise RuntimeError("no chrome here")

    plot_utils.go.Figure.to_image = counting
    try:
        app = start(cell_name="cell-01")
        app.run()
        app.run()
    finally:
        plot_utils.go.Figure.to_image = original

    check("no PNG render happened without being asked",
          len(calls) == 0, f"{len(calls)} renders")
    check("a Prepare a PNG button is offered",
          button_by_label(app, "Prepare a PNG") is not None)
    check("HTML is still available without asking",
          any("save as html" in (d.label or "").lower()
              for d in app.get("download_button")),
          str([d.label for d in app.get("download_button")]))


def case_guided_order_follows_the_work():
    print("parts, then range, then work it out, in that order")
    app = start(cell_name="cell-01")
    if not no_exception(app, "guided order"):
        return
    headings = [
        str(m.value).strip() for m in app.get("markdown")
        if str(m.value).strip().startswith("####")
    ]
    order = " | ".join(headings)
    # Range first, because it decides which points the rest is about, then
    # the parts and how they share the load, then the fit.
    # Choices first, in one place and made once; then the fit; then
    # everything the fit produced.
    check("step 1 is the range",
          any("1 · Relative deformation range" in h for h in headings), order)
    check("step 2 is the fit itself",
          any("2 · Fit" in h for h in headings), order)
    positions = [
        next(i for i, h in enumerate(headings) if key in h)
        for key in ("1 · Relative deformation range", "2 · Fit")
    ]
    check("and they are in that order", positions == sorted(positions),
          str(positions))
    check("the results come after both",
          all(positions[-1] < i for i, h in enumerate(headings)
              if "turned out to be" in h), order)
    check("no expander labels are raw markdown",
          not any((e.label or "").startswith("#") for e in app.get("expander")),
          str([e.label for e in app.get("expander")]))

    # The button has to be reachable without opening anything.
    work = button_by_label(app, "Fit this cell")
    check("the button is on the page", work is not None)
    check("and it is the primary action",
          work is not None and work.proto.type == "primary")

    # The part checkboxes must exist exactly once, in Step 2.
    labels = [c.label for c in app.checkbox]
    for term in ("Membrane", "Cytoskeleton", "Nuclear envelope",
                 "Inside the nucleus"):
        matching = [l for l in labels if bare(l).startswith(term)]
        check(f"{term} has exactly one checkbox", len(matching) == 1, str(matching))

    # Unticking a part must disable the button rather than fail later.
    import app as app_module
    for term in app_module.terms_for("Myoblast (C2C12)"):
        app.session_state[f"use_{term}"] = False
    app.run()
    if no_exception(app, "no parts selected"):
        work = button_by_label(app, "Fit this cell")
        check("with nothing ticked the button is disabled",
              work is not None and work.disabled is True)


def case_it_picks_the_arrangement():
    print("the search chooses segmented, side by side or stacked")
    from lulevich_model import LulevichModel as LM, search_arrangements

    eps = np.linspace(0.001, 0.60, 300)

    def segmented():
        g = LM(np.zeros_like(eps), eps, cell_height=8.0e-6)
        m, c, nu = g.composition_terms(eps, 0.15, 0.40, "freeze", "break")
        return m * 0.6e6 + c * 1.2e3 + nu * 3e3

    def side_by_side():
        g = LM(np.zeros_like(eps), eps, cell_height=8.0e-6)
        return g.combined_model(eps, 0.6e6, 1.2e3, 0.0, En=0.0)

    for name, maker, expected in (
        ("a segmented curve", segmented, "segmented"),
        ("a side-by-side curve", side_by_side, "parallel"),
    ):
        force = maker() * (
            1 + 0.01 * np.random.default_rng(0).standard_normal(eps.size)
        )
        model = LM(force, eps, cell_height=8.0e-6)
        found = search_arrangements(model, 0.0, 0.60)
        check(f"{name}: the search ran", found.get("success"),
              str(found.get("error")))
        if not found.get("success"):
            continue
        picked = found["best"]["arrangement"]
        tied = {c["arrangement"] for c in found["candidates"]
                if c.get("tied_with_best")} | {picked}
        check(f"{name} is called {expected}", expected in tied,
              f"picked {picked}, tied {tied}")
        check(f"{name}: all three arrangements were tried",
              len(found["candidates"]) == 3, str(len(found["candidates"])))
        check(f"{name}: the verdict is in plain words",
              "curve" in found["verdict"].lower(), found["verdict"][:80])

    # The winner has to translate into settings the app can apply.
    import app as app_module
    for arrangement, expected_model in (
        ("segmented", "Segmented"),
        ("series", "Stacked"),
        ("parallel", "Side by side"),
    ):
        pending = app_module.settings_from_arrangement(
            {"arrangement": arrangement,
             "composition": {"membrane": "freeze", "cyto_start": "break",
                             "break_1": 0.15, "break_2": 0.40,
                             "use_nucleus": True}},
            0.6,
        )
        check(f"{arrangement} maps to a real model name",
              pending["model_kind"].startswith(expected_model),
              pending["model_kind"])
        check(f"{arrangement} maps to a name the radio offers",
              pending["model_kind"] in app_module.MODEL_KEYS,
              pending["model_kind"])


def case_fitting_applies_what_it_found():
    print("pressing fit fits what is on the page, and moves nothing on it")
    app = start(cell_name="cell-01")
    work = button_by_label(app, "Fit this cell")
    check("the button is there", work is not None)
    if work is None:
        return
    before = {
        "segment_break_1": float(state(app, "segment_break_1")),
        "segment_break_2": float(state(app, "segment_break_2")),
        "membrane_after_break": state(app, "membrane_after_break"),
        "cyto_starts_at": state(app, "cyto_starts_at"),
        "use_membrane": state(app, "use_membrane"),
        "use_interior": state(app, "use_interior"),
        "use_nucleus": state(app, "use_nucleus"),
    }
    work.click().run()
    if not no_exception(app, "fitting"):
        return
    # The bug this catches: fitting used to compare arrangements and write
    # the winner's boundaries back, so pressing it moved ε₁ and ε₂ off the
    # numbers they had just been set to.
    after = {key: (float(state(app, key)) if key.startswith("segment")
                   else state(app, key)) for key in before}
    check("the boundaries are exactly where they were",
          after["segment_break_1"] == before["segment_break_1"]
          and after["segment_break_2"] == before["segment_break_2"],
          f"{before} -> {after}")
    check("and so is the arrangement",
          after["membrane_after_break"] == before["membrane_after_break"]
          and after["cyto_starts_at"] == before["cyto_starts_at"], str(after))
    check("and so are the components",
          all(after[k] == before[k] for k in
              ("use_membrane", "use_interior", "use_nucleus")), str(after))
    fit = state(app, "_last_fit")
    check("and the fit is the fit of those settings",
          fit and fit.get("success")
          and abs(float(fit["break_1"]) - before["segment_break_1"]) < 1e-6
          and abs(float(fit["break_2"]) - before["segment_break_2"]) < 1e-6,
          str((fit.get("break_1"), fit.get("break_2")) if fit else None))
    check("it is fitted as a hand-over, which is the only model with an order",
          state(app, "model_kind").startswith("Segmented"),
          state(app, "model_kind"))
    said = " ".join(str(m.value) for m in
                    list(app.get("markdown")) + list(app.get("success"))
                    + list(app.get("info")))
    check("and what it came to is said next to the button",
          "R²" in said, said[-300:])


def app_module_model_name(arrangement):
    import app as app_module
    return app_module.settings_from_arrangement(
        {"arrangement": arrangement, "composition": {}}, 0.6
    )["model_kind"]


def case_guided_range_is_settable():
    print("the range can be set at either end, by the bar or by typing")
    app = start(cell_name="cell-01")
    if not no_exception(app, "guided range"):
        return
    slider = widget_by_label(app, "slider", "Fitted range")
    check("a range bar sits before the button", slider is not None,
          str([s.label for s in app.slider]))
    if slider is None:
        return
    said = " ".join(str(m.value) for m in
                    list(app.get("markdown")) + list(app.get("caption")))
    check("it is labelled",
          "1 · Relative deformation range" in said, said[:200])
    check("and it comes before the components, since it decides which "
          "points exist at all",
          said.index("1 · Relative deformation range")
          < said.index("Components"), said[:200])
    check("and it has a handle at each end, not just at the far one",
          isinstance(slider.value, (list, tuple)) and len(slider.value) == 2,
          str(slider.value))
    check("with a box for each end beside it",
          widget_by_label(app, "number_input", "from") is not None
          and widget_by_label(app, "number_input", "to") is not None,
          str([n.label for n in app.number_input]))

    # The bar moves both ends at once.
    slider.set_value((0.10, 0.35)).run()
    if not no_exception(app, "narrowed range"):
        return
    check("the near end is where the bar was put",
          abs(app.session_state["window_start"] - 0.10) < 0.01,
          str(app.session_state["window_start"]))
    check("and so is the far end",
          abs(app.session_state["window_end"] - 0.35) < 0.01,
          str(app.session_state["window_end"]))

    # Typing does the same thing, and the bar follows.
    widget_by_label(app, "number_input", "from").set_value(0.0).run()
    if not no_exception(app, "typing the near end"):
        return
    check("typing the near end moves it",
          abs(app.session_state["window_start"]) < 1e-6,
          str(app.session_state["window_start"]))
    check("and leaves the far end alone",
          abs(app.session_state["window_end"] - 0.35) < 0.01,
          str(app.session_state["window_end"]))
    bar = widget_by_label(app, "slider", "Fitted range")
    check("the bar shows what was typed",
          bar is not None and abs(bar.value[0]) < 1e-6
          and abs(bar.value[1] - 0.35) < 0.01, str(bar.value))

    button_by_label(app, "Fit this cell").click().run()
    if not no_exception(app, "search over the chosen range"):
        return
    fit = app.session_state["_last_fit"]
    check("the fit stops where the range stopped",
          fit is not None and abs(fit["epsilon_range"][1] - 0.35) < 0.02,
          str(fit["epsilon_range"]) if fit else "no fit")


def case_typing_the_ends_the_wrong_way_round():
    print("typing the two ends the wrong way round does not break the page")
    app = start(cell_name="cell-01")
    if not no_exception(app, "range boxes"):
        return
    far = widget_by_label(app, "number_input", "to")
    if far is None:
        check("there is a box for the far end", False)
        return
    # Far end typed below the near end. The typed number is the one to
    # honour, so the near end gives way rather than the number snapping
    # back to where it was.
    widget_by_label(app, "number_input", "from").set_value(0.40).run()
    far = widget_by_label(app, "number_input", "to")
    far.set_value(0.20).run()
    if not no_exception(app, "ends typed the wrong way round"):
        return
    lo = app.session_state["window_start"]
    hi = app.session_state["window_end"]
    check("the range is still the right way round", lo < hi, f"{lo} to {hi}")
    check("and the number just typed is the one kept",
          abs(hi - 0.20) < 0.01, str(hi))


def case_search_maths_is_shown():
    print("the maths behind the button is available")
    # In full control: the guided page keeps to curve, numbers, maths,
    # database, and this is the working behind the search rather than the
    # working behind the fit.
    app = start(cell_name="cell-01", ui_mode="Full control · every setting")
    if not no_exception(app, "search maths"):
        return
    blocks = [str(b.value) for b in app.get("latex")]
    joined = " ".join(blocks)
    check("the cross-validation formula is shown",
          "mathrm{CV}" in joined, joined[-200:])
    check("the tie tolerance is shown", "tau" in joined, joined[-200:])
    check("the stacked form is shown", "F^{1/3}" in joined, joined[:200])
    text = " ".join(str(m.value) for m in app.get("markdown"))
    check("it says these are three arrangements",
          "three arrangements" in text, "not found")


def case_the_cortex_carries_the_start():
    print("a cardiomyocyte's cortex carries the load from first contact")
    import app as app_module

    defaults = app_module.DEFAULT_COMPOSITION_BY_TYPE["Cardiomyocyte"]
    # The cortex is the term loaded from ε = 0, so the scaffolding under it
    # has to wait for ε₁. Both from zero and they are one Hertzian term
    # with two names, split arbitrarily by the solver.
    check("the interior carries the load from first contact",
          defaults["cyto_starts_at"] == "from the very start", str(defaults))
    check("and the membrane starts stretching at ε₁ by default",
          defaults["membrane_after_break"] == "starts stretching at ε₁",
          str(defaults))

    # The physics that justifies it: something Hertzian loaded from contact
    # gives a slope near 3/2 there, where a membrane alone gives 3.
    from lulevich_model import LulevichModel as LM
    eps = np.linspace(0.001, 0.60, 300)
    g = LM(np.zeros_like(eps), eps, cell_height=19.0e-6, cell_radius=9.5e-6)
    basis = g.composition_basis(eps, 0.15, 0.40, "late", "break")
    near = (eps > 0.02) & (eps < 0.12)

    def slope(force):
        return float(np.polyfit(np.log(eps[near]), np.log(force[near]), 1)[0])

    cortex_only = basis["cortex"] * 2.0e3
    check("the cortex alone reads as a Hertzian contact",
          abs(slope(cortex_only) - 1.5) < 0.05, f"{slope(cortex_only):.2f}")
    shell_only = basis["membrane"] * 1.5e6
    membrane_first = g.composition_basis(
        eps, 0.15, 0.40, "continue", "break")["membrane"] * 1.5e6
    check("a membrane alone reads as a cube law",
          abs(slope(membrane_first) - 3.0) < 0.05,
          f"{slope(membrane_first):.2f}")
    check("and a late membrane contributes nothing at all near contact",
          float(np.max(shell_only[near])) == 0.0)


def case_schematic_is_a_mechanics_diagram():
    print("the diagram reads as a mechanics schematic")
    import plot_utils
    import app as app_module

    style = plot_utils.PlotStyle(force_unit="nN")
    fig = plot_utils.cell_schematic(
        style, epsilon=0.28, break_1=0.15, break_2=0.40,
        membrane_mode="freeze", cyto_start="break",
        Em_MPa=0.6, Ei_kPa=1.2, En_kPa=3.0,
        labels=app_module.components_for("Cardiomyocyte"),
    )
    text = " ".join(str(getattr(a, "text", "")) for a in fig.layout.annotations)
    check("the cantilever is drawn", "cantilever" in text, text[:120])
    check("the applied force is labelled", "<b>F</b>" in text, text[:120])
    check("the squash is stated", "ε = 0.280" in text, text[:160])
    # Wrapped captions can break inside a phrase, so compare with the line
    # breaks taken out rather than demanding one particular wrapping.
    flat = text.replace("<br>", " ")
    check("a component not yet reached shows its gap",
          "gap closes at ε = 0.40" in flat, flat[-200:])
    check("a component that handed over shows as locked",
          "locked" in text, text[-260:])
    check("it uses the cell type's own names",
          "Sarcolemma" in flat and "myofibrils" in flat.lower(),
          flat[-260:])

    # Springs must hang straight, not lean: a precedence bug once drew them
    # diagonally across the page.
    for trace in fig.data:
        xs = [v for v in (trace.x or []) if v is not None]
        if len(xs) > 5:
            check("the spring hangs straight",
                  abs(xs[0] - xs[-1]) < 1e-9 and max(xs) - min(xs) < 20,
                  f"x from {min(xs):.1f} to {max(xs):.1f}")
            break

    stacked = plot_utils.cell_schematic(
        style, epsilon=0.4, coupling="series", Em_MPa=0.6, Ei_kPa=1.2,
        En_kPa=3.0, labels=app_module.components_for("Myoblast (C2C12)"),
    )
    check("the stacked arrangement says so",
          "Stacked" in str(stacked.layout.title.text),
          str(stacked.layout.title.text))


def case_fit_only_plot():
    print("the fit can be plotted without the data")
    import plot_utils

    eps, force = synthetic()
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.60, 0.15, 0.40)
    mb, cb, nb = model.composition_terms(eps, 0.15, 0.40)
    fitted = mb * fit["Em"] + cb * fit["Ei"] + nb * fit["En"]

    def names(**flags):
        style = plot_utils.PlotStyle(force_unit="N", **flags)
        fig = plot_utils.force_curve_figure(eps, force, style, fit_force_N=fitted)
        # The model is drawn twice, a white halo under a coloured line, so
        # that it reads over a dense band of points. Only one of them is a
        # trace anybody is meant to see named.
        return [t.name for t in fig.data if t.showlegend is not False]

    check("both by default",
          set(names()) == {"Experimental data", "Model"}, str(names()))
    check("data off leaves only the model",
          names(show_data=False) == ["Model"], str(names(show_data=False)))
    check("fit off leaves only the data",
          names(show_fit_line=False) == ["Experimental data"],
          str(names(show_fit_line=False)))

    app = start(cell_name="cell-01")
    boxes = [c.label for c in app.checkbox
             if "measured points" in (c.label or "").lower()
             or "fitted curve" in (c.label or "").lower()]
    check("no checkbox can take the data or the curve off the plot",
          not boxes, str(boxes))
    check("and both are drawn",
          state(app, "show_data_and_fit") is True,
          str(state(app, "show_data_and_fit")))


def case_springs_share_a_pitch():
    print("springs can be compared because their coils match")
    from plot_utils import _zigzag, COIL_PITCH

    def pitch_of(y_bottom, y_top):
        xs, ys = _zigzag(50, y_bottom, y_top, 12)
        turns = sum(
            1 for a, b, c in zip(xs, xs[1:], xs[2:])
            if (b - a) * (c - b) < 0
        )
        return (max(ys) - min(ys)) / max(turns, 1)

    tall, short = pitch_of(10, 70), pitch_of(30, 70)
    check("a short spring keeps the pitch of a tall one",
          abs(tall - short) / tall < 0.25, f"{tall:.2f} vs {short:.2f}")
    check("the pitch is near the declared one",
          abs(tall - COIL_PITCH) / COIL_PITCH < 0.6, f"{tall:.2f} vs {COIL_PITCH}")
    check("a short element gets fewer coils, not tighter ones",
          len(_zigzag(50, 30, 70, 12)[0]) < len(_zigzag(50, 10, 70, 12)[0]))


def case_start_a_new_cell():
    print("a new cell clears the last one but keeps the setup")
    app = start(cell_name="cell-01", cell_notes="first cell")
    check("there is a fit to clear", app.session_state["_last_fit"] is not None)
    app.session_state["fit_color"] = "#123456"
    app.session_state["cell_height_um"] = 11.5
    app.run()

    button = button_by_label(app, "Start a new cell")
    check("the button is there", button is not None)
    if button is None:
        return
    button.click().run()
    if not no_exception(app, "start a new cell"):
        return

    for key in ("data", "_last_fit", "arrangement_search", "video_saved_frame"):
        check(f"{key} was cleared", app.session_state[key] is None,
              str(app.session_state[key])[:40])
    check("the name was cleared", app.session_state["cell_name"] == "",
          repr(app.session_state["cell_name"]))
    check("the notes were cleared", app.session_state["cell_notes"] == "")
    check("display settings survive", app.session_state["fit_color"] == "#123456")
    check("the geometry survives", app.session_state["cell_height_um"] == 11.5)


def case_manual_cell_and_probe_scale():
    print("the cell can be drawn by hand and the probe sets the scale")
    try:
        import cv2
        import video_analysis as va
    except Exception as exc:
        print(f"  skip (no OpenCV: {exc})")
        return

    frame = np.full((300, 420, 3), 150, np.uint8)
    det = va.manual_detection(frame, (0.25, 0.30, 0.55, 0.70))
    check("a hand-drawn box is a detection", det["found"] and det["manual"])
    check("its height is the box height",
          abs(det["height_px"] - 0.40 * 300) < 2, str(det["height_px"]))
    check("its width is the box width",
          abs(det["width_px"] - 0.30 * 420) < 2, str(det["width_px"]))
    check("a reversed box is still read correctly",
          va.manual_detection(frame, (0.55, 0.70, 0.25, 0.30))["bbox"]
          == det["bbox"])

    scale, detail = va.scale_from_probe((0.10, 0, 0.60, 0), frame.shape, 60.0)
    check("the probe gives a scale", scale is not None)
    if scale:
        check("and it is right",
              abs(scale - 60.0 / (0.5 * 420)) < 1e-9, f"{scale:.5f}")
        check("the detail says how it was worked out",
              "µm per pixel" in detail, detail)
        # The whole point: pixels become micrometres.
        check("the cell height converts to micrometres",
              abs(det["height_px"] * scale - 120 * (60.0 / 210)) < 0.5,
              f"{det['height_px'] * scale:.2f} µm")

    check("too narrow a probe box is refused",
          va.scale_from_probe((0.1, 0, 0.1, 0), frame.shape, 60.0)[0] is None)
    check("a missing width is refused",
          va.scale_from_probe((0.1, 0, 0.6, 0), frame.shape, 0)[0] is None)

    # The switches live inside the video tab, which only draws its controls
    # once a video is loaded, so AppTest cannot reach them from a bare
    # start(). Check that the app is wired to them instead.
    src = pathlib.Path(__file__).with_name("app.py").read_text()
    check("the manual switch is wired up",
          'key="video_manual_cell"' in src)
    check("the probe switch is wired up",
          'key="video_use_probe_scale"' in src)
    check("the app calls the hand-drawn detector",
          "va.manual_detection(" in src)
    check("the app calls the probe scale",
          "va.scale_from_probe(" in src)

    app = start(cell_name="cell-01")
    check("the manual switch starts off",
          app.session_state["video_manual_cell"] is False)
    check("the probe switch starts off",
          app.session_state["video_use_probe_scale"] is False)


def case_zero_modulus_explains_itself():
    print("a membrane driven to zero says why")
    eps = np.linspace(0.001, 0.60, 300)
    g = LulevichModel(np.zeros_like(eps), eps, cell_height=12.0e-6)
    mb, cb, nb = g.composition_terms(eps, 0.15, 0.40, "continue", "zero")
    force = mb * 1.4e6 + cb * 3.1e3 + nb * 9.0e3
    model = LulevichModel(force, eps, cell_height=12.0e-6)

    right = model.fit_composition(0.0, 0.60, 0.15, 0.40, "continue", "zero")
    check("the right combination recovers the membrane",
          abs(right["Em_MPa"] - 1.4) / 1.4 < 0.1, f"{right['Em_MPa']:.4f} MPa")

    wrong = model.fit_composition(0.0, 0.60, 0.15, 0.40, "freeze", "zero")
    check("holding plus starting at zero kills the membrane",
          wrong["Em_MPa"] <= 0, f"{wrong['Em_MPa']:.4f} MPa")

    app = start(cell_name="cell-01", cell_type="Cardiomyocyte",
                membrane_after_break="holds what it reached",
                cyto_starts_at="from the very start")
    if not no_exception(app, "zero membrane"):
        return
    warnings = " ".join(str(w.value) for w in app.warning)
    if any("zero" in warnings for _ in [1]) and "Eₘ" in warnings:
        check("the app says the two choices cancel",
              "cancelling each other" in warnings, warnings[:200])
        check("and names the fix",
              "keeps stiffening" in warnings, warnings[:260])
    else:
        print("       (this curve did not zero the membrane; logic tested above)")


def case_fit_quality():
    print("the fit actually follows the data")
    for membrane, cyto in (("freeze", "break"), ("continue", "zero")):
        eps, force = synthetic(membrane, cyto)
        model = LulevichModel(force, eps, cell_height=8.0e-6)
        fit = model.fit_composition(0.0, 0.60, 0.15, 0.40, membrane, cyto)
        check(f"R² > 0.99 for {membrane}/{cyto}",
              fit["success"] and fit["r_squared"] > 0.99,
              f"R²={fit.get('r_squared')}")
        check(f"Eₘ recovered for {membrane}/{cyto}",
              abs(fit["Em_MPa"] - 0.6) / 0.6 < 0.15, f"{fit['Em_MPa']:.3f} MPa")
        check(f"E_c recovered for {membrane}/{cyto}",
              abs(fit["Ei_kPa"] - 1.2) / 1.2 < 0.20, f"{fit['Ei_kPa']:.3f} kPa")


def four_element_curve(T0=1.2e-3, seed=3, noise=0.003, n=500):
    """A cardiomyocyte curve with all four springs in it."""
    eps = np.linspace(0.002, 0.65, n)
    geometry = dict(cell_height=14.0e-6, shell_thickness=200e-9,
                    deep_uses_cell_radius=True)
    blank = LulevichModel(np.zeros_like(eps), eps, **geometry)
    basis = blank.composition_basis(eps, 0.15, 0.40, "continue", "zero")
    force = (
        basis["tension"] * T0
        + basis["membrane"] * 1.4e6
        + basis["interior"] * 3.1e3
        + basis["nucleus"] * 9.0e3
    )
    rng = np.random.default_rng(seed)
    noisy = force + rng.normal(0.0, noise * force.max(), force.size)
    return eps, noisy, LulevichModel(noisy, eps, **geometry), geometry


def case_four_element_model():
    print("the membrane is two springs and the fit can tell them apart")
    eps, force, model, geometry = four_element_curve()

    blank = LulevichModel(np.zeros_like(eps), eps, **geometry)
    basis = blank.composition_basis(eps, 0.15, 0.40, "continue", "zero")
    check("there are six basis functions", len(basis) == 6, str(sorted(basis)))

    # The two membrane laws must not be the same shape, or the split between
    # them is arbitrary and the numbers wander from cell to cell.
    a = basis["tension"] / max(basis["tension"].max(), 1e-30)
    b = basis["membrane"] / max(basis["membrane"].max(), 1e-30)
    check("the two membrane springs have different shapes",
          abs(np.corrcoef(a, b)[0, 1]) < 0.95,
          f"correlation {np.corrcoef(a, b)[0, 1]:.4f}")
    # And the crossover is the point of the pair: the taut network answers
    # first, the shell's elasticity takes over once the strain is large.
    taut = basis["tension"] * 1.2e-3
    elastic = basis["membrane"] * 1.4e6
    check("the taut one leads near first contact",
          taut[5] > elastic[5] * 10, f"{taut[5]:.3g} vs {elastic[5]:.3g}")
    check("the elastic one leads deep in",
          elastic[-1] > taut[-1], f"{elastic[-1]:.3g} vs {taut[-1]:.3g}")

    fit = model.fit_composition(
        0.0, 0.65, 0.15, 0.40, "continue", "zero", use_tension=True
    )
    check("the four-term fit succeeds", fit.get("success"))
    check("T0 is recovered", abs(fit["T0_mN_m"] - 1.2) / 1.2 < 0.25,
          f"{fit['T0_mN_m']:.3f} mN/m")
    check("Em is recovered", abs(fit["Em_MPa"] - 1.4) / 1.4 < 0.12,
          f"{fit['Em_MPa']:.3f} MPa")
    check("Ec is recovered", abs(fit["Ei_kPa"] - 3.1) / 3.1 < 0.20,
          f"{fit['Ei_kPa']:.3f} kPa")
    check("En is recovered", abs(fit["En_kPa"] - 9.0) / 9.0 < 0.15,
          f"{fit['En_kPa']:.3f} kPa")
    check("it says which terms it used",
          set(fit["terms"]) == {"tension", "membrane", "interior", "nucleus"},
          str(fit["terms"]))
    check("the breakpoints still count as parameters", fit["n_params"] == 6,
          str(fit["n_params"]))

    # A tension is a force per length. Turning it into a modulus needs the
    # coat thickness, and the model must not pretend otherwise.
    check("the tension prefactor has no thickness in it",
          abs(model.At - 2 * np.pi * model.R0 ** 2 / model.cell_height)
          < 1e-12 * model.At)
    check("the equivalent modulus divides by the coat",
          abs(fit["T0_as_modulus_kPa"]
              - (fit["T0_mN_m"] * 1e-3 / 200e-9) / 1e3) < 1e-6)

    # Switching the spring off must leave the classic model exactly as it was.
    three = model.fit_composition(0.0, 0.65, 0.15, 0.40, "continue", "zero")
    check("the classic three-term fit still runs", three.get("success"))
    check("and reports no tension", three.get("T0_mN_m", 0.0) == 0.0)
    check("and is a term shorter", three["n_params"] == 5, str(three["n_params"]))


def wt_cardiomyocyte():
    """The measured WT cardiomyocyte curve, thinned but not smoothed.

    20230713_WT_2, 3.3 N/m cantilever, 2 um/s, 19 um cell. Kept in the repo
    because a model that only ever meets curves this code generated has not
    met anything: every synthetic curve in this file was built from the same
    basis functions the fit uses, so it can only ever confirm the arithmetic.
    This one was measured, and it is the reason the confinement term exists.
    """
    path = pathlib.Path(__file__).with_name("reference_WT_cardiomyocyte.csv")
    if not path.exists():
        return None, None
    frame = pd.read_csv(path)
    return (frame["relative_deformation"].to_numpy(float),
            frame["force_N"].to_numpy(float))


def wt_model(q=1.25):
    """That curve with the cardiomyocyte geometry the app now defaults to."""
    eps, force = wt_cardiomyocyte()
    if eps is None:
        return None
    return LulevichModel(
        force, eps, cell_height=19.0e-6, cell_radius=9.5e-6,
        membrane_thickness=8.0e-9, shell_thickness=200e-9,
        deep_uses_cell_radius=True, sarcomere_length=2.1e-6, confinement=q,
    )


def vcm_curves():
    """The four corrected WT ventricular cardiomyocyte curves, thinned.

    Real measurements, and the reason the cardiomyocyte model looks the way
    it does. Cell 3 carries a bad contact near ε = 0.2, which is not a defect
    in the file: it is what the automatic range picker exists to find.
    """
    path = pathlib.Path(__file__).with_name("reference_WT_vcm_four.csv")
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out = {}
    for n in (3, 5, 11, 14):
        eps = frame[f"eps_{n}"].to_numpy(float)
        force = frame[f"force_{n}"].to_numpy(float)
        good = np.isfinite(eps) & np.isfinite(force)
        out[n] = (eps[good], force[good])
    return out


def vcm_model(eps, force, q=1.10):
    return LulevichModel(
        force, eps, cell_height=19.0e-6, cell_radius=9.5e-6,
        membrane_thickness=8.0e-9, deep_uses_cell_radius=True,
        sarcomere_length=2.1e-6, confinement=q,
    )


def case_component_heights_survive_a_log_axis():
    print("component height labels do not blank the plot on a log axis")
    import plot_utils
    eps, force = synthetic()
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.60, 0.15, 0.40)
    basis = model.composition_basis(eps, 0.15, 0.40, "freeze", "break")
    fitted = (basis["membrane"] * fit["Em"] + basis["interior"] * fit["Ei"]
              + basis["nucleus"] * fit["En"])

    def build(log):
        style = plot_utils.PlotStyle(
            force_unit="nN", show_components=True,
            show_component_heights=True, log_scale=log,
        )
        return plot_utils.force_curve_figure(
            eps, force, style, fit_force_N=fitted,
            membrane_N=basis["membrane"] * fit["Em"],
            interior_N=basis["interior"] * fit["Ei"],
            nucleus_N=basis["nucleus"] * fit["En"],
        )

    linear, logged = build(False), build(True)

    def heights(figure):
        return [
            (float(a.y), a.text) for a in figure.layout.annotations
            if a.text and a.text.startswith("<b>") and "N" in a.text
        ]

    flat, curved = heights(linear), heights(logged)
    check("labels are drawn on a linear axis", len(flat) >= 3, str(len(flat)))
    check("and on a log one too", len(curved) >= 3, str(len(curved)))

    # Plotly places annotations in axis coordinates, and on a log axis those
    # are log10 of the value. Passing 1.6e-8 asks for 10^(1.6e-8), which is
    # 1 — off the top of any real curve, and enough to blank the chart.
    for y, text in curved:
        check(f"{text} sits at log10 of its value, not at the value itself",
              -12 < y < 6, f"y = {y:.4g}")
    pairs = dict(flat)
    for y, text in curved:
        if text in pairs:
            check(f"{text} is exactly log10 of the linear position",
                  abs(y - np.log10(pairs[text])) < 1e-9,
                  f"{y:.6g} vs {np.log10(pairs[text]):.6g}")

    # A value at or below zero has no place on a log axis, and must be
    # dropped rather than drawn at minus infinity.
    style = plot_utils.PlotStyle(force_unit="nN", show_components=True,
                                 show_component_heights=True, log_scale=True)
    zeroed = plot_utils.force_curve_figure(
        eps, force, style, fit_force_N=np.zeros_like(fitted),
        membrane_N=basis["membrane"] * fit["Em"],
    )
    check("a zero height is left off a log axis rather than drawn at -inf",
          all(np.isfinite(float(a.y)) for a in zeroed.layout.annotations))


def case_the_tab_is_stripped_back():
    print("the analysis tab is down to the work, and the extras are off the bar")
    import app as app_module
    source = pathlib.Path(__file__).with_name("app.py").read_text()
    check("the video tab is off the bar for now",
          app_module.SHOW_VIDEO_TAB is False)
    check("and the database tab too",
          app_module.SHOW_DATABASE_TAB is False)
    # Hidden, not deleted: the code that fills them still runs, so it cannot
    # rot while it is out of sight.
    check("but their code still runs",
          "with tab_video:" in source and "with tab_db:" in source)
    check("they are hidden by hiding their buttons",
          'button[data-baseweb="tab"]' in source)

    app = start(cell_name="cell-01")
    if not no_exception(app, "stripped tab"):
        return
    labels = [e.label or "" for e in app.get("expander")]
    for gone in ("Explore the curve",):
        check(f"“{gone}” is not in the guided flow",
              not any(gone in l for l in labels), str(labels))


def case_sharing_controls_sit_with_the_parts():
    print("how they share the load is chosen where the parts are chosen")
    app = start(cell_name="cell-01")
    if not no_exception(app, "sharing controls"):
        return
    headings = [
        str(m.value).strip() for m in app.get("markdown")
        if str(m.value).strip().startswith("####")
    ]
    check("the materials and the range are chosen in one place",
          any("1 · Relative deformation range" in h for h in headings), str(headings))

    radios = [r.label for r in app.get("radio")]
    for wanted in ("How the cell is modelled", "After ε₁ the membrane…",
                   "The cytoskeleton starts…"):
        check(f"“{wanted[:28]}…” is on the page", wanted in radios, str(radios))

    source = pathlib.Path(__file__).with_name("app.py").read_text()
    check("and a diagram is drawn from those choices, next to them",
          'key="sharing_preview"' in source)
    check("there is no separate load-sharing section left",
          source.count('open_panel(\n            "⚙️ How the elements share') <= 1)

    # Changing a sharing choice must change the diagram, or putting them
    # together achieves nothing.
    import plot_utils
    style = plot_utils.PlotStyle()

    def caption_for(term, cyto_start, at):
        figure = plot_utils.cell_schematic(
            style, epsilon=at, break_1=0.15, break_2=0.40,
            membrane_mode="continue", cyto_start=cyto_start,
            Em_MPa=1.0, Ei_kPa=1.0, En_kPa=1.0,
        )
        wanted = {"membrane": "Membrane", "interior": "Cytoskeleton",
                  "nucleus": "Nucleus"}[term]
        for a in figure.layout.annotations:
            text = str(getattr(a, "text", ""))
            if text.startswith(f"<b>{wanted}</b>"):
                return text
        return ""

    # Below ε₁ the two choices differ; above it they cannot, because the
    # cytoskeleton is loading either way. That is exactly why the preview
    # has a slider rather than one fixed deformation.
    early_break = caption_for("interior", "break", 0.08)
    early_zero = caption_for("interior", "zero", 0.08)
    check("below ε₁, starting at ε₁ draws a gap under the cytoskeleton",
          "gap closes" in early_break, early_break)
    check("and starting at zero does not",
          "gap closes" not in early_zero, early_zero)
    late_break = caption_for("interior", "break", 0.30)
    check("above ε₁ the two look the same, as they should",
          "gap closes" not in late_break, late_break)

    # The preview is drawn just past the first boundary, where every choice
    # is visible at once. It used to have a slider of its own; one fewer
    # widget on a crowded page is worth more than sweeping it by hand.
    check("the preview is drawn where the choices show",
          "preview_at = float(np.clip(" in SOURCE)


def case_boundaries_are_checked_against_the_power_law():
    print("the boundary scan is checked against the curve's own slope")
    app = start(cell_name="cell-01", ui_mode="Full control · every setting")
    button = button_by_label(app, "Find the boundaries from the data")
    check("the button is there", button is not None)
    if button is None:
        return
    button.click().run()
    if not no_exception(app, "boundary scan"):
        return
    try:
        profile = app.session_state["_power_law_check"]
    except Exception:
        profile = None
    check("a power-law profile was measured", profile is not None
          and len(profile["epsilon"]) > 10,
          str(len(profile["epsilon"]) if profile else None))
    said = " ".join(str(c.value) for c in app.get("caption"))
    check("and reported next to the boundaries it chose",
          "log-log slope" in said, said[:160])
    check("with what the numbers mean",
          "3 is a membrane on its own" in said, said[:200])
    # The two are arrived at differently on purpose: one asks which
    # boundaries fit best, the other where the curve's shape changes.
    check("the scan and the slope are separate measurements",
          "scan_segment_breaks(" in pathlib.Path(__file__)
          .with_name("app.py").read_text()
          and "local_exponent(" in pathlib.Path(__file__)
          .with_name("app.py").read_text())


def case_the_curve_comes_first():
    print("the curve is at the top, and the diagram is with its own question")
    app = start(cell_name="cell-01")
    if not no_exception(app, "layout"):
        return
    source = pathlib.Path(__file__).with_name("app.py").read_text()

    # The curve is drawn into a container staked out before the settings,
    # so it appears above them however far down the code that builds it is.
    # The plot is staked out after the choices, so the page reads choose,
    # fit, look rather than look, scroll, choose.
    check("the curve is staked out after the choices",
          source.index("curve_slot = st.container()")
          > source.index('st.markdown("#### 1 · Relative deformation range")'),
          "curve_slot is too early")
    check("and the plot is drawn into it",
          "plot_col = curve_slot" in source)
    check("the diagram goes with the question it answers, not beside the plot",
          'key="sharing_preview"' in source
          and "panel_cols = []" in source)
    check("and the video frame goes under the curve",
          "video_target = video_slot" in source)

    check("the fit button sits with the choices",
          button_by_label(app, "Fit this cell") is not None,
          str([b.label for b in app.button]))
    check("and there is exactly one fit button in guided mode",
          source.count('"🔬 Fit this cell"') == 1,
          str(source.count('"🔬 Fit this cell"')))

    # Explore the curve is gone from the guided flow.
    labels = [e.label or "" for e in app.get("expander")]
    check("Explore the curve is not in the guided flow",
          not any("Explore the curve" in l for l in labels), str(labels))
    # There is one page now, so it is gone for good: the boundaries are
    # found by the fit and moved by the two sliders in the sidebar, and a
    # second, manual way of finding them was a section to scroll past.
    check("and not reachable by a mode switch either, because there is none",
          "How much to show" not in source, "the mode radio is still there")

    # The three tables that were asked to go.
    for gone in ("The pictures it compared", "How the arrangements compared",
                 "Every combination it tried"):
        check(f"“{gone}…” is gone", gone not in source, gone)


def case_elements_carry_emojis_but_tables_do_not():
    print("emojis where you choose, plain names where you read")
    import app as app_module
    for cell_type in ("Myoblast (C2C12)", "Cardiomyocyte"):
        for term in app_module.terms_for(cell_type):
            name = app_module.term_name(term, cell_type)
            plain = app_module.plain_name(term, cell_type)
            check(f"{cell_type} {term} has an emoji", plain != name, name)
            check(f"and a plain form under it", plain and plain[0].isalpha(),
                  plain)
    check("two cell types do not share an emoji for the same slot",
          app_module.term_name("membrane", "Myoblast (C2C12)")
          != app_module.term_name("membrane", "Cardiomyocyte"))

    app = start(cell_name="cell-01")
    if not no_exception(app, "emoji labels"):
        return
    ticks = [c.label for c in app.checkbox if bare(c.label) != (c.label or "")]
    check("the checkboxes carry them", len(ticks) >= 3, str(ticks))
    tiles = [m.label for m in app.get("metric")
             if m.label.startswith(("Eₘ ", "Ec ", "Eₙ ", "T₀ "))]
    check("the modulus tiles do not", tiles and all(bare(t) == t for t in tiles),
          str(tiles))
    table = table_with(app, "range", "membrane")
    if table is not None:
        check("nor do the table headers",
              all(bare(c) == c for c in table.columns), str(list(table.columns)))


def case_bending_is_the_same_column_as_the_spring():
    print("a bending term is the in-plane spring under another name")
    eps, force, model, _ = four_element_curve()
    check("the model can state the bending prefactor", model.Ab > 0)
    check("and it is far smaller than the tension one",
          model.At > model.Ab * 1e6, f"{model.At / model.Ab:.4g}")

    # Both laws are linear in eps, so they are the same column: whatever one
    # can fit, the other fits identically, with a rescaled coefficient.
    basis = model.composition_basis(np.linspace(0.01, 0.6, 60), 0.15, 0.40,
                                    "continue", "zero")
    spring = basis["tension"]
    bending = spring * (model.Ab / model.At)
    ratio = bending / np.maximum(spring, 1e-300)
    check("one is an exact multiple of the other",
          float(np.ptp(ratio)) < 1e-12, f"{float(np.ptp(ratio)):.3g}")

    fit = model.fit_composition(0.0, 0.65, 0.15, 0.40, "continue", "zero",
                                use_tension=True)
    check("so the fit reports both readings of the same number",
          np.isfinite(fit["T0_as_bending_MPa"]) and fit["T0_as_bending_MPa"] > 0,
          str(fit.get("T0_as_bending_MPa")))
    check("and they are related by exactly At/Ab",
          abs(fit["T0_as_bending_MPa"] * 1e6
              - fit["T0"] * model.At / model.Ab) < 1e-6,
          f"{fit['T0_as_bending_MPa']:.6g}")

    source = pathlib.Path(__file__).with_name("app.py").read_text()
    check("and the app says so where the maths is shown",
          "**Bending.**" in source)


def case_cortical_actin_can_carry_it_first():
    print("the cortical network can carry the load before the membrane stretches")
    from lulevich_model import compare_hypotheses
    import app as app_module
    eps = np.linspace(0.002, 0.65, 400)
    blank = LulevichModel(np.zeros_like(eps), eps, cell_height=19.0e-6,
                          cell_radius=9.5e-6, membrane_thickness=8.0e-9,
                          deep_uses_cell_radius=True, confinement=1.1)

    # The membrane's cube law measured from ε₁, not from first contact.
    late = blank.composition_basis(eps, 0.09, 0.42, "late", "zero")
    early = blank.composition_basis(eps, 0.09, 0.42, "continue", "zero")
    check("a late membrane contributes nothing before ε₁",
          float(np.max(late["membrane"][eps < 0.09])) == 0.0)
    check("and something after it",
          float(late["membrane"][-1]) > 0)
    check("while a membrane loading from the start does contribute before it",
          float(np.max(early["membrane"][eps < 0.09])) > 0)
    check("the late one is always the softer of the two",
          np.all(late["membrane"] <= early["membrane"] + 1e-30))
    check("and it still only ever stiffens",
          np.all(np.diff(late["membrane"]) >= -1e-30))

    # Near contact a late membrane must read as a Hertzian network alone.
    force = (late["membrane"] * 1.4e6 + late["interior"] * 2.0e3
             + late["nucleus"] * 4.0e3)
    good = (eps > 0.02) & (eps < 0.08)
    slope = float(np.polyfit(np.log(eps[good]), np.log(force[good]), 1)[0])
    check("which reads as a slope near 3/2, not 3, near contact",
          1.3 < slope < 2.2, f"{slope:.2f}")

    # The evidence for it is the real curves, not a synthetic. A synthetic
    # built from these same basis functions is too forgiving: the other
    # terms absorb a late membrane and every picture ties.
    curves = vcm_curves()
    if not curves:
        print("  skip (no reference curves)")
        return
    # With a cortex in the model the start of the curve is carried by the
    # cortex whichever order wins, so which of the two the curve prefers is
    # no longer the same question it was. What must hold is that both are
    # offered, that the winner fits, and that ε₁ lands inside the curve
    # rather than at either end, which is what a real hand-over looks like.
    fitted = 0
    for n in (11, 14):
        if n not in curves:
            continue
        eps_n, force_n = curves[n]
        model = vcm_model(eps_n, force_n)
        window = model.suggest_window()
        found = compare_hypotheses(
            model, window["epsilon_min"], window["epsilon_max"],
            app_module.cardiomyocyte_hypotheses(),
            weighting="relative", cv_repeats=1, n_grid=8, scan_q=True,
        )
        if not found.get("success"):
            continue
        fitted += 1
        rows = {r["key"]: r for r in found["candidates"]}
        check(f"cell {n} was offered both orders",
              {"interior_first", "coupled"} <= set(rows), str(sorted(rows)))
        best = found["best"]
        check(f"cell {n}: the winner follows the curve",
              best["r_squared"] > 0.9995, f"{best['r_squared']:.6f}")
        check(f"cell {n} hands over part-way in, not at either end",
              window["epsilon_min"] < best["break_1"] < window["epsilon_max"],
              f"ε₁ = {best['break_1']:.3f}")
        check(f"cell {n}: the interior is carrying load",
              best["fit"].get("Ei_kPa", 0.0) > 0.0,
              str(best["fit"].get("Ei_kPa")))
    check("both clean curves were fitted", fitted == 2, f"{fitted} of 2")

    check("and a late membrane is offered as a named picture",
          any(p["membrane"] == "late"
              for p in app_module.cardiomyocyte_hypotheses()))
    check("with a plain-words label rather than a crash",
          "stretch" in app_module.composition_label("late", "zero").lower(),
          app_module.composition_label("late", "zero"))
    check("and an unknown composition still gets words, not a KeyError",
          bool(app_module.composition_label("something-new", "zero")))


def case_the_cardiomyocyte_picture_holds_fluid():
    print("a cardiomyocyte is a balloon of fluid, with nothing called a nucleus")
    import plot_utils
    style = plot_utils.PlotStyle()
    labels = {
        "tension": ("Extra membrane protein", ""),
        "membrane": ("Membrane and cortex", ""),
        "interior": ("Non-sarcomeric cytoskeleton", ""),
        "nucleus": ("Sarcomeric myofibrils", ""),
    }

    fluid = plot_utils.balloon_figure(
        style, epsilon=0.5, cell_height_um=19.0, deep_onset=0.40,
        labels=labels, interior="fluid",
    )
    text = " ".join(str(getattr(a, "text", "")) for a in fluid.layout.annotations)
    check("it says the inside is fluid that does not compress",
          "does not compress" in text, text[:160])
    check("and the deep layer is named for what it is",
          "Sarcomeric myofibrils" in text)
    check("nothing in it is called a nucleus", "nucleus" not in text.lower(),
          text[:200])
    # A body at the centre would claim the model separates one, and it does
    # not: the myofibrils are drawn lying along the cell instead.
    check("there is no round body drawn at the centre",
          not any(sh.type == "circle" for sh in fluid.layout.shapes))
    lying = [
        t for t in fluid.data
        if t.y is not None and len(t.y) == 2 and t.y[0] == t.y[1]
    ]
    check("the myofibrils are drawn lying along the cell", len(lying) >= 3,
          str(len(lying)))

    # A myoblast keeps its spring and its nucleus, because it has both.
    spring = plot_utils.balloon_figure(
        style, epsilon=0.5, cell_height_um=8.0, deep_onset=0.40,
        interior="spring",
    )
    check("a myoblast still gets a spring inside",
          "spring inside" in str(spring.layout.title.text))
    # Not a bead at the centre any more: a shorter balloon with a spring of
    # its own, because that is what the model now says a nucleus is.
    closed = [
        tr for tr in spring.data
        if getattr(tr, "fill", None) == "toself" and tr.x is not None
    ]
    check("and a balloon of its own inside it", len(closed) >= 2,
          str(len(closed)))
    inner = plot_utils.balloon_figure(
        style, epsilon=0.5, cell_height_um=8.0, deep_onset=0.40,
        interior="spring", show_nucleus=True, show_nucleus_shell=True,
    )
    # One spring for the cell and one inside the nucleus. Two, not three:
    # the cytoskeleton is a single material, so it gets a single coil, moved
    # aside rather than split around the nucleus.
    coils = [
        tr for tr in inner.data
        if tr.x is not None and len(tr.x) > 20
        and getattr(tr, "fill", None) != "toself"
    ]
    check("with a spring drawn inside that one too", len(coils) == 2,
          str(len(coils)))

    import app as app_module
    check("a cardiomyocyte is drawn as the balloon by default",
          app_module.CELL_TYPES["Cardiomyocyte"]["schematic_style"]
          .startswith("Balloon"))
    check("and a myoblast as the schematic",
          app_module.CELL_TYPES["Myoblast (C2C12)"]["schematic_style"]
          .startswith("Mechanics"))


def case_schematic_labels_do_not_collide():
    print("four elements' labels do not land on top of each other")
    import plot_utils
    labels = {
        "tension": ("Extra membrane protein", ""),
        "membrane": ("Membrane and cortex", ""),
        "interior": ("Non-sarcomeric cytoskeleton", ""),
        "nucleus": ("Sarcomeric myofibrils", ""),
    }
    figure = plot_utils.cell_schematic(
        plot_utils.PlotStyle(), epsilon=0.30, labels=labels,
        show_tension=True, show_nucleus=True, Em_MPa=1.4, Ei_kPa=3.1,
        En_kPa=9.0, T0_mN_m=1.2, break_1=0.15, break_2=0.40,
        membrane_mode="continue", cyto_start="zero",
    )
    captions = [a for a in figure.layout.annotations if a.yanchor == "top"]
    check("every element gets a caption", len(captions) == 4, str(len(captions)))
    rows = sorted({round(float(a.y), 3) for a in captions})
    check("they are staggered onto two rows", len(rows) == 2, str(rows))
    check("with real space between the rows",
          abs(rows[1] - rows[0]) >= 10.0, f"{abs(rows[1] - rows[0]):.1f}")

    # Neighbours on the same row must be far apart; wrapped names keep each
    # caption narrow enough that they are.
    for row in rows:
        same = sorted(float(a.x) for a in captions if round(float(a.y), 3) == row)
        for left, right in zip(same, same[1:]):
            check("captions sharing a row are well separated",
                  right - left >= 30.0, f"{left:.1f} and {right:.1f}")
    check("long names are wrapped rather than run on",
          all("<br>" in a.text for a in captions
              if "Non-sarcomeric" in a.text or "Extra membrane" in a.text))
    check("and the figure makes room for the dropped row",
          figure.layout.yaxis.range[0] <= min(rows) - 12,
          f"{figure.layout.yaxis.range[0]} vs {min(rows)}")


def case_the_fit_never_softens():
    print("no combination of elements can make the cell soften under load")
    curves = vcm_curves()
    if not curves:
        print("  skip (no reference curves)")
        return
    eps, force = curves[11]
    model = vcm_model(eps, force, q=1.2)
    fit = model.fit_composition(0.0, 0.65, 0.15, 0.42, "continue", "zero",
                                use_tension=True, weighting="relative")
    basis = model.composition_basis(eps, 0.15, 0.42, "continue", "zero")
    predicted = (
        basis["membrane"] * fit["Em"] + basis["tension"] * fit["T0"]
        + basis["interior"] * fit["Ei"] + basis["nucleus"] * fit["En"]
    )
    order = np.argsort(eps)
    e, f = eps[order], predicted[order]
    check("the force never falls as the cell is squashed",
          np.all(np.diff(f) >= -1e-18), f"{np.min(np.diff(f)):.3g} N")
    slope = np.diff(f) / np.maximum(np.diff(e), 1e-12)
    check("and the stiffness never falls either",
          np.min(np.diff(slope)) >= -abs(np.max(slope)) * 1e-6,
          f"{np.min(np.diff(slope)):.3g}")

    # Each basis function on its own, which is where the guarantee comes from.
    for name, values in model.composition_basis(
        np.linspace(0.001, 0.7, 400), 0.15, 0.42, "continue", "zero"
    ).items():
        check(f"the {name} basis is non-decreasing",
              np.all(np.diff(values) >= -1e-18),
              f"{np.min(np.diff(values)):.3g}")
    check("and confinement can only stiffen, never soften",
          model.confinement >= 0
          and np.all(np.diff(model.confinement_factor(
              np.linspace(0.0, 0.9, 200))) >= 0))


def case_weighting_decides_where_the_fit_is_good():
    print("weighting is what trades the top of the curve against the bottom")
    from lulevich_model import composition_weights
    curves = vcm_curves()
    if not curves:
        print("  skip (no reference curves)")
        return
    eps, force = curves[11]

    flat = composition_weights(eps, force, "uniform")
    check("uniform weights everything the same", float(np.ptp(flat)) == 0.0)
    rel = composition_weights(eps, force, "relative")
    check("relative gives the small forces far more weight",
          rel[np.argmin(force)] > rel[np.argmax(force)] * 100,
          f"{rel[np.argmin(force)] / rel[np.argmax(force)]:.3g}")
    noisy = composition_weights(eps, force, "noise")
    check("noise weighting is finite everywhere", np.all(np.isfinite(noisy)))
    check("and never zero, which would drop a point entirely",
          np.all(noisy > 0))

    def miss_below(weighting, edge=0.10):
        model = vcm_model(eps, force, q=1.2)
        fit = model.fit_composition(0.0, 0.65, 0.15, 0.42, "continue", "zero",
                                    use_tension=True, weighting=weighting)
        e, y, _ = model._select(0.0, 0.65)
        basis = model.composition_basis(e, 0.15, 0.42, "continue", "zero")
        predicted = (
            basis["membrane"] * fit["Em"] + basis["tension"] * fit["T0"]
            + basis["interior"] * fit["Ei"] + basis["nucleus"] * fit["En"]
        )
        low = (e > 0.02) & (e < edge)
        return abs(float(np.mean(predicted[low] - y[low]) / np.mean(np.abs(y[low]))))

    uniform_miss, relative_miss = miss_below("uniform"), miss_below("relative")
    check("uniform misses the low-force end badly on a real curve",
          uniform_miss > 0.10, f"{100 * uniform_miss:.1f} %")
    check("and relative brings it back",
          relative_miss < uniform_miss * 0.7,
          f"{100 * uniform_miss:.1f} % -> {100 * relative_miss:.1f} %")

    import app as app_module
    check("so a cardiomyocyte is fitted with relative by default",
          app_module.CELL_TYPES["Cardiomyocyte"]["weighting"] == "relative")
    check("and a myoblast, whose curve spans far less, is not",
          app_module.CELL_TYPES["Myoblast (C2C12)"]["weighting"] == "uniform")


def case_the_whole_range_is_used_and_fits():
    print("the full range is fitted, and it fits, band by band")
    curves = vcm_curves()
    if not curves:
        print("  skip (no reference curves)")
        return
    for n, (eps, force) in curves.items():
        model = vcm_model(eps, force)
        window = model.suggest_window()
        # Everything from the start it chose to the end of the usable curve.
        check(f"cell {n} uses the whole usable range",
              window["epsilon_max"] >= float(eps.max()) - 0.02,
              f"{window['epsilon_max']:.3f} of {eps.max():.3f}")

        # The same path the app takes: measure the confinement, then let the
        # named pictures fit their own boundaries. Fitting at a guessed
        # boundary would be testing something the app never does.
        import app as app_module
        from lulevich_model import compare_hypotheses
        # Exactly what the app does: every picture profiled at its own
        # confinement, its own boundaries found at that q. Measuring q once
        # for all of them and fitting at a guessed boundary is testing
        # something the app stopped doing.
        picks_here = app_module.cardiomyocyte_hypotheses()
        found = compare_hypotheses(
            model, window["epsilon_min"], window["epsilon_max"], picks_here,
            weighting="relative", cv_repeats=1, n_grid=8, scan_q=True,
        )
        if not found.get("success"):
            check(f"cell {n} gets a picture", False)
            continue
        check(f"cell {n} needs a positive confinement",
              found["best"].get("confinement", 0.0) > 0.5,
              f"q = {found['best'].get('confinement', 0.0):.2f}")
        model.confinement = float(found["best"]["confinement"])
        fit = found["best"]["fit"]
        check(f"cell {n} reaches R² above 0.9999",
              fit["r_squared"] > 0.9999, f"{fit['r_squared']:.6f}")

        e, y, _ = model._select(window["epsilon_min"], window["epsilon_max"])
        # Every column the winner carried, rebuilt through the model rather
        # than by hand: a term left out of this sum reads as a fit that
        # misses the data, which is a test failing for the wrong reason.
        predicted = model.composition_curve(e, fit)
        worst = 0.0
        for lo_b, hi_b in ((0.05, 0.10), (0.10, 0.20), (0.20, 0.35),
                           (0.35, 0.50), (0.50, 1.01)):
            band = (e >= lo_b) & (e < hi_b)
            if band.sum() < 10:
                continue
            scale = float(np.mean(np.abs(y[band])))
            if scale <= 0:
                continue
            worst = max(worst, abs(float(np.mean(predicted[band] - y[band]) / scale)))
        # Above the first hundredth of the curve, where a few hundred pN of
        # contact-point error is the whole signal, it should track to a few
        # per cent everywhere rather than only at the top.
        check(f"cell {n} tracks the curve to better than 8 % throughout",
              worst < 0.08, f"worst band {100 * worst:.1f} %")


def case_real_vcm_curves_start_near_three_halves():
    print("the corrected VCM curves start near 1.7, not 3")
    curves = vcm_curves()
    if not curves:
        print("  skip (no reference curves)")
        return
    for n, (eps, force) in curves.items():
        good = (eps > 0.05) & (force > 0)
        e, f = eps[good], force[good]

        def slope(lo, hi):
            m = (e >= lo) & (e < hi)
            if m.sum() < 6:
                return float("nan")
            return float(np.polyfit(np.log(e[m]), np.log(f[m]), 1)[0])

        early, late = slope(0.15, 0.30), slope(0.50, 0.62)
        # 1.5 is a Hertzian network alone, 3 is a membrane alone. Between
        # them is both together, which is the claim the model makes.
        check(f"cell {n} starts between 3/2 and 3", 1.4 < early < 3.0,
              f"{early:.2f}")
        check(f"cell {n} is far steeper deep in", late > 3.3, f"{late:.2f}")


def case_the_range_is_picked_and_a_bad_contact_is_found():
    print("the range is chosen from the curve, artefacts and all")
    curves = vcm_curves()
    if not curves:
        print("  skip (no reference curves)")
        return
    for n, (eps, force) in curves.items():
        window = vcm_model(eps, force).suggest_window()
        check(f"cell {n} gets a window", window.get("success"), str(window))
        if not window.get("success"):
            continue
        check(f"cell {n} stops at or before the data ends",
              window["epsilon_max"] <= eps.max() + 1e-9)
        if n == 3:
            # The one with a probe that caught and let go near ε = 0.2.
            check("cell 3's bad contact is found",
                  window.get("bad_contact") is not None, str(window["why_start"]))
            check("and the fit is started after it",
                  window["epsilon_min"] > 0.2, f"{window['epsilon_min']:.3f}")
        else:
            check(f"cell {n} starts at first contact",
                  window["epsilon_min"] < 0.05, f"{window['epsilon_min']:.3f}")

    # And it matters: fitting through the artefact ruins the answer.
    eps, force = curves[3]
    model = vcm_model(eps, force, q=0.9)
    window = model.suggest_window()
    through = model.fit_composition(0.0, window["epsilon_max"], 0.15, 0.45,
                                    "continue", "zero")
    around = model.fit_composition(window["epsilon_min"], window["epsilon_max"],
                                   0.15, 0.45, "continue", "zero")
    check("fitting through the bad contact is visibly worse",
          around["r_squared"] > through["r_squared"],
          f"{through['r_squared']:.6f} -> {around['r_squared']:.6f}")
    # And not by a little: on the full 7290-point curve fitting through it
    # drove the membrane modulus to exactly zero. Thinned, it survives, but
    # every modulus still moves by more than the artefact is worth.
    check("and it moves the answer, not just the residual",
          abs(around["En_kPa"] - through["En_kPa"])
          > 0.5 * max(around["En_kPa"], through["En_kPa"]),
          f"En {through['En_kPa']:.3f} -> {around['En_kPa']:.3f} kPa")


def case_named_hypotheses_are_compared():
    print("the guess is a named picture of the cell, not a setting")
    import app as app_module
    from lulevich_model import compare_hypotheses
    curves = vcm_curves()
    if not curves:
        print("  skip (no reference curves)")
        return

    picks = app_module.hypotheses_for("Cardiomyocyte")
    check("there are named hypotheses for a cardiomyocyte", len(picks) >= 3)
    # The cortical network loading from first contact is not one of the
    # things that varies: it is tied to the membrane through the costameres
    # and the measured slope near contact says so. What the membrane does is
    # a real question, though — it can load with it from the start, or only
    # begin to stretch once the cell has been flattened enough to stretch it.
    check("the interior always carries the load from first contact",
          all(p["cyto_start"] == "zero" for p in picks),
          str([(p["key"], p["cyto_start"]) for p in picks]))
    check("and every picture carries the interior",
          all("interior" in p["terms"] for p in picks),
          str([p["terms"] for p in picks]))
    check("and the membrane either loads with it or starts stretching later",
          all(p["membrane"] in ("continue", "late") for p in picks),
          str([(p["key"], p["membrane"]) for p in picks]))
    check("one picture has the cortical network carrying it alone at first",
          any(p["membrane"] == "late" for p in picks))
    check("the first names the order in words",
          "first" in picks[0]["label"].lower(), picks[0]["label"])
    # The deep slot is the myofibrils, never a nucleus, and whether they are
    # reached at all inside the range is one of the things being compared.
    check("some pictures reach the myofibrils and some do not",
          any("nucleus" in p["terms"] for p in picks)
          and any("nucleus" not in p["terms"] for p in picks),
          str([p["terms"] for p in picks]))
    check("none of them offers an in-plane spring",
          not any("tension" in p["terms"] for p in picks))

    # hypotheses_for marks each picture with whether it matches the cell
    # type's prior, so compare the pictures themselves.
    def bare(specs):
        return [{k: v for k, v in spec.items() if k != "expected"}
                for spec in specs]

    check("the argument passed in is accepted and ignored for now",
          bare(app_module.cardiomyocyte_hypotheses("anything")) == bare(picks))

    for n, (eps, force) in curves.items():
        model = vcm_model(eps, force)
        window = model.suggest_window()
        found = compare_hypotheses(
            model, window["epsilon_min"], window["epsilon_max"], picks,
            weighting="relative", cv_repeats=1, n_grid=6,
        )
        check(f"cell {n} gets a verdict", found.get("success"))
        if not found.get("success"):
            continue
        check(f"cell {n} names its pick", bool(found["best"]["label"]))
        check(f"cell {n} keeps the interior carrying it from first contact",
              found["best"]["cyto_start"] == "zero",
              str(found["best"]["cyto_start"]))


def case_springs_are_round_and_the_balloon_exists():
    print("the springs are coils, and there is a balloon to look at instead")
    import plot_utils

    xs, ys = plot_utils._zigzag(50.0, 10.0, 70.0, 14.0)
    check("a spring is drawn with enough points to be smooth", len(xs) > 60,
          str(len(xs)))
    check("it hangs straight: it starts and ends on its own axis",
          abs(xs[0] - 50.0) < 1e-9 and abs(xs[-1] - 50.0) < 1e-9)
    check("it stays inside its width",
          max(abs(x - 50.0) for x in xs) <= 7.0 + 1e-9,
          f"{max(abs(x - 50.0) for x in xs):.3f}")
    inner = xs[len(xs) // 4: 3 * len(xs) // 4]
    check("and it really oscillates rather than zigzagging once",
          sum(1 for a, b in zip(inner, inner[1:]) if (a - 50) * (b - 50) < 0) > 3)

    style = plot_utils.PlotStyle()
    figure = plot_utils.balloon_figure(
        style, epsilon=0.35, cell_height_um=19.0, deep_onset=0.45,
        show_tension=True,
    )
    check("the balloon renders", len(figure.data) >= 2)
    text = " ".join(str(getattr(a, "text", "")) for a in figure.layout.annotations)
    check("it says how far the cell was squashed", "ε = 0.350" in text, text[:120])
    check("and that it spreads because it keeps its volume",
          "keeps its volume" in text, text[:200])

    # A squashed balloon must actually get wider, not just shorter.
    def width_of(eps):
        fig = plot_utils.balloon_figure(style, epsilon=eps, cell_height_um=19.0)
        outline = fig.data[0]
        return max(outline.x) - min(outline.x)

    check("squashing it makes it wider", width_of(0.5) > width_of(0.0),
          f"{width_of(0.0):.1f} -> {width_of(0.5):.1f}")

    # The deeper element is drawn as not yet reached below its onset.
    early = plot_utils.balloon_figure(style, epsilon=0.10, deep_onset=0.45)
    late = plot_utils.balloon_figure(style, epsilon=0.60, deep_onset=0.45)
    check("and the deeper layer says when it has not been reached",
          "not reached yet" in " ".join(
              str(getattr(a, "text", "")) for a in early.layout.annotations)
          and "not reached yet" not in " ".join(
              str(getattr(a, "text", "")) for a in late.layout.annotations))


def case_switching_cell_type_and_back_changes_nothing():
    print("a myoblast fitted after a cardiomyocyte is still a myoblast")
    app = start(cell_name="myo-01")
    if not no_exception(app, "myoblast first"):
        return
    before = app.session_state["_last_fit"]
    check("it fits to start with", before and before.get("success"))
    if not before:
        return

    picker = widget_by_label(app, "selectbox", "Cell type")
    check("the cell type can be changed", picker is not None)
    if picker is None:
        return
    picker.set_value("Cardiomyocyte").run()
    if not no_exception(app, "switched to cardiomyocyte"):
        return
    check("the cardiomyocyte geometry is applied",
          app.session_state["cell_height_um"] == 19.0,
          str(app.session_state["cell_height_um"]))

    widget_by_label(app, "selectbox", "Cell type").set_value(
        "Myoblast (C2C12)"
    ).run()
    if not no_exception(app, "switched back"):
        return

    # Geometry going back is the easy half. The composition is the half that
    # was silently left behind: a myoblast was then fitted as though its
    # membrane kept stiffening and its cytoskeleton loaded from zero, which
    # still fits, still looks good, and gives the wrong modulus for every
    # element.
    check("the geometry comes back",
          app.session_state["cell_height_um"] == 8.0
          and app.session_state["confinement"] == 0.0
          and app.session_state["membrane_thickness_nm"] == 4.0,
          f"h {app.session_state['cell_height_um']} "
          f"q {app.session_state['confinement']} "
          f"hm {app.session_state['membrane_thickness_nm']}")
    # A C2C12 carries its sarcolemma and cytoskeleton throughout, which is
    # its own composition default, not the app's blank one.
    check("and so does the composition",
          app.session_state["membrane_after_break"] == "keeps stiffening"
          and app.session_state["cyto_starts_at"] == "from the very start",
          f"{app.session_state['membrane_after_break']} / "
          f"{app.session_state['cyto_starts_at']}")
    check("the cardiomyocyte's extra spring does not follow it home",
          app.session_state["use_tension"] is False)
    check("nor does its measured confinement",
          float(app.session_state["confinement"]) == 0.0,
          str(app.session_state["confinement"]))

    after = app.session_state["_last_fit"]
    check("and the moduli are exactly what they were", (
        after and after.get("success")
        and abs(after["Em_MPa"] - before["Em_MPa"]) < 1e-9
        and abs(after["Ei_kPa"] - before["Ei_kPa"]) < 1e-9
        and abs(after["En_kPa"] - before["En_kPa"]) < 1e-9
    ), f"{before['Em_MPa']:.4f}/{before['Ei_kPa']:.4f} -> "
       f"{after['Em_MPa']:.4f}/{after['Ei_kPa']:.4f}" if after else "no fit")


def case_no_nucleus_wording_for_a_cardiomyocyte():
    print("the word nucleus never reaches a cardiomyocyte's screen")
    import app as app_module
    check("the component set has no nucleus name",
          "Nucleus" not in [
              v[0] for v in app_module.COMPONENT_SETS["Cardiomyocyte"].values()
          ])
    check("but a myoblast still names one",
          "nucleus" in app_module.plain_name(
              "nucleus", "Myoblast (C2C12)").lower(),
          app_module.term_name("nucleus", "Myoblast (C2C12)"))
    check("and a cardiomyocyte's deep slot is its myofibrils",
          "myofibril" in app_module.plain_name(
              "nucleus", "Cardiomyocyte").lower(),
          app_module.term_name("nucleus", "Cardiomyocyte"))
    check("its materials are a sarcolemma and two cytoskeletons",
          [app_module.plain_name(term, "Cardiomyocyte")
           for term in app_module.terms_for("Cardiomyocyte")]
          == ["Sarcolemma", "Non-sarcomeric cytoskeleton",
              "Sarcomeric cytoskeleton (myofibrils)"],
          str([app_module.plain_name(t2, "Cardiomyocyte")
               for t2 in app_module.terms_for("Cardiomyocyte")]))
    check("the stored model name no longer names a myoblast's parts",
          not any("nucleus" in k.lower() for k in app_module.MODELS),
          str(list(app_module.MODELS)[:1]))
    check("and the old name still resolves, so old records load",
          app_module.MODEL_KEYS_ANY.get(
              "Segmented (membrane → cytoskeleton → nucleus)") == "segmented")

    app = start(cell_name="WT", cell_type="Cardiomyocyte",
                ui_mode="Full control · every setting")
    if not no_exception(app, "cardiomyocyte page"):
        return
    shown = []
    for kind in ("markdown", "caption", "metric", "expander", "checkbox",
                 "selectbox", "slider", "number_input", "radio"):
        for element in app.get(kind):
            for attribute in ("value", "label"):
                text = getattr(element, attribute, None)
                # A radio's value is the stored key, which is not shown.
                if isinstance(text, str) and not (
                    kind == "radio" and attribute == "value"
                ):
                    shown.append(text)
    offenders = [t for t in shown if "nucleus" in t.lower()]
    check("nothing on the page says nucleus", not offenders,
          str(offenders)[:200])

    plain = start(cell_name="myo-01", ui_mode="Full control · every setting")
    if no_exception(plain, "myoblast page"):
        said = " ".join(
            str(x.value) for kind in ("markdown", "caption")
            for x in plain.get(kind)
        ) + " ".join(m.label for m in plain.get("metric"))
        check("a myoblast still does, because it has one",
              "nucleus" in said.lower())


def case_components_are_recommended():
    print("the search says which components to use, and can apply them")
    from lulevich_model import recommend_components
    import app as app_module
    eps, force, model, _ = four_element_curve()

    # A curve whose in-plane spring carries real force. The default one in
    # four_element_curve carries about 5 % near contact, which the tie rule
    # correctly calls indistinguishable from not having it at all, and that
    # is tested below.
    _, _, loud, _ = four_element_curve(T0=6.0e-3, seed=9)
    found = recommend_components(
        loud, 0.0, 0.65,
        candidates=("membrane", "interior", "nucleus", "tension"),
        e1=0.15, e2=0.40, membrane="continue", cyto_start="zero",
        cv_repeats=2,
    )
    check("it runs", found.get("success"))
    if not found.get("success"):
        return
    check("it tried every combination", len(found["candidates"]) == 8,
          str(len(found["candidates"])))
    check("the membrane is never dropped",
          all("membrane" in r["terms"] for r in found["candidates"]))
    check("a spring that carries real force is recommended",
          "tension" in found["recommended"], str(found["recommended"]))
    check("and so are the others it was built from",
          {"membrane", "interior", "nucleus"} <= set(found["recommended"]),
          str(found["recommended"]))
    check("exactly one row is the recommendation",
          sum(r["recommended"] for r in found["candidates"]) == 1)
    check("they are ranked, best first",
          found["candidates"] == sorted(found["candidates"],
                                        key=lambda r: r["cv_rmse"]))

    # The other direction, and the more important one: a term that earns
    # almost nothing must not be sold as needed just because it fits.
    # A spring an order of magnitude smaller than the one above: it fits,
    # because any extra column fits, and it must still be left out.
    _, _, faint, _ = four_element_curve(T0=1.5e-4, seed=9)
    quiet = recommend_components(
        faint, 0.0, 0.65,
        candidates=("membrane", "interior", "nucleus", "tension"),
        e1=0.15, e2=0.40, membrane="continue", cyto_start="zero",
        cv_repeats=2,
    )
    check("a barely-there spring is left out rather than kept",
          quiet.get("success") and "tension" not in quiet["recommended"],
          str(quiet.get("recommended")))
    check("and the app says it was a close call, not a clear one",
          quiet.get("clear_cut") is False)

    # A curve with nothing deep in it must not be told to include a deep term.
    blank = LulevichModel(np.zeros_like(eps), eps, cell_height=14.0e-6,
                          shell_thickness=200e-9, deep_uses_cell_radius=True)
    basis = blank.composition_basis(eps, 0.15, 0.40, "continue", "zero")
    two = basis["membrane"] * 1.4e6 + basis["interior"] * 3.1e3
    rng = np.random.default_rng(4)
    simple = LulevichModel(two + rng.normal(0, 0.004 * two.max(), two.size), eps,
                           cell_height=14.0e-6, shell_thickness=200e-9,
                           deep_uses_cell_radius=True)
    lean = recommend_components(
        simple, 0.0, 0.65,
        candidates=("membrane", "interior", "nucleus", "tension"),
        e1=0.15, e2=0.40, membrane="continue", cyto_start="zero", cv_repeats=2,
    )
    check("a two-element curve is not sold four elements",
          lean.get("success") and len(lean["recommended"]) <= 3,
          str(lean.get("recommended")))
    check("and the interior is kept", "interior" in lean["recommended"],
          str(lean["recommended"]))

    # It stays a library routine. The page no longer offers it: which
    # components are in the model is a statement about the sample, and a
    # button that cleared a tick box on a cross-validation score was the
    # app arguing with the person about their own cell.
    app = start(cell_name="WT", cell_type="Cardiomyocyte")
    if not no_exception(app, "the page without an element search"):
        return
    check("no button clears the component ticks for you",
          button_by_label(app, "Find the elements") is None,
          str([b.label for b in app.button][:10]))
    check("the components are still ticked and still editable",
          all(state(app, f"use_{t}", False)
              for t in app_module.terms_for("Cardiomyocyte")),
          str([c.label for c in app.checkbox][:6]))
    check("no picture picker is left on the page",
          not any("Which picture of the cell" in (r.label or "")
                  for r in app.get("radio")),
          str([r.label for r in app.get("radio")]))
    check("and the boundaries are still something you can move",
          any((s.label or "").startswith("ε₁") for s in app.slider),
          str([s.label for s in app.slider][:8]))


def case_a_model_that_cannot_carry_a_term_says_so():
    print("a model asked for a term it has no place for reports the loss")
    from lulevich_model import classic_terms, dropped_term_warning

    kept, dropped = classic_terms(("tension", "membrane", "interior",
                                   "cortex", "nucleus"))
    check("the classic path keeps only what it can carry",
          kept == ("membrane", "interior", "nucleus"), str(kept))
    check("and names everything it had to drop",
          set(dropped) == {"tension", "cortex"}, str(dropped))
    message = " ".join(dropped_term_warning(dropped))
    check("the warning names them in words",
          "tension" in message and "cortical" in message, message)
    check("and says the fit went ahead without them",
          "left out" in message.lower(), message)
    check("nothing is said when nothing was dropped",
          not dropped_term_warning(()))

    # And a cardiomyocyte page never mentions the in-plane spring at all,
    # because the term is not offered for it at the moment.
    app = start(cell_name="WT", cell_type="Cardiomyocyte",
                ui_mode="Full control · every setting")
    if not no_exception(app, "a cardiomyocyte in full control"):
        return
    said = " ".join(
        [str(w.value) for w in app.get("warning")]
        + [str(m.value) for m in app.get("markdown")]
    )
    check("no T₀ anywhere on the page", "T₀" not in said,
          said[said.find("T₀") - 60:said.find("T₀") + 60] if "T₀" in said
          else "")


def case_search_says_when_a_winner_drops_a_spring():
    print("an arrangement that cannot carry every element admits it")
    from lulevich_model import search_arrangements
    eps, force, model, _ = four_element_curve()
    found = search_arrangements(
        model, 0.0, 0.65, terms=("tension", "membrane", "interior", "nucleus"),
        tension_mode="always", n_folds=4, cv_repeats=1,
    )
    check("the search runs", found.get("success"))
    if not found.get("success"):
        return
    by_name = {c["arrangement"]: c for c in found["candidates"]}
    check("segmented carries everything",
          by_name.get("segmented", {}).get("dropped") == ())
    for name in ("parallel", "series"):
        if name in by_name:
            check(f"{name} admits it drops the tension spring",
                  by_name[name].get("dropped") == ("tension",),
                  str(by_name[name].get("dropped")))
    if found["best"].get("dropped"):
        check("and the verdict says so when such a one wins",
              "T₀" in found["verdict"], found["verdict"][-160:])
    else:
        check("the winner here carries everything", True)


def case_real_curve_is_steeper_than_any_fixed_power():
    print("the measured curve does what no fixed power law can")
    eps, force = wt_cardiomyocyte()
    if eps is None:
        print("  skip (no reference curve)")
        return
    good = (eps > 0.05) & (force > 0)
    e, f = eps[good], force[good]

    def exponent(lo, hi):
        m = (e >= lo) & (e < hi)
        if m.sum() < 6:
            return float("nan")
        return float(np.polyfit(np.log(e[m]), np.log(f[m]), 1)[0])

    early = exponent(0.15, 0.35)
    late = exponent(0.55, 0.71)
    check("early on it follows the membrane's cube law",
          2.7 < early < 3.7, f"{early:.2f}")
    check("but deep in it is far steeper than any term in the model",
          late > 4.5, f"{late:.2f}")
    check("and steeper than early on, not flatter",
          late > early + 1.0, f"{early:.2f} then {late:.2f}")


def case_confinement_earns_its_place_on_real_data():
    print("confinement is what closes that gap, measured on the real curve")
    model = wt_model()
    if model is None:
        print("  skip (no reference curve)")
        return
    scan = model.scan_confinement(
        0.05, 0.70, e1=0.30, e2=0.475, membrane="continue", cyto_start="break",
        use_tension=True,
    )
    check("the scan succeeds", scan.get("success"))
    if not scan.get("success"):
        return
    check("it lands near q = 1.3, not at zero",
          1.0 < scan["q"] < 1.7, f"q = {scan['q']:.2f}")
    check("and q is well determined by this curve",
          scan["q_high"] - scan["q_low"] < 0.4,
          f"{scan['q_low']:.2f} to {scan['q_high']:.2f}")

    with_q, without = scan["fit"], scan["baseline"]
    check("the classic model cannot follow this curve",
          without["r_squared"] < 0.999, f"R² {without['r_squared']:.6f}")
    check("with confinement it can",
          with_q["r_squared"] > 0.9999, f"R² {with_q['r_squared']:.6f}")
    # Residual sum, not chi-squared per point. This reference curve is
    # thinned, so successive differences measure the curve's own slope rather
    # than the noise, the estimated sigma comes out far too large, and every
    # chi-squared computed from it is meaningless. On the full 8030-point
    # curve chi-squared per point improves about 200-fold.
    check("the residual sum improves by more than tenfold",
          without["ss_res"] > 10 * with_q["ss_res"],
          f"{without['ss_res']:.4g} -> {with_q['ss_res']:.4g}")
    check("and the typical miss shrinks with it",
          without["rmse"] > 5 * with_q["rmse"],
          f"{without['rmse']:.3g} -> {with_q['rmse']:.3g} N")

    # q = 0 must reproduce the old model exactly, or every myoblast moves.
    plain = wt_model(q=0.0)
    check("q = 0 leaves the basis functions untouched",
          plain.confinement_factor(np.array([0.0, 0.3, 0.6])) == 1.0)
    a = plain.fit_composition(0.05, 0.70, 0.30, 0.475, "continue", "break",
                              use_tension=True)
    check("and gives exactly the classic answer",
          abs(a["r_squared"] - without["r_squared"]) < 1e-12,
          f"{a['r_squared']!r} vs {without['r_squared']!r}")


def case_real_curve_gives_believable_numbers():
    print("the fitted numbers land where a cardiomyocyte's should")
    model = wt_model()
    if model is None:
        print("  skip (no reference curve)")
        return
    fit = model.fit_composition(0.0, 0.70, 0.30, 0.475, "continue", "break",
                                use_tension=True)
    check("the fit succeeds", fit.get("success"))
    check("it describes the curve", fit["r_squared"] > 0.9999,
          f"{fit['r_squared']:.6f}")
    # Cortical tension of a cell is tenths of a mN/m. Orders of magnitude,
    # not decimal places: the point is that nothing came out absurd.
    check("cortical tension is in the range a cell's is",
          0.01 < fit["T0_mN_m"] < 5.0, f"{fit['T0_mN_m']:.3f} mN/m")
    check("the membrane modulus is sub-MPa to MPa",
          0.01 < fit["Em_MPa"] < 20.0, f"{fit['Em_MPa']:.3f} MPa")
    check("the cytoskeleton is a few kPa",
          0.05 < fit["Ei_kPa"] < 100.0, f"{fit['Ei_kPa']:.3f} kPa")
    check("every modulus is positive, none pinned at zero",
          min(fit["T0_mN_m"], fit["Em_MPa"], fit["Ei_kPa"]) > 0,
          f"{fit['T0_mN_m']:.3g} {fit['Em_MPa']:.3g} {fit['Ei_kPa']:.3g}")

    # The areal modulus is what the cube-law term really measures, and it
    # says plainly that this "membrane" is not a bare bilayer.
    areal = fit["membrane_areal_modulus"] * 1e3
    check("the areal modulus is far below a lipid bilayer's 240 mN/m",
          areal < 240.0, f"{areal:.2f} mN/m")


def case_cardiomyocyte_defaults_match_the_experiment():
    print("choosing Cardiomyocyte sets the geometry that cell actually has")
    import app as app_module
    preset = app_module.CELL_TYPES["Cardiomyocyte"]
    check("19 um tall", preset["cell_height_um"] == 19.0)
    # Everything shared with a myoblast is shared. A parameter invented for
    # one cell type is a difference that turns up in the answer and cannot
    # be traced back, so the only ones that differ are the two that really
    # do: the cell is taller and the sarcolemma is twice a bare bilayer.
    myoblast = app_module.CELL_TYPES["Myoblast (C2C12)"]
    for key in ("radius_aspect", "nucleus_fraction", "nucleus_onset",
                "cell_shape"):
        check(f"{key} is the same as a myoblast's",
              preset[key] == myoblast[key],
              f"{preset[key]!r} against {myoblast[key]!r}")
    check("the sarcolemma is twice a bare bilayer",
          preset["membrane_thickness_nm"]
          == 2 * myoblast["membrane_thickness_nm"],
          f"{preset['membrane_thickness_nm']} against "
          f"{myoblast['membrane_thickness_nm']}")
    check("and confined, not free", preset["confinement"] > 1.0)
    check("at roughly what four corrected WT curves measured",
          0.9 <= preset["confinement"] <= 1.4, str(preset["confinement"]))
    check("a myoblast is unconfined",
          app_module.CELL_TYPES["Myoblast (C2C12)"]["confinement"] == 0.0)

    app = start(cell_name="WT_2", cell_type="Cardiomyocyte")
    if not no_exception(app, "cardiomyocyte defaults"):
        return
    # Confinement starts from the preset. Measuring it from the curve is a
    # button now, like the boundary search: nothing on a freshly loaded page
    # is a search result nobody asked for.
    check("confinement starts from the cell type",
          app_module.CELL_TYPES["Cardiomyocyte"]["confinement"] == 1.10)
    check("and it is what a loaded curve is fitted at",
          abs(float(app.session_state["confinement"]) - 1.10) < 1e-9,
          str(app.session_state["confinement"]))
    check("with a button to measure it from this curve instead",
          button_by_label(app, "Measure q from this curve") is not None,
          str([b.label for b in app.button][:12]))
    for key, want in (("cell_height_um", 19.0), ("radius_aspect", 0.55),
                      ("membrane_thickness_nm", 8.0)):
        check(f"{key} reaches the page as {want}",
              app.session_state[key] == want, str(app.session_state[key]))
    check("the approach speed is recorded",
          app.session_state["approach_speed_um_s"] == 2.0)
    check("and there is somewhere to put the probe diameter",
          "probe_diameter_um" in app.session_state)

    plain = start(cell_name="myo-01")
    if no_exception(plain, "myoblast defaults"):
        check("a myoblast is left unconfined",
              plain.session_state["confinement"] == 0.0)
        check("and 4 nm", plain.session_state["membrane_thickness_nm"] == 4.0)


def case_offset_is_available_and_signed():
    print("a baseline offset can be fitted, and it is the one signed column")
    eps, force = synthetic("continue", "zero")
    shifted = force + 2.0e-9          # 2 nN of baseline, the wrong way up
    model = LulevichModel(shifted, eps, cell_height=8.0e-6)
    without = model.fit_composition(0.0, 0.60, 0.15, 0.40, "continue", "zero")
    withit = model.fit_composition(0.0, 0.60, 0.15, 0.40, "continue", "zero",
                                   fit_offset=True)
    check("fitting the offset recovers it",
          abs(withit["force_offset"] - 2.0e-9) < 0.5e-9,
          f"{withit['force_offset'] * 1e9:.3f} nN")
    check("and it fits better than pretending there is none",
          withit["ss_res"] < without["ss_res"])
    check("the offset counts as a parameter",
          withit["n_params"] == without["n_params"] + 1)
    check("without it the offset is exactly zero",
          without["force_offset"] == 0.0)

    # Negative baselines happen just as often, and a bound at zero would
    # silently refuse them.
    low = LulevichModel(force - 2.0e-9, eps, cell_height=8.0e-6)
    negative = low.fit_composition(0.0, 0.60, 0.15, 0.40, "continue", "zero",
                                   fit_offset=True)
    check("a negative baseline is allowed", negative["force_offset"] < 0,
          f"{negative['force_offset'] * 1e9:.3f} nN")


def case_sarcomere_length():
    print("the squash is reported as a sarcomere length")
    eps, force, model, _ = four_element_curve()
    check("the relaxed length is 2.1 um by default",
          abs(model.L_sarcomere - 2.1e-6) < 1e-12, str(model.L_sarcomere))
    check("unsquashed, the sarcomere is its relaxed length",
          abs(model.sarcomere_at(0.0) - 2.1e-6) < 1e-15)

    # The direction matters and is easy to get backwards: squashing a
    # cardiomyocyte lengthens its sarcomeres, because the cell spreads
    # sideways and the myofibrils run that way.
    check("squashing lengthens them, it does not shorten them",
          model.sarcomere_at(0.30) > model.sarcomere_at(0.0),
          f"{model.sarcomere_at(0.30) * 1e9:.0f} nm")
    check("constant volume gives the (1-e)^-1/2 stretch",
          abs(model.sarcomere_at(0.30) - 2.1e-6 / np.sqrt(0.70)) < 1e-15,
          f"{model.sarcomere_at(0.30) * 1e9:.1f} nm")
    check("a cell held at its ends keeps them at rest length",
          abs(model.sarcomere_at(0.50, spread=0.0) - 2.1e-6) < 1e-15)
    check("and half the spreading gives half the exponent",
          abs(model.sarcomere_at(0.30, spread=0.5)
              - 2.1e-6 * 0.70 ** -0.25) < 1e-15)
    check("it stays finite at full compression",
          np.isfinite(model.sarcomere_at(1.0)))

    report = model.sarcomere_report(0.65, onset=0.40)
    check("the report gives the length at the top of the range",
          abs(report["at_epsilon_max_nm"] - model.sarcomere_at(0.65) * 1e9) < 1e-6)
    check("and where the myofibrils engage",
          abs(report["at_onset_nm"] - model.sarcomere_at(0.40) * 1e9) < 1e-6)
    check("the working limit follows the relaxed length",
          abs(report["working_limit_nm"] - 2310.0) < 1e-6,
          f"{report['working_limit_nm']:.1f} nm")
    check("a deep squash is flagged as past it",
          report["beyond_working_range"])
    check("and it says where that started",
          0.0 < report["epsilon_at_limit"] < 0.65,
          f"{report['epsilon_at_limit']:.3f}")
    check("the flag agrees with the length",
          report["at_epsilon_max_nm"] > report["working_limit_nm"])

    shallow = model.sarcomere_report(0.10, onset=0.40)
    check("a shallow squash is not flagged", not shallow["beyond_working_range"],
          f"{shallow['at_epsilon_max_nm']:.0f} nm")

    check("it counts the sarcomeres along the cell",
          abs(report["n_along_cell"] - (2 * model.R0) / 2.1e-6) < 1e-9,
          f"{report['n_along_cell']:.1f}")

    # None of this may touch the fit.
    changed = LulevichModel(model.force, model.epsilon, cell_height=14.0e-6,
                            shell_thickness=200e-9, deep_uses_cell_radius=True,
                            sarcomere_length=1.8e-6)
    a = model.fit_composition(0.0, 0.65, 0.15, 0.40, "continue", "zero",
                              use_tension=True)
    b = changed.fit_composition(0.0, 0.65, 0.15, 0.40, "continue", "zero",
                                use_tension=True)
    for key in ("T0_mN_m", "Em_MPa", "Ei_kPa", "En_kPa", "r_squared"):
        check(f"changing the sarcomere length leaves {key} alone",
              abs(a[key] - b[key]) < 1e-12, f"{a[key]!r} vs {b[key]!r}")
    check("but the clone carries it", model._clone(
        model.force, model.epsilon).L_sarcomere == model.L_sarcomere)

    # And it has to reach the page, for cardiomyocytes only. It is working
    # rather than answer, so guided mode leaves it out; full control shows
    # it, which is where this is checked.
    app = start(cell_name="cardio-01", cell_type="Cardiomyocyte")
    if not no_exception(app, "sarcomere panel"):
        return
    # Captions are their own element type in AppTest, not markdown.
    def page_text(a):
        return " ".join(
            str(x.value) for kind in ("markdown", "caption")
            for x in a.get(kind)
        )

    said = " ".join(str(x.value) for kind in ("markdown", "caption")
                    for x in app.get(kind))
    check("the sarcomere working is shown for a cardiomyocyte",
          "sarcomeres" in said.lower(), said[-400:])
    labels = [m.label for m in app.get("metric")]
    check("the relaxed length is on the page", "Relaxed" in labels, str(labels))
    check("so is the length at the top of the range",
          any(l.startswith("At ε =") for l in labels), str(labels))
    check("it says which way the length goes",
          "lengthens them" in page_text(app))

    plain = start(cell_name="myo-01")
    if no_exception(plain, "no sarcomere panel"):
        check("and not for a myoblast",
              "sarcomere" not in page_text(plain).lower())


def case_extra_terms_never_crash_old_paths():
    print("a fourth spring reaching three-spring code is dropped, loudly")
    import lulevich_model as lm
    eps, force, model, _ = four_element_curve()
    four = ("tension", "membrane", "interior", "nucleus")

    kept, dropped = lm.classic_terms(four)
    check("the classic paths keep three", kept == lm.CLASSIC_TERMS, str(kept))
    check("and report what they could not take", dropped == ("tension",),
          str(dropped))
    check("an all-classic list is untouched",
          lm.classic_terms(("membrane", "interior"))[0] == ("membrane", "interior"))
    check("an empty list falls back to all three",
          lm.classic_terms(())[0] == lm.CLASSIC_TERMS)

    # This is the exact path that raised a KeyError on Streamlit Cloud for
    # every cardiomyocyte: the "Find the segments" button.
    explored = model.explore_segments(0.0, 0.65, terms=four, n_grid=8)
    check("exploring the curve survives the extra term",
          explored.get("success"), str(explored.get("error")))

    for name, call in (
        ("fit", lambda: model.fit(0.0, 0.65, terms=four)),
        ("fit_segmented",
         lambda: model.fit_segmented(0.0, 0.65, 0.15, 0.40, terms=four)),
        ("fit_series", lambda: model.fit_series(0.01, 0.65, terms=four)),
        ("fit_staged", lambda: model.fit_staged([
            {"terms": ("tension", "membrane"), "range": (0.20, 0.65)},
            {"terms": ("interior", "nucleus"), "range": (0.00, 0.20)},
        ])),
    ):
        try:
            out = call()
        except Exception as exc:
            check(f"{name} survives the extra term", False,
                  f"{type(exc).__name__}: {exc}")
            continue
        check(f"{name} survives the extra term", out.get("success"))
        check(f"{name} reports only the terms it really used",
              "tension" not in (out.get("terms") or []), str(out.get("terms")))
        check(f"{name} says the spring was left out",
              any("T₀" in w for w in (out.get("warnings") or [])),
              str(out.get("warnings"))[:120])

    # Scans go through those same fits and must not raise either.
    for name, call in (
        ("scan_segment_breaks",
         lambda: model.scan_segment_breaks(0.0, 0.65, terms=four, n_grid=6)),
        ("scan_nucleus_onset",
         lambda: model.scan_nucleus_onset(0.0, 0.65, terms=four, n_trials=5)),
        ("scan_crossover",
         lambda: model.scan_crossover(0.02, 0.65, terms=four, n_trials=4)),
    ):
        try:
            check(f"{name} survives the extra term", call().get("success"))
        except Exception as exc:
            check(f"{name} survives the extra term", False,
                  f"{type(exc).__name__}: {exc}")

    # And the composition fit, which does implement it, still does.
    real = model.fit_composition(0.0, 0.65, 0.15, 0.40, "continue", "zero",
                                 use_tension=True)
    check("the segmented fit still carries the spring",
          "tension" in real["terms"] and real["T0_mN_m"] > 0,
          str(real["terms"]))


def case_four_element_search():
    print("the search places the boundaries inside the four-spring model")
    eps, force, model, _ = four_element_curve(seed=11)
    found = model.search_compositions(0.0, 0.65, tension_mode="always", n_grid=10)
    check("the search succeeded", found.get("success"))
    if not found.get("success"):
        return
    best = found["best"]
    check("every candidate keeps the tension spring",
          all(row["use_tension"] for row in found["candidates"]))
    check("the membrane is not made to hand over",
          best["membrane"] == "continue", best["membrane"])
    check("the deeper layer is found near where it was put",
          abs(best["break_2"] - 0.40) < 0.05, f"{best['break_2']:.3f}")
    check("Em survives the search", abs(best["Em_MPa"] - 1.4) / 1.4 < 0.15,
          f"{best['Em_MPa']:.3f}")
    check("En survives the search", abs(best["En_kPa"] - 9.0) / 9.0 < 0.15,
          f"{best['En_kPa']:.3f}")

    off = model.search_compositions(0.0, 0.65, n_grid=8)
    check("with the spring off, no candidate carries it",
          off.get("success")
          and not any(row["use_tension"] for row in off["candidates"]))


def case_breakpoint_spread_is_the_real_error_bar():
    print("the boundaries are fitted too, and the numbers say how much that matters")
    eps, force, model, _ = four_element_curve(seed=5, noise=0.004)

    # Two boundary placements that fit this curve equally well give very
    # different tensions. A standard error worked out at fixed boundaries
    # cannot see that, which is the whole reason this exists.
    a = model.fit_composition(0.0, 0.65, 0.024, 0.40, "continue", "break",
                              use_tension=True)
    b = model.fit_composition(0.0, 0.65, 0.0, 0.40, "continue", "zero",
                              use_tension=True)
    check("both placements fit about as well",
          abs(a["ss_res"] - b["ss_res"]) / a["ss_res"] < 0.05,
          f"{a['ss_res']:.3e} vs {b['ss_res']:.3e}")
    check("but they disagree about the tension",
          abs(a["T0_mN_m"] - b["T0_mN_m"]) / max(b["T0_mN_m"], 1e-12) > 0.5,
          f"{a['T0_mN_m']:.3f} vs {b['T0_mN_m']:.3f}")
    check("while the fixed-boundary error bar looks small",
          a["T0_mN_m_std"] / a["T0_mN_m"] < 0.2,
          f"± {a['T0_mN_m_std']:.3f} on {a['T0_mN_m']:.3f}")

    spread = model.breakpoint_spread(0.0, 0.65, 0.024, 0.40, "continue",
                                     "break", use_tension=True)
    check("the spread is measurable", spread.get("success"))
    if not spread.get("success"):
        return
    check("more than one placement is accepted", spread["n_accepted"] > 1,
          str(spread["n_accepted"]))
    bands = spread["ranges"]
    for key in ("T0_mN_m", "Em_MPa", "Ei_kPa", "En_kPa"):
        band = bands[key]
        check(f"{key} is bracketed", band["low"] <= band["value"] <= band["high"],
              f"{band['low']:.4g} .. {band['value']:.4g} .. {band['high']:.4g}")
    check("the tension is the loose one here",
          bands["T0_mN_m"]["relative"] > bands["Em_MPa"]["relative"],
          f"T0 {bands['T0_mN_m']['relative']:.2f} vs "
          f"Em {bands['Em_MPa']['relative']:.2f}")
    check("and it is not cheap to compute",
          spread["break_1_range"][0] <= 0.024 <= spread["break_1_range"][1])

    # It has to reach the page, or it is a diagnostic nobody sees.
    source = pathlib.Path(__file__).with_name("app.py").read_text()
    check("the app computes it once per fit",
          'fit["breakpoint_spread"] = model.breakpoint_spread(' in source)
    check("and shows it", "but anywhere in" in source)


def case_error_bars_are_reported():
    print("every modulus comes with a standard error")
    eps, force = synthetic("freeze", "break")
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.60, 0.15, 0.40, "freeze", "break")
    for key in ("Em_MPa_std", "Ei_kPa_std", "En_kPa_std"):
        check(f"{key} is a number", np.isfinite(fit[key]), str(fit[key]))
    check("a clean curve gets tight bars",
          fit["Em_MPa_std"] / fit["Em_MPa"] < 0.1,
          f"± {fit['Em_MPa_std']:.4g} on {fit['Em_MPa']:.4g}")
    check("and no complaint about identifiability",
          not fit["warnings"], str(fit["warnings"]))

    # A term pinned at the zero bound is not free, and must not make the
    # covariance singular or produce a bogus error bar for itself.
    zeroed = model.fit_composition(0.0, 0.60, 0.15, 0.40, "continue", "zero")
    check("a fit with a zeroed term still returns", zeroed.get("success"))
    check("and the fast path skips the error bars entirely",
          not np.isfinite(model.fit_composition(
              0.0, 0.60, 0.15, 0.40, "freeze", "break", with_stats=False
          )["Em_MPa_std"]))


def case_clone_keeps_the_whole_geometry():
    print("a cross-validation clone is the model it is standing in for")
    eps, force, model, _ = four_element_curve()
    twin = model._clone(model.force, model.epsilon)
    for name in ("cell_height", "R0", "R_nucleus", "h_membrane", "h_shell",
                 "Am", "Ai", "An", "At"):
        mine, theirs = getattr(model, name), getattr(twin, name)
        check(f"{name} survives the clone", abs(theirs - mine) <= abs(mine) * 1e-12,
              f"{theirs!r} vs {mine!r}")
    check("so does the myofibril radius choice",
          twin.deep_uses_cell_radius is model.deep_uses_cell_radius)


def case_the_cardiomyocyte_has_three_materials():
    print("a cardiomyocyte is a sarcolemma, a packed interior and myofibrils")
    import app as app_module
    here = app_module.terms_for("Cardiomyocyte")
    check("three materials", len(here) == 3, str(here))
    check("and they are the three asked for",
          here == ("membrane", "interior", "nucleus"), str(here))
    check("no cortical layer of its own", "cortex" not in here, str(here))
    check("no in-plane spring either", "tension" not in here, str(here))
    names = app_module.components_for("Cardiomyocyte")
    check("the membrane is named the sarcolemma",
          "sarcolemma" in names["membrane"][0].lower(), str(names["membrane"]))
    check("the interior is the non-sarcomeric cytoskeleton",
          "non-sarcomeric" in names["interior"][0].lower(),
          str(names["interior"]))
    check("and the deep layer is the myofibrils",
          "myofibril" in names["nucleus"][0].lower(), str(names["nucleus"]))
    check("nothing anywhere in its names says nucleus",
          not any("nucleus" in v[0].lower() for v in names.values()),
          str([v[0] for v in names.values()]))
    check("the interior carries the load from first contact",
          app_module.DEFAULT_COMPOSITION_BY_TYPE["Cardiomyocyte"][
              "cyto_starts_at"] == "from the very start")

    app = start(cell_name="cardio-01", cell_type="Cardiomyocyte")
    if not no_exception(app, "three materials"):
        return
    fitted = (app.session_state["_last_fit"] or {}).get("terms") or []
    check("all three are fitted",
          set(fitted) == {"membrane", "interior", "nucleus"}, str(fitted))
    # Named on the page as its own three components, in the metrics row and
    # on the component ticks. No table of parts any more.
    labels = " ".join(str(m.label or "") for m in app.get("metric"))
    boxes = " ".join(str(c.label or "") for c in app.checkbox)
    check("all three are named on the page",
          "sarcolemma" in (labels + boxes).lower(), labels + " | " + boxes)
    check("and nothing on it says nucleus",
          "nucleus" not in (labels + boxes).lower(), labels + " | " + boxes)

    plain = start(cell_name="myo-01")
    if no_exception(plain, "a myoblast"):
        myo = " ".join(str(c.label or "") for c in plain.checkbox)
        check("a myoblast still offers four of its own",
              sum(1 for term in ("Membrane", "Cytoskeleton",
                                 "Nuclear envelope", "Inside the nucleus")
                  if term in myo) == 4, myo)


def case_q_and_the_boundaries_are_searched_together():
    print("the confinement and the boundaries are found jointly, not in turn")
    eps = np.linspace(0.002, 0.62, 340)
    seed = LulevichModel(np.zeros_like(eps), eps, cell_height=19.0e-6,
                         cell_radius=9.5e-6, confinement=1.4)
    basis = seed.composition_basis(eps, 0.18, 0.50, "late", "zero")
    force = basis["membrane"] * 1.5e6 + basis["interior"] * 2.0e3
    model = LulevichModel(force, eps, cell_height=19.0e-6, cell_radius=9.5e-6,
                          confinement=0.0)
    q, e1, e2 = model.best_confinement_and_breaks(
        0.0, 0.62, membrane="late", cyto_start="zero", use_nucleus=False,
        weighting="relative", n_grid=8,
    )
    check("q is recovered", abs(q - 1.4) < 0.25, f"{q:.2f}")
    check("and so is the boundary", abs(e1 - 0.18) < 0.04, f"{e1:.3f}")
    check("the model's own q is left alone", model.confinement == 0.0,
          str(model.confinement))
    fitted = LulevichModel(force, eps, cell_height=19.0e-6,
                           cell_radius=9.5e-6, confinement=q).fit_composition(
        0.0, 0.62, e1, e2, "late", "zero", use_nucleus=False,
        weighting="relative")
    check("and the pair fits the curve", fitted["r_squared"] > 0.9999,
          f"R2 {fitted['r_squared']:.6f}")

    # Each picture must be allowed its own q inside a comparison, or the one
    # fitted first sets the terms of the argument for the rest.
    from lulevich_model import compare_hypotheses
    scored = compare_hypotheses(
        model, 0.0, 0.62,
        [{"key": "cyto_first", "label": "interior first",
          "terms": ("membrane", "interior"),
          "membrane": "late", "cyto_start": "zero"},
         {"key": "together", "label": "together",
          "terms": ("membrane", "interior"),
          "membrane": "continue", "cyto_start": "zero"}],
        weighting="relative", cv_repeats=2, n_grid=8, scan_q=True,
    )
    check("the comparison ran with q scanned per picture", scored.get("success"),
          str(scored.get("error")))
    if scored.get("success"):
        check("every candidate reports the q it competed at",
              all(np.isfinite(r.get("confinement", np.nan))
                  for r in scored["candidates"]))
        check("and the built-in ordering is the one that wins",
              scored["best"]["key"] == "cyto_first", scored["best"]["key"])


def case_one_fitting_routine():
    print("what loads and what the button does are the same routine")
    source = pathlib.Path(APP).read_text()
    check("there is one routine", source.count("def analyse_curve(") == 1)
    check("and the page calls it", source.count("analyse_curve(") >= 1,
          str(source.count("analyse_curve(")))
    check("the arrangement search no longer competes with it",
          "search_arrangements(" not in source.split("def analyse_curve")[-1]
          or source.count("search_arrangements(") <= 1,
          str(source.count("search_arrangements(")))

    curves = vcm_curves()
    if not curves:
        check("the VCM reference curves are in the repository", False)
        return
    eps, force = curves[11]
    app = AppTest.from_file(APP, default_timeout=900)
    app.run()
    app.session_state["cell_type"] = "Cardiomyocyte"
    app.session_state["cell_name"] = "vcm-11"
    app.session_state["data"] = {
        "epsilon": eps, "force_N": force, "source": "vcm_11.csv", "n_dropped": 0,
    }
    app.run()
    if not no_exception(app, "loading a cardiomyocyte"):
        return
    on_load = app.session_state["_last_fit"]
    check("a curve that loads is already fitted",
          on_load and on_load.get("success"))
    if not (on_load and on_load.get("success")):
        return
    check("with all three materials",
          set(on_load["terms"]) == {"membrane", "interior", "nucleus"},
          str(on_load["terms"]))
    check("and it fits the measured curve", on_load["r_squared"] > 0.9995,
          f"R2 {on_load['r_squared']:.6f}")

    work = button_by_label(app, "Fit this cell")
    check("the fit button is there", work is not None)
    if work is None:
        return
    work.click().run()
    if not no_exception(app, "pressing fit"):
        return
    after = app.session_state["_last_fit"]
    check("pressing it fits at least as well, not worse",
          after["r_squared"] >= on_load["r_squared"] - 1e-4,
          f"{on_load['r_squared']:.6f} -> {after['r_squared']:.6f}")
    check("and the answer is still a fit of this curve",
          after["r_squared"] > 0.9995, f"R2 {after['r_squared']:.6f}")
    # The one number that says the search did not wander somewhere silly.
    check("chi-squared per point stays in a sane range",
          0.02 < after["chi_squared_reduced"] < 20.0,
          f"{after['chi_squared_reduced']:.3g}")

    check("the table of pictures compared is gone from the page",
          table_with(app, "Picture of the cell") is None)
    said = " ".join(str(m.value) for m in
                    list(app.get("markdown")) + list(app.get("success"))
                    + list(app.get("info")))
    check("the verdict is still said in one line",
          "curve" in said.lower(), said[-200:])


def case_materials_are_explained_by_their_law():
    print("every material on the page says what law it obeys and why it separates")
    import app as app_module
    for term in ("tension", "membrane", "interior", "nucleus"):
        law = app_module.MATERIAL_LAWS[term]
        for field in ("role", "law", "exponent", "separable"):
            check(f"{term} has a {field}", bool(law.get(field)))
    check("the two Hertzian terms are separated by onset alone",
          "onset" in app_module.MATERIAL_LAWS["nucleus"]["separable"],
          app_module.MATERIAL_LAWS["nucleus"]["separable"])
    check("and the in-plane spring by being linear",
          "linear" in app_module.MATERIAL_LAWS["tension"]["separable"])

    app = start(cell_name="cardio-01", cell_type="Cardiomyocyte")
    if not no_exception(app, "the materials table"):
        return
    table = table_with(app, "Material", "Force law", "What makes it separable")
    check("the table of materials is on the page", table is not None)
    if table is not None:
        ticked = [t for t in app_module.terms_for("Cardiomyocyte")
                  if app.session_state[f"use_{t}"]]
        check("one row per ticked material", len(table) == len(ticked),
              f"{len(table)} rows for {ticked}")
        check("each row carries its exponent",
              all("ε^" in str(v) for v in table["Rises as"]),
              str(list(table["Rises as"])))
        check("and says when it starts carrying load",
              all(str(v).strip() for v in table["Carries load"]),
              str(list(table["Carries load"])))
    text = " ".join(str(m.value) for m in app.get("markdown"))
    everything = text + " ".join(str(i.value) for i in app.get("info"))
    check("the separation rule is stated in words",
          "different shapes" in everything or "wearing two names" in everything)


def case_the_nucleus_is_a_balloon_too():
    print("the nucleus is an envelope around a filling, and both are found")
    eps = np.linspace(0.002, 0.62, 320)
    seed = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
    basis = seed.composition_basis(eps, 0.15, 0.40, "freeze", "break")
    check("the envelope has a basis function of its own",
          "nucleus_shell" in basis)
    # It must not be the same shape as what it contains, or the split
    # between them is arbitrary. Same onset, so the exponents do the work.
    shell = basis["nucleus_shell"] / max(basis["nucleus_shell"].max(), 1e-30)
    inside = basis["nucleus"] / max(basis["nucleus"].max(), 1e-30)
    # Normalised to the same peak, u^3 and u^1.5 differ by exactly 0.25 at
    # their furthest apart. That gap is the whole reason a fit can split
    # them, so it is checked rather than assumed.
    check("it is a cube law where the filling is Hertzian",
          float(np.max(np.abs(shell - inside))) > 0.2,
          f"largest difference {float(np.max(np.abs(shell - inside))):.3f}")
    check("both start at ε₂ and not before",
          float(np.max(np.abs(shell[eps < 0.40]))) == 0.0
          and float(np.max(np.abs(inside[eps < 0.40]))) == 0.0)

    truth = {"Em": 0.6e6, "Ec": 1.2e3, "Ene": 0.4e6, "En": 3.0e3}
    force = (basis["membrane"] * truth["Em"] + basis["interior"] * truth["Ec"]
             + basis["nucleus_shell"] * truth["Ene"]
             + basis["nucleus"] * truth["En"])
    rng = np.random.default_rng(0)
    noisy = force * (1.0 + 0.01 * rng.standard_normal(eps.size))
    model = LulevichModel(noisy, eps, cell_height=8.0e-6)
    fit = model.fit_composition(
        0.0, 0.62, 0.15, 0.40, "freeze", "break",
        use_nucleus=True, use_nucleus_shell=True, weighting="uniform",
    )
    for key, want, tol in (("Em_MPa", 0.6, 0.05), ("Ei_kPa", 1.2, 0.1),
                           ("Ene_MPa", 0.4, 0.05), ("En_kPa", 3.0, 0.3)):
        check(f"{key} is recovered", abs(fit[key] - want) < tol,
              f"{fit[key]:.4g} against {want}")
    # The one that catches a term left out of the predicted sum: a modulus
    # can come back right while the curve drawn from it is wrong.
    check("and the fit follows the curve it was given",
          fit["r_squared"] > 0.999, f"R2 {fit['r_squared']:.6f}")
    rebuilt = model.composition_curve(eps, fit)
    check("the rebuilt curve carries the envelope too",
          float(np.max(np.abs(rebuilt - force))) < 0.02 * float(force.max()),
          f"worst gap {float(np.max(np.abs(rebuilt - force))):.3e}")

    # And the search finds the whole picture on its own.
    import app as app_module
    from lulevich_model import compare_hypotheses
    scored = compare_hypotheses(
        model, 0.0, 0.62, app_module.HYPOTHESES["Myoblast (C2C12)"],
        weighting="uniform", cv_repeats=2, n_grid=8,
    )
    check("the comparison ran", scored.get("success"), str(scored.get("error")))
    if scored.get("success"):
        best = scored["best"]
        check("it picks the picture the curve was built from",
              best["key"] == "handover_with_envelope", best["key"])
        check("with ε₁ where it belongs", abs(best["break_1"] - 0.15) < 0.03,
              f"{best['break_1']:.3f}")
        check("and ε₂ where it belongs", abs(best["break_2"] - 0.40) < 0.03,
              f"{best['break_2']:.3f}")
        check("and the envelope carrying load",
              best.get("Ene_MPa", 0.0) > 0.2, str(best.get("Ene_MPa")))


def case_the_myoblast_nucleus_reaches_the_page():
    print("a myoblast is fitted with four materials, envelope included")
    import app as app_module
    check("the myoblast has four materials",
          len(app_module.terms_for("Myoblast (C2C12)")) == 4,
          str(app_module.terms_for("Myoblast (C2C12)")))
    names = app_module.components_for("Myoblast (C2C12)")
    check("the envelope is named as a skin",
          "skin" in names["nucleus_shell"][1], str(names["nucleus_shell"]))
    check("and what it contains is named separately",
          names["nucleus"][0] != names["nucleus_shell"][0])
    check("the envelope has a law of its own",
          app_module.MATERIAL_LAWS["nucleus_shell"]["exponent"] == "3")

    eps = np.linspace(0.002, 0.62, 300)
    seed = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
    # Built the way a C2C12 is now understood to behave: both outer
    # elements loading from first contact, the nucleus met at 0.50, inside
    # the 44 % to 70 % band.
    basis = seed.composition_basis(eps, 0.15, 0.50, "continue", "zero")
    # 30 kPa inside the nucleus, not 3. Its prefactor is the nucleus radius
    # squared (Lulevich eq 6), and at 3 kPa a body that size contributes
    # less than the noise: the search would be right to drop it, so a test
    # that demanded it be kept would be demanding a wrong answer.
    force = (basis["membrane"] * 0.6e6 + basis["interior"] * 1.2e3
             + basis["nucleus_shell"] * 0.4e6 + basis["nucleus"] * 30.0e3)
    rng = np.random.default_rng(0)
    force = force * (1.0 + 0.01 * rng.standard_normal(eps.size))

    app = AppTest.from_file(APP, default_timeout=900)
    app.run()
    app.session_state["c2c12_fit_mode"] = ADVANCED_MODE
    app.session_state["cell_name"] = "myo-nucleus"
    app.session_state["data"] = {
        "epsilon": eps, "force_N": force, "source": "myo.csv", "n_dropped": 0,
    }
    app.run()
    if not no_exception(app, "a myoblast with an envelope"):
        return
    fit = app.session_state["_last_fit"]
    check("it is fitted", fit and fit.get("success"))
    if not (fit and fit.get("success")):
        return
    check("the envelope is one of the fitted terms",
          "nucleus_shell" in fit["terms"], str(fit["terms"]))
    # The curve was built with its nucleus met at 0.50 and the page starts
    # at the middle of the band, so the envelope only carries load once the
    # boundary has been estimated. That is the search's job.
    work = button_by_label(app, "Find the boundaries from the log curve")
    if work is not None:
        work.click().run()
        no_exception(app, "finding the onset")
        fit = state(app, "_last_fit") or fit
    check("the boundary is recovered inside the band",
          abs(float(fit["break_2"]) - 0.50) < 0.06, str(fit.get("break_2")))
    check("and it carries load", fit.get("Ene_MPa", 0.0) > 0.1,
          str(fit.get("Ene_MPa")))
    check("the fit follows the curve", fit["r_squared"] > 0.999,
          f"R2 {fit['r_squared']:.6f}")
    table = table_with(app, "Material", "Force law")
    check("both halves of the nucleus are in the materials table",
          table is not None and len(table) == 4,
          "none" if table is None else str(len(table)))


def case_component_heights_are_readable():
    print("the height labels land inside the axes and do not sit on each other")
    import plot_utils
    eps = np.linspace(0.002, 0.60, 260)
    seed = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
    basis = seed.composition_basis(eps, 0.15, 0.40, "freeze", "break")
    force = (basis["membrane"] * 0.6e6 + basis["interior"] * 1.2e3
             + basis["nucleus_shell"] * 0.4e6 + basis["nucleus"] * 3.0e3)
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.60, 0.15, 0.40, "freeze", "break",
                                use_nucleus=True, use_nucleus_shell=True)

    def build(log):
        style = plot_utils.PlotStyle(
            force_unit="nN", show_components=True,
            show_component_heights=True, log_scale=log, height=520,
        )
        return plot_utils.force_curve_figure(
            eps, force, style, fit_force_N=model.composition_curve(eps, fit),
            membrane_N=basis["membrane"] * fit["Em"],
            interior_N=basis["interior"] * fit["Ei"],
            nucleus_shell_N=basis["nucleus_shell"] * fit["Ene"],
            nucleus_N=basis["nucleus"] * fit["En"],
            deep_label="inside the nucleus",
        )

    for log in (True, False):
        fig = build(log)
        labels = [a for a in fig.layout.annotations
                  if a.text and " nN" in str(a.text)]
        check(f"{'log' if log else 'linear'}: every curve is labelled",
              len(labels) == 5, str(len(labels)))
        # Anchored inside the axes. Hung off the right-hand end with
        # xanchor="left" they were clipped by the plot area, and ticking the
        # box appeared to do nothing at all.
        check(f"{'log' if log else 'linear'}: they sit inside the axes",
              all(a.xanchor == "right" and (a.xshift or 0) <= 0
                  for a in labels),
              str([(a.xanchor, a.xshift) for a in labels]))
        ys = sorted(float(a.y) for a in labels)
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        check(f"{'log' if log else 'linear'}: none lands on another",
              all(g > 0 for g in gaps), str([round(g, 4) for g in gaps]))
        check(f"{'log' if log else 'linear'}: each is boxed to be readable",
              all(a.bgcolor for a in labels))

    # With the heights off, no labels at all.
    plain = plot_utils.PlotStyle(force_unit="nN", show_components=True,
                                 show_component_heights=False)
    quiet = plot_utils.force_curve_figure(
        eps, force, plain, fit_force_N=model.composition_curve(eps, fit),
        membrane_N=basis["membrane"] * fit["Em"],
    )
    check("switched off, nothing is labelled",
          not [a for a in quiet.layout.annotations
               if a.text and " nN" in str(a.text)])


def case_the_membrane_protein_is_put_aside():
    print("the in-plane spring is out of the cardiomyocyte for now")
    import app as app_module
    check("it is not one of the cardiomyocyte's materials",
          "tension" not in app_module.terms_for("Cardiomyocyte"),
          str(app_module.terms_for("Cardiomyocyte")))
    check("no picture of the cell offers one",
          not any("tension" in p["terms"]
                  for p in app_module.cardiomyocyte_hypotheses()),
          str([p["terms"] for p in app_module.cardiomyocyte_hypotheses()]))
    source = SOURCE
    check("and the genotype selector is off the sidebar",
          'key="membrane_protein"' not in source)
    # The term itself is still in the model layer, so putting the control
    # back is one entry in OPTIONAL_TERMS rather than a rewrite.
    from lulevich_model import COMPOSITION_TERMS
    check("the term is still in the model, ready to come back",
          "tension" in COMPOSITION_TERMS, str(COMPOSITION_TERMS))

    app = start(cell_name="cm-01", cell_type="Cardiomyocyte")
    if not no_exception(app, "no in-plane spring"):
        return
    fitted = (app.session_state["_last_fit"] or {}).get("terms") or []
    check("nothing fits one", "tension" not in fitted, str(fitted))
    labels = " ".join(str(c.label) for c in app.checkbox)
    check("and nothing on the page offers one",
          "spring protein" not in labels.lower(), labels[:200])


def case_the_interior_carries_first_contact():
    print("the interior is loaded from contact and the sarcolemma may wait")
    eps = np.linspace(0.002, 0.62, 340)
    seed = LulevichModel(np.zeros_like(eps), eps, cell_height=19.0e-6,
                         cell_radius=10.45e-6, confinement=1.1,
                         deep_uses_cell_radius=True)
    basis = seed.composition_basis(eps, 0.18, 0.45, "late", "zero")
    check("the interior is loaded from first contact",
          float(basis["interior"][0]) > 0.0)
    check("a late sarcolemma contributes nothing before ε₁",
          float(np.max(basis["membrane"][eps < 0.18])) == 0.0)
    check("and the myofibrils wait for ε₂",
          float(np.max(basis["nucleus"][eps < 0.45])) == 0.0)

    truth = {"Em": 1.5e6, "Ec": 2.0e3, "En": 3.0e3}
    force = (basis["membrane"] * truth["Em"] + basis["interior"] * truth["Ec"]
             + basis["nucleus"] * truth["En"])
    rng = np.random.default_rng(1)
    noisy = force * (1.0 + 0.01 * rng.standard_normal(eps.size))
    model = LulevichModel(noisy, eps, cell_height=19.0e-6,
                          cell_radius=10.45e-6, confinement=1.1,
                          deep_uses_cell_radius=True)
    fit = model.fit_composition(
        0.0, 0.62, 0.18, 0.45, "late", "zero", use_nucleus=True,
        weighting="relative",
    )
    for key, want, tol in (("Em_MPa", 1.5, 0.15), ("Ei_kPa", 2.0, 0.3),
                           ("En_kPa", 3.0, 0.4)):
        check(f"{key} is recovered", abs(fit[key] - want) < tol,
              f"{fit[key]:.4g} against {want}")
    check("and the curve is followed", fit["r_squared"] > 0.999,
          f"R2 {fit['r_squared']:.6f}")


def case_the_cardiomyocyte_curves_fit():
    print("the cardiomyocyte model fits the measured curves")
    curves = vcm_curves()
    if not curves:
        check("the VCM reference curves are in the repository", False)
        return
    import app as app_module
    for n, (eps, force) in curves.items():
        app = AppTest.from_file(APP,
                                default_timeout=900)
        app.run()
        app.session_state["cell_type"] = "Cardiomyocyte"
        app.session_state["cell_name"] = f"vcm-{n}"
        app.session_state["data"] = {
            "epsilon": eps, "force_N": force, "source": f"vcm_{n}.csv",
            "n_dropped": 0,
        }
        app.run()
        if not no_exception(app, f"cell {n}"):
            continue
        # A loaded curve is fitted at this cell type's stated boundaries.
        # Moving them onto the curve is the optimisation button's job --
        # fitting fits what it is given -- so both are pressed, in the
        # order a person would press them.
        button = button_by_label(app, "Find the boundaries from the log curve")
        if button is not None:
            button.click().run()
            if not no_exception(app, f"cell {n} refined"):
                continue
        button = button_by_label(app, "Fit this cell")
        if button is not None:
            button.click().run()
            if not no_exception(app, f"cell {n} fitted"):
                continue
        fit = app.session_state["_last_fit"]
        check(f"cell {n} is fitted", fit and fit.get("success"))
        if not (fit and fit.get("success")):
            continue
        check(f"cell {n}: R² is at least 0.9999",
              fit["r_squared"] > 0.9999, f"{fit['r_squared']:.6f}")
        # The number that says the residuals are the size of the noise, not
        # merely small next to a curve spanning four decades.
        check(f"cell {n}: chi-squared per point is near one",
              fit["chi_squared_reduced"] < 3.0,
              f"{fit['chi_squared_reduced']:.3g}")
        check(f"cell {n}: the interior carries load",
              fit.get("Ei_kPa", 0.0) > 0.0, str(fit.get("Ei_kPa")))
        check(f"cell {n}: nothing is called a nucleus",
              "nucleus" not in " ".join(
                  str(m.value) for m in app.get("markdown")).lower())


def case_legacy_record_refits():
    print("a record saved under the old model names still resolves")
    import app as app_module
    for old, key in app_module.LEGACY_MODEL_NAMES.items():
        check(f"{old!r} resolves", app_module.MODEL_KEYS_ANY.get(old) == key)
    for new, key in app_module.MODEL_KEYS.items():
        check(f"{new!r} resolves", app_module.MODEL_KEYS_ANY.get(new) == key)


def case_one_video_two_doors():
    print("uploading in either place loads the same video")
    source = pathlib.Path(__file__).with_name("app.py").read_text()
    check("both uploaders go through one adopter",
          source.count("adopt_video(") >= 3, str(source.count("adopt_video(")))
    check("only the adopter and the link fetcher set the path",
          source.count('st.session_state["video_path"] = ') == 2,
          str(source.count('st.session_state["video_path"] = ')))
    check("every frame is read through one place",
          source.count("cached_detection(") == 2,
          str(source.count("cached_detection(")))
    check("the side panel no longer calls the detector directly",
          "vframe, vdet, vnuc, vprobe, vscale = detection_at(" in source)

    # The adopter must act only when its own box changes. Keyed on the loaded
    # video's name instead, two boxes holding different files overwrite each
    # other on alternate reruns, one winning per pass.
    class Fake:
        def __init__(self, name, data):
            self.name, self._data, self.size = name, data, len(data)

        def getvalue(self):
            return self._data

    seen, taken = {}, []

    def adopt(uploaded, widget_key):
        signature = None if uploaded is None else (uploaded.name, uploaded.size)
        if signature == seen.get(widget_key):
            return False
        seen[widget_key] = signature
        if uploaded is None:
            return False
        taken.append((widget_key, uploaded.name))
        return True

    a, b = Fake("a.mp4", b"aaa"), Fake("b.mp4", b"bbbb")
    check("the first upload is taken", adopt(a, "main") is True)
    check("the same box unchanged is ignored", adopt(a, "main") is False)
    check("the other box with a new file is taken", adopt(b, "tab") is True)
    check("and the first box does not grab it back", adopt(a, "main") is False)
    check("only two adoptions happened", len(taken) == 2, str(taken))
    check("starting a new cell forgets what each box handed over",
          'st.session_state["_video_seen"] = {}' in source,
          "otherwise re-uploading the same file is ignored")


def case_tables_are_not_clipped():
    print("no table can lose a column off the right-hand edge")
    source = pathlib.Path(__file__).with_name("app.py").read_text()
    check("the fixed-layout style is defined", "table-layout: fixed" in source)
    check("cells wrap rather than overflow", "overflow-wrap: anywhere" in source)

    app = start(cell_name="cell-01", cell_type="Cardiomyocyte")
    if not no_exception(app, "tables"):
        return
    tables = flat_tables(app)
    check("the result tables are drawn as flat tables", len(tables) >= 2,
          str(len(tables)))
    for frame in tables:
        check(f"every column of the {len(frame.columns)}-column table is named",
              all(str(c).strip() for c in frame.columns), str(list(frame.columns)))
        check("no row is short of cells",
              all(len(row) == len(frame.columns)
                  for row in frame.itertuples(index=False)),
              str(frame.shape))


def case_plot_clutter_toggles():
    print("each piece of clutter can be switched off")
    import plot_utils

    eps, force = synthetic()
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.60, 0.15, 0.40)
    mb, cb, nb = model.composition_terms(eps, 0.15, 0.40)
    fitted = mb * fit["Em"] + cb * fit["Ei"] + nb * fit["En"]

    def build(**flags):
        style = plot_utils.PlotStyle(force_unit="N", **flags)
        return plot_utils.force_curve_figure(
            eps, force, style, fit_force_N=fitted,
            fit_window=[{"range": (0.0, 0.6), "label": "fit"}],
            highlight_window=(0.15, 0.40, "segment 2"),
            highlight=(0.30, float(force[len(force) // 2])),
            rupture_epsilon=0.55,
        )

    on = build()
    check("shading on by default", len(on.layout.shapes) >= 2)
    check("video marker on by default",
          any(t.name == "video frame" for t in on.data))

    off = build(show_fit_window=False)
    check("shading gone", not [s for s in off.layout.shapes if s.type == "rect"])

    off = build(show_video_marker=False)
    check("video frame marker gone",
          not any(t.name == "video frame" for t in off.data))

    off = build(show_rupture_marker=False)
    labels = [getattr(a, "text", "") for a in off.layout.annotations]
    check("rupture marker gone", "rupture" not in labels, str(labels))

    with_moduli = plot_utils.cell_schematic(
        plot_utils.PlotStyle(force_unit="N"), epsilon=0.3,
        break_1=0.15, break_2=0.40, Em_MPa=0.6, Ei_kPa=1.2, En_kPa=3.0,
    )
    without = plot_utils.cell_schematic(
        plot_utils.PlotStyle(force_unit="N", show_schematic_moduli=False),
        epsilon=0.3, break_1=0.15, break_2=0.40,
        Em_MPa=0.6, Ei_kPa=1.2, En_kPa=3.0,
    )

    def caption(fig):
        return " ".join(getattr(a, "text", "") or "" for a in fig.layout.annotations)

    check("moduli printed by default", "E<sub>m</sub>" in caption(with_moduli))
    check("moduli gone when switched off",
          "E<sub>m</sub>" not in caption(without) and "E<sub>c</sub>" not in caption(without))
    check("the diagram still says where it is",
          "ε =" in caption(without), caption(without)[:80])


def case_one_range_control_for_the_whole_page():
    print("one fitted range, set in one place, whichever model is chosen")
    app = start()
    bars = [s for s in app.slider if (s.label or "").startswith("Fitted range")]
    check("exactly one range bar on the page", len(bars) == 1,
          str([s.label for s in app.slider]))
    check("and it has two handles", len(bars) == 1
          and isinstance(bars[0].value, (list, tuple)), str(bars[0].value if bars else None))
    if bars:
        bars[0].set_value((0.0, 0.35)).run()
        no_exception(app, "moving the range")
        lo, hi = app.session_state["window_combined"]
        check("the combined window follows the bar", lo == 0.0 and abs(hi - 0.35) < 0.01,
              f"{lo} to {hi}")
        check("and so do the two numbers behind it",
              app.session_state["window_start"] == 0.0
              and abs(app.session_state["window_end"] - 0.35) < 0.01,
              f"{app.session_state['window_start']} to "
              f"{app.session_state['window_end']}")

    # Switching model does not move the range or produce a second control.
    app2 = start()
    widget_by_label(app2, "radio", "how the cell is modelled").set_value(
        "Side by side (every element acts everywhere)"
    ).run()
    bars = [s for s in app2.slider if (s.label or "").startswith("Fitted range")]
    check("still one range bar for the other models", len(bars) == 1,
          str([s.label for s in app2.slider]))


def case_fit_stops_at_the_end_of_the_range():
    print("the model line stops where the range stops")
    import plot_utils

    eps, force = synthetic()
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.35, 0.15, 0.30)
    check("fit succeeded on a short range", fit.get("success"))
    if not fit.get("success"):
        return
    mb, cb, nb = model.composition_terms(eps, 0.15, 0.30)
    full = mb * fit["Em"] + cb * fit["Ei"] + nb * fit["En"]
    lo, hi = fit["epsilon_range"]
    clipped = np.array(full, dtype=float)
    clipped[(eps < lo) | (eps > hi)] = np.nan

    fig = plot_utils.force_curve_figure(
        eps, force, plot_utils.PlotStyle(force_unit="N"), fit_force_N=clipped
    )
    line = next(t for t in fig.data if t.name == "Model")
    drawn = np.asarray(line.y, dtype=float)
    finite = np.isfinite(drawn)
    check("nothing drawn past the end of the range",
          not finite[eps > hi + 1e-9].any())
    check("the whole range is drawn", finite[(eps >= lo) & (eps <= hi)].all())
    check("the line reaches the end of the range",
          abs(eps[finite].max() - hi) < 0.01, f"{eps[finite].max():.3f} vs {hi:.3f}")


def case_composition_curve_rebuilds_the_fit():
    print("a finished fit can be redrawn at any deformations")
    eps, force = synthetic()
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.60, 0.15, 0.40, "freeze", "break")
    basis = model.composition_basis(eps, 0.15, 0.40, "freeze", "break")
    by_hand = (basis["membrane"] * fit["Em"] + basis["interior"] * fit["Ei"]
               + basis["nucleus"] * fit["En"])
    rebuilt = model.composition_curve(eps, fit)
    check("the rebuilt curve is the fitted curve",
          np.allclose(rebuilt, by_hand, rtol=1e-10, atol=0.0),
          f"max gap {float(np.max(np.abs(rebuilt - by_hand))):.3e}")
    # And on a grid finer than the data, which is what it is for.
    fine = np.linspace(0.0, 0.60, 1000)
    smooth = model.composition_curve(fine, fit)
    check("it works on a grid the data does not have",
          smooth.size == fine.size and np.all(np.isfinite(smooth)))
    check("it rises the whole way", np.all(np.diff(smooth) >= -1e-18))


def case_breakpoints_are_searched_when_they_matter():
    print("ε₁ is fitted whenever it is in the model, not only with a deep layer")
    from lulevich_model import compare_hypotheses

    eps = np.linspace(0.002, 0.60, 300)
    seed = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
    basis = seed.composition_basis(eps, 0.22, 0.45, "freeze", "break")
    force = basis["membrane"] * 0.6e6 + basis["interior"] * 1.5e3
    # A model whose stored boundary is nowhere near the truth. If the search
    # is skipped for a two-term hypothesis, this is the number that comes
    # back, and the fit is poor for a reason nobody can see.
    model = LulevichModel(force, eps, cell_height=8.0e-6,
                          segment_break_1=0.05, segment_break_2=0.55)
    out = compare_hypotheses(
        model, 0.0, 0.60,
        [{"key": "handover", "label": "handover",
          "terms": ("membrane", "interior"),
          "membrane": "freeze", "cyto_start": "break"}],
        weighting="relative", cv_repeats=2, n_grid=8,
    )
    check("the two-term hypothesis was fitted", out.get("success"),
          str(out.get("error")))
    if out.get("success"):
        found = out["best"]["break_1"]
        check("ε₁ moved off the stored value", abs(found - 0.05) > 0.02,
              f"{found:.3f}")
        check("and landed near the truth", abs(found - 0.22) < 0.05,
              f"{found:.3f}")
        check("so the fit is good", out["best"]["r_squared"] > 0.9995,
              f"R2 {out['best']['r_squared']:.5f}")


def case_ordering_is_a_question_about_two_springs():
    print("the ordering comparison separates the balloon from the spring")
    from lulevich_model import (
        ORDERINGS, compare_orderings, near_contact_exponent, ordering_of,
    )

    check("four orderings are offered", len(ORDERINGS) == 4,
          f"got {len(ORDERINGS)}")
    keys = {row["key"] for row in ORDERINGS}
    check("membrane first and cytoskeleton first are both there",
          {"membrane_first", "cyto_first", "together"} <= keys, str(keys))
    # Each has to be a distinct composition, or two of them are the same fit
    # under two names and the comparison means nothing.
    pairs = {(row["membrane"], row["cyto_start"]) for row in ORDERINGS}
    check("no two orderings are the same composition",
          len(pairs) == len(ORDERINGS), str(pairs))
    for row in ORDERINGS:
        found = ordering_of(row["membrane"], row["cyto_start"])
        check(f"ordering_of finds {row['key']}",
              found is not None and found["key"] == row["key"])

    # A curve built as a pure Hertzian contact with the membrane switched on
    # late must be read as cytoskeleton first, and one built the classic way
    # as membrane first. This is the arithmetic, checked on curves whose
    # answer is known by construction.
    for membrane, cyto_start, expect in (
        ("late", "zero", "cyto_first"),
        ("freeze", "break", "membrane_first"),
    ):
        eps = np.linspace(0.002, 0.60, 320)
        model = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
        basis = model.composition_basis(eps, 0.18, 0.42, membrane, cyto_start)
        force = basis["membrane"] * 0.6e6 + basis["interior"] * 1.5e3
        built = LulevichModel(force, eps, cell_height=8.0e-6)
        out = compare_orderings(
            built, 0.0, 0.60, terms=("membrane", "interior"),
            weighting="relative", cv_repeats=2, n_grid=8,
        )
        check(f"a curve built {membrane}/{cyto_start} is read as {expect}",
              out.get("success") and out["best"]["key"] == expect,
              str(out.get("best", {}).get("key")))
        if out.get("success"):
            check(f"  and it fits it well ({expect})",
                  out["best"]["r_squared"] > 0.999,
                  f"R2 {out['best']['r_squared']:.5f}")

    # The near-contact slope is the model-free half of the answer, so it has
    # to read 3/2 on a Hertzian start and 3 on a membrane start.
    eps = np.linspace(0.002, 0.60, 320)
    model = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
    hertz = LulevichModel(
        model.composition_basis(eps, 0.18, 0.42, "late", "zero")["interior"] * 1.5e3,
        eps, cell_height=8.0e-6,
    )
    slope, r2, upto = near_contact_exponent(hertz, 0.0, 0.60)
    check("a Hertzian start reads about 3/2", 1.35 < slope < 1.65,
          f"{slope:.2f}")
    shell = LulevichModel(
        model.composition_basis(eps, 0.18, 0.42, "freeze", "break")["membrane"] * 0.6e6,
        eps, cell_height=8.0e-6,
    )
    slope_m, _, _ = near_contact_exponent(shell, 0.0, 0.60)
    check("a membrane start reads about 3", 2.7 < slope_m < 3.3, f"{slope_m:.2f}")

    # And it refuses the question when it cannot be asked.
    refused = compare_orderings(
        hertz, 0.0, 0.60, terms=("membrane",), weighting="relative",
    )
    check("one spring alone is refused, with a reason",
          not refused.get("success") and "membrane" in refused.get("error", ""),
          str(refused.get("error"))[:60])


def case_ordering_reads_the_real_cardiomyocytes():
    print("the four measured VCM curves are read the way the data says")
    from lulevich_model import compare_orderings

    curves = vcm_curves()
    if not curves:
        check("the VCM reference curves are in the repository", False)
        return

    verdicts = {}
    for n, (eps, force) in curves.items():
        model = vcm_model(eps, force)
        window = model.suggest_window()
        lo = window["epsilon_min"] if window.get("success") else 0.0
        hi = window["epsilon_max"] if window.get("success") else float(eps.max())
        out = compare_orderings(
            model, lo, hi, terms=("membrane", "interior", "nucleus"),
            weighting="relative", cv_repeats=2, n_grid=8,
        )
        if not out.get("success"):
            check(f"cell {n} produced an answer", False, str(out.get("error")))
            continue
        verdicts[n] = out
        check(f"cell {n}: every ordering was fitted",
              len(out["candidates"]) == 4, str(len(out["candidates"])))
        check(f"cell {n}: the winner fits it",
              out["best"]["r_squared"] > 0.999,
              f"R2 {out['best']['r_squared']:.5f}")
        check(f"cell {n}: the near-contact slope was measured",
              np.isfinite(out["near_contact_exponent"]))

    # Cells 11 and 14 are the clean ones: they start at contact and their
    # first stretch is a Hertzian slope, so both halves of the answer have to
    # come out cytoskeleton first. This is the finding the tab exists to
    # show, and it is checked against measured data rather than asserted.
    for n in (11, 14):
        if n not in verdicts:
            continue
        out = verdicts[n]
        check(f"cell {n} is read as cytoskeleton first",
              out["best"]["key"] == "cyto_first", out["best"]["key"])
        check(f"cell {n} starts on the Hertzian slope",
              1.3 < out["near_contact_exponent"] < 1.9,
              f"{out['near_contact_exponent']:.2f}")
        check(f"cell {n}: the slope and the fit agree",
              "agree" in out["reading"], out["reading"][:70])
        beaten = [r for r in out["candidates"] if r["key"] == "membrane_first"]
        check(f"cell {n} beats the classic membrane-first order",
              beaten and beaten[0]["cv_rmse"] > out["best"]["cv_rmse"] * 1.5,
              "")

    # Cell 3 has a bad contact, so its range does not start at contact and
    # the slope measured inside it cannot answer the question. Saying so is
    # the point: a number quoted from the middle of a squash as if it were
    # the start is how a wrong answer looks right.
    if 3 in verdicts:
        check("a range that misses contact says the slope cannot answer",
              "cannot say which spring answered first" in verdicts[3]["reading"],
              verdicts[3]["reading"][-90:])


def case_balloon_and_spring_tab_answers_by_itself():
    print("the log curve tab holds the picture, the algebra and the search")
    source = pathlib.Path(APP).read_text()
    check("the balloon tab is gone from the bar",
          "🎈 Balloon and spring" not in source)
    check("and the log curve has its place instead",
          '"📈 Log curve and boundaries"' in source, source[:0])
    check("the hidden-tab indices were left alone",
          "((3, SHOW_VIDEO_TAB), (5, SHOW_DATABASE_TAB))" in source)

    curves = vcm_curves()
    if not curves:
        check("the VCM reference curves are in the repository", False)
        return
    eps, force = curves[11]
    app = AppTest.from_file(APP, default_timeout=900)
    app.run()
    app.session_state["cell_type"] = "Cardiomyocyte"
    app.session_state["data"] = {
        "epsilon": eps, "force_N": force, "source": "vcm_11.csv", "n_dropped": 0,
    }
    app.run()
    if not no_exception(app, "the log curve tab runs"):
        return

    said = " ".join(str(m.value) for m in
                    list(app.get("markdown")) + list(app.get("caption")))
    for wanted in ("Where the curve changes its power law",
                   "What the curve is fitting, stretch by stretch",
                   "Find the boundaries from this curve"):
        check(f"“{wanted[:34]}…” is on the tab", wanted in said, said[:200])

    # The algebra of each stretch, which is what says one term or two.
    formulas = " ".join(str(e.value) for e in app.get("latex"))
    check("each stretch is written as the terms carrying it",
          formulas.count(r"\le \varepsilon <") >= 2, formulas[:300])
    check("and the slope reading is defined from the data",
          r"\frac{d\log F}{d\log \varepsilon}" in formulas, formulas[:300])
    check("one stretch is named as a single term",
          "one term" in said, said[-600:])
    check("and another as two carrying together",
          "carrying together" in said, said[-600:])

    # The search lives here now, and it moves the boundaries.
    was = (float(state(app, "segment_break_1")),
           float(state(app, "segment_break_2")))
    work = button_by_label(app, "Find the boundaries from the log curve")
    check("the boundary search is on this tab", work is not None,
          str([b.label for b in app.button][:10]))
    if work is None:
        return
    work.click().run()
    if not no_exception(app, "searching from the log curve"):
        return
    now = (float(state(app, "segment_break_1")),
           float(state(app, "segment_break_2")))
    check("pressing it moves the boundaries", now != was, f"{was} -> {now}")
    rows = state(app, "boundary_candidates")
    check("and the placements it weighed are listed",
          rows and len(rows) >= 3, str(len(rows or [])))


def case_the_prefactors_are_the_papers():
    print("the prefactors are Lulevich 2006 eq 3 and eq 6, not near enough")
    model = LulevichModel(np.ones(50), np.linspace(0.01, 0.60, 50),
                          8.0e-6, cell_radius=4.4e-6,
                          membrane_thickness=4.0e-9,
                          poisson_membrane=0.5, poisson_interior=0.0)
    # eq 3: F_m = 2 pi Em h R0 e^3 / (1 - nu_m)
    want_m = 2 * np.pi * 4.0e-9 * 4.4e-6 / (1 - 0.5)
    check("Am is eq 3", abs(model.Am - want_m) < want_m * 1e-12,
          f"{model.Am:.6g} against {want_m:.6g}")
    # eq 6: F_i = sqrt(2) Ei R0^2 e^1.5 / (3 (1 - nu_i^2))
    want_i = np.sqrt(2) * 4.4e-6 ** 2 / (3 * (1 - 0.0 ** 2))
    check("Ai is eq 6, with R0 squared",
          abs(model.Ai - want_i) < want_i * 1e-12,
          f"{model.Ai:.6g} against {want_i:.6g}")
    # The form it used to have. Kept as a check rather than a comment,
    # because the two agree only for h0 = 2 R0 and differ by about 2.5x for
    # the cell shapes this app is used on, straight into every interior
    # modulus it reports.
    old_form = np.sqrt(2) * np.sqrt(4.4e-6) * 8.0e-6 ** 1.5 / 3.0
    check("and that is not the old sqrt(R0) h0^1.5 form",
          abs(model.Ai - old_form) > old_form * 0.5,
          f"{model.Ai:.6g} against the old {old_form:.6g}")
    # The two forms never agree: their ratio is (h0/R0)^1.5, which is 2.83
    # for a sphere and about 2.45 for the shapes here.
    sphere = LulevichModel(np.ones(50), np.linspace(0.01, 0.60, 50),
                           8.0e-6, cell_radius=4.0e-6, poisson_interior=0.0)
    old_sphere = np.sqrt(2) * np.sqrt(4.0e-6) * 8.0e-6 ** 1.5 / 3.0
    check("for a sphere the old form was 2 sqrt(2) times too big",
          abs(old_sphere / sphere.Ai - 2 * np.sqrt(2)) < 1e-6,
          f"{old_sphere / sphere.Ai:.4f}")

    # eq 5 and eq 2: bending, which the paper writes out and then drops.
    check("bending goes as the square root of the deformation",
          abs(model.bending_force(0.25, 30e6)
              / model.bending_force(1.0, 30e6) - np.sqrt(0.25)) < 1e-9)
    at_thirty = model.bending_force(0.30, 30e6)
    check("and at 30 % with a 30 MPa membrane it is under a nanonewton",
          at_thirty < 1e-9, f"{at_thirty * 1e9:.3g} nN")
    check("eq 2 puts it well under a twentieth of the stretching term",
          model.bending_share(0.30) < 0.05,
          f"{model.bending_share(0.30):.4f}")


def case_the_written_equation_is_the_fitted_curve():
    print("the equation printed on the page evaluates to the fitted curve")
    import app as app_module

    eps, force = synthetic()
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    model.confinement = 0.8
    fit = model.fit_composition(0.0, 0.85, 0.15, 0.40, membrane="freeze",
                                cyto_start="break")
    check("the fit succeeded", fit.get("success"))
    if not fit.get("success"):
        return

    pieces = app_module.equation_pieces(fit)
    check("every fitted term is in the equation",
          {p["term"] for p in pieces} == set(fit["terms"]),
          f"{[p['term'] for p in pieces]} vs {fit['terms']}")
    check("and it carries the confinement it was fitted at",
          abs(float(fit["confinement"]) - 0.8) < 1e-9,
          str(fit.get("confinement")))

    # Rebuild the curve from the printed coefficients alone: coefficient
    # times shape of epsilon, times the confinement factor. If this does not
    # land on the fitted curve then the equation on the page is not the
    # model that was fitted, which is worse than printing nothing.
    e1, e2 = float(fit["break_1"]), float(fit["break_2"])
    held = np.clip(np.minimum(eps, e1), 0.0, None)
    shapes = {
        "tension": held,
        "membrane": held ** 3,
        "cortex": np.clip(eps, 0.0, None) ** 1.5,
        "interior": np.clip(eps - e1, 0.0, None) ** 1.5,
        "nucleus_shell": np.clip(eps - e2, 0.0, None) ** 3,
        "nucleus": np.clip(eps - e2, 0.0, None) ** 1.5,
    }
    by_hand = np.zeros_like(eps)
    for piece in pieces:
        by_hand = by_hand + piece["coefficient_N"] * shapes[piece["term"]]
    by_hand = by_hand * (1.0 - np.clip(eps, 0.0, 0.999)) ** (-0.8)
    theirs = model.composition_curve(eps, fit)
    scale = float(np.max(np.abs(theirs))) or 1.0
    worst = float(np.max(np.abs(by_hand - theirs))) / scale
    check("the printed coefficients rebuild the fitted curve",
          worst < 1e-9, f"worst difference {worst:.3g} of full scale")

    # And the coefficient really is prefactor times modulus, in newtons.
    for piece in pieces:
        if piece["term"] != "interior":
            continue
        check("the interior coefficient is Ai times Ec",
              abs(piece["coefficient_N"] - fit["Ai"] * fit["Ei"])
              < 1e-12 * max(1.0, abs(fit["Ai"] * fit["Ei"])),
              f"{piece['coefficient_N']:.6g} vs {fit['Ai'] * fit['Ei']:.6g}")


def case_the_equation_reaches_the_page():
    print("the equation is on the page after a fit, with numbers in it")
    app = start(cell_name="cell-01")
    if not no_exception(app, "the equation on the page"):
        return
    formulas = [str(getattr(e, "value", "")) for e in app.get("latex")]
    joined = " ".join(formulas)
    check("an equation for F is printed", "F(\\varepsilon)" in joined,
          joined[:160])
    check("with the moduli named in it", "E_c" in joined or "E_m" in joined,
          joined[:160])
    check("and a numeric version beside it",
          "frac{F(\\varepsilon)}" in joined, joined[:200])


def case_the_plot_says_which_range_was_used():
    print("the plot carries a note saying which range the numbers came from")
    import plot_utils

    eps, force = synthetic()
    style = plot_utils.PlotStyle(force_unit="nN",
                                 range_note="fitted ε = 0.000 to 0.600")
    fig = plot_utils.force_curve_figure(eps, force, style)
    notes = [getattr(a, "text", "") for a in fig.layout.annotations]
    check("the note is drawn", any("fitted ε" in str(n) for n in notes),
          str(notes))
    check("against the figure, not the axes, so zooming keeps it",
          any(getattr(a, "xref", "") == "paper" for a in fig.layout.annotations))

    bare = plot_utils.force_curve_figure(
        eps, force, plot_utils.PlotStyle(force_unit="nN"))
    check("and nothing is drawn when there is no note",
          not any("fitted ε" in str(getattr(a, "text", ""))
                  for a in bare.layout.annotations))

    # The app's own note, off the fit.
    app = start(cell_name="cell-01")
    if not no_exception(app, "the range note"):
        return
    style_note = None
    try:
        import app as app_module
        style_note = app_module.fit_range_note()
    except Exception as exc:  # no running app around this helper
        style_note = f"unavailable: {exc}"
    check("the app writes a note naming a range",
          "ε" in str(style_note) and "to" in str(style_note),
          str(style_note))


def case_the_zoom_survives_a_refit():
    print("zooming in and refitting keeps the view")
    import plot_utils

    eps, force = synthetic()
    first = plot_utils.force_curve_figure(
        eps, force, plot_utils.PlotStyle(force_unit="nN", uirevision="cell-a"))
    check("the figure carries a view token", first.layout.uirevision == "cell-a",
          str(first.layout.uirevision))
    again = plot_utils.force_curve_figure(
        eps, force,
        plot_utils.PlotStyle(force_unit="nN", uirevision="cell-a",
                             fit_color="#c0392b"))
    check("redrawing the same curve keeps the same token, so the zoom stays",
          again.layout.uirevision == first.layout.uirevision)
    other = plot_utils.force_curve_figure(
        eps, force, plot_utils.PlotStyle(force_unit="nN", uirevision="cell-b"))
    check("another curve gets a different one, so the view starts over",
          other.layout.uirevision != first.layout.uirevision)

    app = start(cell_name="cell-01")
    if not no_exception(app, "the view token"):
        return
    before = state(app, "_view_token_seen")
    import app as app_module
    # What matters is that the token does not move when the curve is
    # refitted: that is what keeps the zoom. Two runs of the same page with
    # the same curve must produce the same one.
    first = str(app_module.view_token())
    button = button_by_label(app, "Fit this cell")
    if button is not None:
        button.click().run()
        no_exception(app, "refitting while zoomed")
    check("the token is the same after a fit, so the view is kept",
          str(app_module.view_token()) == first,
          f"{first} then {app_module.view_token()}")
    check("and it names the curve, not the fit", "|" in first, first)
    del before


def case_axis_numbers_are_powers_not_letters():
    print("tick labels use powers of ten, never SI letters")
    import plot_utils

    eps, force = synthetic()
    fig = plot_utils.force_curve_figure(
        eps, force, plot_utils.PlotStyle(force_unit="N"))
    check("the force axis is in powers of ten",
          fig.layout.yaxis.exponentformat == "power",
          str(fig.layout.yaxis.exponentformat))
    check("and so is the deformation axis",
          fig.layout.xaxis.exponentformat == "power",
          str(fig.layout.xaxis.exponentformat))
    check("the unit stays in the axis title, where it belongs",
          "N" in str(fig.layout.yaxis.title.text),
          str(fig.layout.yaxis.title.text))


def case_a_fixed_cell_can_have_several_hertzian_terms():
    print("a fixed cell can be given more than one Hertzian term")
    import app as app_module

    offered = app_module.OPTIONAL_TERMS["Fixed cell"]
    check("three Hertzian slots are offered",
          set(offered) == {"cortex", "interior", "nucleus"}, str(offered))
    check("and no shell term among them",
          not any(t in offered for t in ("membrane", "tension", "nucleus_shell")),
          str(offered))
    picks = app_module.HYPOTHESES["Fixed cell"]
    check("one, two and three solids are all offered", len(picks) == 3,
          str([p["key"] for p in picks]))
    check("only one of them is on by default",
          sum(1 for on in app_module.DEFAULT_TERMS_BY_TYPE["Fixed cell"].values()
              if on) == 1,
          str(app_module.DEFAULT_TERMS_BY_TYPE["Fixed cell"]))
    # Every picture must give its terms different onsets, or two of them are
    # one term with two names and the split between them is arbitrary.
    for spec in picks:
        if "cortex" in spec["terms"] and "interior" in spec["terms"]:
            check("with the cortex in, the interior starts at ε₁",
                  spec["cyto_start"] == "break", str(spec))
        elif spec["terms"] == ("interior",) or "cortex" not in spec["terms"]:
            check("without it, the one from contact starts at zero",
                  spec["cyto_start"] == "zero", str(spec))

    # Two Hertzian bodies, the deeper one stiffer, met at eps2. This is the
    # fixed cardiomyocyte: cross-linked bundles inside cross-linked cytoplasm.
    eps = np.linspace(0.002, 0.70, 420)
    blank = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
    basis = blank.composition_basis(eps, 0.15, 0.35, "continue", "zero")
    force = basis["interior"] * 120e3 + basis["nucleus"] * 400e3
    rng = np.random.default_rng(3)
    force = force * (1.0 + 0.005 * rng.standard_normal(eps.size))

    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.70, 0.15, 0.35, membrane="continue",
                                cyto_start="zero", use_membrane=False,
                                use_interior=True, use_nucleus=True,
                                weighting="relative")
    check("two Hertzian terms can be fitted together", fit.get("success"))
    if not fit.get("success"):
        return
    check("the first solid comes back", abs(fit["Ei_kPa"] - 120.0) < 12.0,
          f"{fit['Ei_kPa']:.4g} kPa")
    check("and the deeper, stiffer one too",
          abs(fit["En_kPa"] - 400.0) < 40.0, f"{fit['En_kPa']:.4g} kPa")
    check("with no membrane term in the answer",
          "membrane" not in fit["terms"], str(fit["terms"]))


def case_each_boundary_moves_on_its_own():
    print("each boundary can be set by itself, by bar or by typing")
    app = start(cell_name="cell-01")
    if not no_exception(app, "boundary controls"):
        return
    first = [s for s in app.slider if (s.label or "").startswith("ε₁")]
    second = [s for s in app.slider if (s.label or "").startswith("ε₂")]
    check("there is a bar for ε₁", len(first) == 1,
          str([s.label for s in app.slider][:8]))
    check("and one for ε₂", len(second) == 1,
          str([s.label for s in app.slider][:8]))
    if not (first and second):
        return
    was_2 = float(state(app, "segment_break_2", 0.0))
    first[0].set_value(0.22).run()
    if not no_exception(app, "moving the first boundary"):
        return
    check("moving ε₁ moves ε₁",
          abs(float(state(app, "segment_break_1", 0.0)) - 0.22) < 0.01,
          str(state(app, "segment_break_1")))
    check("and leaves ε₂ where it was",
          abs(float(state(app, "segment_break_2", 0.0)) - was_2) < 1e-6,
          f"{was_2} then {state(app, 'segment_break_2')}")

    was_1 = float(state(app, "segment_break_1", 0.0))
    boxes = [n for n in app.number_input if (n.label or "").startswith("ε₂")]
    if boxes:
        boxes[0].set_value(0.55).run()
        no_exception(app, "typing the second boundary")
        check("typing ε₂ moves ε₂",
              abs(float(state(app, "segment_break_2", 0.0)) - 0.55) < 0.01,
              str(state(app, "segment_break_2")))
        check("and leaves ε₁ alone",
              abs(float(state(app, "segment_break_1", 0.0)) - was_1) < 1e-6,
              f"{was_1} then {state(app, 'segment_break_1')}")
    check("neither can be pushed past the other",
          float(state(app, "segment_break_1", 0.0))
          < float(state(app, "segment_break_2", 1.0)),
          f"{state(app, 'segment_break_1')} / {state(app, 'segment_break_2')}")


def case_the_controls_sit_above_the_curve():
    print("the settings are under the curve, the maths and the answer below")
    import app as app_module
    source = SOURCE

    # Order in the source is order on the page: Streamlit draws a container
    # where it was created, so the slots have to be staked out in the order
    # the page should read.
    check("in guided mode the settings are one collapsed line, not three panels",
          source.count("flat=guided") >= 3, str(source.count("flat=guided")))
    check("and that line is in the sidebar, off the page entirely",
          "st.sidebar.expander(\n                \"⚙️ Change what the fit assumed\""
          in source or 'settings_box = st.sidebar.expander(' in source,
          "settings box is not in the sidebar")
    check("the equation is written after the results heading",
          source.index('st.markdown("##### The equation that was fitted")')
          > source.index('section("4 · Fitting results")'))
    check("and the power-law picture has a tab of its own",
          'st.markdown("#### Where the curve changes its power law")'
          in source and '"📈 Log curve and boundaries"' in source)

    app = start(cell_name="cell-01")
    if not no_exception(app, "the reordered page"):
        return
    text = " ".join(str(m.value) for m in app.get("markdown"))
    for wanted in ("Fitting results", "The equation that was fitted",
                   "Where the curve changes its power law"):
        check(f"“{wanted}” is on the page", wanted in text, text[:300])
    labels = [e.label for e in app.expander] if hasattr(app, "expander") else []
    check("the settings are behind one line, not three",
          sum(1 for l in labels if "Change what the fit assumed" in str(l)) == 1,
          str(labels))
    for gone in ("What each material is, and how they are told apart",
                 "🔍 Fit diagnostics",
                 "📊 How well it fits, stretch by stretch"):
        check(f"“{gone[:36]}…” is not on the guided page",
              not any(gone in str(l) for l in labels), str(labels))
    for gone in ("⚙️ Change how the materials share the load",
                 "📏 Where each material takes over", "🔧 Fitting options"):
        check(f"“{gone}” is no longer a panel of its own",
              gone not in labels, str(labels))
    # Still reachable, still working, just not in the way.
    check("the combination search is still reachable",
          button_by_label(app, "Find the best combination") is not None,
          str([b.label for b in app.button][:12]))
    check("so is the boundary search",
          button_by_label(app, "Find the boundaries from the data") is not None,
          str([b.label for b in app.button][:12]))
    check("and the advanced fitting options",
          "Advanced fitting options" in text, text[:200])


def case_the_button_says_what_it_will_do():
    print("the fit button says what it assumes before it is pressed")
    app = start(cell_name="cell-01")
    if not no_exception(app, "the default plan"):
        return
    text = " ".join(str(m.value) for m in
                    list(app.get("markdown")) + list(app.get("caption")))
    check("the equation is with the results, where what was done belongs",
          "The equation that was fitted" in text, text[:300])
    check("and the table of defaults is gone",
          table_with(app, "setting", "default") is None)
    check("one sentence says the range and the weighting instead",
          "Fitted over ε" in text and "weighted" in text, text[:400])
    formulas = " ".join(str(getattr(e, "value", "")) for e in app.get("latex"))
    check("with the model written as an equation",
          "F(\\varepsilon)" in formulas, formulas[:160])
    check("it says the boundaries were found rather than assumed",
          "found from the curve" in text, text[:400])
    check("and how the winner among the arrangements is chosen",
          "predicts points it was not fitted to" in text, text[:600])
    # Everything the fit measured, in one block that pastes into a
    # spreadsheet one number per cell.
    panels = [e.label for e in app.get("expander")]
    check("the results can be copied into a spreadsheet",
          any("Copy these results" in (p or "") for p in panels), str(panels))
    check("and the panel of workings is gone",
          not any("How this fit was calculated" in (p or "") for p in panels),
          str(panels))
    blocks = [str(c.value) for c in app.get("code")]
    record = next((b for b in blocks if b.startswith("Cell ID\t")), None)
    check("the block is two tab separated lines, headers then values",
          record is not None and len(record.split("\n")) == 2
          and len(record.split("\n")[0].split("\t"))
          == len(record.split("\n")[1].split("\t")),
          str(record)[:120])
    if record:
        headers = record.split("\n")[0].split("\t")
        check("every modulus is followed by its own uncertainty",
              all(f"± {h}" in headers
                  for h in headers
                  if not h.startswith("±")
                  and (h.endswith("(MPa)") or h.endswith("(kPa)"))),
              str(headers))
        check("and the quality of the fit travels with it",
              {"R2", "chi2 per dof", "n points"} <= set(headers),
              str(headers))


def case_the_data_is_a_field_and_the_model_a_dashed_line():
    print("borderless light blue points, one dashed line over them")
    import app as app_module
    import plot_utils

    eps, force = synthetic()
    model = LulevichModel(force, eps, cell_height=8.0e-6)
    fit = model.fit_composition(0.0, 0.60, 0.15, 0.40)
    mb, cb, nb = model.composition_terms(eps, 0.15, 0.40)
    fitted = mb * fit["Em"] + cb * fit["Ei"] + nb * fit["En"]
    style = plot_utils.PlotStyle(force_unit="N")
    fig = plot_utils.force_curve_figure(eps, force, style, fit_force_N=fitted)

    points = [t for t in fig.data if t.name == "Experimental data"][0]
    check("the markers have no outline at all",
          not getattr(points.marker.line, "width", 0),
          str(points.marker.line))
    check("and they are light blue",
          points.marker.color == "#79c2e8", str(points.marker.color))
    check("big enough to read as one band", points.marker.size >= 8,
          str(points.marker.size))

    lines = [t for t in fig.data if t.name == "Model"]
    check("the model is drawn once, not twice", len(lines) == 1,
          str([t.name for t in fig.data]))
    check("and it is dashed", lines[0].line.dash == "dash",
          str(lines[0].line.dash))
    check("no white halo is left under it",
          not any(str(getattr(t.line, "color", "")).lower() == "white"
                  for t in fig.data if t.mode == "lines"),
          str([getattr(t.line, "color", None) for t in fig.data]))
    check("the app agrees with the figure about the colour",
          app_module.DEFAULTS["data_color"] == "#79c2e8",
          app_module.DEFAULTS["data_color"])


def case_the_range_carries_its_own_maths():
    print("the fitting range says in maths what it does")
    app = start(cell_name="cell-01")
    if not no_exception(app, "range maths"):
        return
    said = " ".join(str(m.value) for m in app.get("markdown"))
    check("the power law is a picture at the end, not a paragraph",
          "Where the curve changes its power law" in said, said[-400:])
    formulas = " ".join(str(getattr(e, "value", "")) for e in app.get("latex"))
    check("the least-squares sum is written with the range in it",
          "arg\\min" in formulas or "argmin" in formulas.replace(" ", ""),
          formulas[:200])
    # The laws sit beside the tick boxes now, one line each, rather than in
    # a table further down: a component is a term in an equation, and the
    # exponent is what makes it a different term from its neighbour.
    import app as app_module
    said_all = " ".join(str(c.value) for c in app.get("caption"))
    laws = [app_module.MATERIAL_LAWS[t]["law"]
            for t in app_module.terms_for("Myoblast (C2C12)")]
    check("and each component carries its own law beside it",
          all(law in said_all for law in laws), str(laws))
    check("with the exponent in it, which is what tells them apart",
          all(mark in said_all for mark in ("ε³", "³ᐟ²")), said_all[:200])


def case_the_page_ends_with_the_answer():
    print("the answer is on the page once, not repeated at the foot")
    app = start(cell_name="cell-01")
    if not no_exception(app, "the results"):
        return
    text = " ".join(str(m.value) for m in
                    list(app.get("markdown")) + list(app.get("caption")))
    check("the block that repeated it all is gone", "In one block" not in text)
    check("the range it was fitted over is still said once",
          "Fitted over ε" in text, text[:400])
    labels = [str(m.label or "") for m in app.get("metric")]
    check("and every modulus is in the metrics row",
          any("Eₘ" in lab for lab in labels), str(labels[:6]))
    check("with the fit quality beside them",
          any("R²" in lab for lab in labels), str(labels[:8]))


def case_a_companion_file_is_found_wherever_it_sits():
    print("a companion file is found even when the app is a folder deeper")
    import app as app_module
    import os
    import shutil
    import tempfile

    check("both companions loaded here",
          app_module.ONEDRIVE_IMPORT_ERROR is None
          and app_module.SHEETS_IMPORT_ERROR is None,
          f"{app_module.ONEDRIVE_IMPORT_ERROR} / "
          f"{app_module.SHEETS_IMPORT_ERROR}")
    seen = app_module.companion_files()
    check("and the app can list what it sees beside itself",
          "app.py" in seen and "onedrive_store.py" in seen, str(seen))

    # The deployment that produced the bug report: the main file one folder
    # below the companion it needs. A plain import cannot see it; the app
    # has to go and find the file.
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "sub"))
        with open(os.path.join(root, "sidecar_probe.py"), "w") as handle:
            handle.write("VALUE = 'found me'\n")
        found = app_module._find_companion_file.__wrapped__ \
            if hasattr(app_module._find_companion_file, "__wrapped__") \
            else app_module._find_companion_file
        # Point the search at the temporary tree the way a deployment would.
        old_app, old_repo = app_module.APP_DIR, app_module.REPO_DIR
        app_module.APP_DIR = os.path.join(root, "sub")
        app_module.REPO_DIR = root
        try:
            path = found("sidecar_probe")
            check("the file one folder up is found",
                  path is not None and path.endswith("sidecar_probe.py"),
                  str(path))
            module, error = app_module.import_companion("sidecar_probe")
            check("and it is actually loaded from there",
                  module is not None and getattr(module, "VALUE", "") == "found me",
                  str(error))
            missing, error = app_module.import_companion("no_such_companion_xyz")
            check("a file that is really absent still reports missing",
                  missing is None and "No module named" in str(error),
                  str(error))
        finally:
            app_module.APP_DIR, app_module.REPO_DIR = old_app, old_repo
            import sys
            sys.modules.pop("sidecar_probe", None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def case_the_missing_file_message_says_what_it_can_see():
    print("the missing-file message lists the folder it actually looked in")
    import app as app_module

    real = app_module.ONEDRIVE_IMPORT_ERROR
    try:
        app_module.ONEDRIVE_IMPORT_ERROR = (
            "ModuleNotFoundError: No module named 'onedrive_store'"
        )
        said = app_module.onedrive_load_problem()
        check("it names the file", "onedrive_store.py" in said, said[:120])
        check("and the folder the app is running from",
              app_module.APP_DIR in said, said[:200])
        check("and lists what is there instead", "app.py" in said, said[:400])
        check("and names the three things that actually cause it",
              "branch" in said and "reboot" in said.lower()
              and ".txt" in said, said)

        app_module.ONEDRIVE_IMPORT_ERROR = "No module named 'requests'"
        said = app_module.onedrive_load_problem()
        check("a missing package is told apart from a missing file",
              "requirements.txt" in said, said[:160])
    finally:
        app_module.ONEDRIVE_IMPORT_ERROR = real


def case_an_element_window_is_an_onset_not_a_mask():
    print("an element's own range moves where its law starts, and holds after")
    eps = np.linspace(0.0, 0.9, 400)
    model = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)

    plain = model.composition_basis(eps, 0.15, 0.40, "continue", "zero")
    moved = model.composition_basis(
        eps, 0.15, 0.40, "continue", "zero",
        term_windows={"nucleus": (0.40, 0.70)},
    )
    check("nothing before the window", float(np.max(moved["nucleus"][eps < 0.40])) == 0.0)
    # The point of this: a body first met at 0.40 answers with (e-0.40)^1.5,
    # which is what the untouched basis already does when e2 is 0.40. A mask
    # would have kept the old onset and blanked the front of it instead.
    check("it is that element's own law measured from its own start",
          np.allclose(moved["nucleus"][(eps >= 0.40) & (eps <= 0.70)],
                      plain["nucleus"][(eps >= 0.40) & (eps <= 0.70)]),
          "windowed term does not match the same onset")

    after = moved["nucleus"][eps > 0.70]
    check("past the far end it holds what it reached, it does not vanish",
          float(np.min(after)) > 0.0, f"min {float(np.min(after)):.3g}")
    check("and holds it flat, so the curve has no step in it",
          float(np.ptp(after)) < 1e-12, f"spread {float(np.ptp(after)):.3g}")

    check("no windows given leaves every term exactly as it was",
          all(np.allclose(plain[t], model.composition_basis(
              eps, 0.15, 0.40, "continue", "zero", term_windows=None)[t])
              for t in plain))


def case_the_placement_search_finds_where_elements_act():
    print("the search places four elements where the curve says they act")
    from lulevich_model import search_term_windows

    eps = np.linspace(0.002, 0.85, 420)
    blank = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
    # A cell built with four elements and known hand-overs: the shell holds
    # at 0.15, the scaffolding takes over there, and the nucleus and its
    # envelope are both met at 0.55.
    basis = blank.composition_basis(eps, 0.15, 0.55, "freeze", "break")
    force = (basis["membrane"] * 0.8e6 + basis["interior"] * 2.0e3
             + basis["nucleus_shell"] * 30e6 + basis["nucleus"] * 20e3)
    rng = np.random.default_rng(1)
    force = force * (1.0 + 0.004 * rng.standard_normal(eps.size))

    model = LulevichModel(force, eps, cell_height=8.0e-6)
    found = search_term_windows(
        model, 0.0, 0.85,
        ("membrane", "interior", "nucleus_shell", "nucleus"),
        membrane="freeze", cyto_start="break",
        # Deliberately started at the wrong deep boundary: the search has to
        # move it, which is the whole point of having one.
        e1=0.15, e2=0.35, weighting="relative", n_grid=10, rounds=3,
    )
    check("the search ran", found.get("success"), str(found.get("error")))
    if not found.get("success"):
        return
    windows = found["windows"]
    check("the shell is found to hold near 0.15",
          abs(windows["membrane"][1] - 0.15) < 0.05,
          str([round(v, 3) for v in windows["membrane"]]))
    check("the scaffolding takes over there",
          abs(windows["interior"][0] - 0.15) < 0.05,
          str([round(v, 3) for v in windows["interior"]]))
    check("and the deep pair is met near 0.55, not at the 0.35 it started on",
          abs(windows["nucleus"][0] - 0.55) < 0.06
          and abs(windows["nucleus_shell"][0] - 0.55) < 0.06,
          f"{windows['nucleus_shell'][0]:.3f} / {windows['nucleus'][0]:.3f}")

    started = found["trials"][0]["score"]
    check("and it ends better than where it started",
          found["cv_rmse"] < started,
          f"{started:.4g} -> {found['cv_rmse']:.4g}")

    fit = found["fit"]
    check("the moduli come back with it",
          abs(fit["Em_MPa"] - 0.8) < 0.15 and abs(fit["Ei_kPa"] - 2.0) < 0.4,
          f"Em {fit['Em_MPa']:.4g}, Ei {fit['Ei_kPa']:.4g}")
    check("including the envelope, which only its own window separates",
          abs(fit["Ene_MPa"] - 30.0) < 6.0, f"{fit['Ene_MPa']:.4g} MPa")
    check("the fit carries the windows it was made with",
          fit.get("term_windows") is not None
          and set(fit["term_windows"]) == set(windows), str(fit.get("term_windows")))
    check("every placement tried is reported, not just the winner",
          len(found["trials"]) > 20, str(len(found["trials"])))


def case_a_new_curve_starts_from_the_cell_type_defaults():
    print("a curve loads at its cell type's boundaries, not at a search result")
    import app as app_module

    wanted = app_module.default_boundaries("Myoblast (C2C12)")
    # ε₂ starts in the middle of the band a C2C12's nucleus is met in.
    band = app_module.deep_onset_band("Myoblast (C2C12)")
    check("a C2C12 has boundaries of its own",
          abs(wanted["segment_break_1"] - 0.15) < 1e-9
          and band[0] <= wanted["segment_break_2"] <= band[1], str(wanted))
    # The nucleus shows as a bump at about half the squash that runs on for
    # another quarter, so the onset band is that, not a third of the curve.
    check("and that band is where the nucleus bump starts",
          abs(band[0] - 0.44) < 1e-9 and abs(band[1] - 0.62) < 1e-9, str(band))
    bump = app_module.bump_window("Myoblast (C2C12)")
    check("the bump itself is recorded, 50 % running to 75 %",
          bump is not None and abs(bump[0] - 0.50) < 1e-9
          and abs(bump[1] - 0.75) < 1e-9, str(bump))
    check("and an unknown cell type falls back to the app's",
          app_module.default_boundaries("Something else")["segment_break_1"]
          == app_module.DEFAULTS["segment_break_1"])

    app = start(cell_name="cell-01")
    if not no_exception(app, "a freshly loaded curve"):
        return
    check("the loaded curve sits on those boundaries",
          abs(float(state(app, "segment_break_1"))
              - wanted["segment_break_1"]) < 1e-6
          and abs(float(state(app, "segment_break_2"))
                  - wanted["segment_break_2"]) < 1e-6,
          f"{state(app, 'segment_break_1')} / {state(app, 'segment_break_2')}")
    check("the component ranges follow them",
          state(app, "element_window_membrane")[0] == 0.0
          and abs(state(app, "element_window_nucleus")[0]
                  - wanted["segment_break_2"]) < 1e-6,
          f"{state(app, 'element_window_membrane')} "
          f"{state(app, 'element_window_nucleus')}")
    check("and the two a C2C12 carries throughout do run throughout",
          state(app, "element_window_membrane")[1]
          == state(app, "element_window_interior")[1]
          == float(state(app, "window_end")),
          f"{state(app, 'element_window_membrane')} "
          f"{state(app, 'element_window_interior')}")
    fit = state(app, "_last_fit")
    check("and it is still fitted, at those boundaries",
          fit and fit.get("success")
          and abs(float(fit["break_2"])
                  - wanted["segment_break_2"]) < 1e-6, str(fit.get("break_2")))
    check("no search ran on load, so nothing was applied unasked",
          state(app, "hypothesis_search") in (None, {}),
          str(type(state(app, "hypothesis_search"))))

    # And the button is what goes looking.
    button = button_by_label(app, "Find the boundaries from the log curve")
    check("there is a button to find them from the curve", button is not None,
          str([b.label for b in app.button][:10]))
    if button is None:
        return
    button.click().run()
    if not no_exception(app, "the boundary search"):
        return
    moved = (float(state(app, "segment_break_1")),
             float(state(app, "segment_break_2")))
    check("pressing it moves them off the defaults", moved != (
        wanted["segment_break_1"], wanted["segment_break_2"]), str(moved))
    check("but never outside the band the biology allows",
          band[0] - 1e-6 <= moved[1] <= band[1] + 1e-6, str(moved))
    check("and the deep range moves with the boundary",
          abs(state(app, "element_window_nucleus")[0] - moved[1]) < 1e-6,
          f"{state(app, 'element_window_nucleus')} against {moved}")
    # Pressed twice it must land in the same place: a search that answers
    # differently each time is not an estimate.
    button = button_by_label(app, "Find the boundaries from the log curve")
    button.click().run()
    again = (float(state(app, "segment_break_1")),
             float(state(app, "segment_break_2")))
    check("and pressing it again does not move it", again == moved,
          f"{moved} then {again}")
    check("the total range is left alone",
          abs(float(state(app, "window_end"))
              - float(fit["epsilon_range"][1])) < 1e-6,
          f"{state(app, 'window_end')} against {fit['epsilon_range'][1]}")


def case_the_ranges_follow_the_fit():
    print("each material's range shows where the fit actually put it")
    app = start(cell_name="cell-01")
    if not no_exception(app, "ranges after the first fit"):
        return
    fit = state(app, "_last_fit")
    check("there is a fit to follow", fit and fit.get("success"))
    if not (fit and fit.get("success")):
        return

    def bar(term):
        return state(app, f"element_window_{term}")

    e1, e2 = float(fit["break_1"]), float(fit["break_2"])
    hi = float(fit["epsilon_range"][1])
    check("the shell's bar ends where the fit hands over",
          fit.get("membrane") != "freeze" or abs(bar("membrane")[1] - e1) < 1e-3,
          f"{bar('membrane')} against ε₁ = {e1:.3f}")
    check("the scaffolding starts where the fit says it starts",
          fit.get("cyto_start") == "zero"
          or abs(bar("interior")[0] - e1) < 1e-3,
          f"{bar('interior')} against ε₁ = {e1:.3f}")
    if state(app, "use_nucleus"):
        check("and the deeper material starts at ε₂",
              abs(bar("nucleus")[0] - e2) < 1e-3,
              f"{bar('nucleus')} against ε₂ = {e2:.3f}")
    check("nothing runs past the fitted range",
          all(bar(t)[1] <= hi + 1e-6 for t in ("membrane", "interior")),
          f"{bar('membrane')} {bar('interior')} against {hi:.3f}")

    # Refit over a shorter range: the bars have to move with it, or they are
    # describing a model that is no longer the one on the page.
    before = bar("interior")
    widget_by_label(app, "slider", "Fitted range").set_value((0.0, 0.40)).run()
    button = button_by_label(app, "Fit this cell")
    if button is not None:
        button.click().run()
        if not no_exception(app, "refit over a shorter range"):
            return
    after_fit = state(app, "_last_fit")
    check("the refit happened", after_fit and after_fit.get("success"))
    if not (after_fit and after_fit.get("success")):
        return
    check("and every bar followed it",
          abs(bar("interior")[1] - float(after_fit["epsilon_range"][1])) < 1e-3,
          f"{before} then {bar('interior')} for a fit ending at "
          f"{after_fit['epsilon_range'][1]:.3f}")
    check("including where the new ε₁ landed",
          after_fit.get("cyto_start") == "zero"
          or abs(bar("interior")[0] - float(after_fit["break_1"])) < 1e-3,
          f"{bar('interior')} against ε₁ = {after_fit['break_1']:.3f}")

    # And once they are yours, a fit no longer moves them - but the page
    # says so and offers to put them back.
    app.session_state["element_window_interior"] = (0.05, 0.30)
    app.session_state["_window_touched_interior"] = True
    app.run()
    if not no_exception(app, "a range moved by hand"):
        return
    said = " ".join(str(c.value) for c in app.get("caption"))
    check("it says that range no longer follows the boundaries",
          "Moved by hand" in said, said[-300:])
    check("and offers to put it back",
          button_by_label(app, "Put them back on the boundaries")
          is not None, str([b.label for b in app.button][:14]))


def case_a_sheet_that_will_not_open_says_why():
    print("a Google Sheet that cannot be opened says so, and what to do")
    from google_sheets_manager import GoogleSheetsManager

    manager = GoogleSheetsManager.__new__(GoogleSheetsManager)
    manager.spreadsheet = None
    manager.worksheet = None
    check("a manager with no worksheet says that plainly",
          "no worksheet" in manager.describe().lower(), manager.describe())
    ok, said = manager.check()
    check("and its check fails rather than pretending", not ok, said)
    check("naming the thing to do about it",
          "reconnect" in said.lower(), said)

    class _Book:
        title = "AFM cells"

    class _Tab:
        title = "Cells"

        def row_values(self, n):
            return ["Cell ID", "Experiment Date"]

        def get_all_values(self):
            return [["Cell ID", "Experiment Date"], ["c1", "2026-01-01"]]

        def update(self, values=None, range_name=None):
            return None

    manager.spreadsheet, manager.worksheet = _Book(), _Tab()
    check("a working one names the file and the tab",
          "AFM cells" in manager.describe() and "Cells" in manager.describe(),
          manager.describe())
    ok, said = manager.check()
    check("and its check passes", ok, said)
    check("reporting what is in the sheet",
          "2 columns" in said and "1 row" in said, said)
    check("and that writing works", "Writing works" in said, said)

    class _ReadOnly(_Tab):
        def update(self, values=None, range_name=None):
            raise RuntimeError("The caller does not have permission")

    manager.worksheet = _ReadOnly()
    ok, said = manager.check()
    check("a read-only share is caught before a row goes missing", not ok, said)
    check("and named as the Viewer/Editor mistake it is",
          "Editor" in said and "Viewer" in said, said)


def case_each_element_gets_its_own_bar():
    print("each element has a bar of its own, and a button that places them")
    app = start(cell_name="cell-01")
    if not no_exception(app, "before the element windows"):
        return
    check("there is no switch to choose between the ranges and the "
          "boundaries, because they are the same numbers",
          not any("Set the ranges myself" in (c.label or "")
                  for c in app.checkbox),
          str([c.label for c in app.checkbox][:8]))
    fit = state(app, "_last_fit")
    check("the fit is made with each component on its own range",
          fit is not None and fit.get("term_windows"),
          str(fit.get("term_windows") if fit else "no fit"))
    check("and those ranges are the boundaries, until one is moved",
          abs(fit["term_windows"]["nucleus"][0] - float(fit["break_2"])) < 1e-6,
          f"{fit['term_windows']['nucleus']} against ε₂ = {fit['break_2']:.3f}")
    terms = [t for t in ("membrane", "interior", "nucleus", "nucleus_shell")
             if state(app, f"use_{t}")]
    bars = [s for s in app.slider if (s.label or "").startswith("acts over ε")]
    check("there is a bar for every component offered",
          len(bars) >= len(terms), f"{len(bars)} bars for {len(terms)} components")

    # Moving one element's bar moves that element and nothing else.
    others = {t: state(app, f"element_window_{t}") for t in terms[1:]}
    if bars:
        bars[0].set_value((0.10, 0.45)).run()
        no_exception(app, "moving one element's bar")
        moved = [t for t in terms
                 if state(app, f"element_window_{t}") == (0.1, 0.45)]
        check("the bar that was moved is the one that changed",
              len(moved) == 1, str(moved))
        check("and the others stayed where they were",
              all(state(app, f"element_window_{t}") == w
                  for t, w in others.items() if w is not None),
              str(others))

    fit = state(app, "_last_fit")
    check("moving one keeps it",
          fit is not None and fit.get("term_windows"),
          str(fit.get("term_windows") if fit else "no fit"))

    # The unconstrained placement search moved to the settings, because it
    # overwrites every range at once and knows nothing about the cell type.
    button = button_by_label(app, "Search the ranges freely")
    check("and one button that places them by arithmetic",
          button is not None, str([b.label for b in app.button][:14]))
    if button is None:
        return
    button.click().run()
    if not no_exception(app, "the placement search"):
        return
    found = state(app, "element_window_search")
    check("the search ran and applied its answer",
          found is not None and found.get("success"), str(found))
    if found and found.get("success"):
        applied = {t: state(app, f"element_window_{t}") for t in found["windows"]}
        check("every element sits where the search put it",
              all(abs(applied[t][0] - w[0]) < 1e-3
                  and abs(applied[t][1] - w[1]) < 1e-3
                  for t, w in found["windows"].items()),
              f"{applied} vs {found['windows']}")
        after = state(app, "_last_fit")
        check("and the curve is refitted with them",
              after is not None and after.get("term_windows") is not None
              and after.get("success"), str(after.get("term_windows") if after else None))


def case_the_fit_colour_moves_with_the_range():
    print("light blue data, a black fit, and a new fit colour per range")
    import app as app_module
    import plot_utils

    check("data is a light blue by default",
          app_module.DEFAULTS["data_color"] == "#79c2e8",
          app_module.DEFAULTS["data_color"])
    check("the fit is black by default",
          app_module.DEFAULTS["fit_color"] == "#000000",
          app_module.DEFAULTS["fit_color"])
    check("and the figures agree with the app about both",
          plot_utils.PlotStyle().data_color == app_module.DEFAULTS["data_color"]
          and plot_utils.PlotStyle().fit_color == app_module.DEFAULTS["fit_color"],
          f"{plot_utils.PlotStyle().data_color} / {plot_utils.PlotStyle().fit_color}")
    check("the colour walk starts on black",
          app_module.FIT_COLORS[0] == "#000000", app_module.FIT_COLORS[0])
    check("and every colour in it is different",
          len(set(app_module.FIT_COLORS)) == len(app_module.FIT_COLORS),
          str(app_module.FIT_COLORS))

    app = start(cell_name="cell-01")
    if not no_exception(app, "colour walk"):
        return
    first = state(app, "_fit_colour_step", 0)
    bar = widget_by_label(app, "slider", "Fitted range")
    if bar is None:
        check("there is a range bar to move", False)
        return
    bar.set_value((0.0, 0.40)).run()
    second = state(app, "_fit_colour_step", 0)
    check("moving the range takes the fit to the next colour",
          second == first + 1, f"{first} then {second}")
    widget_by_label(app, "slider", "Fitted range").set_value((0.0, 0.30)).run()
    third = state(app, "_fit_colour_step", 0)
    check("and moving it again takes it to the one after",
          third == second + 1, f"{second} then {third}")
    check("which are three different colours",
          len({app_module.FIT_COLORS[i % len(app_module.FIT_COLORS)]
               for i in (first, second, third)}) == 3)

    # A pass that changes nothing must not change the colour, or the fit
    # would be a different colour in every screenshot of the same range.
    app.run()
    check("a rerun that changes nothing leaves the colour alone",
          state(app, "_fit_colour_step", 0) == third,
          str(state(app, "_fit_colour_step")))


def case_the_fixed_tick_locks_out_the_other_materials():
    print("ticking fixed leaves one material and switches the rest off")
    app = start(cell_name="cell-01")
    if not no_exception(app, "before fixing"):
        return
    boxes = [c for c in app.checkbox if "fixed" in (c.label or "").lower()]
    check("the tick is on the page beside the materials", len(boxes) >= 1,
          str([c.label for c in app.checkbox][:12]))
    if not boxes:
        return
    boxes[0].set_value(True).run()
    if not no_exception(app, "ticking fixed"):
        return
    check("the membrane is switched off", not app.session_state["use_membrane"])
    check("the deep element is switched off", not app.session_state["use_nucleus"])
    check("the one material left is on", app.session_state["use_interior"])
    # The tick is a statement about the sample, so it wins over a box left
    # behind by an earlier cell: even with the membrane forced back on, the
    # fit is one Hertzian solid.
    app.session_state["use_membrane"] = True
    app.run()
    check("and a material switched back on behind its back is ignored",
          state(app, "_last_fit") is not None
          and tuple(state(app, "_last_fit")["terms"]) == ("interior",),
          str(state(app, "_last_fit", {}).get("terms")))
    app.session_state["use_membrane"] = False
    app.run()
    fit = state(app, "_last_fit")
    check("the fit that follows has one term",
          fit and tuple(fit["terms"]) == ("interior",),
          str(fit["terms"]) if fit else "no fit")

    # Unticking puts back what this cell type normally starts with, rather
    # than leaving every box cleared.
    boxes = [c for c in app.checkbox if "fixed" in (c.label or "").lower()]
    boxes[0].set_value(False).run()
    if not no_exception(app, "unticking fixed"):
        return
    check("unticking it brings the membrane back",
          app.session_state["use_membrane"])
    check("and the deep element",
          app.session_state["use_nucleus"]
          or app.session_state["use_nucleus_shell"],
          f"{app.session_state['use_nucleus']} / "
          f"{app.session_state['use_nucleus_shell']}")


class _StubSheet:
    """A Google Sheet with three cells in it, without a Google account."""

    ROWS = [
        {"Cell ID": "sheet-01", "Experiment Date": "2026-01-04",
         "Young's Modulus (Em, MPa)": "1.4",
         "Young's Modulus (Ei, kPa)": "5.1", "Operator": "DM"},
        {"Cell ID": "sheet-02", "Experiment Date": "2026-01-05",
         "Young's Modulus (Em, MPa)": "1.8",
         "Young's Modulus (Ei, kPa)": "6.3", "Operator": "DM"},
        {"Cell ID": "other-03", "Experiment Date": "2026-01-06",
         "Young's Modulus (Em, MPa)": "2.2",
         "Young's Modulus (Ei, kPa)": "7.7", "Operator": "AB"},
    ]

    def get_all_cells(self):
        return pd.DataFrame(self.ROWS)

    def get_spreadsheet_url(self):
        return "https://docs.google.com/spreadsheets/d/stub"

    def export_to_csv(self):
        return self.get_all_cells().to_csv(index=False)

    def export_to_json(self):
        return self.get_all_cells().to_json(orient="records")

    def get_statistics(self):
        return {}


class _FakeTab:
    """One worksheet, in memory, with just the calls the manager makes."""

    def __init__(self, title, values=None):
        self.title = title
        self.values = [list(row) for row in (values or [])]

    def row_values(self, n):
        return list(self.values[n - 1]) if len(self.values) >= n else []

    def insert_row(self, row, index):
        self.values.insert(index - 1, list(row))

    def append_row(self, row, value_input_option=None):
        self.values.append(list(row))

    def get_all_values(self):
        return [list(row) for row in self.values]

    def clear(self):
        self.values = []

    def update(self, values=None, range_name=None):
        if not self.values:
            self.values = [list(r) for r in values]
            return
        self.values[: len(values)] = [list(r) for r in values]


class _FakeBook:
    def __init__(self):
        self.tabs = {}

    def worksheet(self, title):
        import gspread
        if title not in self.tabs:
            raise gspread.WorksheetNotFound(title)
        return self.tabs[title]

    def add_worksheet(self, title, rows=0, cols=0):
        self.tabs[title] = _FakeTab(title)
        return self.tabs[title]


def _fake_manager():
    from google_sheets_manager import GoogleSheetsManager
    manager = GoogleSheetsManager()
    manager.spreadsheet = _FakeBook()
    manager.worksheet = manager.spreadsheet.add_worksheet("Cells")
    manager._initialize_headers()
    return manager


def case_the_sheet_has_a_summary_tab_and_a_working_tab():
    print("the spreadsheet is a readable summary plus a tab of working")
    from google_sheets_manager import GoogleSheetsManager as G

    wanted = [
        "Experiment Date", "Cell ID", "Cell Height (μm)",
        "Spring Constant, K (N/m)", "Young's Modulus (Em, MPa)", "Em range (ε)",
        "Young's Modulus (Ecx cortex, kPa)", "Young's Modulus (Ei, kPa)",
        "Ei range (ε)", "Young's Modulus (Ene envelope, MPa)",
        "Young's Modulus (En, kPa)", "En range (ε)", "Fit Quality (R²)",
        "Chi squared", "Height (um)", "Video Comment",
    ]
    first = [name for _, name in G.main_columns()]
    check("the first tab is exactly the asked-for columns, in order",
          first == wanted, str(first))
    second = [name for _, name in G.extra_columns()]
    check("the second tab starts with what identifies the cell",
          second[:3] == ["Cell ID", "Experiment Date",
                         "Spring Constant, K (N/m)"], str(second[:3]))
    check("nothing is lost between the two tabs",
          set(first) | set(second) == {name for _, name in G.COLUMNS},
          str(set(name for _, name in G.COLUMNS) - (set(first) | set(second))))
    check("and the working is not repeated on the summary",
          not ({"Points fitted", "Weighting", "Timestamp"} & set(first)),
          str(first))

    manager = _fake_manager()
    ok, message = manager.append_cell_data({
        "cell_id": "C2C12_001", "experiment_date": "2022-04-13",
        "cell_height": 8.06, "spring_constant": 0.0,
        "Em": 8.070523, "Em_range": "0.000 to 0.960",
        "Ei": 11.559871, "Ei_range": "0.246 to 0.960",
        "Ene": 155.090322, "En": 7112.678218, "En_range": "0.868 to 0.960",
        "fit_quality": 0.99984, "chi_squared": 1581300,
        "video_height_um": 9.1, "video_comment": "probe slipped once",
        "n_points": 812, "weighting": "relative",
    })
    check("the row is written", ok, message)
    rows = manager.worksheet.get_all_values()
    check("the summary tab has a header and one row", len(rows) == 2,
          str(len(rows)))
    summary = dict(zip(rows[0], rows[1]))
    check("with the cell named in it", summary["Cell ID"] == "C2C12_001",
          str(summary.get("Cell ID")))
    check("its membrane modulus", str(summary["Young's Modulus (Em, MPa)"])
          .startswith("8.07"), str(summary.get("Young's Modulus (Em, MPa)")))
    check("the height measured off the video",
          str(summary["Height (um)"]) == "9.1", str(summary.get("Height (um)")))
    check("and the comment about it",
          summary["Video Comment"] == "probe slipped once",
          str(summary.get("Video Comment")))

    extra = manager.extra_worksheet(create=False)
    check("a second tab was made", extra is not None)
    if extra is None:
        return
    erows = extra.get_all_values()
    check("it has the same one row", len(erows) == 2, str(len(erows)))
    working = dict(zip(erows[0], erows[1]))
    check("keyed by the same cell", working["Cell ID"] == "C2C12_001",
          str(working.get("Cell ID")))
    check("with its date and spring constant repeated",
          working["Experiment Date"] == "2022-04-13"
          and working["Spring Constant, K (N/m)"] in ("0", "0.0", 0, 0.0),
          f"{working.get('Experiment Date')} / "
          f"{working.get('Spring Constant, K (N/m)')}")
    check("and the working on it", str(working["Points fitted"]) == "812",
          str(working.get("Points fitted")))


def case_an_old_one_tab_sheet_is_split_without_losing_anything():
    print("an existing single-tab sheet splits in two without losing a column")
    from google_sheets_manager import GoogleSheetsManager as G

    manager = _fake_manager()
    # A sheet as it was written before the split: every column on tab one,
    # plus a column of the user's own that this app has never heard of.
    header = [name for _, name in G.COLUMNS] + ["My own note"]
    row = [f"v{i}" for i in range(len(header))]
    row[header.index("Cell ID")] = "old-01"
    row[header.index("Points fitted")] = "404"
    manager.worksheet.values = [header, row]

    ok, message = manager.reorder_columns()
    check("the sheet is rewritten", ok, message)
    first = manager.worksheet.get_all_values()
    check("the summary tab keeps its row", len(first) == 2, str(len(first)))
    check("in the asked-for order",
          first[0][:2] == ["Experiment Date", "Cell ID"], str(first[0][:3]))
    check("a column of the user's own is kept, at the end",
          "My own note" in first[0], str(first[0][-3:]))
    summary = dict(zip(first[0], first[1]))
    check("and the row is still that cell", summary["Cell ID"] == "old-01",
          str(summary.get("Cell ID")))

    extra = manager.extra_worksheet(create=False)
    check("the working moved to the second tab", extra is not None)
    if extra is None:
        return
    working = dict(zip(*extra.get_all_values()[:2]))
    check("carrying the value that was under that heading",
          working["Points fitted"] == "404", str(working.get("Points fitted")))
    check("still keyed by the cell", working["Cell ID"] == "old-01",
          str(working.get("Cell ID")))


def case_a_sheet_with_repeated_headings_still_opens():
    print("a spreadsheet with a repeated or blank heading still shows")
    import app as app_module
    from google_sheets_manager import GoogleSheetsManager as G

    made = G.unique_headers(["Cell ID", "Notes", "", "Notes", "Cell ID"])
    check("no two columns share a name", len(set(made)) == len(made), str(made))
    check("the first of each keeps its name",
          made[0] == "Cell ID" and made[1] == "Notes", str(made))
    check("a blank heading becomes a named column",
          made[2] == "Column 3", str(made))
    check("and the repeats are numbered, not dropped",
          made[3] == "Notes (2)" and made[4] == "Cell ID (2)", str(made))

    # The crash: Arrow refuses a DataFrame with two columns of one name, and
    # the whole page went down naming none of them.
    frame = pd.DataFrame([["a", "b", "c"]], columns=["x", "y", "x"])
    fixed = app_module.safe_frame(frame)
    check("a frame with a repeated column comes back usable",
          len(set(fixed.columns)) == 3, str(list(fixed.columns)))
    check("with every value still in it",
          list(fixed.iloc[0]) == ["a", "b", "c"], str(list(fixed.iloc[0])))
    try:
        import pyarrow
        pyarrow.Table.from_pandas(fixed, preserve_index=False)
        arrow_ok = True
    except ImportError:
        arrow_ok = True  # nothing to prove without pyarrow installed
    except Exception as exc:
        arrow_ok = False
        print(f"  arrow refused it: {exc}")
    check("and Arrow accepts it, which is what the table widget needs",
          arrow_ok)
    check("a frame that was already fine is handed back untouched",
          app_module.safe_frame(frame[["x", "y"]]) is not None)

    # End to end: a stub sheet whose header repeats itself.
    class _Twice(_StubSheet):
        """A sheet whose header really does repeat a name."""

        def get_all_cells(self):
            frame = pd.DataFrame(self.ROWS)
            frame["Cell ID "] = frame["Cell ID"]
            frame.columns = list(frame.columns[:-1]) + ["Cell ID"]
            return frame

        # The real manager de-duplicates inside get_all_cells, so its own
        # exports are already safe; this stub deliberately does not, and the
        # exports would fail for a reason that is not what is being tested.
        def export_to_csv(self):
            return pd.DataFrame(self.ROWS).to_csv(index=False)

        def export_to_json(self):
            return pd.DataFrame(self.ROWS).to_json(orient="records")

    app = AppTest.from_file(APP, default_timeout=600)
    app.run()
    app.session_state["db_enabled"] = True
    app.session_state["gs_manager"] = _Twice()
    app.run()
    check("the database tab opens on a sheet with a repeated heading",
          no_exception(app, "a sheet with two Cell ID columns"))


def case_the_sheet_can_be_the_database():
    print("a connected sheet is where the database tab reads from")
    app = AppTest.from_file(APP, default_timeout=600)
    app.run()
    app.session_state["db_enabled"] = True
    app.session_state["gs_manager"] = _StubSheet()
    app.run()
    if not no_exception(app, "the sheet as the database"):
        return
    shown = []
    for element in app.get("dataframe"):
        frame = getattr(element, "value", None)
        if frame is not None and "Cell ID" in getattr(frame, "columns", []):
            shown.append(frame)
    check("the cells in the sheet are listed",
          bool(shown) and len(shown[0]) == 3,
          "none" if not shown else str(len(shown[0])))
    text = " ".join(str(m.value) for m in app.get("markdown"))
    captions = " ".join(str(c.value) for c in app.get("caption"))
    check("it says where the rows came from",
          "sheet" in (text + captions).lower())
    check("and links to the sheet itself",
          "docs.google.com/spreadsheets" in (text + captions))
    check("with no complaint about OneDrive not being connected",
          "No database connected" not in text)


def case_a_fixed_cell_is_one_hertzian_solid():
    print("a fixed cell is fitted as one cross-linked solid")
    import app as app_module
    # Fixation is a state of the cell, not a kind of cell, so it is a tick
    # beside the materials and not an entry in the dropdown.
    check("it is not offered as a cell type",
          "Fixed cell" not in app_module.SELECTABLE_CELL_TYPES,
          str(app_module.SELECTABLE_CELL_TYPES))
    check("its plausibility band is still there to switch to",
          "Fixed cell" in app_module.CELL_TYPES, str(list(app_module.CELL_TYPES)))
    check("and it is off unless you tick it",
          app_module.DEFAULTS["fixed_cell"] is False)
    here = app_module.OPTIONAL_TERMS["Fixed cell"]
    check("with Hertzian materials only", set(here) <= {"cortex", "interior",
                                                        "nucleus"}, str(here))
    check("named for what it is",
          "fixed cell" in app_module.COMPONENT_SETS["Fixed cell"]["interior"][0].lower(),
          app_module.COMPONENT_SETS["Fixed cell"]["interior"][0])
    picks = app_module.HYPOTHESES["Fixed cell"]
    check("the simplest picture is one solid over the whole curve",
          picks[0]["terms"] == ("interior",) and picks[0]["cyto_start"] == "zero",
          str(picks[0]))

    # The paper's own numbers: fixation cross-links the proteins and the
    # cell comes out 20 to 50 times stiffer, 150 to 230 kPa.
    eps = np.linspace(0.002, 0.60, 320)
    blank = LulevichModel(np.zeros_like(eps), eps, cell_height=8.0e-6)
    force = blank.composition_basis(
        eps, 0.15, 0.40, "continue", "zero")["interior"] * 190e3
    rng = np.random.default_rng(0)
    force = force * (1.0 + 0.01 * rng.standard_normal(eps.size))

    app = AppTest.from_file(APP, default_timeout=900)
    app.run()
    app.session_state["c2c12_fit_mode"] = ADVANCED_MODE
    app.session_state["fixed_cell"] = True
    # Set directly rather than through the tick, so the tick's own callback
    # has not run: say what it would have said.
    app.session_state["use_nucleus"] = False
    app.session_state["use_nucleus_shell"] = False
    app.session_state["use_membrane"] = False
    app.session_state["cyto_starts_at"] = "from the very start"
    app.session_state["cell_name"] = "fixed-01"
    app.session_state["data"] = {
        "epsilon": eps, "force_N": force, "source": "fixed.csv",
        "n_dropped": 0,
    }
    app.run()
    if not no_exception(app, "a fixed cell"):
        return
    fit = app.session_state["_last_fit"]
    check("it is fitted", fit and fit.get("success"))
    if not (fit and fit.get("success")):
        return
    check("with the one Hertzian term and nothing else",
          tuple(fit["terms"]) == ("interior",), str(fit["terms"]))
    check("and the modulus comes back",
          abs(fit["Ei_kPa"] - 190.0) < 10.0, f"{fit['Ei_kPa']:.4g} kPa")
    check("which is in the range the paper reports for fixed cells",
          150.0 <= fit["Ei_kPa"] <= 230.0, f"{fit['Ei_kPa']:.4g} kPa")
    import app as app_module
    rows = app_module.result_rows(fit)
    check("the others are marked as not in this model",
          any("not in this model" in row[2] for row in rows),
          str([row[2] for row in rows]))
    # The living cell type underneath is untouched: fixation changed the
    # chemistry, not the geometry the prefactors are built from.
    check("and the cell type underneath is left alone",
          app.session_state["cell_type"] == "Myoblast (C2C12)",
          str(app.session_state["cell_type"]))


def case_unticking_a_material_fits_without_it():
    print("unticking a material fits without it, rather than putting it back")
    import app as app_module
    curves = vcm_curves()
    if not curves:
        check("the VCM reference curves are in the repository", False)
        return
    eps, force = curves[11]
    app = AppTest.from_file(APP, default_timeout=900)
    app.run()
    app.session_state["cell_type"] = "Cardiomyocyte"
    app.session_state["cell_name"] = "vcm-11"
    app.session_state["data"] = {
        "epsilon": eps, "force_N": force, "source": "vcm_11.csv",
        "n_dropped": 0,
    }
    app.run()
    if not no_exception(app, "a cardiomyocyte"):
        return
    before = (app.session_state["_last_fit"] or {}).get("terms") or []
    check("all three are fitted to start with",
          set(before) == {"membrane", "interior", "nucleus"}, str(before))

    names = app_module.components_for("Cardiomyocyte")
    box = next((c for c in app.checkbox
                if (c.label or "").startswith(names["membrane"][0])), None)
    check("the membrane has a checkbox", box is not None,
          str([c.label for c in app.checkbox][:4]))
    if box is None:
        return
    box.uncheck().run()
    if not no_exception(app, "unticking the membrane"):
        return
    check("it stays unticked", app.session_state["use_membrane"] is False)
    # Fitting fits what it is given, so the boundaries are optimised for
    # the model that is now on the page before the fit is judged.
    tune = button_by_label(app, "Find the boundaries from the log curve")
    if tune is not None:
        tune.click().run()
        no_exception(app, "refining without the membrane")
    work = button_by_label(app, "Fit this cell")
    if work is None:
        check("the fit button is there", False)
        return
    work.click().run()
    if not no_exception(app, "fitting without the membrane"):
        return
    # The bug this catches: every picture carried its own list of materials
    # and the winner wrote them straight back, so the box you had just
    # cleared reappeared with a modulus beside it.
    check("the membrane is still unticked after fitting",
          app.session_state["use_membrane"] is False)
    after = app.session_state["_last_fit"]
    check("and it is not in the fit", "membrane" not in after["terms"],
          str(after["terms"]))
    # Less well than with it, which is the honest consequence of removing
    # a term that carries real load: what matters is that it still fits
    # rather than collapsing.
    check("the rest still fit the curve", after["r_squared"] > 0.995,
          f"{after['r_squared']:.6f}")
    check("the membrane modulus reads zero",
          after["Em_MPa"] == 0.0, str(after["Em_MPa"]))

    # And the other half of the same rule: a component that is ticked stays
    # ticked. Fitting decides the arrangement and the boundaries; which
    # components are in the model is the search button's question, asked on
    # purpose, with its own criterion.
    box = next((c for c in app.checkbox
                if (c.label or "").startswith(names["membrane"][0])), None)
    if box is None:
        return
    box.check().run()
    if not no_exception(app, "ticking the membrane again"):
        return
    ticked = {t for t in ("membrane", "interior", "nucleus")
              if state(app, f"use_{t}", False)}
    work = button_by_label(app, "Fit this cell")
    if work is None:
        return
    work.click().run()
    if not no_exception(app, "fitting with all three"):
        return
    still = {t for t in ("membrane", "interior", "nucleus")
             if state(app, f"use_{t}", False)}
    check("fitting never unticks a component you asked for",
          still == ticked, f"{sorted(ticked)} -> {sorted(still)}")
    end = state(app, "_last_fit") or {}
    check("and every one of them is in the fit",
          set(end.get("terms") or ()) == ticked, str(end.get("terms")))
    check("which still follows the curve", end.get("r_squared", 0) > 0.999,
          f"{end.get('r_squared', 0):.6f}")


def case_a_new_curve_arrives_ready_to_fit():
    print("a loaded curve comes with its components ticked and a range that "
          "says where it came from")
    import app as app_module
    curves = vcm_curves()
    if not curves:
        check("the VCM reference curves are in the repository", False)
        return

    app = AppTest.from_file(APP, default_timeout=900)
    app.run()
    app.session_state["cell_type"] = "Cardiomyocyte"
    app.session_state["cell_name"] = "vcm-3"
    eps, force = curves[3]
    app.session_state["data"] = {
        "epsilon": eps, "force_N": force, "source": "vcm_3.csv", "n_dropped": 0,
    }
    app.run()
    if not no_exception(app, "loading a curve"):
        return

    here = app_module.terms_for("Cardiomyocyte")
    check("every component of this cell type is ticked on arrival",
          all(state(app, f"use_{t}", False) for t in here),
          str({t: state(app, f"use_{t}", False) for t in here}))

    suggested = state(app, "_suggested_window")
    check("the curve's own suggested range was worked out",
          suggested is not None, str(suggested))
    if not suggested:
        return
    check("and it is the range on screen",
          abs(float(state(app, "window_start")) - suggested["start"]) < 5e-4
          and abs(float(state(app, "window_end")) - suggested["end"]) < 5e-4,
          f"{state(app, 'window_start')} {state(app, 'window_end')} "
          f"vs {suggested}")
    said = " ".join(str(c.value) for c in app.get("caption"))
    check("the page says so, rather than leaving it to be guessed",
          "This is the suggested range" in said, said[:300])
    # Cell 3 has the bad contact, so the reason is worth reading.
    check("and says why it does not start at contact",
          "bad contact" in said, said[:400])

    # Moved by hand, it stops claiming to be the suggestion and offers it back.
    app.session_state["window_start"] = 0.30
    app.session_state["window_combined"] = (0.30, float(suggested["end"]))
    app.run()
    if not no_exception(app, "after moving the range"):
        return
    said = " ".join(str(c.value) for c in app.get("caption"))
    check("a moved range is called yours, not the suggestion",
          "This range is yours" in said, said[:300])
    back = button_by_label(app, "Back to the suggested range")
    check("with the suggestion one button away", back is not None,
          str([b.label for b in app.button][:10]))
    if back is not None:
        back.click().run()
        check("and pressing it restores the suggested range",
              abs(float(state(app, "window_start")) - suggested["start"]) < 5e-4,
              str(state(app, "window_start")))

    # A search that drops a component answers about that curve, not the next.
    app.session_state["use_nucleus"] = False
    eps, force = curves[5]
    app.session_state["cell_name"] = "vcm-5"
    app.session_state["data"] = {
        "epsilon": eps, "force_N": force, "source": "vcm_5.csv", "n_dropped": 0,
    }
    app.run()
    if not no_exception(app, "loading the next curve"):
        return
    check("the next curve starts with its components back",
          all(state(app, f"use_{t}", False) for t in here),
          str({t: state(app, f"use_{t}", False) for t in here}))

    # And every modulus is quoted with its uncertainty, in the tile the
    # modulus is in rather than in a second table of the same numbers.
    fit = state(app, "_last_fit")
    check("it was fitted on arrival", fit and fit.get("success"))
    rows = app_module.result_rows(fit)
    keys = [row[0] for row in rows]
    check("there is a row for every component of this cell type",
          all(f"modulus_{t}" in keys for t in here), str(keys))
    check("and the uncertainty is a column of its own",
          any(row[3].startswith("±") for row in rows),
          str([row[3] for row in rows]))
    check("with the interval beside it",
          any(" to " in row[4] for row in rows),
          str([row[4] for row in rows]))
    check("the goodness of fit is a row too",
          "quality" in keys and "boundaries" in keys, str(keys))
    check("no table of tiles repeats the moduli",
          table_with(app, "modulus", "± (standard error)") is None)
    # The half-open reading of a range was the bug: an element that stops
    # taking more load has not stopped carrying it.
    supports = [row[5] for row in rows if row[0].startswith("modulus_")]
    check("a component that holds is said to hold, not to stop",
          all("stiffening to the end" in note or "holding from" in note
              or "not in this model" in note for note in supports),
          str(supports))


def case_the_boundaries_come_from_the_log_curve():
    print("the boundaries are read off the log slope, and the placements "
          "tried are shown")
    import app as app_module
    app = start(cell_name="cell-01")
    if not no_exception(app, "before the boundary search"):
        return

    # The boundaries section on the analysis tab keeps only the setting for
    # the whole experiment; the search itself lives on the log curve tab.
    source = pathlib.Path(APP).read_text()
    body = source.split('st.markdown("##### Boundaries")')[-1]
    check("the defaults panel is what is left under Boundaries",
          body.index("set_default_boundaries_control()")
          < body.index("def "), body[:200])
    check("and the search is on the log curve tab",
          "refine_boundaries_control(model_ex" in source)

    work = button_by_label(app, "Find the boundaries from the log curve")
    check("the button says where the boundaries come from", work is not None,
          str([b.label for b in app.button][:8]))
    if work is None:
        return
    work.click().run()
    if not no_exception(app, "the log-slope search"):
        return

    rows = state(app, "boundary_candidates")
    check("several placements were tried, not one", rows and len(rows) >= 3,
          str(len(rows or [])))
    if not rows:
        return
    check("one of them is the log curve's own answer",
          any("log slope" in row["why"] for row in rows),
          str([row["why"] for row in rows]))
    check("and one is what was on the page, so nothing changes unseen",
          any("on the page" in row["why"] for row in rows),
          str([row["why"] for row in rows]))
    check("every row carries what it came to",
          all(np.isfinite(row["r_squared"]) for row in rows),
          str([row.get("r_squared") for row in rows]))
    applied = (round(float(state(app, "segment_break_1")), 3),
               round(float(state(app, "segment_break_2")), 3))
    check("the best one was applied",
          any((round(row["break_1"], 3), round(row["break_2"], 3)) == applied
              for row in rows), str(applied))

    # And any other row can be taken with one press.
    others = [i for i, row in enumerate(rows)
              if (round(row["break_1"], 3), round(row["break_2"], 3)) != applied]
    if others:
        pick = rows[others[0]]
        take = button_by_label(app, "Use")
        check("the others are one button away", take is not None,
              str([b.label for b in app.button][:10]))
        if take is not None:
            take.click().run()
            no_exception(app, "taking another placement")
            check("and pressing it applies that placement",
                  abs(float(state(app, "segment_break_2"))
                      - pick["break_2"]) < 1e-3,
                  f"{state(app, 'segment_break_2')} vs {pick['break_2']}")

    # The slope profile is measured from the data, with no fit in it.
    profile = state(app, "_slope_profile")
    check("the log slope was measured along the curve",
          profile is not None and len(profile.get("epsilon", [])) > 5,
          str(len(profile.get("epsilon", [])) if profile else 0))

    said = " ".join(str(m.value) for m in app.get("markdown"))
    check("and the prose about power laws is gone",
          "What the power law says" not in said)


def case_the_plot_carries_what_is_sent_to_it():
    print("the plot draws the data and the fit, plus whatever is sent to it")
    import app as app_module
    app = start(cell_name="cell-01")
    if not no_exception(app, "before anything is sent"):
        return

    # A curve arrives with its boundaries drawn: they are the first thing
    # anybody checks against the shape of the curve.
    started = state(app, "plot_layers") or []
    check("a new curve arrives with its boundaries on the plot",
          [row["kind"] for row in started] == ["boundaries"], str(started))
    said = " ".join(str(m.value) for m in
                    list(app.get("markdown")) + list(app.get("caption")))
    check("the figure says what it is carrying", "On the plot" in said,
          said[:200])
    # The panel of switches is gone: the data and the fit are not optional,
    # and everything else arrives by being sent.
    check("there is no “what is drawn” panel any more",
          "What is drawn on the curve" not in said, said[:200])
    labels = [c.label for c in app.checkbox]
    check("the extras left are the bands and the range",
          any("Shaded segment bands" in (l or "") for l in labels)
          and any("range was fitted" in (l or "") for l in labels),
          str(labels[:12]))
    check("and data and fit is not a switch, because it is always on",
          not any("measured points and the fitted curve" in (l or "")
                  for l in labels), str(labels[:12]))

    sends = [b for b in app.button if (b.label or "").startswith("📤")]
    check("every part of the results can be sent to the plot",
          len(sends) >= 4, str([b.label for b in app.button][:12]))
    if not sends:
        return
    sends[0].click().run()
    if not no_exception(app, "sending a piece to the plot"):
        return
    layers = state(app, "plot_layers") or []
    check("what was sent joins what was there",
          len(layers) == len(started) + 1, str(layers))
    check("and the list under the figure names it",
          any(layers[0]["kind"] in row["kind"] for row in layers), str(layers))

    # A layer that follows the page: moving a boundary moves what is drawn,
    # rather than leaving a line where the boundary used to be.
    sends = [b for b in app.button if (b.label or "").startswith("📤")]
    for button in sends:
        button.click().run()
        break
    layers = state(app, "plot_layers") or []
    check("a second piece goes on beside the first",
          len(layers) == len(started) + 2,
          str([row["kind"] for row in layers]))
    if any(row["kind"] == "boundaries" for row in layers):
        app.session_state["segment_break_2"] = 0.47
        app.run()
        said = " ".join(str(m.value) for m in app.get("markdown"))
        check("the boundary layer follows the page rather than freezing",
              "ε₂ = 0.470" in said, said[-300:])

    drops = [b for b in app.button if b.label == "✕"]
    check("each piece comes off again", len(drops) == len(layers),
          f"{len(drops)} crosses for {len(layers)} layers")
    if drops:
        drops[0].click().run()
        no_exception(app, "taking a piece off the plot")
        check("and taking one off leaves the rest",
              len(state(app, "plot_layers") or []) == len(layers) - 1,
              str(state(app, "plot_layers")))

    clear = button_by_label(app, "Clear the plot")
    if clear is not None:
        clear.click().run()
        check("and the plot can be cleared in one press",
              not (state(app, "plot_layers") or []),
              str(state(app, "plot_layers")))


def case_the_plot_says_what_the_ranges_say():
    print("the ranges sit above the curve, and the curve draws what they say")
    import app as app_module
    source = pathlib.Path(APP).read_text()
    body = source.split('section("3 · Nonlinear fitting")')[-1]
    check("the curve is staked out under the component ranges",
          body.index("curve_slot = st.container()")
          > body.index("component_controls("), body[:200])
    check("and above the fit button",
          body.index("curve_slot = st.container()")
          < body.index('st.markdown("#### 2 · Fit")'), body[:200])

    # A curve whose deep element carries real force, so every element has a
    # curve of its own to check.
    eps, force = synthetic(En_kPa=30.0)
    app = AppTest.from_file(APP, default_timeout=600)
    app.run()
    app.session_state["c2c12_fit_mode"] = ADVANCED_MODE
    app.session_state["cell_name"] = "bands"
    app.session_state["show_components"] = True
    app.session_state["data"] = {
        "epsilon": eps, "force_N": force, "source": "s.csv", "n_dropped": 0,
    }
    app.run()
    if not no_exception(app, "a curve with every element carrying"):
        return

    fit = state(app, "_last_fit")
    results = state(app, "results")
    windows = (fit or {}).get("term_windows") or {}
    check("the fit carries a range per element", bool(windows), str(windows))
    if not (windows and results):
        return

    # The bands behind the curve are the element ranges, not a hard-coded
    # membrane / cytoskeleton / nucleus story that stopped being true the
    # moment an element was given a range of its own.
    bands = results.get("fit_windows") or []
    edges = sorted({round(float(v), 4)
                    for band in bands for v in band["range"]})
    wanted = sorted({round(float(v), 4)
                     for window in windows.values() for v in window})
    check("the bands break where the elements do",
          all(any(abs(edge - value) < 1e-3 for edge in edges)
              for value in wanted), f"{edges} against {wanted}")
    deep = [band for band in bands
            if "nucleus" in band["label"] or "nuclear" in band["label"]]
    check("the band carrying the deep element starts at its onset",
          deep and abs(float(deep[0]["range"][0])
                       - float(windows["nucleus"][0])) < 1e-3,
          str([band["range"] for band in bands]))
    check("and every band names who is carrying it",
          all(band["label"] and band["label"] != "nothing carrying"
              for band in bands), str([band["label"] for band in bands]))

    # Each element's own curve starts where the element starts.
    for name, term in (("membrane_N", "membrane"), ("interior_N", "interior"),
                       ("nucleus_N", "nucleus")):
        drawn = results.get(name)
        if drawn is None:
            continue
        good = np.isfinite(drawn)
        if not good.any():
            continue
        first = float(np.asarray(results["epsilon"])[good].min())
        check(f"the {term} curve begins at its own onset",
              first >= float(windows[term][0]) - 1e-3,
              f"drawn from {first:.3f}, window {windows[term]}")


def case_the_video_is_not_a_plot_marking():
    print("the video belongs to the record and the morphology, not the plot")
    app = start(cell_name="cell-01")
    if not no_exception(app, "plot options"):
        return
    keys = [c.key for c in app.checkbox]
    check("no video frame marker among the plot switches",
          "show_video_marker" not in keys, str(keys))
    check("and no video panel switch either",
          "video_show_panel" not in keys, str(keys))
    check("the marker is off in the style the plot is drawn with",
          "show_video_marker=False" in SOURCE)
    # The video tab itself is untouched: it is what links a cell to its
    # record and measures the cell's shape.
    check("the video tab still exists in the source",
          "adopt_video(" in SOURCE and "detect_nucleus" in SOURCE)


if __name__ == "__main__":
    for case in (
        case_c2c12_opens_on_the_four_regime_fit,
        case_loads_clean,
        case_legacy_record_refits,
        case_companion_file_guard,
        case_a_half_updated_deploy_says_so,
        case_fit_quality,
        case_fit_only_plot,
        case_springs_share_a_pitch,
        case_start_a_new_cell,
        case_manual_cell_and_probe_scale,
        case_four_element_model,
        case_component_heights_survive_a_log_axis,
        case_the_tab_is_stripped_back,
        case_sharing_controls_sit_with_the_parts,
        case_boundaries_are_checked_against_the_power_law,
        case_the_curve_comes_first,
        case_elements_carry_emojis_but_tables_do_not,
        case_bending_is_the_same_column_as_the_spring,
        case_cortical_actin_can_carry_it_first,
        case_the_cardiomyocyte_picture_holds_fluid,
        case_schematic_labels_do_not_collide,
        case_the_fit_never_softens,
        case_weighting_decides_where_the_fit_is_good,
        case_the_whole_range_is_used_and_fits,
        case_real_vcm_curves_start_near_three_halves,
        case_the_range_is_picked_and_a_bad_contact_is_found,
        case_named_hypotheses_are_compared,
        case_springs_are_round_and_the_balloon_exists,
        case_switching_cell_type_and_back_changes_nothing,
        case_no_nucleus_wording_for_a_cardiomyocyte,
        case_components_are_recommended,
        case_a_model_that_cannot_carry_a_term_says_so,
        case_search_says_when_a_winner_drops_a_spring,
        case_real_curve_is_steeper_than_any_fixed_power,
        case_confinement_earns_its_place_on_real_data,
        case_real_curve_gives_believable_numbers,
        case_cardiomyocyte_defaults_match_the_experiment,
        case_offset_is_available_and_signed,
        case_sarcomere_length,
        case_extra_terms_never_crash_old_paths,
        case_four_element_search,
        case_breakpoint_spread_is_the_real_error_bar,
        case_error_bars_are_reported,
        case_clone_keeps_the_whole_geometry,
        case_the_cardiomyocyte_has_three_materials,
        case_the_prefactors_are_the_papers,
        case_each_boundary_moves_on_its_own,
        case_the_controls_sit_above_the_curve,
        case_the_button_says_what_it_will_do,
        case_an_element_window_is_an_onset_not_a_mask,
        case_the_placement_search_finds_where_elements_act,
        case_a_new_curve_starts_from_the_cell_type_defaults,
        case_the_ranges_follow_the_fit,
        case_a_sheet_that_will_not_open_says_why,
        case_each_element_gets_its_own_bar,
        case_a_companion_file_is_found_wherever_it_sits,
        case_the_missing_file_message_says_what_it_can_see,
        case_the_data_is_a_field_and_the_model_a_dashed_line,
        case_the_range_carries_its_own_maths,
        case_the_page_ends_with_the_answer,
        case_the_fit_colour_moves_with_the_range,
        case_the_written_equation_is_the_fitted_curve,
        case_the_equation_reaches_the_page,
        case_the_plot_says_which_range_was_used,
        case_the_zoom_survives_a_refit,
        case_axis_numbers_are_powers_not_letters,
        case_a_fixed_cell_can_have_several_hertzian_terms,
        case_the_fixed_tick_locks_out_the_other_materials,
        case_the_sheet_can_be_the_database,
        case_a_sheet_with_repeated_headings_still_opens,
        case_the_sheet_has_a_summary_tab_and_a_working_tab,
        case_an_old_one_tab_sheet_is_split_without_losing_anything,
        case_a_fixed_cell_is_one_hertzian_solid,
        case_unticking_a_material_fits_without_it,
        case_a_new_curve_arrives_ready_to_fit,
        case_the_boundaries_come_from_the_log_curve,
        case_the_plot_carries_what_is_sent_to_it,
        case_the_plot_says_what_the_ranges_say,
        case_the_video_is_not_a_plot_marking,
        case_the_nucleus_is_a_balloon_too,
        case_the_myoblast_nucleus_reaches_the_page,
        case_component_heights_are_readable,
        case_the_membrane_protein_is_put_aside,
        case_the_interior_carries_first_contact,
        case_the_cardiomyocyte_curves_fit,
        case_q_and_the_boundaries_are_searched_together,
        case_one_fitting_routine,
        case_materials_are_explained_by_their_law,
        case_tables_are_not_clipped,
        case_one_video_two_doors,
        case_zero_modulus_explains_itself,
        case_guided_range_is_settable,
        case_search_maths_is_shown,
        case_the_cortex_carries_the_start,
        case_schematic_is_a_mechanics_diagram,
        case_guided_order_follows_the_work,
        case_it_picks_the_arrangement,
        case_fitting_applies_what_it_found,
        case_search_stays_fast,
        case_png_is_not_rendered_every_run,
        case_fit_statistics,
        case_chi_squared_reaches_the_page,
        case_axis_ranges,
        case_nucleus_spring_is_shorter,
        case_component_names_follow_the_cell_type,
        case_cardiomyocyte_model_is_flagged_provisional,
        case_a_zero_modulus_is_a_measurement,
        case_plain_language_helpers,
        case_guided_mode_is_the_default,
        case_full_control_shows_everything,
        case_curve_saved_as_a_tab,
        case_plot_options_are_under_the_plot,
        case_save_the_plot,
        case_fit_maths_box,
        case_sheet_row_matches_the_header,
        case_sheet_reorder_keeps_the_data,
        case_fit_line_and_heights_toggle,
        case_all_three_moduli_always_reported,
        case_load_share_table,
        case_download_when_box_is_absent,
        case_clear_cell_wins_over_dark_debris,
        case_fit_survives_a_rerun,
        case_database_section_without_a_fit,
        case_send_without_a_video,
        case_fit_stops_at_the_end_of_the_range,
        case_plot_clutter_toggles,
        case_preset_round_trip,
        case_model_names,
        case_one_range_control_for_the_whole_page,
        case_typing_the_ends_the_wrong_way_round,
        case_composition_radios,
        case_highlight,
        case_search_beats_the_old_grid,
        case_search_flags_what_it_cannot_see,
        case_search_applies_in_one_press,
        case_bare_plot,
        case_buttons_do_not_break_widgets,
        case_ordering_is_a_question_about_two_springs,
        case_ordering_reads_the_real_cardiomyocytes,
        case_composition_curve_rebuilds_the_fit,
        case_breakpoints_are_searched_when_they_matter,
        case_balloon_and_spring_tab_answers_by_itself,
    ):
        case()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failing: {FAILURES}")
        sys.exit(1)
    print("all passing")
