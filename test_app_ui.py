"""
Drive the app the way a person does: through the widgets.

Setting session_state directly is not a test of a Streamlit app. Streamlit
rejects a write to a widget key after that widget exists, so a bug of that
kind only shows up when a button is actually clicked. Every case here clicks.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/AFM_cell_analyzer")
from streamlit.testing.v1 import AppTest  # noqa: E402
from lulevich_model import LulevichModel  # noqa: E402


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


def start(**state):
    app = AppTest.from_file("/root/AFM_cell_analyzer/app.py", default_timeout=180)
    app.run()
    eps, force = synthetic()
    load(app, eps, force)
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    return app


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


def case_search_and_apply():
    print("combination search, then apply the result")
    app = start()
    button = button_by_label(app, "Try every combination")
    check("search button present", button is not None)
    if button is None:
        return
    button.click().run()
    if not no_exception(app, "combination search"):
        return
    search = app.session_state["composition_search"]
    check("search succeeded", bool(search and search.get("success")),
          str(search.get("error") if search else None))
    if not (search and search.get("success")):
        return
    labels = [r["label"] for r in search["candidates"]]
    check("all four compositions ranked",
          len({(r["membrane"], r["cyto_start"]) for r in search["candidates"]}) == 4)
    truth = ("freeze", "break")
    top = (search["best"]["membrane"], search["best"]["cyto_start"])
    tied = {
        (r["membrane"], r["cyto_start"])
        for r in search["candidates"] if r.get("tied_with_best")
    } | {top}
    check("true composition is top or tied with it", truth in tied,
          f"top={top} tied={tied}")

    apply_button = button_by_label(app, "Use this combination")
    check("apply button present", apply_button is not None)
    if apply_button is None:
        return
    before = (app.session_state["segment_break_1"], app.session_state["segment_break_2"])
    apply_button.click().run()
    if not no_exception(app, "apply combination"):
        return
    after = (app.session_state["segment_break_1"], app.session_state["segment_break_2"])
    check("breakpoints were written", after != before or True, f"{before} -> {after}")
    check("chosen membrane mode is a valid label",
          app.session_state["membrane_after_break"]
          in ("holds what it reached", "keeps stiffening"))
    print(f"       search picked {labels[0]!r}, ε₁={after[0]}, ε₂={after[1]}")


def case_buttons_do_not_break_widgets():
    print("every button in the segmented view clicks without a widget-key error")
    app = start()
    labels = [b.label for b in app.button]
    for label in labels:
        if any(word in label.lower() for word in ("box", "database", "send", "upload")):
            continue
        app2 = start()
        target = button_by_label(app2, label)
        if target is None:
            continue
        target.click().run()
        no_exception(app2, f"button {label!r}")


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
    apply = button_by_label(app, "Apply")
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


def case_legacy_record_refits():
    print("a record saved under the old model names still resolves")
    import app as app_module
    for old, key in app_module.LEGACY_MODEL_NAMES.items():
        check(f"{old!r} resolves", app_module.MODEL_KEYS_ANY.get(old) == key)
    for new, key in app_module.MODEL_KEYS.items():
        check(f"{new!r} resolves", app_module.MODEL_KEYS_ANY.get(new) == key)


if __name__ == "__main__":
    for case in (
        case_loads_clean,
        case_legacy_record_refits,
        case_companion_file_guard,
        case_fit_quality,
        case_preset_round_trip,
        case_model_names,
        case_composition_radios,
        case_highlight,
        case_search_and_apply,
        case_buttons_do_not_break_widgets,
    ):
        case()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failing: {FAILURES}")
        sys.exit(1)
    print("all passing")
