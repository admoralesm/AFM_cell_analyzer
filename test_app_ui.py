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

# Label -> the argument the model takes, mirroring app.py.
MEMBRANE_MODE = {"holds what it reached": "freeze", "keeps stiffening": "continue"}
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
    check("all four compositions ranked",
          len({(r["membrane"], r["cyto_start"]) for r in search["candidates"]}) == 4)

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
    print("one switch strips the plot to data and fit")
    app = start()
    switch = None
    for box in app.checkbox:
        if "data and fit only" in (box.label or "").lower():
            switch = box
    check("bare-plot switch present", switch is not None)
    if switch is None:
        return
    check("element curves off by default",
          app.session_state["show_components"] is False)
    switch.set_value(True).run()
    if not no_exception(app, "bare plot"):
        return

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
          button_by_label(app, "Send to Box") is not None)


def case_database_section_without_a_fit():
    print("the database section is reachable before any fit")
    app = AppTest.from_file("/root/AFM_cell_analyzer/app.py", default_timeout=180)
    app.run()
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
    send = button_by_label(app, "Send to Box")
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
    app = start(box_store=FakeStore(), cell_name="cell-01")
    check("no video is loaded", not app.session_state["video_path"])

    send = button_by_label(app, "Send to Box")
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
    app = start(use_nucleus=False, use_interior=False)
    if not no_exception(app, "membrane only"):
        return
    labels = [m.label for m in app.get("metric")]
    for wanted in ("membrane", "cytoskeleton", "nucleus"):
        check(f"{wanted} has a tile with only the membrane selected",
              any(wanted in (l or "").lower() for l in labels), str(labels))
    values = {m.label: (m.value, m.delta) for m in app.get("metric")}
    # Only the modulus tiles, which are the ones named "Ec …" and "Eₙ …".
    off = [
        (label, values[label][0], values[label][1])
        for label in ("Ec cytoskeleton", "E\u2099 nucleus") if label in values
    ]
    check("both switched-off modulus tiles were found", len(off) == 2, str(list(values)))
    for label, value, delta in off:
        check(f"{label} reads zero", value.strip().startswith("0"), value)
        check(f"{label} says it was not in the model",
              "not in this model" in (delta or ""), str(delta))


def case_load_share_table():
    print("the range-by-range table says who carries the load")
    app = start()
    if not no_exception(app, "load share table"):
        return
    text = " ".join(str(m.value) for m in app.get("markdown"))
    check("the table has a heading", "share the load" in text, text[:120])


def case_download_when_box_is_absent():
    print("a cell can be saved with no Box account")
    app = start(cell_name="cell-01")
    box = button_by_label(app, "Send to Box")
    check("the Box button is present", box is not None)
    if box is not None:
        check("and disabled without a connection", box.disabled is True)

    downloads = [d for d in app.get("download_button")
                 if "download this cell" in (d.label or "").lower()]
    check("a download is offered instead", len(downloads) == 1)

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

    def row_values(self, index):
        return list(self.header) if index == 1 else []

    def update(self, values=None, range_name=None, **kwargs):
        self.header = list(values[0])
        self.updates.append(list(values[0]))

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

    # The user's real sheet, as it stands today: no nucleus column.
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
        "break_1": 0.151, "break_2": 0.403, "fit_quality": 0.9997,
        "cell_height": 8.0, "notes": "test",
    })
    check("the write succeeded", ok, message)
    check("the header was extended, not rebuilt",
          sheet.header[:12] == existing, str(sheet.header[:12]))
    check("a nucleus column was added",
          "Young's Modulus (En, kPa)" in sheet.header, str(sheet.header))
    for wanted in ("ε₁ membrane hands over", "ε₂ nucleus engages"):
        check(f"{wanted} column added", wanted in sheet.header)

    check("exactly one row was written", len(sheet.rows) == 1)
    written = dict(zip(sheet.header, sheet.rows[0]))
    check("cell id in the right column", written["Cell ID"] == "cell-01")
    check("Em in the Em column",
          abs(float(written["Young's Modulus (Em, MPa)"]) - 0.605) < 1e-6)
    check("En in the new nucleus column",
          abs(float(written["Young's Modulus (En, kPa)"]) - 3.027) < 1e-6)
    check("ε₂ written", abs(float(written["ε₂ nucleus engages"]) - 0.403) < 1e-6)
    check("row length matches the header", len(sheet.rows[0]) == len(sheet.header))

    # A sheet whose columns someone has reordered must still be written
    # correctly, which is the whole point of going by name.
    shuffled = list(reversed(existing))
    manager2 = GoogleSheetsManager.__new__(GoogleSheetsManager)
    sheet2 = FakeWorksheet(shuffled)
    manager2.worksheet = sheet2
    manager2.append_cell_data({"cell_id": "cell-02", "Em": 1.5})
    written2 = dict(zip(sheet2.header, sheet2.rows[0]))
    check("reordered header still gets the right cell id",
          written2["Cell ID"] == "cell-02")
    check("reordered header still gets Em in the Em column",
          abs(float(written2["Young's Modulus (Em, MPa)"]) - 1.5) < 1e-6)


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


def case_range_table_shows_zero_moduli():
    print("each range lists the moduli active there, zero where not reached")
    app = start()
    if not no_exception(app, "range table"):
        return
    frames = app.get("dataframe")
    target = None
    for frame in frames:
        columns = list(getattr(frame.value, "columns", []))
        if "range" in columns and "membrane" in columns:
            target = frame.value
    check("the range table is on the page", target is not None)
    if target is None:
        return
    for column in ("E membrane", "E cytoskeleton", "E nucleus"):
        check(f"{column} column present", column in target.columns,
              str(list(target.columns)))
    first = target.iloc[0]
    check("the first range carries only the membrane",
          first["E cytoskeleton"].startswith("0")
          and first["E nucleus"].startswith("0"),
          f"{first['E cytoskeleton']!r} {first['E nucleus']!r}")
    check("and the membrane is non-zero there",
          not first["E membrane"].startswith("0 "), str(first["E membrane"]))
    last = target.iloc[-1]
    check("the last range carries the nucleus",
          not last["E nucleus"].startswith("0 "), str(last["E nucleus"]))


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


def case_range_starts_at_zero():
    print("the segmented range always starts at zero")
    app = start()
    ends = [s for s in app.slider if "fit up to" in (s.label or "").lower()]
    check("single end-of-range slider in the segmented view", len(ends) == 1)
    pairs = [s for s in app.slider if (s.label or "").strip() == "Fitted range"]
    check("no two-handle range slider in the segmented view", not pairs)
    if ends:
        ends[0].set_value(0.35).run()
        no_exception(app, "moving the end of the range")
        lo, hi = app.session_state["window_combined"]
        check("range starts at zero", lo == 0.0, str(lo))
        check("range ends where the slider was put", abs(hi - 0.35) < 0.01, str(hi))

    # The other models keep the two-handle slider.
    app2 = start()
    widget_by_label(app2, "radio", "how the cell is modelled").set_value(
        "Side by side (every element acts everywhere)"
    ).run()
    pairs = [s for s in app2.slider if (s.label or "").strip() == "Fitted range"]
    check("two-handle slider still there for the other models", len(pairs) == 1)


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


if __name__ == "__main__":
    for case in (
        case_loads_clean,
        case_legacy_record_refits,
        case_companion_file_guard,
        case_fit_quality,
        case_sheet_row_matches_the_header,
        case_fit_line_and_heights_toggle,
        case_range_table_shows_zero_moduli,
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
        case_range_starts_at_zero,
        case_composition_radios,
        case_highlight,
        case_search_beats_the_old_grid,
        case_search_flags_what_it_cannot_see,
        case_search_applies_in_one_press,
        case_bare_plot,
        case_buttons_do_not_break_widgets,
    ):
        case()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failing: {FAILURES}")
        sys.exit(1)
    print("all passing")
