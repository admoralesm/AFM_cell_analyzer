"""
Tests for the four-regime, C0-anchored C2C12 fit in piecewise_fit.py.

Plain numpy/scipy, no Streamlit, so they run anywhere:

    python tests_piecewise.py        # or: python -m pytest tests_piecewise.py
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from piecewise_fit import (  # noqa: E402
    C2C12_BOUNDARIES_PCT,
    component_curve,
    find_boundaries,
    joint_sse,
    Geometry,
    fit_piecewise,
    piecewise_moduli,
    predict_piecewise,
    probe_correction,
)

TRUE = dict(k_align=2e-10, C0=1e-10, K_shell=1e-13, K_cyto=2e-11,
            K_nucleus=5e-12, K_nuc_cyto=2e-10, K_core=2e-9)


def truth(x, t=TRUE, b=(5.0, 40.0, 60.0), membrane_throughout=True):
    """
    The four laws, chained exactly as the fit chains them.

    With the membrane acting throughout, the shell's K_shell*(x - e1)^3 keeps
    rising through regimes 3 and 4 on top of the other elements.
    """
    x = np.asarray(x, dtype=float)
    e1, e2, e3 = b
    d = lambda a: np.clip(x - a, 0.0, None)  # noqa: E731
    shell = t["K_shell"] * d(e1) ** 3
    carry = shell if membrane_throughout else 0.0
    f1 = t["k_align"] * e1 + t["C0"]
    f2 = f1 + t["K_shell"] * (e2 - e1) ** 3 + t["K_cyto"] * (e2 - e1) ** 1.5
    f3 = (f2 + t["K_nucleus"] * (e3 - e2) ** 3 + t["K_nuc_cyto"] * (e3 - e2) ** 1.5
          + (t["K_shell"] * ((e3 - e1) ** 3 - (e2 - e1) ** 3)
             if membrane_throughout else 0.0))
    return np.where(
        x < e1, t["k_align"] * x + t["C0"],
        np.where(
            x < e2, f1 + shell + t["K_cyto"] * d(e1) ** 1.5,
            np.where(
                x < e3,
                f2 + t["K_nucleus"] * d(e2) ** 3 + t["K_nuc_cyto"] * d(e2) ** 1.5
                + (carry - t["K_shell"] * (e2 - e1) ** 3
                   if membrane_throughout else 0.0),
                f3 + t["K_core"] * d(e3) ** 1.5
                + (carry - t["K_shell"] * (e3 - e1) ** 3
                   if membrane_throughout else 0.0),
            ),
        ),
    )


def curve(noise=0.0, top=91.2, n=1200, seed=0, t=TRUE, b=(5.0, 40.0, 60.0),
          membrane_throughout=True):
    x = np.linspace(0.0, top, n)
    rng = np.random.default_rng(seed)
    return x / 100.0, truth(x, t, b, membrane_throughout) + rng.normal(0.0, noise, n)


GEOMETRY = Geometry(cell_height=8.09e-6, cell_radius=4.45e-6,
                    nucleus_radius=1.5575e-6, probe_radius=20e-6)


# ------------------------------------------------------------------ fit ---

def test_recovers_every_coefficient_on_a_clean_curve():
    eps, f = curve()
    r = fit_piecewise(eps, f)
    assert r["success"], r
    for name, want in TRUE.items():
        got = r["coefficients"][name]
        assert np.isclose(got, want, rtol=1e-6, atol=1e-18), (name, got, want)
    assert r["r_squared"] > 0.999999


def test_the_spec_as_first_written_is_still_there():
    # carry=() holds the shell at the force it reached at e2, which is the
    # specification's R3 and R4 exactly.
    eps, f = curve(membrane_throughout=False)
    r = fit_piecewise(eps, f, carry=())
    for name, want in TRUE.items():
        assert np.isclose(r["coefficients"][name], want, rtol=1e-6, atol=1e-18), name


def test_the_membrane_acts_throughout():
    eps, f = curve(noise=0.2e-9)
    r = fit_piecewise(eps, f)
    reg = {x["key"]: x for x in r["regimes"]}
    # Carried into R3 and R4 with the value R2 measured, not refitted.
    for key in ("R3", "R4"):
        carried = reg[key]["carried"]["K_shell"]
        assert carried["value"] == reg["R2"]["params"]["K_shell"]["value"]
        assert carried["onset_pct"] == 5.0
    # Its own curve runs from e1 to the end of the fit.
    x, shell = component_curve(r, "K_shell")
    assert x[0] == 5.0 and np.isclose(x[-1], 91.2)
    assert shell[0] == 0.0 and np.all(np.diff(shell) >= 0)
    # And ignoring it on a curve that has it is measurably worse.
    held = fit_piecewise(eps, f, carry=())
    assert r["rmse"] < held["rmse"]


def test_the_curve_is_continuous_at_every_boundary():
    eps, f = curve(noise=0.5e-9)
    r = fit_piecewise(eps, f)
    gaps = r["continuity_gaps_N"]
    assert set(gaps) == {"5%", "40%", "60%"}
    assert all(abs(g) < 1e-20 for g in gaps.values()), gaps
    # And the fitted line is continuous where a plot would join it.
    for b in (5.0, 40.0, 60.0):
        left = predict_piecewise(np.array([(b - 1e-9) / 100.0]), r)[0]
        right = predict_piecewise(np.array([b / 100.0]), r)[0]
        assert abs(left - right) < 1e-15, (b, left, right)


def test_anchors_are_constants_passed_forward():
    eps, f = curve(noise=0.5e-9)
    r = fit_piecewise(eps, f)
    reg = {x["key"]: x for x in r["regimes"]}
    k = reg["R1"]["params"]["k_align"]["value"]
    c0 = reg["R1"]["params"]["C0"]["value"]
    assert np.isclose(r["anchors"]["F_5pct"], k * 5 + c0)
    assert reg["R2"]["anchor_in_N"] == r["anchors"]["F_5pct"]
    assert reg["R3"]["anchor_in_N"] == r["anchors"]["F_40pct"]
    assert reg["R4"]["anchor_in_N"] == r["anchors"]["F_60pct"]
    # The anchors are not parameters of the regimes they start.
    assert "F_5pct" not in reg["R2"]["params"]


def test_initial_guess_does_not_change_the_answer():
    eps, f = curve(noise=0.5e-9)
    base = fit_piecewise(eps, f)
    wild = fit_piecewise(eps, f, settings={
        "K_shell": {"p0": 1e3}, "K_cyto": {"p0": 0.0},
        "K_nucleus": {"p0": 1e-30}, "K_nuc_cyto": {"p0": 5.0},
        "K_core": {"p0": 1e-2},
    })
    for name in ("K_shell", "K_cyto", "K_nucleus", "K_nuc_cyto", "K_core"):
        assert np.isclose(base["coefficients"][name],
                          wild["coefficients"][name], rtol=1e-6), name


def test_coefficients_stay_non_negative_and_are_flagged_on_the_bound():
    # A regime 2 that bends the wrong way: the best unconstrained cubic term
    # would be negative, so it must sit on zero and say so.
    t = dict(TRUE, K_shell=-2e-13)
    eps, f = curve(t=t)
    r = fit_piecewise(eps, f)
    p = next(x for x in r["regimes"] if x["key"] == "R2")["params"]
    assert p["K_shell"]["value"] >= 0.0
    assert p["K_shell"]["at_bound"]
    assert any("K_shell" in w for w in r["warnings"])
    assert all(v >= 0 for k, v in r["coefficients"].items()
               if k.startswith("K_"))


def test_user_bounds_are_honoured():
    eps, f = curve(noise=0.5e-9)
    cap = 1e-10
    r = fit_piecewise(eps, f, settings={"K_nuc_cyto": {"upper": cap}})
    assert r["coefficients"]["K_nuc_cyto"] <= cap * (1 + 1e-9)


def test_a_point_on_a_boundary_belongs_to_the_regime_that_starts_there():
    x = np.array([0.0, 1, 2, 3, 4, 5, 10, 20, 30, 39, 40, 45, 50, 55, 59,
                  60, 70, 80, 91.2])
    r = fit_piecewise(x / 100.0, truth(x))
    counts = {reg["key"]: reg["n_points"] for reg in r["regimes"]}
    assert counts == {"R1": 5, "R2": 5, "R3": 5, "R4": 4}, counts


def test_a_curve_that_stops_early_fits_what_it_has():
    eps, f = curve(top=55.0)
    r = fit_piecewise(eps, f)
    assert r["success"]
    reg = {x["key"]: x for x in r["regimes"]}
    assert reg["R3"]["fitted"] and not reg["R4"]["fitted"]
    assert any("stops at" in w for w in r["warnings"])
    assert np.isnan(predict_piecewise(np.array([0.70]), r)[0])


def test_boundaries_must_increase():
    eps, f = curve()
    r = fit_piecewise(eps, f, boundaries_pct=(0, 40, 5, 60, 91.2))
    assert not r["success"] and "increase" in r["error"]


def test_nothing_is_fitted_past_the_end():
    eps, f = curve(top=100.0)
    r = fit_piecewise(eps, f)
    assert np.isclose(r["epsilon_range"][1], 0.912)
    assert np.isnan(predict_piecewise(np.array([0.95]), r)[0])


def test_boundaries_can_be_moved():
    eps, f = curve(noise=0.5e-9)
    r = fit_piecewise(eps, f, boundaries_pct=(0, 4, 35, 65, 85))
    assert r["success"]
    assert r["boundaries_pct"] == (0.0, 4.0, 35.0, 65.0, 85.0)
    assert set(r["anchors"]) == {"F_4pct", "F_35pct", "F_65pct"}


# ------------------------------------------------------ boundary search ---

def test_the_search_finds_boundaries_that_were_moved():
    true = (7.0, 36.0, 64.0)
    eps, f = curve(noise=0.5e-9, n=1500, b=true)
    found = find_boundaries(eps, f)
    assert found["success"]
    assert found["strength"] == "strong", found["delta_bic"]
    for key, want in zip(("eps1", "eps2", "eps3"), true):
        iv = found["intervals"][key]
        assert iv["lo95"] - 0.3 <= want <= iv["hi95"] + 0.3, (key, iv, want)
    # The two boundaries the curve pins down, pinned down.
    assert abs(found["best_pct"][1] - 36.0) < 0.6
    assert abs(found["best_pct"][2] - 64.0) < 0.6


def test_the_search_keeps_the_spec_when_the_spec_is_right():
    # Boundaries exactly at 5 / 40 / 60: moving them must not pay for itself.
    for seed in range(3):
        eps, f = curve(noise=0.5e-9, n=1500, seed=seed)
        found = find_boundaries(eps, f)
        assert found["delta_bic"] < 0, (seed, found["delta_bic"])
        assert found["strength"] == "none"
        assert abs(found["best_pct"][2] - 60.0) < 0.6


def test_the_search_stays_inside_its_bands_and_can_hold_one_fixed():
    eps, f = curve(noise=0.5e-9, n=1200, b=(7.0, 36.0, 64.0))
    found = find_boundaries(eps, f, bands_pct=((5.0, 5.0), (38.0, 45.0), (55.0, 62.0)))
    b1, b2, b3 = found["best_pct"]
    assert b1 == 5.0
    assert 38.0 <= b2 <= 45.0 and 55.0 <= b3 <= 62.0
    # ε₃ wants 64 % and is not allowed past 62 %: it says so.
    assert b3 == 62.0 and found["intervals"]["eps3"]["at_band_edge"]


def test_the_search_uses_the_page_settings():
    # A bound that forbids the nuclear envelope changes S, so it is used.
    eps, f = curve(noise=0.5e-9, n=1200)
    free = joint_sse(eps, f)
    capped = joint_sse(eps, f, settings={"K_nucleus": {"upper": 0.0}})
    assert capped > free
    held = joint_sse(eps, f, carry=())
    assert held != free


def test_the_joint_curve_is_never_worse_than_the_chain():
    # The joint S is the best continuous curve of the same family, so the
    # sequential fit at the same boundaries can only match or exceed it.
    eps, f = curve(noise=0.5e-9, n=1200)
    for b in ((0, 5, 40, 60, 91.2), (0, 4, 37, 63, 91.2)):
        r = fit_piecewise(eps, f, boundaries_pct=b)
        x = eps * 100
        m = (x >= 0) & (x <= 91.2)
        seq = np.nansum((f[m] - predict_piecewise(eps[m], r)) ** 2)
        assert joint_sse(eps, f, boundaries_pct=b) <= seq * (1 + 1e-9)


# --------------------------------------------------------------- moduli ---

def test_probe_correction():
    assert probe_correction(4.45e-6, None) == 1.0
    assert probe_correction(4.45e-6, 0.0) == 1.0
    assert np.isclose(probe_correction(4.45e-6, 1.0), 1.0, atol=1e-3)
    c = probe_correction(4.45e-6, 20e-6)
    assert 0.94 < c < 0.96, c
    # A smaller probe is a sharper contact and softer still.
    assert probe_correction(4.45e-6, 5e-6) < c


def test_prefactors_match_the_lulevich_model():
    from lulevich_model import LulevichModel

    m = LulevichModel(np.zeros(5), np.linspace(0, 0.5, 5), cell_height=8.09e-6,
                      cell_radius=4.45e-6, nucleus_radius=1.5575e-6)
    flat = Geometry(cell_height=8.09e-6, cell_radius=4.45e-6,
                    nucleus_radius=1.5575e-6, probe_radius=None)
    eps, f = curve()
    rows = piecewise_moduli(fit_piecewise(eps, f), flat)
    assert np.isclose(rows["K_shell"]["prefactor_N_per_Pa"], m.Am)
    assert np.isclose(rows["K_cyto"]["prefactor_N_per_Pa"], m.Ai)
    assert np.isclose(rows["K_nucleus"]["prefactor_N_per_Pa"], m.An_shell)
    assert np.isclose(rows["K_core"]["prefactor_N_per_Pa"], m.An)
    assert np.isclose(rows["k_align"]["prefactor_N_per_Pa"], m.At)


def test_moduli_round_trip():
    # Pick moduli, turn them into coefficients with the prefactors, build
    # the curve, fit it, and get the moduli back.
    g = GEOMETRY
    want = {"K_shell": 5e5, "K_cyto": 2e3, "K_nucleus": 3e6,
            "K_nuc_cyto": 1e4, "K_core": 5e4}
    eps, f = curve()
    probe = piecewise_moduli(fit_piecewise(eps, f), g)
    t = dict(TRUE)
    for name, E in want.items():
        power = probe[name]["power"]
        t[name] = E * probe[name]["prefactor_N_per_Pa"] / 100.0 ** power
    eps, f = curve(t=t)
    rows = piecewise_moduli(fit_piecewise(eps, f), g)
    for name, E in want.items():
        assert np.isclose(rows[name]["E_Pa"], E, rtol=1e-5), (name, rows[name]["E_Pa"])


def test_every_coefficient_has_a_modulus():
    eps, f = curve(noise=0.5e-9)
    rows = piecewise_moduli(fit_piecewise(eps, f), GEOMETRY)
    assert set(rows) == {"k_align", "K_shell", "K_cyto", "K_nucleus",
                         "K_nuc_cyto", "K_core"}
    for row in rows.values():
        assert np.isfinite(row["E_Pa"]) and row["E_Pa"] >= 0, row
        assert np.isfinite(row["E_se_Pa"]), row
    assert "tension_N_per_m" in rows["k_align"]


def test_spec_defaults():
    assert C2C12_BOUNDARIES_PCT == (0.0, 5.0, 40.0, 60.0, 91.2)
    eps, f = curve()
    r = fit_piecewise(eps, f)
    p0 = {name: p["p0"] for reg in r["regimes"] for name, p in reg["params"].items()}
    assert p0["K_shell"] == 1e-9 and p0["K_cyto"] == 1e-9
    assert p0["K_nucleus"] == 1e-8 and p0["K_nuc_cyto"] == 1e-8
    assert p0["K_core"] == 1e-7
    # C0's starting value is the mean of the first ten points.
    assert np.isclose(p0["C0"], np.mean(f[:10]))


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"  ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc!r}")
    print("all passing" if not failures else f"{failures} failing")
    sys.exit(1 if failures else 0)
