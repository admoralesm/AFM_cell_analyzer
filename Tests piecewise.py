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
    Geometry,
    fit_piecewise,
    piecewise_moduli,
    predict_piecewise,
    probe_correction,
)

TRUE = dict(k_align=2e-10, C0=1e-10, K_shell=1e-13, K_cyto=2e-11,
            K_nucleus=5e-12, K_nuc_cyto=2e-10, K_core=2e-9)


def truth(x, t=TRUE):
    """The spec's four laws, chained exactly as the fit chains them."""
    x = np.asarray(x, dtype=float)
    f5 = t["k_align"] * 5 + t["C0"]
    f40 = f5 + t["K_shell"] * 35 ** 3 + t["K_cyto"] * 35 ** 1.5
    f60 = f40 + t["K_nucleus"] * 20 ** 3 + t["K_nuc_cyto"] * 20 ** 1.5
    d = lambda a: np.clip(x - a, 0.0, None)  # noqa: E731
    return np.where(
        x < 5, t["k_align"] * x + t["C0"],
        np.where(
            x < 40, f5 + t["K_shell"] * d(5) ** 3 + t["K_cyto"] * d(5) ** 1.5,
            np.where(
                x < 60,
                f40 + t["K_nucleus"] * d(40) ** 3 + t["K_nuc_cyto"] * d(40) ** 1.5,
                f60 + t["K_core"] * d(60) ** 1.5,
            ),
        ),
    )


def curve(noise=0.0, top=91.2, n=1200, seed=0, t=TRUE):
    x = np.linspace(0.0, top, n)
    rng = np.random.default_rng(seed)
    return x / 100.0, truth(x, t) + rng.normal(0.0, noise, n)


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
