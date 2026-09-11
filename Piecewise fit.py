"""
Sequential, C0-anchored piecewise fit of a whole-cell compression curve.

Built for C2C12 myoblasts squashed by a large spherical probe (40 um), where
the curve is read as four regimes met one after another as the relative
deformation x (in PERCENT) grows:

    R1  contact artefact / alignment    0 %  .. < 5 %
            F1(x) = k_align * x + C0
    R2  parallel cell stretch + cytoskeleton    5 % .. < 40 %
            F2(x) = K_shell * dx^3 + K_cyto * dx^1.5 + F_5pct,      dx = x - 5
    R3  nuclear envelope stretch (dual, parallel)   40 % .. < 60 %
            F3(x) = K_nucleus * dx^3 + K_nuc_cyto * dx^1.5 + F_40pct, dx = x - 40
    R4  dense intranuclear packing    60 % .. 91.2 %
            F4(x) = K_core * dx^1.5 + F_60pct,                     dx = x - 60

The regimes are fitted in that order. Each one ends by evaluating itself at
its right-hand boundary, and that force is passed to the next regime as a
fixed constant, never a free parameter. The whole curve is therefore
continuous (C0) across [0, end] by construction, whatever the data do.

Why this is easy to trust
-------------------------
Once the anchor is fixed every regime is LINEAR in its coefficients, so each
bounded fit is a convex problem with a single minimum. The coefficients are
found with Trust-Region Reflective (``scipy.optimize.least_squares``,
``method="trf"``) started from the user's initial guesses and held inside
the user's bounds (non-negative by default), and are cross-checked against
the exact bounded linear solution (``scipy.optimize.lsq_linear``). The
lower-cost answer is kept, so a poor initial guess can slow the solver down
but cannot change the result.

Units
-----
Force in newtons, x in percent. A coefficient K multiplying dx^p therefore
has units of N / %^p. Converting to the dimensionless deformation e = x/100
used by the Lulevich prefactors multiplies it by 100^p.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares, lsq_linear

__all__ = [
    "Term",
    "Regime",
    "Geometry",
    "C2C12_REGIMES",
    "C2C12_BOUNDARIES_PCT",
    "fit_piecewise",
    "predict_piecewise",
    "regime_curves",
    "piecewise_moduli",
    "probe_correction",
    "with_settings",
    "parameter_table",
]


# ============================================================ specification ==

@dataclass(frozen=True)
class Term:
    """One coefficient of one regime: K * dx^power."""

    name: str
    power: float
    p0: float
    lower: float = 0.0
    upper: float = np.inf
    # Which element this coefficient describes, for the modulus conversion.
    element: str = ""
    label: str = ""


@dataclass(frozen=True)
class Regime:
    """One stretch of the curve and the law fitted on it."""

    key: str
    title: str
    terms: tuple
    # Only the first regime has a free intercept (C0). Every other regime
    # starts from the force the previous one reached, as a hard constant.
    free_offset: bool = False
    equation: str = ""


# Relative deformation, in percent, where each regime starts, and where the
# last one ends. Five numbers for four regimes.
C2C12_BOUNDARIES_PCT = (0.0, 5.0, 40.0, 60.0, 91.2)

C2C12_REGIMES = (
    Regime(
        key="R1",
        title="Contact artefact & alignment",
        terms=(
            # k_align's starting value is computed from the data (the slope
            # over the first 5 %), so the number here is only a placeholder.
            Term("k_align", 1.0, p0=np.nan, lower=-np.inf, upper=np.inf,
                 element="alignment", label="alignment slope"),
        ),
        free_offset=True,
        equation="F₁(x) = k_align·x + C₀",
    ),
    Regime(
        key="R2",
        title="Parallel cell stretch + cytoskeleton",
        terms=(
            Term("K_shell", 3.0, p0=1.0e-9, element="membrane",
                 label="cell shell (membrane/cortex) stretch"),
            Term("K_cyto", 1.5, p0=1.0e-9, element="cytoskeleton",
                 label="cytoskeleton compression"),
        ),
        equation="F₂(x) = K_shell·(x−5)³ + K_cyto·(x−5)^1.5 + F_5%",
    ),
    Regime(
        key="R3",
        title="Dual-cubic parallel nuclear envelope stretch",
        terms=(
            Term("K_nucleus", 3.0, p0=1.0e-8, element="nuclear_envelope",
                 label="nuclear envelope stretch"),
            Term("K_nuc_cyto", 1.5, p0=1.0e-8, element="perinuclear_cytoskeleton",
                 label="perinuclear cytoskeleton compression"),
        ),
        equation="F₃(x) = K_nucleus·(x−40)³ + K_nuc_cyto·(x−40)^1.5 + F_40%",
    ),
    Regime(
        key="R4",
        title="Dense intranuclear packing",
        terms=(
            Term("K_core", 1.5, p0=1.0e-7, element="nuclear_core",
                 label="nuclear interior compression"),
        ),
        equation="F₄(x) = K_core·(x−60)^1.5 + F_60%",
    ),
)


def with_settings(regimes, settings):
    """
    The regimes with the user's initial guesses and bounds applied.

    ``settings`` maps a coefficient name to any of ``p0``, ``lower`` and
    ``upper``. Names that are not in the regimes are ignored, so a stored
    setting for a coefficient that no longer exists cannot break a fit.
    """
    settings = settings or {}
    out = []
    for regime in regimes:
        terms = []
        for term in regime.terms:
            wanted = settings.get(term.name) or {}
            changes = {}
            for key in ("p0", "lower", "upper"):
                if key in wanted and wanted[key] is not None:
                    try:
                        changes[key] = float(wanted[key])
                    except (TypeError, ValueError):
                        continue
            terms.append(replace(term, **changes) if changes else term)
        out.append(replace(regime, terms=tuple(terms)))
    return tuple(out)


def parameter_table(regimes=C2C12_REGIMES):
    """Rows of (regime, name, power, p0, lower, upper) for display/editing."""
    rows = []
    for regime in regimes:
        for term in regime.terms:
            rows.append({
                "regime": regime.key,
                "name": term.name,
                "power": term.power,
                "p0": term.p0,
                "lower": term.lower,
                "upper": term.upper,
                "label": term.label,
            })
    return rows


# ================================================================== solving ==

def _statistics(y, predicted, n_params):
    """R², RMSE and the rest for one stretch of curve."""
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = int(y.size)
    if n == 0:
        return {"r_squared": float("nan"), "rmse": float("nan"),
                "ss_res": 0.0, "n_points": 0}
    residual = y - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(ss_res / n)),
        "ss_res": ss_res,
        "n_points": n,
    }


def _covariance(A, residual, free):
    """Parameter covariance from the Jacobian of the free parameters."""
    k = A.shape[1]
    cov = np.full((k, k), np.nan)
    idx = np.flatnonzero(free)
    dof = A.shape[0] - idx.size
    if idx.size == 0 or dof <= 0:
        return cov
    J = A[:, idx]
    s2 = float(np.sum(residual ** 2)) / dof
    try:
        sub = s2 * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:  # pragma: no cover - pinv rarely fails
        return cov
    cov[np.ix_(idx, idx)] = sub
    return cov


def _bounded_linear_fit(A, y, p0, lower, upper):
    """
    min ||A p - y||  subject to  lower <= p <= upper.

    Solved in a scaled space (unit-norm columns, unit-size target) so that
    coefficients spanning ten orders of magnitude are all of order one to the
    solver. Trust-Region Reflective from the initial guess, cross-checked
    against the exact bounded linear solution; the lower cost wins.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    col = np.linalg.norm(A, axis=0)
    col[~np.isfinite(col) | (col == 0)] = 1.0
    f_scale = float(np.max(np.abs(y))) if y.size and np.any(y) else 1.0
    As = A / col
    ys = y / f_scale
    lb = lower * col / f_scale
    ub = upper * col / f_scale

    start = np.asarray(p0, dtype=float) * col / f_scale
    start = np.where(np.isfinite(start), start, 0.0)
    # Strictly inside the box: TRF needs a feasible start, and a guess on or
    # past a bound is moved just inside it rather than refused.
    span = np.where(np.isfinite(ub - lb), ub - lb, 1.0)
    eps = 1e-10 * np.maximum(span, 1.0)
    start = np.clip(start, lb + eps, ub - eps)

    candidates = []
    try:
        trf = least_squares(
            lambda q: As @ q - ys, start, jac=lambda q: As,
            bounds=(lb, ub), method="trf", x_scale="jac",
            ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=2000,
        )
        candidates.append(("trust-region reflective (from p0)", trf.x))
    except Exception:  # pragma: no cover - fall through to the exact solver
        pass
    try:
        exact = lsq_linear(As, ys, bounds=(lb, ub), method="trf",
                           tol=1e-14, lsmr_tol="auto", max_iter=2000)
        candidates.append(("bounded linear least squares (TRF)", exact.x))
    except Exception:  # pragma: no cover
        pass
    if not candidates:
        raise RuntimeError("both bounded solvers failed")

    def cost(q):
        return float(np.sum((As @ q - ys) ** 2))

    engine, q = min(candidates, key=lambda item: cost(item[1]))
    q = np.clip(q, lb, ub)
    params = q * f_scale / col
    # A coefficient sitting on a bound is not a measurement of that
    # coefficient: the data wanted it further out and were not allowed.
    tol = 1e-9 * np.maximum(np.abs(q), 1.0)
    at_bound = (np.abs(q - lb) <= tol) | (np.abs(q - ub) <= tol)
    return params, engine, at_bound


# ============================================================ the pipeline ==

def fit_piecewise(
    epsilon,
    force_N,
    boundaries_pct=C2C12_BOUNDARIES_PCT,
    regimes=C2C12_REGIMES,
    settings=None,
    min_points=None,
):
    """
    Fit every regime in turn, each anchored to where the last one ended.

    Parameters
    ----------
    epsilon : array
        Relative deformation as a FRACTION (0.40 is 40 %), the app's unit.
    force_N : array
        Force in newtons.
    boundaries_pct : sequence of len(regimes) + 1 floats
        Where each regime starts, in percent, then where the last one ends.
    regimes : tuple of Regime
    settings : dict, optional
        Per-coefficient overrides of p0 / lower / upper, see
        :func:`with_settings`.
    min_points : int, optional
        Fewest points a regime may be fitted on. Defaults to its number of
        free parameters plus two.

    Returns
    -------
    dict with ``success``, per-regime results under ``regimes``, the flat
    ``coefficients``, the ``anchors``, whole-curve statistics and
    ``warnings``.
    """
    regimes = with_settings(regimes, settings)
    bounds = [float(b) for b in boundaries_pct]
    if len(bounds) != len(regimes) + 1:
        return {"success": False,
                "error": f"need {len(regimes) + 1} boundaries, got {len(bounds)}"}
    if any(b1 <= b0 for b0, b1 in zip(bounds, bounds[1:])):
        return {"success": False,
                "error": "the boundaries must increase: "
                         + " < ".join(f"{b:g} %" for b in bounds)}

    eps = np.asarray(epsilon, dtype=float).ravel()
    force = np.asarray(force_N, dtype=float).ravel()
    if eps.shape != force.shape:
        return {"success": False, "error": "deformation and force differ in length"}
    good = np.isfinite(eps) & np.isfinite(force)
    order = np.argsort(eps[good], kind="stable")
    x_all = eps[good][order] * 100.0
    f_all = force[good][order]
    if x_all.size < 5:
        return {"success": False, "error": "fewer than five usable points"}

    warnings = []
    data_top = float(x_all.max())
    end = bounds[-1]
    if data_top < end:
        warnings.append(
            f"The curve stops at {data_top:.1f} %, before the {end:g} % the "
            f"last regime is set to reach. It is fitted on what is there."
        )

    anchor = None
    anchors = {}
    coefficients = {}
    regime_out = []
    chain_broken = False

    for i, regime in enumerate(regimes):
        a, b = bounds[i], bounds[i + 1]
        last = i == len(regimes) - 1
        inside = (x_all >= a) & ((x_all <= b) if last else (x_all < b))
        x = x_all[inside]
        f = f_all[inside]
        names = [t.name for t in regime.terms]
        powers = np.array([t.power for t in regime.terms], dtype=float)
        n_free = len(regime.terms) + (1 if regime.free_offset else 0)
        need = int(min_points) if min_points else n_free + 2

        entry = {
            "key": regime.key,
            "title": regime.title,
            "equation": regime.equation,
            "domain_pct": (a, b),
            "n_points": int(x.size),
            "anchor_in_N": None if regime.free_offset else anchor,
            "anchor_out_N": None,
            "params": {},
            "fitted": False,
            "engine": "",
        }

        if chain_broken or x.size < need:
            if not chain_broken:
                warnings.append(
                    f"{regime.key} ({a:g}–{b:g} %) has {x.size} points, fewer "
                    f"than the {need} it needs, so "
                    + ("it was not fitted." if last else
                       "it and every regime after it were not fitted: each "
                       "one needs the anchor the one before it ends on.")
                )
            chain_broken = True
            for t in regime.terms:
                entry["params"][t.name] = {
                    "value": float("nan"), "se": float("nan"), "p0": t.p0,
                    "lower": t.lower, "upper": t.upper, "power": t.power,
                    "at_bound": False,
                }
                coefficients[t.name] = float("nan")
            if regime.free_offset:
                coefficients["C0"] = float("nan")
            entry.update(_statistics([], [], n_free))
            regime_out.append(entry)
            continue

        if regime.free_offset:
            # R1: an ordinary straight-line least-squares fit. Its starting
            # values are reported for the record; a straight line has one
            # answer and needs none.
            k0 = (f[-1] - f[0]) / (x[-1] - x[0]) if x[-1] > x[0] else 0.0
            c0 = float(np.mean(f_all[:10]))
            design = np.column_stack([x ** p for p in powers] + [np.ones_like(x)])
            sol, *_ = np.linalg.lstsq(design, f, rcond=None)
            predicted = design @ sol
            cov = _covariance(design, f - predicted, np.ones(design.shape[1], bool))
            slope, offset = float(sol[0]), float(sol[1])
            entry["params"][names[0]] = {
                "value": slope, "se": float(np.sqrt(max(cov[0, 0], 0.0))),
                "p0": float(k0), "lower": -np.inf, "upper": np.inf,
                "power": 1.0, "at_bound": False,
            }
            entry["params"]["C0"] = {
                "value": offset, "se": float(np.sqrt(max(cov[1, 1], 0.0))),
                "p0": c0, "lower": -np.inf, "upper": np.inf, "power": 0.0,
                "at_bound": False,
            }
            coefficients[names[0]] = slope
            coefficients["C0"] = offset
            entry["engine"] = "linear least squares"
            anchor = slope * b + offset
        else:
            dx = x - a
            design = np.column_stack([dx ** p for p in powers])
            target = f - anchor
            p0 = [t.p0 for t in regime.terms]
            lower = [t.lower for t in regime.terms]
            upper = [t.upper for t in regime.terms]
            params, engine, at_bound = _bounded_linear_fit(
                design, target, p0, lower, upper,
            )
            predicted = design @ params + anchor
            cov = _covariance(design, f - predicted, ~at_bound)
            for j, t in enumerate(regime.terms):
                entry["params"][t.name] = {
                    "value": float(params[j]),
                    "se": float(np.sqrt(cov[j, j])) if np.isfinite(cov[j, j])
                    and cov[j, j] >= 0 else float("nan"),
                    "p0": float(t.p0), "lower": float(t.lower),
                    "upper": float(t.upper), "power": float(t.power),
                    "at_bound": bool(at_bound[j]),
                }
                coefficients[t.name] = float(params[j])
                if at_bound[j]:
                    where = "its lower bound" if np.isclose(
                        params[j], t.lower, atol=1e-300) else "its upper bound"
                    warnings.append(
                        f"{t.name} in {regime.key} settled on {where} "
                        f"({params[j]:.3g}). The data wanted it further out; "
                        f"the {t.label or t.name} is not resolved on this curve."
                    )
            entry["engine"] = engine
            anchor = anchor + float(
                sum(params[j] * (b - a) ** powers[j] for j in range(len(powers)))
            )

        entry["anchor_out_N"] = float(anchor)
        entry.update(_statistics(f, predicted, n_free))
        entry["fitted"] = True
        regime_out.append(entry)
        if not last:
            anchors[f"F_{b:g}pct"] = float(anchor)

    fitted_regimes = [r for r in regime_out if r["fitted"]]
    if not fitted_regimes:
        return {"success": False, "error": "no regime had enough points to fit",
                "warnings": warnings, "regimes": regime_out}

    reach = min(end, data_top)
    result = {
        "success": True,
        "model": "piecewise",
        "boundaries_pct": tuple(bounds),
        "regimes": regime_out,
        "coefficients": coefficients,
        "anchors": anchors,
        "epsilon_range": (bounds[0] / 100.0, reach / 100.0),
        "warnings": warnings,
    }
    # Whole-curve quality over everything that was fitted.
    lo = bounds[0]
    hi = max(r["domain_pct"][1] for r in fitted_regimes)
    covered = (x_all >= lo) & (x_all <= hi)
    predicted = predict_piecewise(x_all[covered] / 100.0, result)
    ok = np.isfinite(predicted)
    n_params = sum(
        len(r["params"]) for r in fitted_regimes
    )
    stats = _statistics(f_all[covered][ok], predicted[ok], n_params)
    result.update({
        "r_squared": stats["r_squared"],
        "rmse": stats["rmse"],
        "n_points": stats["n_points"],
        "n_params": int(n_params),
    })
    try:  # the chi-squared the rest of the app reports, when available
        from lulevich_model import fit_statistics

        full = fit_statistics(
            f_all[covered][ok], predicted[ok], n_params,
            epsilon=x_all[covered][ok] / 100.0,
        )
        result.update({
            "adj_r_squared": full["adj_r_squared"],
            "chi_squared": full["chi_squared"],
            "chi_squared_reduced": full["chi_squared_reduced"],
            "noise_sigma": full["noise_sigma"],
        })
    except Exception:  # pragma: no cover - optional companion
        pass
    result["continuity_gaps_N"] = continuity_gaps(result)
    return result


def _regime_force(regime, x):
    """One fitted regime's law evaluated at x (percent), NaN if not fitted."""
    x = np.asarray(x, dtype=float)
    if not regime["fitted"]:
        return np.full(x.shape, np.nan)
    a = regime["domain_pct"][0]
    params = regime["params"]
    if regime["anchor_in_N"] is None:
        name = next(k for k in params if k != "C0")
        return params[name]["value"] * x + params["C0"]["value"]
    dx = np.clip(x - a, 0.0, None)
    out = np.full(x.shape, float(regime["anchor_in_N"]))
    for p in params.values():
        out = out + p["value"] * dx ** p["power"]
    return out


def predict_piecewise(epsilon, result):
    """
    The fitted curve at ``epsilon`` (fraction), in newtons.

    NaN outside the fitted domain and inside any regime that was not fitted,
    so a plot stops the line instead of extrapolating a power law.
    """
    eps = np.asarray(epsilon, dtype=float)
    x = eps * 100.0
    out = np.full(x.shape, np.nan)
    regimes = result.get("regimes") or []
    for i, regime in enumerate(regimes):
        a, b = regime["domain_pct"]
        last = i == len(regimes) - 1
        mask = (x >= a) & ((x <= b) if last else (x < b))
        if mask.any():
            out[mask] = _regime_force(regime, x[mask])
    return out


def regime_curves(result, n=200):
    """Dense (x_pct, F_N) arrays per fitted regime, for drawing."""
    curves = []
    for regime in result.get("regimes") or []:
        if not regime["fitted"]:
            continue
        a, b = regime["domain_pct"]
        x = np.linspace(a, b, n)
        curves.append((regime["key"], x, _regime_force(regime, x)))
    return curves


def continuity_gaps(result):
    """F_i(b) - F_{i+1}(b) at every internal boundary, in newtons. All zero."""
    gaps = {}
    regimes = result.get("regimes") or []
    for left, right in zip(regimes, regimes[1:]):
        if not (left["fitted"] and right["fitted"]):
            continue
        b = left["domain_pct"][1]
        gaps[f"{b:g}%"] = float(
            _regime_force(left, np.array([b]))[0]
            - _regime_force(right, np.array([b]))[0]
        )
    return gaps


# ============================================================ the moduli ==

@dataclass
class Geometry:
    """Everything the analytical modulus conversion needs. Metres."""

    cell_height: float
    cell_radius: float
    nucleus_radius: float
    # None or 0 means a flat plate (no curvature correction).
    probe_radius: float | None = 20e-6
    membrane_thickness: float = 4e-9
    envelope_thickness: float = 40e-9
    # Only used to express the linear (alignment) term as an equivalent
    # modulus, the same way the app treats its in-plane tension spring.
    coat_thickness: float = 200e-9
    nu_membrane: float = 0.5
    nu_interior: float = 0.5
    nu_nucleus: float = 0.5
    extras: dict = field(default_factory=dict)


def probe_correction(radius, probe_radius):
    """
    How much softer a curved probe makes a Hertzian contact than a flat plate.

    The Lulevich interior term is a sphere squashed between two flat plates:
    two identical Hertz contacts in series, each taking half the
    indentation. With a sphere of radius Rp on top, the upper contact has
    the reduced radius R* = R Rp / (R + Rp) while the lower one, on the dish,
    keeps R. Adding the two indentations at equal force gives

        F = (4/3) E* d^(3/2) / (R*^(-1/3) + R^(-1/3))^(3/2)

    and dividing by the two-plate result gives this factor,

        c = [ 2 R^(-1/3) / (R*^(-1/3) + R^(-1/3)) ]^(3/2)   (<= 1)

    which is exactly 1 for a flat probe (Rp -> infinity). For a 20 um probe
    radius on a 4.5 um cell it is about 0.95, i.e. the flat-plate formula
    would under-report a Hertzian modulus by about 5 %.
    """
    R = float(radius)
    if not probe_radius or not np.isfinite(probe_radius) or probe_radius <= 0:
        return 1.0
    Rp = float(probe_radius)
    r_top = R * Rp / (R + Rp)
    return float(
        (2.0 * R ** (-1.0 / 3.0) / (r_top ** (-1.0 / 3.0) + R ** (-1.0 / 3.0)))
        ** 1.5
    )


def _prefactors(g: Geometry):
    """Newtons per pascal for each element, at e = 1 (e dimensionless).

    The same expressions as LulevichModel (Am, Ai, An_shell, An, At), with
    the probe-curvature correction applied to the three Hertzian terms.
    """
    R0, Rn, h0 = g.cell_radius, g.nucleus_radius, g.cell_height
    c_cell = probe_correction(R0, g.probe_radius)
    c_nuc = probe_correction(Rn, g.probe_radius)
    hertz_cell = np.sqrt(2.0) * R0 ** 2 / (3.0 * (1.0 - g.nu_interior ** 2))
    hertz_nuc = np.sqrt(2.0) * Rn ** 2 / (3.0 * (1.0 - g.nu_nucleus ** 2))
    return {
        # F = At * T * e, T in N/m. Reported as an apparent tension and, via
        # the coat thickness, an apparent modulus.
        "alignment": {
            "A": 2.0 * np.pi * R0 ** 2 / h0,
            "law": "in-plane tension (linear), 2πR₀²/h₀",
            "radius": "R₀", "correction": 1.0,
        },
        "membrane": {
            "A": 2.0 * np.pi * g.membrane_thickness * R0 / (1.0 - g.nu_membrane),
            "law": "shell stretch (ε³), 2π·hₘ·R₀/(1−νₘ)",
            "radius": "R₀", "correction": 1.0,
        },
        "cytoskeleton": {
            "A": hertz_cell * c_cell,
            "law": "Hertz (ε^1.5), √2·R₀²/(3(1−νᵢ²))·c_probe",
            "radius": "R₀", "correction": c_cell,
        },
        "nuclear_envelope": {
            "A": 2.0 * np.pi * g.envelope_thickness * Rn / (1.0 - g.nu_nucleus),
            "law": "shell stretch (ε³), 2π·h_ne·Rₙ/(1−νₙ)",
            "radius": "Rₙ", "correction": 1.0,
        },
        "perinuclear_cytoskeleton": {
            "A": hertz_cell * c_cell,
            "law": "Hertz (ε^1.5), √2·R₀²/(3(1−νᵢ²))·c_probe",
            "radius": "R₀", "correction": c_cell,
        },
        "nuclear_core": {
            "A": hertz_nuc * c_nuc,
            "law": "Hertz (ε^1.5), √2·Rₙ²/(3(1−νₙ²))·c_probe",
            "radius": "Rₙ", "correction": c_nuc,
        },
    }


ELEMENT_NAMES = {
    "alignment": "Contact / alignment (apparent)",
    "membrane": "Cell shell (membrane + cortex)",
    "cytoskeleton": "Cytoskeleton",
    "nuclear_envelope": "Nuclear envelope",
    "perinuclear_cytoskeleton": "Perinuclear cytoskeleton",
    "nuclear_core": "Nuclear interior (core)",
}

MODULUS_SYMBOLS = {
    "k_align": "E_align", "K_shell": "E_shell", "K_cyto": "E_cyto",
    "K_nucleus": "E_ne", "K_nuc_cyto": "E_nc", "K_core": "E_core",
}


def piecewise_moduli(result, geometry: Geometry, regimes=C2C12_REGIMES):
    """
    Young's moduli (Pa) from the fitted coefficients.

    A coefficient K in N/%^p is first converted to C = K * 100^p, the force
    per unit e^p with e the dimensionless deformation, then divided by the
    element's analytical prefactor A (N/Pa): E = C / A.

    Returns a dict keyed by coefficient name, each with the modulus, its
    standard error, the prefactor used and a note.
    """
    pref = _prefactors(geometry)
    out = {}
    for regime in result.get("regimes") or []:
        spec = next((r for r in regimes if r.key == regime["key"]), None)
        if spec is None:
            continue
        for term in spec.terms:
            p = regime["params"].get(term.name)
            if p is None:
                continue
            element = pref[term.element]
            factor = 100.0 ** term.power
            value = p["value"] * factor / element["A"]
            se = p["se"] * factor / element["A"] if np.isfinite(p["se"]) else float("nan")
            row = {
                "coefficient": term.name,
                "symbol": MODULUS_SYMBOLS.get(term.name, "E"),
                "element": ELEMENT_NAMES.get(term.element, term.element),
                "regime": regime["key"],
                "power": term.power,
                "K": p["value"],
                "K_se": p["se"],
                "C_per_e": p["value"] * factor,
                "prefactor_N_per_Pa": element["A"],
                "law": element["law"],
                "radius_used": element["radius"],
                "probe_correction": element["correction"],
                "E_Pa": float(value),
                "E_se_Pa": float(se),
                "at_bound": bool(p.get("at_bound")),
                "fitted": bool(regime["fitted"]),
            }
            if term.element == "alignment":
                # A slope is a tension first: N/m. It becomes a modulus only
                # by dividing by a thickness, and that is an assumption.
                row["tension_N_per_m"] = float(value)
                row["E_Pa"] = float(value / geometry.coat_thickness)
                row["E_se_Pa"] = float(se / geometry.coat_thickness) \
                    if np.isfinite(se) else float("nan")
                row["note"] = (
                    "contact/alignment artefact; tension / coat thickness, "
                    "not a material property"
                )
            elif term.power == 3.0:
                thickness = (geometry.membrane_thickness
                             if term.element == "membrane"
                             else geometry.envelope_thickness)
                row["areal_N_per_m"] = float(value * thickness)
                row["note"] = "shell: E·h is what the curve measures"
            else:
                row["note"] = ""
            out[term.name] = row
    return out
