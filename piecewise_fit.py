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

By default the membrane acts throughout: the K_shell*(x - 5)^3 measured on
R2 keeps rising, with that same K_shell, through R3 and R4, which fit what
is left on top of it (``carry``; pass ``carry=()`` for the laws exactly as
written above).

Each element also has a range of its own (:func:`component_ranges`): it
starts where its regime starts and stops adding load at its own ``until``,
holding what it reached after. By default that is its regime's end (the
fit's end for the membrane); an earlier ``until`` stops it inside its
regime, a later one carries it on into the next. Written out, the model is

    F(x) = F(e1) + sum_j K_j * [min(x, until_j) - start_j]_+ ^ p_j,  x >= e1

The three inner boundaries (e1, e2, e3 = 5, 40, 60 %) can be placed by
:func:`find_boundaries`, which profiles the likelihood of the same model
over them and compares the result with the specification by BIC.

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
    "component_curve",
    "component_force",
    "component_ranges",
    "lamina_summary",
    "find_boundaries",
    "power_law_profile",
    "boundaries_from_power_law",
    "joint_design",
    "joint_sse",
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
    # Where this element stops adding load, in percent. NaN means the end of
    # its own regime (or the end of the fit for a carried element). Past it
    # the element holds the force it reached, so the curve has no step.
    until: float = float("nan")
    # "power": K (x - start)^power, rising and then holding.
    # "lump":  A sin^2(pi (x - start) / (end - start)) inside its regime and
    #          zero outside: a transient bump that rises and falls back,
    #          zero with zero slope at both ends of the regime.
    shape: str = "power"


LUMP = "lump"


def _basis(shape, power, x, start, until):
    """
    One element's shape at x (percent), per unit coefficient.

    power: [min(x, until) - start]_+ ^ power
    lump:  sin^2(pi (x - start) / (until - start)) on [start, until], else 0
    """
    x = np.asarray(x, dtype=float)
    if shape == LUMP:
        width = max(float(until) - float(start), 1e-9)
        inside = (x >= start) & (x <= until)
        return np.where(inside, np.sin(np.pi * (x - start) / width) ** 2, 0.0)
    return np.clip(np.minimum(x, until) - start, 0.0, None) ** power


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
            # The little lump at about 50 %: the nuclear lamina taking load
            # and giving way again. A bump confined to the regime, so it
            # leaves both anchors exactly where they were.
            Term("A_lamina", 0.0, p0=1.0e-9, element="nuclear_lamina",
                 label="nuclear lamina lump", shape=LUMP),
        ),
        equation="F₃(x) = K_nucleus·(x−40)³ + K_nuc_cyto·(x−40)^1.5 "
                 "+ A_lamina·sin²(π(x−40)/20) + F_40%",
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

    ``settings`` maps a coefficient name to any of ``p0``, ``lower``,
    ``upper`` and ``until`` (where the element stops adding load, percent). Names that are not in the regimes are ignored, so a stored
    setting for a coefficient that no longer exists cannot break a fit.
    """
    settings = settings or {}
    out = []
    for regime in regimes:
        terms = []
        for term in regime.terms:
            wanted = settings.get(term.name) or {}
            changes = {}
            for key in ("p0", "lower", "upper", "until"):
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

    # A coefficient whose bounds meet is not fitted: it is held there, and
    # the rest are fitted around it. Both solvers refuse an empty box.
    fixed = np.isfinite(lower) & (lower == upper)
    if fixed.any():
        params = lower.copy()
        at_bound = fixed.copy()
        if (~fixed).any():
            sub, engine, sub_bound = _bounded_linear_fit(
                A[:, ~fixed], y - A[:, fixed] @ lower[fixed],
                np.asarray(p0, dtype=float)[~fixed], lower[~fixed], upper[~fixed],
            )
            params[~fixed] = sub
            at_bound[~fixed] = sub_bound
        else:
            engine = "held at its bounds"
        return params, engine, at_bound

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

def _resolve_ranges(regimes, bounds, carry=()):
    """
    {name: (start, until)} in percent, for every coefficient.

    An element starts where its own regime starts: that is the point its
    law is measured from, dx = x - start. It stops adding load at
    ``until``: by default the end of its regime, or the end of the fit for
    an element in ``carry``. An ``until`` inside the regime stops it early;
    one past the regime carries it, with the value it was fitted at, into
    the regimes after. Past ``until`` it holds the force it reached.
    """
    end = float(bounds[-1])
    carry = tuple(carry or ())
    out = {}
    for i, regime in enumerate(regimes):
        a, b = float(bounds[i]), float(bounds[i + 1])
        for term in regime.terms:
            if regime.free_offset or term.shape == LUMP:
                # A lump lives and dies inside its own regime.
                out[term.name] = (a, b)
                continue
            u = term.until
            if u is None or not np.isfinite(u) or u <= a:
                u = end if term.name in carry else b
            out[term.name] = (a, float(min(max(u, a + 1e-6), end)))
    return out


def component_ranges(boundaries_pct=C2C12_BOUNDARIES_PCT, regimes=C2C12_REGIMES,
                     settings=None, carry=("K_shell",)):
    """
    Where each element acts, {name: (start_pct, until_pct)}.

    The one place this is decided. The fit, the boundary search, the
    curves drawn for each element and the range controls on the page all
    read it, so they cannot disagree.
    """
    return _resolve_ranges(with_settings(regimes, settings),
                           [float(b) for b in boundaries_pct], carry)


def fit_piecewise(
    epsilon,
    force_N,
    boundaries_pct=C2C12_BOUNDARIES_PCT,
    regimes=C2C12_REGIMES,
    settings=None,
    min_points=None,
    carry=("K_shell",),
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
    carry : tuple of coefficient names
        Elements that keep acting after the regime they were fitted in. The
        default carries the cell shell (K_shell): once it has been measured
        on regime 2 it goes on stiffening, K_shell*(x - e1)^3 with that same
        K_shell, through regimes 3 and 4 as a known term, and each later
        regime fits only what is left on top of it. Continuity is kept,
        because the carried term is written relative to the regime's start:

            F3(x) = F_40 + K_shell*[(x - e1)^3 - (e2 - e1)^3]
                         + K_nucleus*(x - e2)^3 + K_nuc_cyto*(x - e2)^1.5

        Pass ``()`` for the specification exactly as first written, where
        the shell's force is held at the value it reached at e2.

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
    carry = tuple(carry or ())
    ranges = _resolve_ranges(regimes, bounds, carry)
    carried = []   # [{"name", "power", "onset_pct", "until_pct", "value"}]

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
            "carried": {} if regime.free_offset else {
                c["name"]: dict(c) for c in carried
            },
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
            term0 = regime.terms[0]
            held = bool(np.isfinite(term0.lower) and term0.lower == term0.upper)
            if held:
                # The contact slope switched off (or pinned): held at its
                # bound, and only the offset C0 is fitted, F = k x + C0.
                slope = float(term0.lower)
                offset = float(np.mean(f - slope * x))
                predicted = slope * x + offset
                resid = f - predicted
                se_c0 = (float(np.std(resid, ddof=1) / np.sqrt(resid.size))
                         if resid.size > 1 else float("nan"))
                se_k = 0.0
            else:
                design = np.column_stack([x ** p for p in powers] + [np.ones_like(x)])
                sol, *_ = np.linalg.lstsq(design, f, rcond=None)
                predicted = design @ sol
                cov = _covariance(design, f - predicted, np.ones(design.shape[1], bool))
                slope, offset = float(sol[0]), float(sol[1])
                se_k = float(np.sqrt(max(cov[0, 0], 0.0)))
                se_c0 = float(np.sqrt(max(cov[1, 1], 0.0)))
            entry["params"][names[0]] = {
                "value": slope, "se": se_k,
                "p0": float(k0), "lower": term0.lower if held else -np.inf,
                "upper": term0.upper if held else np.inf,
                "power": 1.0, "at_bound": held,
                "start": float(a), "until": float(b),
            }
            entry["params"]["C0"] = {
                "value": offset, "se": se_c0,
                "p0": c0, "lower": -np.inf, "upper": np.inf, "power": 0.0,
                "at_bound": False,
            }
            coefficients[names[0]] = slope
            coefficients["C0"] = offset
            entry["engine"] = "linear least squares"
            anchor = slope * b + offset
        else:
            # Each element's law runs from the regime's start to its own
            # end, and holds what it reached after that.
            untils = np.array([ranges[t.name][1] for t in regime.terms])
            design = np.column_stack([
                _basis(t.shape, t.power, x, a, untils[j])
                for j, t in enumerate(regime.terms)
            ])
            # What the carried elements add on top of the anchor here. Zero
            # at the regime's start, so the anchor is still the force there.
            known = _carried_force(carried, x, a)
            target = f - anchor - known
            p0 = [t.p0 for t in regime.terms]
            lower = [t.lower for t in regime.terms]
            upper = [t.upper for t in regime.terms]
            params, engine, at_bound = _bounded_linear_fit(
                design, target, p0, lower, upper,
            )
            predicted = design @ params + anchor + known
            cov = _covariance(design, f - predicted, ~at_bound)
            for j, t in enumerate(regime.terms):
                entry["params"][t.name] = {
                    "value": float(params[j]),
                    "se": float(np.sqrt(cov[j, j])) if np.isfinite(cov[j, j])
                    and cov[j, j] >= 0 else float("nan"),
                    "p0": float(t.p0), "lower": float(t.lower),
                    "upper": float(t.upper), "power": float(t.power),
                    "at_bound": bool(at_bound[j]),
                    "start": float(a), "until": float(untils[j]),
                    "shape": t.shape,
                }
                coefficients[t.name] = float(params[j])
                if t.shape == LUMP and not (t.lower == t.upper):
                    # A lump at zero is an answer ("no lamina lump on this
                    # curve"), not a coefficient the data pushed on.
                    continue
                if at_bound[j] and t.lower == t.upper:
                    warnings.append(
                        f"{t.name} in {regime.key} is held at {t.lower:.3g} "
                        f"by its bounds, so it was not fitted."
                    )
                elif at_bound[j]:
                    near_lower = (not np.isfinite(t.upper)) or (
                        np.isfinite(t.lower)
                        and abs(params[j] - t.lower) <= abs(params[j] - t.upper))
                    where = "its lower bound" if near_lower else "its upper bound"
                    warnings.append(
                        f"{t.name} in {regime.key} settled on {where} "
                        f"({params[j]:.3g}). The data wanted it further out; "
                        f"the {t.label or t.name} is not resolved on this curve."
                    )
            entry["engine"] = engine
            anchor = anchor + float(
                sum(params[j] * float(_basis(t.shape, t.power, np.array([b]),
                                             a, untils[j])[0])
                    for j, t in enumerate(regime.terms) if t.shape != LUMP)
            ) + float(_carried_force(carried, np.array([b]), a)[0])
            for j, t in enumerate(regime.terms):
                if t.shape != LUMP and untils[j] > b + 1e-9:
                    carried.append({
                        "name": t.name, "power": float(t.power),
                        "onset_pct": float(a), "until_pct": float(untils[j]),
                        "value": float(params[j]),
                        "se": entry["params"][t.name]["se"],
                        "from_regime": regime.key,
                    })

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
        "carry": carry,
        "ranges": {k: (float(v[0]), float(v[1])) for k, v in ranges.items()},
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


def _carried_force(carried, x, a):
    """
    Force the carried elements add inside a regime that starts at ``a``.

    Each carried element keeps its own law from its own onset,
    K*(x - onset)^p, and is written relative to the regime's start so it is
    zero there and the regime's anchor stays the force at its start.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape)
    for c in carried or ():
        value = float(c["value"])
        if not np.isfinite(value) or value == 0.0:
            continue
        onset, power = float(c["onset_pct"]), float(c["power"])
        until = float(c.get("until_pct", np.inf))
        out = out + value * (
            np.clip(np.minimum(x, until) - onset, 0.0, None) ** power
            - max(min(a, until) - onset, 0.0) ** power
        )
    return out


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
    out = np.full(x.shape, float(regime["anchor_in_N"]))
    for p in params.values():
        until = float(p.get("until", np.inf))
        out = out + p["value"] * _basis(p.get("shape", "power"), p["power"],
                                        x, a, until)
    return out + _carried_force(list((regime.get("carried") or {}).values()), x, a)


def component_curve(result, name, n=300):
    """
    One element's own force over the stretch it acts on: (x_pct, F_N).

    From its start to its own ``until``, the same range the fit gave it, so
    the line on the plot and the range on the page are one number. None
    when the element was not fitted. The contact term is its straight line
    including C0; every other element is measured from its own start.
    """
    regimes = result.get("regimes") or []
    home = next((r for r in regimes if name in r["params"]), None)
    if home is None or not home["fitted"]:
        return None
    p = home["params"][name]
    if not np.isfinite(p["value"]):
        return None
    start = float(p.get("start", home["domain_pct"][0]))
    until = float(p.get("until", home["domain_pct"][1]))
    x = np.linspace(start, until, n)
    if home["anchor_in_N"] is None:
        return x, p["value"] * x + home["params"]["C0"]["value"]
    return x, p["value"] * _basis(p.get("shape", "power"), p["power"], x,
                                  start, until)


def component_force(result, name, x_pct):
    """
    One element's force at any x (percent), in newtons, for stacking.

    Zero before the element starts, K times its shape up to its own
    ``until`` and the force it reached after that (the lump returns to
    zero), NaN outside the fitted domain. The contact term is
    C0 + k_align min(x, until). Summed over every element this is the
    fitted curve exactly, so the layers of a stacked plot end on it.
    None when the element was not fitted.
    """
    regimes = result.get("regimes") or []
    home = next((r for r in regimes if name in r["params"]), None)
    if home is None or not home["fitted"]:
        return None
    p = home["params"][name]
    if not np.isfinite(p["value"]):
        return None
    x = np.asarray(x_pct, dtype=float)
    start = float(p.get("start", home["domain_pct"][0]))
    until = float(p.get("until", home["domain_pct"][1]))
    if home["anchor_in_N"] is None:
        out = p["value"] * np.minimum(x, until) + home["params"]["C0"]["value"]
    else:
        out = p["value"] * _basis(p.get("shape", "power"), p["power"], x,
                                  start, until)
    lo = float(regimes[0]["domain_pct"][0])
    hi = float(regimes[-1]["domain_pct"][1])
    return np.where((x >= lo) & (x <= hi), out, np.nan)


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



# ======================================================= finding boundaries ==
#
# The boundaries are the three numbers the fit cannot find by itself: the
# model is linear in its coefficients only once e1, e2 and e3 are fixed. So
# they are found the way any nonlinear parameter of an otherwise linear
# model is: by profiling the likelihood over them.
#
# The model family is the one on the page, the same four laws, the same
# carried membrane, the same bounds, and continuous at every boundary. Its
# force is LINEAR in all seven coefficients at once, because every anchor is
# itself a linear combination of the coefficients upstream of it:
#
#     F_hat(x) = X(x; e1, e2, e3) . theta,
#     theta = (k_align, C0, K_shell, K_cyto, K_nucleus, K_nuc_cyto, K_core)
#
# so for each trial placement the best continuous curve is one bounded
# linear least-squares problem, solved exactly:
#
#     S(e1, e2, e3) = min_theta || F - X theta ||^2,   lower <= theta <= upper
#     (e1, e2, e3)^ = argmin S    within bands around the specification
#
# Why jointly here when the fit itself is sequential: the sequential chain
# is the specification's estimator, but its anchors inherit the noise of
# the regimes upstream, so its S can be lowered by moving a boundary to
# re-aim an anchor even when the boundaries are right. On curves generated
# with the boundaries exactly at 5 / 40 / 60 % that alone reported strong
# evidence for moving them. The joint S has no such lever: it is the
# profile likelihood of the boundaries, and on those same curves it keeps
# them (Delta BIC around -20). The coefficients shown are still the
# sequential ones, fitted at the boundaries found here.
#
# With Gaussian noise, -2 ln L = n ln(S/n) + const, which gives a
# profile-likelihood interval for each boundary (n ln(S/S_min) <= 1 for
# 68 %, <= 3.84 for 95 %) and a Bayesian information criterion comparing
# the found boundaries with the specification's, paying 3 ln n for the
# three numbers the search was allowed to choose.

def _coefficient_names(regimes):
    names, lower, upper = [], [], []
    for regime in regimes:
        for term in regime.terms:
            names.append(term.name)
            lower.append(term.lower)
            upper.append(term.upper)
        if regime.free_offset:
            names.append("C0")
            lower.append(-np.inf)
            upper.append(np.inf)
    return names, np.array(lower, dtype=float), np.array(upper, dtype=float)


def joint_design(x, bounds, regimes=C2C12_REGIMES, carry=("K_shell",)):
    """
    The design matrix of the continuous four-regime model: F_hat = X theta.

    Row i holds what each coefficient contributes at x_i, including through
    the anchors it feeds downstream and the carried elements. Returns
    (X, covered, names, lower, upper); ``covered`` marks the points inside
    [bounds[0], bounds[-1]].
    """
    x = np.asarray(x, dtype=float)
    names, lower, upper = _coefficient_names(regimes)
    col = {name: i for i, name in enumerate(names)}
    X = np.zeros((x.size, len(names)))
    covered = np.zeros(x.size, dtype=bool)
    anchor = np.zeros(len(names))
    carried = []
    carry = tuple(carry or ())
    ranges = _resolve_ranges(regimes, bounds, carry)
    for i, regime in enumerate(regimes):
        a, b = bounds[i], bounds[i + 1]
        last = i == len(regimes) - 1
        mask = (x >= a) & ((x <= b) if last else (x < b))
        covered |= mask
        xm = x[mask]
        if regime.free_offset:
            term = regime.terms[0]
            X[mask, col[term.name]] = xm ** term.power
            X[mask, col["C0"]] = 1.0
            anchor = np.zeros(len(names))
            anchor[col[term.name]] = b ** term.power
            anchor[col["C0"]] = 1.0
            continue
        rows = np.tile(anchor, (xm.size, 1))
        after = anchor.copy()
        for term in regime.terms:
            u = ranges[term.name][1]
            rows[:, col[term.name]] += _basis(term.shape, term.power, xm, a, u)
            if term.shape != LUMP:
                after[col[term.name]] += (min(b, u) - a) ** term.power
        for name, power, onset, u in carried:
            base = max(min(a, u) - onset, 0.0) ** power
            rows[:, col[name]] += (np.clip(np.minimum(xm, u) - onset, 0.0, None)
                                   ** power - base)
            after[col[name]] += max(min(b, u) - onset, 0.0) ** power - base
        for term in regime.terms:
            u = ranges[term.name][1]
            if term.shape != LUMP and u > b + 1e-9:
                carried.append((term.name, float(term.power), float(a), float(u)))
        X[mask] = rows
        anchor = after
    return X, covered, names, lower, upper


def _bounded_lsq_exact(X, y, lower, upper):
    """
    min ||X theta - y||^2 with box bounds, exactly, and fast when it can be.

    Unbounded coefficients (k_align, C0) are projected out; coefficients
    bounded only below are shifted to a zero lower bound, and the rest is a
    non-negative least-squares problem (Lawson-Hanson, exact). Finite upper
    bounds fall back to ``lsq_linear``.
    """
    from scipy.optimize import nnls

    fixed = np.isfinite(lower) & (lower == upper)
    if fixed.any():
        theta = lower.copy()
        rest = y - X[:, fixed] @ lower[fixed]
        if (~fixed).any():
            sub, cost = _bounded_lsq_exact(X[:, ~fixed], rest,
                                           lower[~fixed], upper[~fixed])
            theta[~fixed] = sub
            return theta, cost
        return theta, float(rest @ rest)

    col = np.linalg.norm(X, axis=0)
    col[~np.isfinite(col) | (col == 0)] = 1.0
    Xs = X / col
    lo, hi = lower * col, upper * col
    free = ~np.isfinite(lo) & ~np.isfinite(hi)
    if np.any(np.isfinite(hi[~free])) or np.any(~np.isfinite(lo[~free])):
        sol = lsq_linear(Xs, y, bounds=(lo, hi), method="trf", tol=1e-12)
        r = Xs @ sol.x - y
        return sol.x / col, float(r @ r)
    U, B = Xs[:, free], Xs[:, ~free]
    shift = lo[~free]
    target = y - B @ shift
    if U.shape[1]:
        Q, _ = np.linalg.qr(U)
        PB = B - Q @ (Q.T @ B)
        Py = target - Q @ (Q.T @ target)
    else:
        PB, Py = B, target
    q, _res = nnls(PB, Py, maxiter=50 * max(B.shape[1], 1))
    theta = np.empty(X.shape[1])
    theta[~free] = q + shift
    if U.shape[1]:
        theta[free] = np.linalg.lstsq(U, target - B @ q, rcond=None)[0]
    r = Xs @ theta - y
    return theta / col, float(r @ r)


def _prepare(epsilon, force_N):
    eps = np.asarray(epsilon, dtype=float).ravel()
    force = np.asarray(force_N, dtype=float).ravel()
    good = np.isfinite(eps) & np.isfinite(force)
    order = np.argsort(eps[good], kind="stable")
    return eps[good][order] * 100.0, force[good][order]


def joint_sse(epsilon, force_N, boundaries_pct=C2C12_BOUNDARIES_PCT,
              regimes=C2C12_REGIMES, settings=None, carry=("K_shell",)):
    """
    S at one placement: the residual sum of squares, in N^2, of the best
    continuous four-regime curve with those boundaries.
    """
    x_all, f_all = _prepare(epsilon, force_N)
    regimes = with_settings(regimes, settings)
    X, covered, _n, lower, upper = joint_design(
        x_all, [float(b) for b in boundaries_pct], regimes, carry)
    return _bounded_lsq_exact(X[covered], f_all[covered], lower, upper)[1]


def _grid(lo, hi, step):
    lo, hi = float(lo), float(hi)
    if hi <= lo or step <= 0:
        return np.array([lo])
    n = int(np.floor((hi - lo) / step + 1e-9))
    return np.round(lo + step * np.arange(n + 1), 6)


def _interval(grid, delta, best, level):
    """The contiguous stretch around ``best`` where delta <= level."""
    grid = np.asarray(grid)
    delta = np.asarray(delta)
    i0 = int(np.argmin(np.abs(grid - best)))
    lo_i = hi_i = i0
    while lo_i > 0 and delta[lo_i - 1] <= level:
        lo_i -= 1
    while hi_i < grid.size - 1 and delta[hi_i + 1] <= level:
        hi_i += 1
    return (float(grid[lo_i]), float(grid[hi_i]),
            lo_i == 0, hi_i == grid.size - 1)


SEARCH_BANDS_PCT = ((2.0, 10.0), (30.0, 50.0), (50.0, 70.0))


def find_boundaries(
    epsilon,
    force_N,
    end_pct=C2C12_BOUNDARIES_PCT[-1],
    bands_pct=SEARCH_BANDS_PCT,
    spec_pct=C2C12_BOUNDARIES_PCT[1:-1],
    regimes=C2C12_REGIMES,
    settings=None,
    carry=("K_shell",),
    coarse_step=(1.0, 2.0, 2.0),
    fine_step=0.1,
    profile_step=0.25,
    min_width=2.0,
    span_pct=None,
):
    """
    The e1, e2, e3 that maximise the likelihood of the four-regime model.

    Every trial uses the page's equations, bounds and carried elements. The
    end of the fit is not searched: it decides which points are fitted, and
    S only compares placements that fit the same points.

    1. A coarse grid over the three bands (1 %, 2 %, 2 % by default).
    2. Coordinate descent at ``fine_step`` from the three best grid points,
       each boundary scanned in turn with the others held, until none moves.
    3. A profile of each boundary across its band with the others at the
       optimum: Delta(-2 ln L) = n ln(S / S_min), giving 68 % (<= 1) and
       95 % (<= 3.84) intervals.
    4. The reference placement ``spec_pct`` scored the same way:
       Delta BIC = n ln(S_ref / S_min) - 3 ln n.

    ``span_pct`` = (low, high) additionally holds e3 - e2 inside that
    range: for a C2C12, how long the nuclear bump lasts once it is met.
    """
    x_all, f_all = _prepare(epsilon, force_N)
    regimes = with_settings(regimes, settings)
    carry = tuple(carry or ())
    end = float(min(end_pct, x_all.max())) if x_all.size else float(end_pct)
    keep = (x_all >= 0.0) & (x_all <= end)
    x_fit, f_fit = x_all[keep], f_all[keep]
    n = int(x_fit.size)
    if n < 12:
        return {"success": False, "error": "too few points to place boundaries"}
    bands = [tuple(sorted(float(v) for v in band)) for band in bands_pct]
    width = float(min_width)
    needs = [len(r.terms) + (1 if r.free_offset else 0) + 2 for r in regimes]

    span = tuple(float(v) for v in span_pct) if span_pct else None

    def valid(b):
        edges = (0.0,) + tuple(b) + (end,)
        if any(e1 - e0 < width for e0, e1 in zip(edges, edges[1:])):
            return False
        if span and not (span[0] - 1e-9 <= b[2] - b[1] <= span[1] + 1e-9):
            return False
        for i, need in enumerate(needs):
            last = i == len(needs) - 1
            inside = (x_fit >= edges[i]) & (
                (x_fit <= edges[i + 1]) if last else (x_fit < edges[i + 1]))
            if int(inside.sum()) < need:
                return False
        return True

    cache = {}

    def S(b):
        key = tuple(round(float(v), 6) for v in b)
        if key not in cache:
            if valid(key):
                X, covered, _n, lower, upper = joint_design(
                    x_fit, (0.0,) + key + (end,), regimes, carry)
                cache[key] = _bounded_lsq_exact(
                    X[covered], f_fit[covered], lower, upper)[1]
            else:
                cache[key] = float("inf")
        return cache[key]

    # 1 · coarse grid
    grids = [_grid(lo, hi, step) for (lo, hi), step in zip(bands, coarse_step)]
    scored = []
    for b1 in grids[0]:
        for b2 in grids[1]:
            for b3 in grids[2]:
                value = S((b1, b2, b3))
                if np.isfinite(value):
                    scored.append((value, (b1, b2, b3)))
    if not scored:
        return {"success": False,
                "error": "no placement inside the bands leaves every regime "
                         "enough points; widen the bands or check the curve"}
    scored.sort(key=lambda item: item[0])

    # 2 · coordinate descent from the best few
    def descend(start):
        b = [float(v) for v in start]
        for _cycle in range(8):
            moved = False
            for i in range(3):
                reach = 2.0 * coarse_step[i]
                trial = _grid(max(bands[i][0], b[i] - reach),
                              min(bands[i][1], b[i] + reach), fine_step)
                values = [S(tuple(b[:i] + [v] + b[i + 1:])) for v in trial]
                j = int(np.argmin(values))
                if values[j] < S(tuple(b)) and trial[j] != b[i]:
                    b[i] = float(trial[j])
                    moved = True
            if not moved:
                break
        return tuple(b)

    best = min((descend(start) for _v, start in scored[:3]), key=S)

    # 3 · profiles, which may also find a lower S the descent missed
    def profiles(best):
        out = {}
        for i in range(3):
            grid = _grid(bands[i][0], bands[i][1], profile_step)
            values = np.array([S(tuple(best[:i]) + (v,) + tuple(best[i + 1:]))
                               for v in grid])
            out[i] = (grid, values)
        return out

    prof = profiles(best)
    for _round in range(2):
        lower_found = False
        for i, (grid, values) in prof.items():
            j = int(np.argmin(values))
            if values[j] < S(best) * (1 - 1e-12):
                best = descend(tuple(best[:i]) + (float(grid[j]),) + tuple(best[i + 1:]))
                lower_found = True
                break
        if not lower_found:
            break
        prof = profiles(best)

    s_min = S(best)
    names = ("eps1", "eps2", "eps3")
    intervals, curves = {}, {}
    for i in range(3):
        grid, values = prof[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = n * np.log(values / s_min)
        delta = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.inf)
        lo68, hi68, open_lo, open_hi = _interval(grid, delta, best[i], 1.0)
        lo95, hi95, _a, _b = _interval(grid, delta, best[i], 3.84)
        intervals[names[i]] = {
            "best": float(best[i]), "lo68": lo68, "hi68": hi68,
            "lo95": lo95, "hi95": hi95,
            "at_band_edge": bool(np.isclose(best[i], bands[i][0])
                                 or np.isclose(best[i], bands[i][1])),
            "open_low": bool(open_lo), "open_high": bool(open_hi),
        }
        curves[names[i]] = ([float(v) for v in grid],
                            [float(v) if np.isfinite(v) else None for v in delta])

    spec = tuple(float(v) for v in spec_pct)
    s_spec = S(spec)
    if np.isfinite(s_spec) and s_spec > 0:
        delta_m2lnl = float(n * np.log(s_spec / s_min))
        delta_bic = float(delta_m2lnl - 3.0 * np.log(n))
    else:
        delta_m2lnl = delta_bic = float("nan")

    if not np.isfinite(delta_bic):
        verdict = ("The reference boundaries cannot be fitted on this curve, "
                   "so the found ones are the only placement.")
        strength = "only"
    elif delta_bic > 6:
        verdict = ("Strong evidence: the curve itself places the boundaries "
                   "here, by more than the cost of choosing three numbers.")
        strength = "strong"
    elif delta_bic > 2:
        verdict = "Positive evidence that the curve places the boundaries here."
        strength = "positive"
    elif delta_bic > 0:
        verdict = ("Weak evidence: this placement fits a little better than "
                   "the reference, hardly more than choosing it costs.")
        strength = "weak"
    else:
        verdict = ("The curve does not pin the boundaries down more tightly "
                   "than the constraints do: the reference placement fits "
                   "about as well. Read the intervals as how far each one "
                   "could move.")
        strength = "none"

    return {
        "success": True,
        "best_pct": tuple(float(v) for v in best),
        "end_pct": end,
        "sse_best": float(s_min),
        "sse_spec": float(s_spec),
        "spec_pct": spec,
        "n_points": n,
        "delta_m2lnL": delta_m2lnl,
        "delta_bic": delta_bic,
        "verdict": verdict,
        "strength": strength,
        "intervals": intervals,
        "profiles": curves,
        "bands_pct": tuple(bands),
        "span_pct": span,
        "carry": carry,
        "n_evaluations": len(cache),
    }

# ========================================================== the power law ==
#
# The other way to place the boundaries, and the one that does not use the
# model at all: read the curve as a physicist does, on log-log axes. Each
# element is a power law, so the local exponent
#
#     p(x) = d ln F / d ln x
#
# follows whichever laws are carrying load, and a boundary is where it
# changes course. Measured straight off the data, it is evidence the fit
# has to agree with rather than a restatement of it.

def power_law_profile(epsilon, force_N, end_pct=C2C12_BOUNDARIES_PCT[-1],
                      start_pct=0.5, n_log=300, window=21, step_pct=0.25):
    """
    The local exponent p = d ln F / d ln x along the curve.

    ln F is binned on a grid uniform in ln x (medians, so a few bad points
    cannot drag it), smoothed and differentiated with a Savitzky-Golay
    filter, then put back on a grid uniform in x, ``step_pct`` apart, which
    is where boundaries are placed. Points at or below zero force carry no
    logarithm and are left out.

    Returns {"x_pct", "exponent", "log_x", "log_F"} or {} when too little of
    the curve is usable.
    """
    from scipy.signal import savgol_filter

    x, f = _prepare(epsilon, force_N)
    end = float(min(end_pct, x.max())) if x.size else float(end_pct)
    keep = (x >= start_pct) & (x <= end) & (f > 0)
    x, f = x[keep], f[keep]
    if x.size < 20:
        return {}
    lx, lf = np.log(x), np.log(f)
    edges = np.linspace(lx.min(), lx.max(), int(n_log) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(lx, edges) - 1, 0, n_log - 1)
    med = np.full(n_log, np.nan)
    for i in np.unique(idx):
        med[i] = np.median(lf[idx == i])
    ok = np.isfinite(med)
    if ok.sum() < 12:
        return {}
    med = np.interp(centres, centres[ok], med[ok])
    w = int(min(window, (len(med) // 2) * 2 - 1))
    w = max(w, 5)
    slope = savgol_filter(med, w, 2, deriv=1, delta=float(centres[1] - centres[0]))
    grid = np.arange(np.ceil(x.min() / step_pct) * step_pct, end + 1e-9, step_pct)
    p = np.interp(np.log(grid), centres, slope)
    return {"x_pct": grid, "exponent": p, "log_x": centres, "log_F": med}


def boundaries_from_power_law(profile, bands_pct=SEARCH_BANDS_PCT,
                              span_pct=None):
    """
    ε₁, ε₂, ε₃ where the curve's log-log slope bends most sharply.

    A new element starting at a boundary adds K (x - start)^p to the force.
    For the 1.5-power elements that onset is a square-root kink in the
    slope of F, so on log-log axes the exponent p(x) turns upward hardest
    right there. Each boundary is placed at the strongest such turn, the
    maximum of dp / d ln x, inside its C2C12 constraint: ε₂ in its band,
    then ε₃ inside its band and ``span_pct`` after ε₂, and ε₁ in the
    contact band.

    This reads the curve without any model, so it is evidence rather than a
    fit, and it is only as sharp as the smoothing lets it be: on test
    curves it lands a few percent from the true boundary. Fitting
    everything near it (``find_boundaries``) finishes the job.

    Returns {"best_pct", "segments", "profile", "turn"} or
    {"success": False, "error"}.
    """
    if not profile:
        return {"success": False,
                "error": "the log-log slope cannot be measured on this curve"}
    x = np.asarray(profile["x_pct"], dtype=float)
    p = np.asarray(profile["exponent"], dtype=float)
    if x.size < 12:
        return {"success": False, "error": "too little of the slope to read"}
    turn = np.gradient(p, np.log(x))

    def strongest(lo, hi):
        inside = (x >= lo - 1e-9) & (x <= hi + 1e-9)
        if not inside.any():
            return None
        return float(x[np.argmax(np.where(inside, turn, -np.inf))])

    bands = [tuple(float(v) for v in b) for b in bands_pct]
    b2 = strongest(*bands[1])
    b1 = strongest(*bands[0])
    if b2 is None or b1 is None:
        return {"success": False,
                "error": "the constraint bands are outside the measured slope"}
    lo3, hi3 = bands[2]
    if span_pct:
        lo3 = max(lo3, b2 + float(span_pct[0]))
        hi3 = min(hi3, b2 + float(span_pct[1]))
    b3 = strongest(lo3, hi3) if hi3 >= lo3 else None
    if b3 is None:
        return {"success": False,
                "error": "no room for ε₃ after ε₂ inside the constraints"}
    cuts = [x[0], b1, b2, b3, x[-1]]
    segments = []
    for a, z in zip(cuts, cuts[1:]):
        inside = (x >= a) & (x <= z)
        segments.append({"from_pct": float(a), "to_pct": float(z),
                         "mean_exponent": float(p[inside].mean())
                         if inside.any() else float("nan")})
    return {
        "success": True,
        "best_pct": (b1, b2, b3),
        "segments": segments,
        "profile": {"x_pct": x.tolist(), "exponent": p.tolist()},
        "turn": turn.tolist(),
    }

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


def lamina_summary(result, geometry: Geometry | None = None, name="A_lamina"):
    """
    The nuclear lamina lump, in numbers.

    L(x) = A_L sin^2(pi (x - e2) / (e3 - e2)),  e2 <= x <= e3

    Its height A_L (N), where it peaks, x_L = (e2 + e3) / 2, how wide it is,
    w = e3 - e2, and the work it takes up,

        W_L = integral L d(delta) = A_L (w / 2) (h0 / 100)   [J],

    since delta = x h0 / 100. A_L at zero means no lump on this curve.
    """
    for regime in result.get("regimes") or []:
        p = regime["params"].get(name)
        if p is None:
            continue
        a, b = float(p.get("start", regime["domain_pct"][0])), float(
            p.get("until", regime["domain_pct"][1]))
        amp = float(p["value"])
        out = {
            "A_N": amp, "A_se_N": float(p["se"]),
            "peak_pct": 0.5 * (a + b), "width_pct": b - a,
            "from_pct": a, "to_pct": b,
            "fitted": bool(regime["fitted"]) and np.isfinite(amp),
            "present": bool(np.isfinite(amp) and amp > 0
                            and not p.get("at_bound")),
        }
        if geometry is not None and np.isfinite(amp):
            out["work_J"] = amp * 0.5 * (b - a) * geometry.cell_height / 100.0
        return out
    return None


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
            if term.shape == LUMP or term.element not in pref:
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
