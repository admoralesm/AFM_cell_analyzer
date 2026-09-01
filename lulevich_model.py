"""
Lulevich et al. 2006 cell-compression model.

Reference
---------
Lulevich, V., Zink, T., Chen, H.-Y., Liu, F.-T., Liu, G.-Y. (2006).
"Cell Mechanics Using Atomic Force Microscopy-Based Single-Cell Compression."
Langmuir 22(19), 8151-8155.

Physical model
--------------
Total force during whole-cell compression is a sum of a membrane (balloon)
term, an interior/cytoskeleton (Hertzian) term, and optionally a nucleus
term that only engages once the cell has been squashed onto it:

    F(e) = Am * Em * e^3  +  Ai * Ei * e^(3/2)  +  An * En * <e - e0>^(3/2)

with the geometry prefactors

    Am = 2 * pi * h_m * R0 / (1 - nu_m)
    Ai = sqrt(2) * R0^(1/2) * h0^(3/2) / (3 * (1 - nu_i^2))
    An = sqrt(2) * Rn^(1/2) * h0^(3/2) / (3 * (1 - nu_n^2))

<x> is x for x > 0 and zero otherwise, so the nucleus contributes nothing
below the onset deformation e0. That offset is the only thing separating the
nucleus term from the cytoskeleton term, which carries the same 3/2 exponent:
without it the two are identical in shape and their moduli trade off freely.

What the membrane term actually measures is the product Em * h_m. Em is that
divided by whatever bilayer thickness is assumed, so it scales inversely with
that assumption and the areal modulus Em * h_m is reported alongside it.

where
    e    = relative deformation, delta / h0  (dimensionless)
    h0   = initial cell height [m]
    R0   = cell radius [m]
    h_m  = membrane thickness [m]
    delta= absolute indentation [m] = e * h0

Two things this module is deliberate about, because both were wrong in
earlier versions and both silently destroy the result:

1. ALL LENGTHS ARE IN METRES. A cell height of 8.09 um must be passed as
   8.09e-6, not 8.09. Passing micrometres inflates R0 by 1e6 and drives the
   fitted moduli into their bounds. The constructor raises if the value
   looks like micrometres.

2. The Hertzian term needs an absolute indentation, delta = e * h0. Feeding
   it the dimensionless e directly leaves the term with units of
   Pa*m^(1/2) instead of newtons, which is off by h0^(3/2) (~1e-8 for an
   8 um cell).

Fitting
-------
F is LINEAR in Em, Ei and En. There is no need for a non-linear optimiser, an
initial guess, or a convergence check: the fit is a bounded linear least
squares problem with a closed-form normal-equation solution. This makes the
result deterministic, guess-independent, and impossible to "fail to
converge". Uncertainties come from the analytic covariance matrix.

The onset e0 is the single exception, the only non-linear parameter. It is a
bounded scalar, so :meth:`scan_nucleus_onset` sweeps it on a grid where each
trial is one exact linear solve, and reports whether the R2 peak is sharp
enough for the data to have located it at all.

Terms can also be fitted in stages on separate windows (:meth:`fit_staged`),
each stage subtracting the current estimates of the terms it is not solving
for and the whole sequence repeating until it settles.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear
from scipy.ndimage import uniform_filter1d

__all__ = ["LulevichModel"]

# Sanity limits used for warnings only, never to clamp the fit.
PLAUSIBLE_EM_PA = (1e3, 1e9)      # 1 kPa .. 1 GPa
PLAUSIBLE_EI_PA = (1e0, 1e7)      # 1 Pa  .. 10 MPa
K_BOLTZMANN = 1.380649e-23


# ============================================================================
#  Coupling: how the elements share load
# ============================================================================
#
# The model above puts the elements in PARALLEL: every element is squashed by
# the same relative deformation and their forces add. That is the right
# picture for a membrane stretched over a cytoskeleton that is compressed
# with it.
#
# The alternative is SERIES: every element carries the same force and their
# deformations add, a spring stacked on a spring. That is the right picture
# when the load path runs through the elements one after another, for example
# the cantilever compressing the cytoplasm which then bears down on the
# nucleus underneath it.
#
# The two make genuinely different predictions. In parallel the stiffest
# element dominates the force; in series the softest element dominates the
# deformation. Which one the data prefers is an empirical question, which is
# what compare_couplings() is for.
#
# Series is still exactly solvable. Inverting each element's law,
#
#     membrane      F = Am Em d^3     ->  d = (F / (Am Em))^(1/3)
#     cytoskeleton  F = Ai Ec d^1.5   ->  d = (F / (Ai Ec))^(2/3)
#     nucleus       F = An En d^1.5   ->  d = (<F - F0> / (An En))^(2/3)
#
# and adding the deformations gives
#
#     e(F) = a F^(1/3) + b F^(2/3) + c <F - F0>^(2/3)
#
# which is LINEAR in the compliances a, b, c exactly as the parallel model is
# linear in the moduli. The moduli come back as
#
#     Em = 1 / (Am a^3),  Ec = 1 / (Ai b^1.5),  En = 1 / (An c^1.5)
#
# The nucleus needs the force offset F0 for the same reason it needed a
# deformation offset in parallel: without it the cytoskeleton and nucleus
# terms are both F^(2/3) and their compliances are not separable.


class _CouplingMixin:
    """Series and hybrid coupling, mixed into LulevichModel below."""

    # ------------------------------------------------------------- series

    def series_epsilon(self, force, Em, Ec, En=0.0, force_onset=0.0):
        """Relative deformation produced by a given force, elements in series."""
        F = np.clip(np.asarray(force, dtype=float), 0.0, None)
        eps = np.zeros_like(F)
        if Em and Em > 0:
            eps = eps + (F / (self.Am * Em)) ** (1.0 / 3.0)
        if Ec and Ec > 0:
            eps = eps + (F / (self.Ai * Ec)) ** (2.0 / 3.0)
        if En and En > 0:
            excess = np.clip(F - float(force_onset), 0.0, None)
            eps = eps + (excess / (self.An * En)) ** (2.0 / 3.0)
        return eps

    def series_force(self, epsilon, Em, Ec, En=0.0, force_onset=0.0, f_max=None, n_grid=4000):
        """
        Force required for a given deformation, elements in series.

        There is no closed form, but e(F) is monotonically increasing, so the
        inverse is obtained by evaluating it on a dense force grid and
        interpolating. Doing it this way lets a series fit be scored on force
        residuals, the same space the parallel fit uses, so the two are
        directly comparable.
        """
        eps = np.asarray(epsilon, dtype=float)
        top = f_max if f_max else float(np.nanmax(self.force)) * 1.6
        if not np.isfinite(top) or top <= 0:
            return np.zeros_like(eps)
        grid = np.linspace(0.0, top, int(n_grid))
        eps_grid = self.series_epsilon(grid, Em, Ec, En, force_onset)
        # np.interp needs an increasing x; ties at zero force are harmless.
        return np.interp(eps, eps_grid, grid, left=0.0, right=top)

    def _refine_series(self, eps, force, Em, Ec, En, force_onset, terms):
        """Polish the series moduli against force residuals."""
        from scipy.optimize import least_squares

        active = [("membrane" in terms), ("interior" in terms), ("nucleus" in terms)]
        values = [Em, Ec, En]
        # A rigid element (infinite modulus) is not something the optimiser can
        # search over, so it is left as found.
        if any(active[i] and not np.isfinite(values[i]) for i in range(3)):
            return Em, Ec, En
        start = np.log10([max(values[i], 1e-3) if active[i] else 1e-6 for i in range(3)])
        free = np.array(active, dtype=bool)
        scale = max(float(np.nanmax(np.abs(force))), 1e-15)
        f_max = float(np.nanmax(self.force)) * 1.6

        def residual(free_log):
            full = start.copy()
            full[free] = free_log
            params = [10.0 ** full[i] if active[i] else 0.0 for i in range(3)]
            predicted = self.series_force(
                eps, params[0], params[1], params[2], force_onset, f_max=f_max
            )
            return (force - predicted) / scale

        try:
            solution = least_squares(
                residual, start[free],
                bounds=(np.full(free.sum(), -1.0), np.full(free.sum(), 11.0)),
                xtol=1e-10, ftol=1e-10, max_nfev=300,
            )
        except Exception:  # pragma: no cover
            return Em, Ec, En
        full = start.copy()
        full[free] = solution.x
        return tuple(10.0 ** full[i] if active[i] else 0.0 for i in range(3))

    def fit_series(
        self,
        epsilon_min=0.01,
        epsilon_max=0.3,
        terms=("membrane", "interior"),
        force_onset=None,
        weighting="uniform",
        refine=True,
    ):
        """
        Fit the series model: one bounded linear least squares in compliance.

        The linear stage minimises deformation residuals, because that is the
        space the model is linear in. Goodness of fit is then reported on
        force residuals, computed by inverting the model, so R2 and RMSE mean
        the same thing here as they do for the parallel fit.

        ``refine`` then polishes the moduli against those force residuals.
        This matters more than it sounds: Em = 1 / (Am a^3), so a one percent
        error in the fitted compliance becomes three percent in Em, and
        minimising deformation error is not the same as minimising force
        error. On a synthetic series curve the linear stage alone returns
        Em = 3.9 MPa against a true 2.5; refined it returns 2.5.
        """
        eps, force, mask = self._select(epsilon_min, epsilon_max)
        usable = force > 0
        eps, force = eps[usable], force[usable]
        n_params = len(terms)
        if eps.size < n_params + 1:
            return self._failure(
                f"Only {eps.size} points with positive force in e = "
                f"[{epsilon_min:.3f}, {epsilon_max:.3f}]; need at least "
                f"{n_params + 1}."
            )

        if force_onset is None:
            force_onset = 0.5 * float(np.nanmax(force)) if "nucleus" in terms else 0.0

        cols, names = [], []
        if "membrane" in terms:
            cols.append(force ** (1.0 / 3.0))
            names.append("a")
        if "interior" in terms:
            cols.append(force ** (2.0 / 3.0))
            names.append("b")
        if "nucleus" in terms:
            cols.append(np.clip(force - force_onset, 0.0, None) ** (2.0 / 3.0))
            names.append("c")
        X = np.column_stack(cols)

        if weighting == "relative":
            scale = np.maximum(np.abs(eps), np.percentile(np.abs(eps), 10) or 1e-9)
            w = 1.0 / scale
        else:
            w = np.ones_like(eps)
        Xw, yw = X * w[:, None], eps * w

        col_norm = np.linalg.norm(Xw, axis=0)
        col_norm[col_norm == 0] = 1.0
        sol = lsq_linear(Xw / col_norm, yw, bounds=(0.0, np.inf), method="bvls")
        params = dict(zip(names, sol.x / col_norm))

        def modulus(compliance, prefactor, power):
            if compliance is None or compliance <= 0:
                return float("inf")  # zero compliance = rigid element
            return float(1.0 / (prefactor * compliance ** power))

        Em = modulus(params.get("a"), self.Am, 3.0) if "membrane" in terms else 0.0
        Ec = modulus(params.get("b"), self.Ai, 1.5) if "interior" in terms else 0.0
        En = modulus(params.get("c"), self.An, 1.5) if "nucleus" in terms else 0.0

        if refine and np.isfinite([Em, Ec, En]).any():
            Em, Ec, En = self._refine_series(eps, force, Em, Ec, En, force_onset, terms)

        pred = self.series_force(eps, Em, Ec, En, force_onset)
        residuals = force - pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((force - force.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        warnings_list = []
        for value, label in ((Em, "Em"), (Ec, "Ec"), (En, "En")):
            if np.isinf(value):
                warnings_list.append(
                    f"{label} came out rigid (zero compliance): in series this element "
                    f"absorbs no deformation, so the data gives no handle on it."
                )
        if np.isfinite(r_squared) and r_squared < 0.9:
            warnings_list.append(
                f"Series R2 = {r_squared:.3f}. The stacked-spring picture does not "
                f"describe this curve well; compare it against parallel."
            )

        out = {
            "success": True,
            "mode": "series",
            "coupling": "series",
            "Em": Em if np.isfinite(Em) else 0.0,
            "Ei": Ec if np.isfinite(Ec) else 0.0,
            "En": En if np.isfinite(En) else 0.0,
            "Em_MPa": (Em / 1e6) if np.isfinite(Em) else float("inf"),
            "Ei_kPa": (Ec / 1e3) if np.isfinite(Ec) else float("inf"),
            "En_kPa": (En / 1e3) if np.isfinite(En) else float("inf"),
            "Em_MPa_std": float("nan"),
            "Ei_kPa_std": float("nan"),
            "En_kPa_std": float("nan"),
            "compliances": {k: float(v) for k, v in params.items()},
            "force_offset": 0.0,
            "nucleus_force_onset": float(force_onset),
            "Km": 0.0,
            "Km_kT": 0.0,
            "membrane_areal_modulus": (Em * self.h_membrane) if np.isfinite(Em) else float("nan"),
            "r_squared": r_squared,
            "adj_r_squared": float("nan"),
            "rmse": float(np.sqrt(ss_res / eps.size)),
            "residual_std": float(np.std(residuals)),
            "n_points": int(eps.size),
            "n_params": n_params,
            "ss_res": ss_res,
            "epsilon_range": [float(epsilon_min), float(epsilon_max)],
            "terms": list(terms),
            "weighting": weighting,
            "fit_offset": False,
            "condition_number": float("nan"),
            "corr_Em_Ei": float("nan"),
            "membrane_fraction_at_max": float("nan"),
            "interior_fraction_at_max": float("nan"),
            "nucleus_fraction_at_max": float("nan"),
            "nucleus_onset": self.nucleus_onset,
            "R0": self.R0,
            "R_nucleus": self.R_nucleus,
            "cell_height": self.cell_height,
            "Am": self.Am,
            "Ai": self.Ai,
            "An": self.An,
            "mask": mask,
            "warnings": warnings_list,
        }
        self.results["series"] = out
        return out

    # ------------------------------------------------------------- hybrid

    def predict(self, epsilon, params, coupling, crossover=None, order="parallel-then-series"):
        """
        Force predicted at each deformation under any of the couplings.

        ``params`` is ``(Em, Ec, En)`` in pascals. For the hybrid couplings
        the elements act in parallel on one side of ``crossover`` and in
        series on the other, so a single set of moduli produces a curve that
        changes its load path partway along.
        """
        Em, Ec, En = params
        eps = np.asarray(epsilon, dtype=float)
        if coupling == "parallel":
            return self.combined_model(eps, Em, Ec, En=En)
        if coupling == "series":
            return self.series_force(eps, Em, Ec, En, self._nucleus_force_onset(Em, Ec, En))
        if coupling == "hybrid":
            out = np.empty_like(eps)
            low = eps <= float(crossover)
            first, second = (
                ("parallel", "series")
                if order == "parallel-then-series"
                else ("series", "parallel")
            )
            out[low] = self.predict(eps[low], params, first)
            out[~low] = self.predict(eps[~low], params, second)
            # Remove the step at the crossover: the load path changes there,
            # but the force the cantilever reads does not jump.
            if low.any() and (~low).any():
                jump = self.predict(np.array([crossover]), params, first)[0] - self.predict(
                    np.array([crossover]), params, second
                )[0]
                out[~low] = out[~low] + jump
            return out
        raise ValueError(f"Unknown coupling: {coupling}")

    def _nucleus_force_onset(self, Em, Ec, En):
        """Force at which the nucleus engages, from its deformation onset."""
        if not En or En <= 0:
            return 0.0
        return float(self.combined_model(np.array([self.nucleus_onset]), Em, Ec)[0])

    def fit_hybrid(
        self,
        epsilon_min,
        epsilon_max,
        crossover,
        terms=("membrane", "interior"),
        order="parallel-then-series",
        seed=None,
    ):
        """
        One set of moduli, two load paths, split at ``crossover``.

        Parallel is linear in the moduli and series is linear in their
        compliances, so a model that is parallel on one side and series on the
        other is linear in neither. It is still only two or three parameters,
        and seeding from the separate parallel and series fits of each region
        puts the optimiser close enough that this is quick and stable.
        """
        from scipy.optimize import least_squares

        eps, force, mask = self._select(epsilon_min, epsilon_max)
        if eps.size < len(terms) + 2:
            return self._failure("Not enough points for a hybrid fit.")
        if not (eps.min() < crossover < eps.max()):
            return self._failure(
                f"Crossover e = {crossover:.3f} lies outside the fitted window."
            )

        if seed is None:
            base = self.fit(epsilon_min, epsilon_max, terms=terms)
            seed = (
                base.get("Em", 1e6) or 1e6,
                base.get("Ei", 1e3) or 1e3,
                base.get("En", 1e3) or 1e3,
            )
        seed = [max(float(v), 1e-3) for v in seed]

        active = [
            ("membrane" in terms),
            ("interior" in terms),
            ("nucleus" in terms),
        ]
        start = np.log10([seed[i] if active[i] else 1e-6 for i in range(3)])
        free = np.array(active, dtype=bool)
        scale = max(float(np.nanmax(np.abs(force))), 1e-15)

        def residual(free_log):
            full = start.copy()
            full[free] = free_log
            params = tuple(10.0 ** full[i] if active[i] else 0.0 for i in range(3))
            return (force - self.predict(eps, params, "hybrid", crossover, order)) / scale

        try:
            solution = least_squares(
                residual,
                start[free],
                bounds=(np.full(free.sum(), -1.0), np.full(free.sum(), 11.0)),
                xtol=1e-10,
                ftol=1e-10,
                max_nfev=400,
            )
        except Exception as exc:  # pragma: no cover - optimiser edge cases
            return self._failure(f"Hybrid fit failed: {exc}")

        full = start.copy()
        full[free] = solution.x
        Em, Ec, En = (10.0 ** full[i] if active[i] else 0.0 for i in range(3))

        pred = self.predict(eps, (Em, Ec, En), "hybrid", crossover, order)
        residuals = force - pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((force - force.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        out = {
            "success": True,
            "mode": "hybrid",
            "coupling": "hybrid",
            "order": order,
            "crossover": float(crossover),
            "Em": Em, "Ei": Ec, "En": En,
            "Em_MPa": Em / 1e6, "Ei_kPa": Ec / 1e3, "En_kPa": En / 1e3,
            "Em_MPa_std": float("nan"),
            "Ei_kPa_std": float("nan"),
            "En_kPa_std": float("nan"),
            "force_offset": 0.0,
            "Km": Em * self.h_membrane ** 3 / (12.0 * (1.0 - self.nu_m ** 2)),
            "Km_kT": (Em * self.h_membrane ** 3 / (12.0 * (1.0 - self.nu_m ** 2)))
            / (K_BOLTZMANN * 300.0),
            "membrane_areal_modulus": Em * self.h_membrane,
            "r_squared": r_squared,
            "adj_r_squared": float("nan"),
            "rmse": float(np.sqrt(ss_res / eps.size)),
            "residual_std": float(np.std(residuals)),
            "n_points": int(eps.size),
            "n_params": int(free.sum()) + 1,  # + the crossover
            "ss_res": ss_res,
            "epsilon_range": [float(epsilon_min), float(epsilon_max)],
            "terms": list(terms),
            "weighting": "uniform",
            "fit_offset": False,
            "condition_number": float("nan"),
            "corr_Em_Ei": float("nan"),
            "membrane_fraction_at_max": float("nan"),
            "interior_fraction_at_max": float("nan"),
            "nucleus_fraction_at_max": float("nan"),
            "nucleus_onset": self.nucleus_onset,
            "R0": self.R0, "R_nucleus": self.R_nucleus,
            "cell_height": self.cell_height,
            "Am": self.Am, "Ai": self.Ai, "An": self.An,
            "mask": mask,
            "warnings": [],
        }
        self.results["hybrid"] = out
        return out

    def scan_crossover(
        self, epsilon_min, epsilon_max, terms=("membrane", "interior"),
        order="parallel-then-series", n_trials=15,
    ):
        """Grid search for the deformation at which the load path changes."""
        lo = epsilon_min + 0.15 * (epsilon_max - epsilon_min)
        hi = epsilon_min + 0.85 * (epsilon_max - epsilon_min)
        trials, best = [], None
        for crossover in np.linspace(lo, hi, max(3, int(n_trials))):
            result = self.fit_hybrid(
                epsilon_min, epsilon_max, float(crossover), terms=terms, order=order
            )
            if not result.get("success"):
                continue
            trials.append(
                {
                    "crossover": float(crossover),
                    "r_squared": float(result["r_squared"]),
                    "Em_MPa": result["Em_MPa"],
                    "Ei_kPa": result["Ei_kPa"],
                }
            )
            if best is None or result["r_squared"] > best["r_squared"]:
                best = result
        if best is None:
            return {"success": False, "error": "No usable hybrid fit.", "trials": trials}
        return {"success": True, "best": best, "trials": trials,
                "best_crossover": best["crossover"]}


# ---------------------------------------------------------------------------
#  Segmented model
# ---------------------------------------------------------------------------
#
# The compression is treated as three stretches of deformation with different
# structures carrying the load:
#
#     0   -> e1   the membrane alone, the cubic balloon term
#     e1  -> e2   the cytoskeleton takes over; the membrane contributes no
#                 further force and simply holds what it had reached at e1
#     e2  -> end  the cytoskeleton continues and the nucleus joins it
#
# Writing that as one expression,
#
#     F(e) = Am Em min(e, e1)^3 + Ai Ec <e - e1>^1.5 + An En <e - e2>^1.5
#
# The min() freezes the membrane term at its value at e1 instead of letting it
# keep climbing, and each later term starts from zero at its own breakpoint.
# Two things follow, and both matter:
#
# * The curve is continuous at both breakpoints by construction. Nothing has to
#   be stitched together afterwards and there is no step for the fit to chase.
# * It is still LINEAR in Em, Ec and En, so the fit remains one exact bounded
#   least-squares solve. Only the breakpoints are non-linear, and they are two
#   bounded scalars, so they can be scanned on a grid.


class _SegmentedMixin:
    """The segmented three-stage model."""

    def segment_terms(self, epsilon, e1=None, e2=None):
        """The three basis functions, before scaling by their moduli."""
        eps = np.asarray(epsilon, dtype=float)
        e1 = self.segment_break_1 if e1 is None else float(e1)
        e2 = self.segment_break_2 if e2 is None else float(e2)
        membrane = self.Am * np.clip(np.minimum(eps, e1), 0.0, None) ** 3
        cyto = self.Ai * np.clip(eps - e1, 0.0, None) ** 1.5
        nucleus = self.An * np.clip(eps - e2, 0.0, None) ** 1.5
        return membrane, cyto, nucleus

    def segmented_model(self, epsilon, Em, Ec, En=0.0, e1=None, e2=None, force_offset=0.0):
        """Total force under the segmented model."""
        membrane, cyto, nucleus = self.segment_terms(epsilon, e1, e2)
        return membrane * Em + cyto * Ec + nucleus * En + force_offset

    def fit_segmented(
        self,
        epsilon_min=0.0,
        epsilon_max=0.60,
        e1=None,
        e2=None,
        terms=("membrane", "interior", "nucleus"),
        weighting="uniform",
        fit_offset=False,
    ):
        """
        Fit the segmented model: one bounded linear solve.

        Parameters
        ----------
        e1, e2 : float
            Breakpoints. ``e1`` is where the membrane stops adding force and
            the cytoskeleton takes over; ``e2`` is where the nucleus engages.
        terms : tuple
            Which of the three stages to include. Dropping one simply removes
            its column.
        """
        e1 = self.segment_break_1 if e1 is None else float(e1)
        e2 = self.segment_break_2 if e2 is None else float(e2)
        if not (0 <= e1 < e2):
            return self._failure(
                f"Breakpoints must satisfy 0 <= e1 < e2; got e1={e1:.3f}, e2={e2:.3f}."
            )

        eps, force, mask = self._select(epsilon_min, epsilon_max)
        n_params = len(terms) + (1 if fit_offset else 0)
        if eps.size < n_params + 1:
            return self._failure(
                f"Only {eps.size} points in e = [{epsilon_min:.3f}, "
                f"{epsilon_max:.3f}]; need at least {n_params + 1}."
            )

        membrane, cyto, nucleus = self.segment_terms(eps, e1, e2)
        columns, names = [], []
        if "membrane" in terms:
            columns.append(membrane)
            names.append("Em")
        if "interior" in terms:
            columns.append(cyto)
            names.append("Ei")
        if "nucleus" in terms:
            columns.append(nucleus)
            names.append("En")
        if fit_offset:
            columns.append(np.ones_like(eps))
            names.append("F0")
        design = np.column_stack(columns)

        if weighting == "relative":
            scale = np.maximum(np.abs(force), np.percentile(np.abs(force), 10) or 1e-15)
            weights = 1.0 / scale
        else:
            weights = np.ones_like(force)
        design_w, target_w = design * weights[:, None], force * weights

        col_norm = np.linalg.norm(design_w, axis=0)
        col_norm[col_norm == 0] = 1.0
        lower = np.array([-np.inf if n == "F0" else 0.0 for n in names])
        solution = lsq_linear(
            design_w / col_norm, target_w,
            bounds=(lower * col_norm, np.inf), method="bvls",
        )
        params = dict(zip(names, solution.x / col_norm))
        Em = float(params.get("Em", 0.0))
        Ec = float(params.get("Ei", 0.0))
        En = float(params.get("En", 0.0))
        F0 = float(params.get("F0", 0.0))

        predicted = self.segmented_model(eps, Em, Ec, En, e1, e2, F0)
        residuals = force - predicted
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((force - force.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        dof = max(eps.size - n_params, 1)

        std_err, corr, cond = self._covariance(
            design_w / col_norm, target_w, col_norm, names, dof, ss_res, eps.size
        )

        # What each stage contributes at the top of the fitted range.
        top = float(eps.max())
        m_top, c_top, n_top = self.segment_terms(np.array([top]), e1, e2)
        f_mem, f_cyt, f_nuc = float(m_top[0] * Em), float(c_top[0] * Ec), float(n_top[0] * En)
        total = f_mem + f_cyt + f_nuc

        warnings_list = []
        for value, key, unit, scale_to in (
            (Em, "Em", "MPa", 1e6), (Ec, "Ei", "kPa", 1e3), (En, "En", "kPa", 1e3),
        ):
            label = {"Em": "Em", "Ei": "Ec", "En": "En"}[key]
            if key in [{"membrane": "Em", "interior": "Ei", "nucleus": "En"}[t] for t in terms]:
                if value <= 0:
                    warnings_list.append(
                        f"{label} came out zero: its segment shows no rise that the "
                        f"other stages do not already explain. Check the breakpoints."
                    )
                else:
                    lo, hi = self.expected_ranges[key]
                    if not (lo <= value <= hi):
                        warnings_list.append(
                            f"{label} = {value / scale_to:.3g} {unit} is outside the "
                            f"expected {lo / scale_to:.3g} to {hi / scale_to:.3g} "
                            f"{unit} for this cell type."
                        )
        if np.isfinite(r_squared) and r_squared < 0.95:
            warnings_list.append(
                f"R2 = {r_squared:.3f}. Try moving the breakpoints, or run the "
                f"breakpoint scan to place them from the data."
            )

        out = {
            "success": True,
            "mode": "segmented",
            "coupling": "segmented",
            "Em": Em, "Ei": Ec, "En": En,
            "Em_MPa": Em / 1e6, "Ei_kPa": Ec / 1e3, "En_kPa": En / 1e3,
            "Em_std": std_err.get("Em", float("nan")),
            "Ei_std": std_err.get("Ei", float("nan")),
            "En_std": std_err.get("En", float("nan")),
            "Em_MPa_std": std_err.get("Em", float("nan")) / 1e6,
            "Ei_kPa_std": std_err.get("Ei", float("nan")) / 1e3,
            "En_kPa_std": std_err.get("En", float("nan")) / 1e3,
            "force_offset": F0,
            "break_1": e1,
            "break_2": e2,
            "Km": Em * self.h_membrane ** 3 / (12.0 * (1.0 - self.nu_m ** 2)),
            "Km_kT": (Em * self.h_membrane ** 3 / (12.0 * (1.0 - self.nu_m ** 2)))
            / (K_BOLTZMANN * 300.0),
            "membrane_areal_modulus": Em * self.h_membrane,
            "r_squared": r_squared,
            "adj_r_squared": 1.0 - (1.0 - r_squared) * (eps.size - 1) / dof
            if np.isfinite(r_squared) and eps.size > n_params else float("nan"),
            "rmse": float(np.sqrt(ss_res / eps.size)),
            "residual_std": float(np.std(residuals)),
            "n_points": int(eps.size),
            "n_params": n_params,
            "ss_res": ss_res,
            "epsilon_range": [float(epsilon_min), float(epsilon_max)],
            "terms": list(terms),
            "weighting": weighting,
            "fit_offset": bool(fit_offset),
            "condition_number": cond,
            "corr_Em_Ei": corr,
            "membrane_fraction_at_max": f_mem / total if total > 0 else float("nan"),
            "interior_fraction_at_max": f_cyt / total if total > 0 else float("nan"),
            "nucleus_fraction_at_max": f_nuc / total if total > 0 else float("nan"),
            "nucleus_onset": e2,
            "R0": self.R0, "R_nucleus": self.R_nucleus,
            "cell_height": self.cell_height,
            "Am": self.Am, "Ai": self.Ai, "An": self.An,
            "mask": mask,
            "warnings": warnings_list,
        }
        self.results["segmented"] = out
        return out

    def scan_segment_breaks(
        self,
        epsilon_min=0.0,
        epsilon_max=0.60,
        terms=("membrane", "interior", "nucleus"),
        n_grid=14,
        weighting="uniform",
    ):
        """
        Place the two breakpoints from the data.

        Both are bounded scalars and every trial is one exact linear solve, so
        a grid over the pair is affordable and exhaustive. The returned surface
        makes it visible when the optimum is a broad plateau, which means the
        curve does not really locate the breakpoints.
        """
        lo = max(float(epsilon_min), float(self.epsilon.min()))
        hi = min(float(epsilon_max), float(self.epsilon.max()))
        span = hi - lo
        if span <= 0:
            return {"success": False, "error": "Empty deformation range."}

        first = np.linspace(lo + 0.10 * span, lo + 0.55 * span, int(n_grid))
        second = np.linspace(lo + 0.30 * span, lo + 0.92 * span, int(n_grid))

        best, trials = None, []
        for e1 in first:
            for e2 in second:
                if e2 <= e1 + 0.02 * span:
                    continue
                result = self.fit_segmented(
                    epsilon_min, epsilon_max, e1=e1, e2=e2,
                    terms=terms, weighting=weighting,
                )
                if not result.get("success"):
                    continue
                trials.append(
                    {
                        "e1": float(e1), "e2": float(e2),
                        "r_squared": float(result["r_squared"]),
                        "Em_MPa": result["Em_MPa"],
                        "Ec_kPa": result["Ei_kPa"],
                        "En_kPa": result["En_kPa"],
                    }
                )
                if best is None or result["r_squared"] > best["r_squared"]:
                    best = result

        if best is None:
            return {"success": False, "error": "No usable segmented fit on the grid."}
        return {
            "success": True,
            "best": best,
            "best_break_1": best["break_1"],
            "best_break_2": best["break_2"],
            "trials": trials,
        }


class LulevichModel(_CouplingMixin, _SegmentedMixin):
    """Lulevich fit for single-cell compression curves."""

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        force,
        relative_deformation,
        cell_height,
        cell_radius=None,
        membrane_thickness=4e-9,
        poisson_membrane=0.5,
        poisson_interior=0.5,
        radius_from_height=0.55,
        nucleus_radius=None,
        nucleus_from_radius=0.35,
        poisson_nucleus=0.5,
        nucleus_onset=0.15,
        expected_ranges=None,
        active_windows=None,
        segment_break_1=0.15,
        segment_break_2=0.40,
    ):
        """
        Parameters
        ----------
        force : array
            Force in newtons.
        relative_deformation : array
            Relative deformation e = delta / h0, dimensionless.
        cell_height : float
            Initial cell height h0 **in metres** (8.09 um -> 8.09e-6).
        cell_radius : float, optional
            Cell radius R0 in metres. Defaults to ``radius_from_height * h0``.
        membrane_thickness : float
            Membrane thickness h_m in metres (default 4 nm).
        poisson_membrane, poisson_interior : float
            Poisson ratios; 0.5 (incompressible) for living cells.
        radius_from_height : float
            Aspect factor used when ``cell_radius`` is not given.
        nucleus_radius : float, optional
            Nucleus radius in metres. Defaults to ``nucleus_from_radius * R0``.
        nucleus_from_radius : float
            Nucleus radius as a fraction of the cell radius.
        poisson_nucleus : float
            Poisson ratio of the nucleus.
        nucleus_onset : float
            Relative deformation at which the plates begin to feel the nucleus.
            Below it the nucleus term is exactly zero. See
            :meth:`scan_nucleus_onset` to find it from the data.
        active_windows : dict, optional
            Deformation range over which each element carries load, e.g.
            ``{"membrane": (0.0, 0.40), "nucleus": (0.40, 1.0)}``. Elements
            without an entry act everywhere. Use this to describe a curve whose
            load-bearing structures change partway along.
        expected_ranges : dict, optional
            Plausibility bands in pascals used only for warnings, e.g.
            ``{"Em": (5e5, 1e7), "Ei": (3e2, 1e4), "En": (1e3, 5e4)}``. Set
            these per cell type so an out-of-range result is flagged against
            something meaningful rather than a generic 1 kPa to 1 GPa window.
        """
        force = np.asarray(force, dtype=float).ravel()
        epsilon = np.asarray(relative_deformation, dtype=float).ravel()

        if force.shape != epsilon.shape:
            raise ValueError(
                f"force and relative_deformation must be the same length "
                f"({force.size} vs {epsilon.size})"
            )

        cell_height = float(cell_height)
        if not np.isfinite(cell_height) or cell_height <= 0:
            raise ValueError("cell_height must be a positive number of metres")
        if cell_height > 1e-3:
            raise ValueError(
                f"cell_height={cell_height:g} m is larger than 1 mm. Cell "
                f"heights must be given in METRES: {cell_height:g} um should "
                f"be passed as {cell_height * 1e-6:g}. Use "
                f"LulevichModel.from_micrometres() if you have micrometres."
            )

        # Drop non-finite samples and sort by epsilon so that derivative-based
        # diagnostics below are meaningful.
        good = np.isfinite(force) & np.isfinite(epsilon)
        order = np.argsort(epsilon[good], kind="stable")
        self.force = force[good][order]
        self.epsilon = epsilon[good][order]
        self.n_dropped = int((~good).sum())

        self.cell_height = cell_height
        self.h_membrane = float(membrane_thickness)
        self.nu_m = float(poisson_membrane)
        self.nu_i = float(poisson_interior)
        self.R0 = float(cell_radius) if cell_radius else cell_height * radius_from_height

        if self.R0 <= 0:
            raise ValueError("cell_radius must be positive")

        self.nu_n = float(poisson_nucleus)
        self.R_nucleus = (
            float(nucleus_radius) if nucleus_radius else self.R0 * float(nucleus_from_radius)
        )
        self.nucleus_onset = float(nucleus_onset)
        self.segment_break_1 = float(segment_break_1)
        self.segment_break_2 = float(segment_break_2)
        self.active_windows = (
            {k: (float(v[0]), float(v[1])) for k, v in active_windows.items() if v}
            if active_windows
            else None
        )
        self.expected_ranges = {
            "Em": PLAUSIBLE_EM_PA,
            "Ei": PLAUSIBLE_EI_PA,
            "En": (1e1, 1e7),
        }
        if expected_ranges:
            self.expected_ranges.update(
                {k: (float(v[0]), float(v[1])) for k, v in expected_ranges.items() if v}
            )

        self.results = {}

    @classmethod
    def from_micrometres(cls, force, relative_deformation, cell_height_um, **kwargs):
        """Convenience constructor taking the cell height in micrometres."""
        if "cell_radius_um" in kwargs:
            radius_um = kwargs.pop("cell_radius_um")
            kwargs["cell_radius"] = None if not radius_um else radius_um * 1e-6
        return cls(force, relative_deformation, float(cell_height_um) * 1e-6, **kwargs)

    # -------------------------------------------------------- geometry terms

    @property
    def Am(self):
        """Membrane prefactor: F_membrane = Am * Em * e^3  [N/Pa]."""
        return 2.0 * np.pi * self.h_membrane * self.R0 / (1.0 - self.nu_m)

    @property
    def Ai(self):
        """Hertzian prefactor: F_interior = Ai * Ei * e^1.5  [N/Pa]."""
        return (
            np.sqrt(2.0)
            * np.sqrt(self.R0)
            * self.cell_height ** 1.5
            / (3.0 * (1.0 - self.nu_i ** 2))
        )

    @property
    def An(self):
        """Nucleus prefactor: F_nucleus = An * En * <e - e_onset>^1.5  [N/Pa]."""
        return (
            np.sqrt(2.0)
            * np.sqrt(self.R_nucleus)
            * self.cell_height ** 1.5
            / (3.0 * (1.0 - self.nu_n ** 2))
        )

    def nucleus_model(self, epsilon, En, onset=None):
        """
        Nucleus term, force in newtons.

        The nucleus carries no load until the cytoplasm above it has been
        squashed away, so the term is exactly zero below ``onset`` and rises as
        a Hertzian contact in the excess deformation beyond it. That offset is
        what keeps this term distinguishable from the cytoskeleton term, which
        has the same 3/2 exponent but starts at zero deformation.
        """
        onset = self.nucleus_onset if onset is None else float(onset)
        raw = np.asarray(epsilon, dtype=float)
        excess = np.clip(raw - onset, 0.0, None)
        return self.An * En * excess ** 1.5 * self._active(raw, "nucleus")

    def balloon_model_cubic(self, epsilon, Em):
        """Membrane (balloon) term, force in newtons."""
        eps = np.asarray(epsilon, dtype=float)
        return self.Am * Em * eps ** 3 * self._active(eps, "membrane")

    def hertzian_contact_model(self, epsilon, Ei):
        """Interior (Hertzian) term, force in newtons."""
        raw = np.asarray(epsilon, dtype=float)
        eps = np.clip(raw, 0.0, None)
        return self.Ai * Ei * eps ** 1.5 * self._active(raw, "interior")

    def combined_model(self, epsilon, Em, Ei, force_offset=0.0, En=0.0, onset=None):
        """Full model, force in newtons. ``En=0`` gives the two-term Lulevich fit."""
        total = (
            self.balloon_model_cubic(epsilon, Em)
            + self.hertzian_contact_model(epsilon, Ei)
            + force_offset
        )
        if En:
            total = total + self.nucleus_model(epsilon, En, onset)
        return total

    # ---------------------------------------------------------------- fitting

    def _select(self, epsilon_min, epsilon_max):
        mask = (self.epsilon >= epsilon_min) & (self.epsilon <= epsilon_max)
        return self.epsilon[mask], self.force[mask], mask

    def _active(self, eps, term):
        """
        1 where an element carries load, 0 where it does not.

        With ``active_windows`` set, an element contributes only inside its own
        deformation range. That is what lets a curve be described as one set of
        elements up to some deformation and a different set beyond it: membrane
        plus cytoskeleton while the cell is a pressurised balloon, cytoskeleton
        plus nucleus once it is squashed flat and the membrane no longer holds
        the load. The model stays linear in the moduli, because switching an
        element off is just zeroing its column.
        """
        window = (self.active_windows or {}).get(term)
        if not window:
            return np.ones_like(eps)
        lo, hi = window
        return ((eps >= lo) & (eps <= hi)).astype(float)

    def _design_matrix(self, eps, terms, fit_offset):
        cols, names = [], []
        if "membrane" in terms:
            cols.append(self.Am * eps ** 3 * self._active(eps, "membrane"))
            names.append("Em")
        if "interior" in terms:
            cols.append(
                self.Ai * np.clip(eps, 0.0, None) ** 1.5 * self._active(eps, "interior")
            )
            names.append("Ei")
        if "nucleus" in terms:
            cols.append(
                self.An
                * np.clip(eps - self.nucleus_onset, 0.0, None) ** 1.5
                * self._active(eps, "nucleus")
            )
            names.append("En")
        if fit_offset:
            cols.append(np.ones_like(eps))
            names.append("F0")
        return np.column_stack(cols), names

    def fit(
        self,
        epsilon_min=0.01,
        epsilon_max=0.3,
        terms=("membrane", "interior"),
        fit_offset=False,
        weighting="uniform",
        enforce_positive=True,
        fixed=None,
    ):
        """
        Bounded linear least-squares fit of the Lulevich model.

        Parameters
        ----------
        epsilon_min, epsilon_max : float
            Fitting window in relative deformation.
        terms : tuple
            Any of ``"membrane"``, ``"interior"``. Use one for a single-term
            fit, both for the full two-term Lulevich fit.
        fit_offset : bool
            Also fit a constant force offset, absorbing a residual baseline
            or contact-point error. The offset is unbounded in sign.
        weighting : {"uniform", "relative"}
            ``"uniform"`` minimises absolute residuals and is dominated by the
            high-force end. ``"relative"`` weights each point by 1/|F|, so the
            low-deformation region carries comparable weight. Use "relative"
            when the fit visibly ignores the small-e points.
        enforce_positive : bool
            Constrain the moduli to be >= 0. Negative moduli are unphysical;
            hitting the zero bound means that term is not supported by the
            data, which is reported rather than treated as a failure.
        fixed : dict, optional
            Moduli to hold at a known value instead of fitting, e.g.
            ``{"Ei": 800.0}``. The corresponding term is subtracted from the
            force before fitting and added back into the reported model. This
            is what the sequential (two-stage) workflow uses to carry Ei from
            the low-deformation window into the membrane fit.

        Returns
        -------
        dict
            Fit results, diagnostics, and uncertainties. ``success`` is False
            only when the input data is unusable (too few points, degenerate).
        """
        fixed = dict(fixed or {})
        # A term that is held fixed is not a free parameter.
        _fixed_key = {"membrane": "Em", "interior": "Ei", "nucleus": "En"}
        free_terms = tuple(t for t in terms if _fixed_key.get(t) not in fixed)

        eps, force_all, mask = self._select(epsilon_min, epsilon_max)
        force = force_all.copy()
        # Remove the known contributions so the remaining fit is on the residual.
        if "Em" in fixed:
            force = force - self.balloon_model_cubic(eps, fixed["Em"])
        if "Ei" in fixed:
            force = force - self.hertzian_contact_model(eps, fixed["Ei"])
        if "En" in fixed:
            force = force - self.nucleus_model(eps, fixed["En"])

        if not free_terms:
            return self._failure("Every requested term is held fixed; nothing to fit.")

        terms = free_terms
        n_params = len(terms) + (1 if fit_offset else 0)

        if eps.size < n_params + 1:
            return self._failure(
                f"Only {eps.size} points in e = [{epsilon_min:.3f}, "
                f"{epsilon_max:.3f}]; need at least {n_params + 1} for a "
                f"{n_params}-parameter fit. Widen the range."
            )
        if np.ptp(eps) <= 0:
            return self._failure("All points in range share the same e value.")

        X, names = self._design_matrix(eps, terms, fit_offset)

        # Weights
        if weighting == "relative":
            scale = np.maximum(np.abs(force), np.percentile(np.abs(force), 10))
            scale[scale <= 0] = 1.0
            w = 1.0 / scale
        else:
            w = np.ones_like(force)
        Xw, yw = X * w[:, None], force * w

        # Column scaling keeps the normal equations well conditioned even
        # though Am and Ai differ by orders of magnitude.
        col_norm = np.linalg.norm(Xw, axis=0)
        col_norm[col_norm == 0] = 1.0
        Xs = Xw / col_norm

        lo = np.full(len(names), 0.0 if enforce_positive else -np.inf)
        hi = np.full(len(names), np.inf)
        if fit_offset:
            lo[names.index("F0")] = -np.inf  # offset may be negative

        sol = lsq_linear(Xs, yw, bounds=(lo * col_norm, hi), method="bvls")
        params = sol.x / col_norm
        p = dict(zip(names, params))

        Em = float(p.get("Em", fixed.get("Em", 0.0)))
        Ei = float(p.get("Ei", fixed.get("Ei", 0.0)))
        En = float(p.get("En", fixed.get("En", 0.0)))
        F0 = float(p.get("F0", 0.0))

        # Goodness of fit is always measured against the ORIGINAL force with
        # the complete model, so a staged fit is scored on the same footing as
        # a simultaneous one.
        pred = self.combined_model(eps, Em, Ei, F0, En=En)
        residuals = force_all - pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((force_all - force_all.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        dof = max(eps.size - n_params, 1)
        adj_r2 = (
            1.0 - (1.0 - r_squared) * (eps.size - 1) / dof
            if np.isfinite(r_squared) and eps.size > n_params
            else float("nan")
        )
        rmse = float(np.sqrt(ss_res / eps.size))

        # Analytic covariance of the (scaled) linear parameters.
        std_err, corr_EmEi, cond = self._covariance(Xs, yw, col_norm, names, dof, ss_res, eps.size)

        # How much of the force at the top of the window each term explains.
        e_top = eps.max()
        f_mem = float(self.balloon_model_cubic(e_top, Em))
        f_int = float(self.hertzian_contact_model(e_top, Ei))
        f_nuc = float(self.nucleus_model(e_top, En)) if En else 0.0
        f_tot = f_mem + f_int + f_nuc
        membrane_fraction = f_mem / f_tot if f_tot > 0 else float("nan")
        interior_fraction = f_int / f_tot if f_tot > 0 else float("nan")
        nucleus_fraction = f_nuc / f_tot if f_tot > 0 else float("nan")

        Km = Em * self.h_membrane ** 3 / (12.0 * (1.0 - self.nu_m ** 2))

        warnings_list = self._warnings(
            Em, Ei, names, cond, r_squared, eps.size, membrane_fraction, En=En
        )

        out = {
            "success": True,
            "Em": Em,
            "Ei": Ei,
            "En": En,
            "Em_MPa": Em / 1e6,
            "Ei_kPa": Ei / 1e3,
            "En_kPa": En / 1e3,
            "En_std": std_err.get("En", float("nan")),
            "En_kPa_std": std_err.get("En", float("nan")) / 1e3,
            "nucleus_onset": self.nucleus_onset,
            "R_nucleus": self.R_nucleus,
            "An": self.An,
            "membrane_areal_modulus": Em * self.h_membrane,
            "Em_std": std_err.get("Em", float("nan")),
            "Ei_std": std_err.get("Ei", float("nan")),
            "Em_MPa_std": std_err.get("Em", float("nan")) / 1e6,
            "Ei_kPa_std": std_err.get("Ei", float("nan")) / 1e3,
            "force_offset": F0,
            "Km": Km,
            "Km_kT": Km / (K_BOLTZMANN * 300.0),
            "r_squared": r_squared,
            "adj_r_squared": adj_r2,
            "rmse": rmse,
            "residual_std": float(np.std(residuals)),
            "n_points": int(eps.size),
            "epsilon_range": [float(epsilon_min), float(epsilon_max)],
            "epsilon_used": [float(eps.min()), float(eps.max())],
            "terms": list(terms),
            "fixed": {k: float(v) for k, v in fixed.items()},
            "weighting": weighting,
            "fit_offset": bool(fit_offset),
            "condition_number": cond,
            "corr_Em_Ei": corr_EmEi,
            "membrane_fraction_at_max": membrane_fraction,
            "interior_fraction_at_max": interior_fraction,
            "nucleus_fraction_at_max": nucleus_fraction,
            "R0": self.R0,
            "cell_height": self.cell_height,
            "Am": self.Am,
            "Ai": self.Ai,
            "mask": mask,
            "warnings": warnings_list,
        }
        self.results["combined"] = out
        return out

    def _covariance(self, Xs, yw, col_norm, names, dof, ss_res, n):
        """Standard errors, Em/Ei correlation, condition number."""
        std_err, corr, cond = {}, float("nan"), float("nan")
        try:
            cond = float(np.linalg.cond(Xs))
            XtX_inv = np.linalg.inv(Xs.T @ Xs)
            sigma2 = ss_res / dof
            # Undo column scaling: cov(params) = D^-1 cov(scaled) D^-1
            D_inv = np.diag(1.0 / col_norm)
            cov = D_inv @ (sigma2 * XtX_inv) @ D_inv
            diag = np.diag(cov)
            for i, name in enumerate(names):
                std_err[name] = float(np.sqrt(diag[i])) if diag[i] > 0 else float("nan")
            if "Em" in names and "Ei" in names:
                a, b = names.index("Em"), names.index("Ei")
                denom = np.sqrt(diag[a] * diag[b])
                if denom > 0:
                    corr = float(cov[a, b] / denom)
        except np.linalg.LinAlgError:
            pass
        return std_err, corr, cond

    def _warnings(self, Em, Ei, names, cond, r2, n_points, membrane_fraction, En=0.0):
        w = []
        if "En" in names and En <= 0:
            w.append(
                "En collapsed to zero: past the onset deformation the data shows no "
                "extra stiffening, so this curve gives no evidence of the nucleus. "
                "Either it was never engaged, or the onset is set too high."
            )
        if "Em" in names and Em <= 0:
            w.append(
                "Em collapsed to zero: over this e-window the data is fully "
                "explained by the Hertzian term. Extend e_max to include more "
                "of the stiffening region where the membrane term dominates."
            )
        if "Ei" in names and Ei <= 0:
            w.append(
                "Ei collapsed to zero: the low-e region carries no cubic-free "
                "signal. Lower e_min, or check the contact point."
            )
        if np.isfinite(cond) and cond > 30:
            w.append(
                f"e^3 and e^1.5 are nearly collinear over this window "
                f"(condition number {cond:.0f}). The SUM is well determined but "
                f"the Em/Ei split is not; widen the range or fix one term."
            )
        for value, key, unit, scale in (
            (Em, "Em", "MPa", 1e6),
            (Ei, "Ei", "kPa", 1e3),
            (En, "En", "kPa", 1e3),
        ):
            if key not in names:
                continue
            lo, hi = self.expected_ranges[key]
            if value > 0 and not (lo <= value <= hi):
                w.append(
                    f"{key} = {value/scale:.3g} {unit} is outside the expected "
                    f"{lo/scale:.3g} to {hi/scale:.3g} {unit} for this cell type. "
                    f"Check the cell height, radius and force units before "
                    f"trusting it."
                )
        if np.isfinite(r2) and r2 < 0.9:
            w.append(
                f"R2 = {r2:.3f}. The two-term model does not describe this "
                f"window well; try enabling the force offset, switching to "
                f"relative weighting, or trimming past the rupture point."
            )
        if n_points < 20:
            w.append(f"Only {n_points} points in the window; the fit is poorly constrained.")
        if np.isfinite(membrane_fraction):
            if membrane_fraction > 0.98:
                w.append("Membrane term carries >98% of the force; Ei is essentially unconstrained.")
            elif membrane_fraction < 0.02:
                w.append("Hertzian term carries >98% of the force; Em is essentially unconstrained.")
        return w

    @staticmethod
    def _failure(message):
        return {
            "success": False,
            "error": message,
            "Em": 0.0,
            "Ei": 0.0,
            "Em_MPa": 0.0,
            "Ei_kPa": 0.0,
            "r_squared": float("nan"),
            "warnings": [],
        }

    # ------------------------------------------------------- staged fitting

    TERM_KEY = {"membrane": "Em", "interior": "Ei", "nucleus": "En"}
    TERM_LABEL = {"membrane": "membrane", "interior": "cytoskeleton", "nucleus": "nucleus"}

    def fit_staged(
        self,
        stages,
        weighting="uniform",
        fit_offset=False,
        refine_iterations=3,
        seed_parallel=True,
    ):
        """
        Fit groups of terms in sequence, each on its own deformation window.

        This is the general form of the series workflow. Each stage names the
        terms it solves for and the window it solves them on; every other term
        is held at its current estimate and subtracted first. Stages run in
        order, and the whole sequence repeats so that early stages get the
        benefit of what the later ones learned.

        Parameters
        ----------
        stages : list of dict
            ``[{"terms": ("membrane",), "range": (0.20, 0.35)},
               {"terms": ("interior", "nucleus"), "range": (0.01, 0.15)}]``
        refine_iterations : int
            Passes over the whole sequence. Later passes let early stages
            benefit from what the later ones found. 3 is normally converged.
        seed_parallel : bool
            Start from a parallel fit over the union of all windows. Without a
            seed the first stage has nothing to subtract, so it absorbs the
            whole force; the non-negativity constraint then pins the later
            stages at zero and the sequence has no way back. Seeding removes
            that trap. Turn it off only to see the unseeded staged behaviour.

        Returns
        -------
        dict
            Same shape as :meth:`fit`, plus ``stages``, ``iterations`` and a
            joint ``r_squared`` over the union of all windows.
        """
        stages = [
            {"terms": tuple(st["terms"]), "range": (float(st["range"][0]), float(st["range"][1]))}
            for st in stages
            if st.get("terms")
        ]
        if not stages:
            return self._failure("No stages defined.")

        estimates = {"Em": 0.0, "Ei": 0.0, "En": 0.0}
        all_terms = [t for st in stages for t in st["terms"]]
        history, stage_results = [], []

        seed = None
        if seed_parallel and len(stages) > 1:
            union_lo = min(st["range"][0] for st in stages)
            union_hi = max(st["range"][1] for st in stages)
            seed = self.fit(
                union_lo,
                union_hi,
                terms=tuple(dict.fromkeys(all_terms)),
                weighting=weighting,
                fit_offset=fit_offset,
            )
            if seed.get("success"):
                for term in all_terms:
                    key = self.TERM_KEY[term]
                    estimates[key] = seed[key]
            else:
                seed = None

        for iteration in range(max(1, int(refine_iterations))):
            stage_results = []
            for st in stages:
                keys_here = {self.TERM_KEY[t] for t in st["terms"]}
                # Hold every other term of the model at its current value.
                carry = {
                    self.TERM_KEY[t]: estimates[self.TERM_KEY[t]]
                    for t in all_terms
                    if self.TERM_KEY[t] not in keys_here
                }
                result = self.fit(
                    st["range"][0],
                    st["range"][1],
                    terms=st["terms"],
                    weighting=weighting,
                    fit_offset=fit_offset,
                    fixed=carry or None,
                )
                if not result.get("success"):
                    labels = ", ".join(self.TERM_LABEL[t] for t in st["terms"])
                    return self._failure(f"Stage '{labels}': {result['error']}")
                for t in st["terms"]:
                    estimates[self.TERM_KEY[t]] = result[self.TERM_KEY[t]]
                stage_results.append(result)

            history.append(
                {
                    "iteration": iteration + 1,
                    "Em_MPa": estimates["Em"] / 1e6,
                    "Ei_kPa": estimates["Ei"] / 1e3,
                    "En_kPa": estimates["En"] / 1e3,
                }
            )
            if len(history) > 1:
                prev, now = history[-2], history[-1]
                moved = max(
                    abs(now[k] - prev[k]) / max(abs(now[k]), 1e-12)
                    for k in ("Em_MPa", "Ei_kPa", "En_kPa")
                )
                if moved < 1e-3:
                    break

        Em, Ei, En = estimates["Em"], estimates["Ei"], estimates["En"]
        F0 = stage_results[-1].get("force_offset", 0.0)

        union = np.zeros(self.epsilon.shape, dtype=bool)
        for result in stage_results:
            union |= result["mask"]
        eps_u, force_u = self.epsilon[union], self.force[union]
        pred_u = self.combined_model(eps_u, Em, Ei, F0, En=En)
        residual = force_u - pred_u
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((force_u - force_u.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        e_top = float(eps_u.max()) if eps_u.size else 0.0
        f_mem = float(self.balloon_model_cubic(e_top, Em))
        f_int = float(self.hertzian_contact_model(e_top, Ei))
        f_nuc = float(self.nucleus_model(e_top, En)) if En else 0.0
        f_tot = f_mem + f_int + f_nuc

        warnings_list = self._staged_warnings(stages, estimates, r_squared)

        out = {
            "success": True,
            "mode": "staged",
            "Em": Em,
            "Ei": Ei,
            "En": En,
            "Em_MPa": Em / 1e6,
            "Ei_kPa": Ei / 1e3,
            "En_kPa": En / 1e3,
            "Em_std": float("nan"),
            "Ei_std": float("nan"),
            "En_std": float("nan"),
            "force_offset": F0,
            "Km": Em * self.h_membrane ** 3 / (12.0 * (1.0 - self.nu_m ** 2)),
            "Km_kT": (Em * self.h_membrane ** 3 / (12.0 * (1.0 - self.nu_m ** 2)))
            / (K_BOLTZMANN * 300.0),
            "membrane_areal_modulus": Em * self.h_membrane,
            "r_squared": r_squared,
            "adj_r_squared": float("nan"),
            "rmse": float(np.sqrt(ss_res / eps_u.size)) if eps_u.size else float("nan"),
            "residual_std": float(np.std(residual)) if residual.size else float("nan"),
            "n_points": int(eps_u.size),
            "epsilon_range": [
                float(min(st["range"][0] for st in stages)),
                float(max(st["range"][1] for st in stages)),
            ],
            "stage_plan": stages,
            "terms": all_terms,
            "weighting": weighting,
            "fit_offset": bool(fit_offset),
            "condition_number": float("nan"),
            "corr_Em_Ei": float("nan"),
            "membrane_fraction_at_max": f_mem / f_tot if f_tot > 0 else float("nan"),
            "interior_fraction_at_max": f_int / f_tot if f_tot > 0 else float("nan"),
            "nucleus_fraction_at_max": f_nuc / f_tot if f_tot > 0 else float("nan"),
            "nucleus_onset": self.nucleus_onset,
            "R0": self.R0,
            "R_nucleus": self.R_nucleus,
            "cell_height": self.cell_height,
            "Am": self.Am,
            "Ai": self.Ai,
            "An": self.An,
            "mask": union,
            "stages": stage_results,
            "iterations": history,
            "n_iterations": len(history),
            "seeded": seed is not None,
            "warnings": warnings_list,
        }
        for key in ("Em", "Ei", "En"):
            unit = 1e6 if key == "Em" else 1e3
            suffix = "MPa" if key == "Em" else "kPa"
            out[f"{key}_{suffix}_std"] = out[f"{key}_std"] / unit
        self.results["staged"] = out
        return out

    def _staged_warnings(self, stages, estimates, r_squared):
        warnings_list = []
        for i, first in enumerate(stages):
            for second in stages[i + 1 :]:
                lo = max(first["range"][0], second["range"][0])
                hi = min(first["range"][1], second["range"][1])
                if lo < hi:
                    a = "/".join(self.TERM_LABEL[t] for t in first["terms"])
                    b = "/".join(self.TERM_LABEL[t] for t in second["terms"])
                    warnings_list.append(
                        f"The '{a}' and '{b}' windows overlap between e = {lo:.3f} and "
                        f"{hi:.3f}. Separate them so each stage measures where its own "
                        f"terms dominate."
                    )
        for term, key in self.TERM_KEY.items():
            if any(term in st["terms"] for st in stages) and estimates[key] <= 0:
                warnings_list.append(
                    f"The {self.TERM_LABEL[term]} modulus came out zero: its window "
                    f"holds no signal of that term's shape once the others are removed."
                )
        if np.isfinite(r_squared) and r_squared < 0.9:
            warnings_list.append(
                f"Joint R2 = {r_squared:.3f} across all windows. The staged result does "
                f"not describe the whole curve; try different windows or fit in parallel."
            )
        return warnings_list

    def scan_nucleus_onset(self, epsilon_min, epsilon_max, terms=("membrane", "interior", "nucleus"),
                           n_trials=25, weighting="uniform", fit_offset=False):
        """
        Find the onset deformation that best explains the data.

        The onset is the only non-linear parameter in the model, and it is a
        single bounded scalar, so a scan over a grid is both exhaustive and
        cheap: each trial is one exact linear solve. Returns the best onset,
        its fit, and the whole R2 curve so a flat maximum (meaning the data
        does not really locate the nucleus) is visible rather than hidden.
        """
        original = self.nucleus_onset
        lo = epsilon_min + 0.15 * (epsilon_max - epsilon_min)
        hi = epsilon_min + 0.85 * (epsilon_max - epsilon_min)
        trials = []
        best = None
        try:
            for onset in np.linspace(lo, hi, max(3, int(n_trials))):
                self.nucleus_onset = float(onset)
                result = self.fit(
                    epsilon_min, epsilon_max, terms=terms,
                    weighting=weighting, fit_offset=fit_offset,
                )
                if not result.get("success"):
                    continue
                trials.append(
                    {
                        "onset": float(onset),
                        "r_squared": float(result["r_squared"]),
                        "Em_MPa": result["Em_MPa"],
                        "Ei_kPa": result["Ei_kPa"],
                        "En_kPa": result["En_kPa"],
                    }
                )
                if best is None or result["r_squared"] > best["r_squared"]:
                    best = {"onset": float(onset), "r_squared": float(result["r_squared"])}
        finally:
            self.nucleus_onset = original

        if best is None:
            return {"success": False, "error": "No usable fit across the onset scan.",
                    "trials": trials}

        r2_values = np.array([t["r_squared"] for t in trials], dtype=float)
        spread = float(np.nanmax(r2_values) - np.nanmin(r2_values)) if r2_values.size else 0.0
        # Judge the spread against the residual that is left at the optimum,
        # not against an absolute number: on a clean curve the whole scan sits
        # within a thousandth of 1.0 and an absolute threshold would call a
        # sharply located onset "undetermined".
        headroom = max(1.0 - best["r_squared"], 1e-12)
        significance = spread / headroom
        out = {
            "success": True,
            "best_onset": best["onset"],
            "best_r_squared": best["r_squared"],
            "trials": trials,
            "r_squared_spread": spread,
            "significance": float(significance),
            "well_determined": bool(significance > 1.0),
        }
        self.results["nucleus_onset_scan"] = out
        return out

    # ------------------------------------------------------ sequential fit

    def fit_sequential(
        self,
        interior_range=(0.01, 0.10),
        membrane_range=(0.15, 0.30),
        weighting="uniform",
        fit_offset=False,
        order="interior-first",
        refine_iterations=3,
    ):
        """
        Two-stage fit on two separate deformation windows.

        The physical justification is the exponents: at small e the Hertzian
        term (e^1.5) dominates, at large e the membrane term (e^3) takes over.
        So the interior modulus is measured where the membrane contributes
        least, then held fixed while the membrane modulus is measured where it
        dominates. This avoids asking one window to separate two nearly
        collinear basis functions, which is what makes the simultaneous fit
        sensitive to the range.

        Parameters
        ----------
        interior_range : (float, float)
            Low-deformation window used for Ei.
        membrane_range : (float, float)
            High-deformation window used for Em.
        order : {"interior-first", "membrane-first"}
            Which modulus is measured on its own window first. The default
            matches the physics; "membrane-first" is available for curves
            where the high-e region is the cleaner one.
        refine_iterations : int
            A single pass is biased: the first stage has no knowledge of the
            other term, so it absorbs whatever that term contributes inside
            its own window. Repeating the pair of fits, each time subtracting
            the other term's current estimate, removes most of that bias
            (this is backfitting). 1 = plain one-pass sequential fit;
            3 is usually converged. The per-iteration values are returned in
            ``iterations`` so the convergence is visible.

        Returns
        -------
        dict
            Same shape as :meth:`fit`, with an extra ``stages`` entry holding
            the two individual fits. ``r_squared`` is measured over the union
            of the two windows using the complete two-term model.
        """
        interior_range = (float(interior_range[0]), float(interior_range[1]))
        membrane_range = (float(membrane_range[0]), float(membrane_range[1]))

        if order == "membrane-first":
            first_terms, first_range, first_key = ("membrane",), membrane_range, "Em"
            second_terms, second_range, second_key = ("interior",), interior_range, "Ei"
        else:
            first_terms, first_range, first_key = ("interior",), interior_range, "Ei"
            second_terms, second_range, second_key = ("membrane",), membrane_range, "Em"

        stage1 = stage2 = None
        estimates = {"Em": 0.0, "Ei": 0.0}
        history = []

        for iteration in range(max(1, int(refine_iterations))):
            # Stage 1: measure the first modulus on its own window, removing
            # the other term's current estimate (zero on the first pass).
            carry = {second_key: estimates[second_key]} if iteration > 0 else None
            stage1 = self.fit(
                first_range[0],
                first_range[1],
                terms=first_terms,
                weighting=weighting,
                fit_offset=fit_offset,
                fixed=carry,
            )
            if not stage1.get("success"):
                return self._failure(f"Stage 1 ({first_terms[0]} window): {stage1['error']}")
            estimates[first_key] = stage1[first_key]

            # Stage 2: the other modulus on its window, holding stage 1 fixed.
            stage2 = self.fit(
                second_range[0],
                second_range[1],
                terms=second_terms,
                weighting=weighting,
                fit_offset=fit_offset,
                fixed={first_key: estimates[first_key]},
            )
            if not stage2.get("success"):
                return self._failure(f"Stage 2 ({second_terms[0]} window): {stage2['error']}")
            estimates[second_key] = stage2[second_key]

            history.append(
                {
                    "iteration": iteration + 1,
                    "Em_MPa": estimates["Em"] / 1e6,
                    "Ei_kPa": estimates["Ei"] / 1e3,
                }
            )
            # Stop once both moduli move by less than 0.1 %.
            if len(history) > 1:
                prev, now = history[-2], history[-1]
                moved = max(
                    abs(now["Em_MPa"] - prev["Em_MPa"]) / max(abs(now["Em_MPa"]), 1e-12),
                    abs(now["Ei_kPa"] - prev["Ei_kPa"]) / max(abs(now["Ei_kPa"]), 1e-12),
                )
                if moved < 1e-3:
                    break

        Em, Ei = estimates["Em"], estimates["Ei"]
        F0 = stage2.get("force_offset", 0.0)

        # Score the joint model over both windows together.
        union = stage1["mask"] | stage2["mask"]
        eps_u, force_u = self.epsilon[union], self.force[union]
        pred_u = self.combined_model(eps_u, Em, Ei, F0)
        res_u = force_u - pred_u
        ss_res = float(np.sum(res_u ** 2))
        ss_tot = float(np.sum((force_u - force_u.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        e_top = float(eps_u.max()) if eps_u.size else 0.0
        f_mem = float(self.balloon_model_cubic(e_top, Em))
        f_int = float(self.hertzian_contact_model(e_top, Ei))
        membrane_fraction = f_mem / (f_mem + f_int) if (f_mem + f_int) > 0 else float("nan")

        warnings_list = []
        if interior_range[1] > membrane_range[0]:
            warnings_list.append(
                f"The two windows overlap between e = {membrane_range[0]:.3f} and "
                f"{interior_range[1]:.3f}. Separate them so each modulus is measured "
                f"where its own term dominates."
            )
        if Ei <= 0:
            warnings_list.append(
                "Ei came out zero on the low-e window. Lower its start, or check "
                "the contact point."
            )
        if Em <= 0:
            warnings_list.append(
                "Em came out zero on the high-e window. The residual after removing "
                "the Hertzian term has no cubic content there."
            )
        if np.isfinite(r_squared) and r_squared < 0.9:
            warnings_list.append(
                f"Joint R2 = {r_squared:.3f} across both windows. The staged result "
                f"does not describe the whole curve; try different windows or the "
                f"simultaneous fit."
            )
        warnings_list.extend(w for w in stage1["warnings"] if "collinear" not in w)

        Km = Em * self.h_membrane ** 3 / (12.0 * (1.0 - self.nu_m ** 2))

        out = {
            "success": True,
            "mode": "sequential",
            "order": order,
            "Em": Em,
            "Ei": Ei,
            "Em_MPa": Em / 1e6,
            "Ei_kPa": Ei / 1e3,
            "Em_std": stage2.get("Em_std", float("nan")) if "membrane" in second_terms else stage1.get("Em_std", float("nan")),
            "Ei_std": stage1.get("Ei_std", float("nan")) if "interior" in first_terms else stage2.get("Ei_std", float("nan")),
            "force_offset": F0,
            "Km": Km,
            "Km_kT": Km / (K_BOLTZMANN * 300.0),
            "r_squared": r_squared,
            "adj_r_squared": float("nan"),
            "rmse": float(np.sqrt(ss_res / eps_u.size)) if eps_u.size else float("nan"),
            "residual_std": float(np.std(res_u)) if res_u.size else float("nan"),
            "n_points": int(eps_u.size),
            "epsilon_range": [
                float(min(interior_range[0], membrane_range[0])),
                float(max(interior_range[1], membrane_range[1])),
            ],
            "interior_range": list(interior_range),
            "membrane_range": list(membrane_range),
            "terms": ["membrane", "interior"],
            "weighting": weighting,
            "fit_offset": bool(fit_offset),
            "condition_number": float("nan"),
            "corr_Em_Ei": float("nan"),
            "membrane_fraction_at_max": membrane_fraction,
            "interior_fraction_at_max": interior_fraction,
            "nucleus_fraction_at_max": nucleus_fraction,
            "R0": self.R0,
            "cell_height": self.cell_height,
            "Am": self.Am,
            "Ai": self.Ai,
            "mask": union,
            "stages": {"first": stage1, "second": stage2},
            "iterations": history,
            "n_iterations": len(history),
            "warnings": warnings_list,
        }
        out["Em_MPa_std"] = out["Em_std"] / 1e6
        out["Ei_kPa_std"] = out["Ei_std"] / 1e3
        self.results["sequential"] = out
        return out

    def suggest_sequential_windows(self, crossover_fraction=0.5):
        """
        Propose a low-e window for Ei and a high-e window for Em by splitting
        the usable range at the point where the two terms would contribute
        equally under a provisional simultaneous fit.
        """
        auto = self.auto_detect_elastic_range()
        lo, hi = auto["elastic_epsilon_min"], auto["elastic_epsilon_max"]

        provisional = self.fit(lo, hi)
        crossover = None
        if provisional.get("success") and provisional["Em"] > 0 and provisional["Ei"] > 0:
            # Am*Em*e^3 == Ai*Ei*e^1.5  ->  e^1.5 = Ai*Ei / (Am*Em)
            ratio = (self.Ai * provisional["Ei"]) / (self.Am * provisional["Em"])
            if ratio > 0:
                crossover = float(ratio ** (2.0 / 3.0))

        if crossover is None or not (lo < crossover < hi):
            crossover = lo + crossover_fraction * (hi - lo)

        gap = 0.05 * (hi - lo)
        interior = (float(lo), float(max(lo + 1e-3, crossover - gap)))
        membrane = (float(min(hi - 1e-3, crossover + gap)), float(hi))
        out = {
            "interior_range": interior,
            "membrane_range": membrane,
            "crossover": float(crossover),
            "note": (
                f"Below e = {crossover:.3f} the Hertzian term dominates, above it the "
                f"membrane term does."
            ),
        }
        self.results["sequential_suggestion"] = out
        return out

    # --------------------------------------------- backwards-compatible API

    def fit_combined_elasticity(self, epsilon_max=0.3, epsilon_min=0.01, **kwargs):
        """Two-term fit (membrane + interior). Kept for API compatibility."""
        return self.fit(
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
            terms=("membrane", "interior"),
            **kwargs,
        )

    def fit_membrane_elasticity(self, epsilon_max=0.3, epsilon_min=0.02, **kwargs):
        """Single-term balloon fit."""
        res = self.fit(
            epsilon_min=epsilon_min, epsilon_max=epsilon_max, terms=("membrane",), **kwargs
        )
        self.results["membrane"] = res
        return res

    def fit_cytoskeleton_elasticity(self, epsilon_max=0.3, epsilon_min=0.05, **kwargs):
        """Single-term Hertzian fit."""
        res = self.fit(
            epsilon_min=epsilon_min, epsilon_max=epsilon_max, terms=("interior",), **kwargs
        )
        res["Ei_Pa"] = res.get("Ei", 0.0)
        self.results["cytoskeleton"] = res
        return res

    # ------------------------------------------------------------ diagnostics

    def detect_rupture_point(self):
        """
        Locate membrane rupture: the first large *drop* in force after the
        curve has risen appreciably. Falls back to the global force maximum.
        """
        eps, force = self.epsilon, self.force
        result = {"epsilon": float(eps.max()) if eps.size else 0.0,
                  "force": float(force.max()) if force.size else 0.0,
                  "index": int(np.argmax(force)) if force.size else 0,
                  "method": "max-force"}
        if eps.size < 10:
            return self._store_rupture(result)

        smooth = uniform_filter1d(force, size=max(3, eps.size // 40), mode="nearest")
        peak_idx = int(np.argmax(smooth))

        # A rupture is a sustained drop after the peak.
        span = smooth.max() - smooth.min()
        if span > 0 and peak_idx < eps.size - 3:
            after = smooth[peak_idx:]
            drop = (smooth[peak_idx] - after.min()) / span
            if drop > 0.05:
                result = {
                    "epsilon": float(eps[peak_idx]),
                    "force": float(force[peak_idx]),
                    "index": peak_idx,
                    "method": "force-drop",
                }
                return self._store_rupture(result)

        # No drop: the curve is still rising, so there is no rupture in view.
        result["method"] = "no-rupture-detected"
        return self._store_rupture(result)

    def _store_rupture(self, result):
        self.results["rupture"] = result
        return result

    def auto_detect_elastic_range(self, noise_sigma=3.0, max_epsilon=0.35):
        """
        Suggest a fitting window from the data itself.

        e_min: where the force first rises above the pre-contact noise floor.
        e_max: the rupture point (or the largest usable e), capped.
        """
        eps, force = self.epsilon, self.force
        if eps.size < 8:
            return self._store_range(0.01, min(0.3, float(eps.max()) if eps.size else 0.3), None)

        # Noise floor from the lowest-e tenth of the curve.
        n_base = max(5, eps.size // 10)
        base = force[:n_base]
        floor = float(np.mean(base) + noise_sigma * (np.std(base) or 1e-15))

        above = np.where(force > floor)[0]
        eps_min = float(eps[above[0]]) if above.size else float(np.percentile(eps, 5))
        eps_min = float(np.clip(eps_min, 0.005, 0.15))

        rupture = self.detect_rupture_point()
        eps_max = rupture["epsilon"]
        if rupture["method"] == "force-drop":
            eps_max *= 0.95  # stay just below the rupture
        eps_max = float(min(eps_max, max_epsilon, eps.max()))

        # Guarantee a usable span.
        if eps_max <= eps_min * 1.5:
            eps_max = float(min(max(eps_min * 3.0, 0.1), eps.max()))

        return self._store_range(eps_min, eps_max, rupture)

    def _store_range(self, eps_min, eps_max, rupture):
        n = int(((self.epsilon >= eps_min) & (self.epsilon <= eps_max)).sum())
        out = {
            "elastic_epsilon_min": eps_min,
            "elastic_epsilon_max": eps_max,
            "n_points": n,
            "rupture_point": rupture["epsilon"] if rupture else None,
            "rupture_method": rupture["method"] if rupture else None,
            "recommendation": f"Fit over e in [{eps_min:.3f}, {eps_max:.3f}] ({n} points)",
        }
        self.results["auto_range"] = out
        return out

    def range_sensitivity(self, epsilon_min, epsilon_max, n_trials=7, **fit_kwargs):
        """
        Refit over a family of shrinking upper bounds to show how much the
        answer depends on the chosen window. Large spread means the Em/Ei
        split is not identifiable from this curve, not that the fit is buggy.

        Returns
        -------
        dict with per-trial rows and the relative spread of Em and Ei.
        """
        rows = []
        uppers = np.linspace(epsilon_min + 0.6 * (epsilon_max - epsilon_min), epsilon_max, n_trials)
        for upper in uppers:
            r = self.fit(epsilon_min=epsilon_min, epsilon_max=float(upper), **fit_kwargs)
            if r.get("success"):
                rows.append(
                    {
                        "epsilon_max": float(upper),
                        "n_points": r["n_points"],
                        "Em_MPa": r["Em_MPa"],
                        "Ei_kPa": r["Ei_kPa"],
                        "r_squared": r["r_squared"],
                    }
                )

        def spread(key):
            vals = np.array([r[key] for r in rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2 or np.mean(vals) == 0:
                return float("nan")
            return float(np.ptp(vals) / abs(np.mean(vals)))

        out = {
            "trials": rows,
            "Em_relative_spread": spread("Em_MPa"),
            "Ei_relative_spread": spread("Ei_kPa"),
        }
        self.results["range_sensitivity"] = out
        return out

    def get_summary(self):
        """All stored results."""
        return self.results


# ---------------------------------------------------------------------------
#  Choosing between the couplings
# ---------------------------------------------------------------------------


def _aicc(ss_res, n, k):
    """Small-sample corrected Akaike information criterion."""
    if n <= 0 or ss_res <= 0 or n - k - 1 <= 0:
        return float("nan")
    return n * np.log(ss_res / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)


def compare_couplings(
    model,
    epsilon_min,
    epsilon_max,
    terms=("membrane", "interior"),
    n_folds=5,
    seed=0,
):
    """
    Fit every coupling on the same window and report which the data supports.

    This is model selection, not proof. Two criteria are computed because they
    fail in different ways:

    * AICc balances fit against parameter count on the data used for fitting.
      It is quick but assumes the residuals are independent and Gaussian,
      which force curves only roughly satisfy.
    * K-fold cross-validation refits on part of the curve and measures error
      on the part held out. It makes almost no assumptions and directly asks
      which model predicts data it has not seen, which is the question that
      matters when the couplings fit the fitted region about equally well.

    When the two disagree, trust the cross-validation. When the top models are
    within a couple of AICc units of each other, the honest answer is that
    this curve does not distinguish them, and the returned verdict says so.

    Returns
    -------
    dict
        ``candidates`` (one row per coupling, ranked), ``best``, ``verdict``
        and the raw fit of each.
    """
    rng = np.random.default_rng(seed)
    eps_all, force_all, _ = model._select(epsilon_min, epsilon_max)
    n = eps_all.size
    if n < 12:
        return {"success": False, "error": f"Only {n} points in the window; need at least 12 to compare couplings."}

    def fit_for(coupling, order=None, lo=epsilon_min, hi=epsilon_max):
        if coupling == "parallel":
            return model.fit(lo, hi, terms=terms)
        if coupling == "series":
            return model.fit_series(lo, hi, terms=terms)
        scan = model.scan_crossover(lo, hi, terms=terms, order=order)
        return scan["best"] if scan.get("success") else {"success": False, "error": "hybrid scan failed"}

    candidates = {
        "parallel": {"label": "Parallel (forces add)", "order": None},
        "series": {"label": "Series (deformations add)", "order": None},
        "hybrid_ps": {"label": "Parallel below, series above", "order": "parallel-then-series"},
        "hybrid_sp": {"label": "Series below, parallel above", "order": "series-then-parallel"},
    }

    # K-fold indices, shared across candidates so the comparison is paired.
    order_idx = rng.permutation(n)
    folds = np.array_split(order_idx, min(n_folds, n))

    rows = []
    fits = {}
    for key, meta in candidates.items():
        coupling = "parallel" if key == "parallel" else ("series" if key == "series" else "hybrid")
        result = fit_for(coupling, meta["order"])
        if not result.get("success"):
            continue
        fits[key] = result

        k = int(result.get("n_params") or len(terms))
        ss_res = float(result.get("ss_res") or (result["rmse"] ** 2 * result["n_points"]))

        # Cross-validation: refit on the training folds, score the held-out one.
        cv_errors = []
        for fold in folds:
            if fold.size == 0 or fold.size >= n - 2:
                continue
            keep = np.ones(n, dtype=bool)
            keep[fold] = False
            sub = LulevichModel(
                force_all[keep], eps_all[keep], model.cell_height,
                cell_radius=model.R0, membrane_thickness=model.h_membrane,
                poisson_membrane=model.nu_m, poisson_interior=model.nu_i,
                nucleus_radius=model.R_nucleus, poisson_nucleus=model.nu_n,
                nucleus_onset=model.nucleus_onset,
            )
            trained = (
                sub.fit(epsilon_min, epsilon_max, terms=terms)
                if coupling == "parallel"
                else sub.fit_series(epsilon_min, epsilon_max, terms=terms)
                if coupling == "series"
                else sub.fit_hybrid(
                    epsilon_min, epsilon_max, result.get("crossover"),
                    terms=terms, order=meta["order"],
                )
            )
            if not trained.get("success"):
                continue
            params = (trained.get("Em", 0.0), trained.get("Ei", 0.0), trained.get("En", 0.0))
            params = tuple(0.0 if not np.isfinite(v) else v for v in params)
            predicted = model.predict(
                eps_all[fold], params, coupling,
                trained.get("crossover", result.get("crossover")), meta["order"] or "parallel-then-series",
            )
            cv_errors.append(float(np.sqrt(np.mean((force_all[fold] - predicted) ** 2))))

        rows.append(
            {
                "coupling": key,
                "label": meta["label"],
                "r_squared": float(result["r_squared"]),
                "rmse": float(result["rmse"]),
                "n_params": k,
                "aicc": _aicc(ss_res, n, k),
                "cv_rmse": float(np.mean(cv_errors)) if cv_errors else float("nan"),
                "Em_MPa": result.get("Em_MPa"),
                "Ei_kPa": result.get("Ei_kPa"),
                "En_kPa": result.get("En_kPa"),
                "crossover": result.get("crossover"),
            }
        )

    if not rows:
        return {"success": False, "error": "No coupling produced a usable fit."}

    finite = [r["aicc"] for r in rows if np.isfinite(r["aicc"])]
    best_aicc = min(finite) if finite else float("nan")
    for row in rows:
        row["delta_aicc"] = row["aicc"] - best_aicc if np.isfinite(row["aicc"]) else float("nan")
    # Akaike weights: the relative likelihood of each model given the data.
    weights = np.array([np.exp(-0.5 * r["delta_aicc"]) if np.isfinite(r["delta_aicc"]) else 0.0
                        for r in rows])
    total = weights.sum()
    for row, weight in zip(rows, weights):
        row["weight"] = float(weight / total) if total > 0 else float("nan")

    rows.sort(key=lambda r: (np.inf if not np.isfinite(r["aicc"]) else r["aicc"]))
    best = rows[0]

    cv_rows = [r for r in rows if np.isfinite(r["cv_rmse"])]
    cv_best = min(cv_rows, key=lambda r: r["cv_rmse"]) if cv_rows else None

    runner_up = rows[1] if len(rows) > 1 else None
    if runner_up is not None and np.isfinite(runner_up["delta_aicc"]) and runner_up["delta_aicc"] < 2:
        verdict = (
            f"{best['label']} fits best, but {runner_up['label']} is within "
            f"{runner_up['delta_aicc']:.1f} AICc of it. This curve does not "
            f"distinguish them; pick on physical grounds, not on this number."
        )
    elif cv_best is not None and cv_best["coupling"] != best["coupling"]:
        verdict = (
            f"AICc prefers {best['label']} but cross-validation prefers "
            f"{cv_best['label']}. They disagree, which usually means the extra "
            f"structure is fitting noise; the cross-validated choice is the safer one."
        )
    else:
        verdict = (
            f"{best['label']} is preferred, ahead of the next by "
            f"{runner_up['delta_aicc']:.1f} AICc."
            if runner_up is not None and np.isfinite(runner_up["delta_aicc"])
            else f"{best['label']} is preferred."
        )

    return {
        "success": True,
        "candidates": rows,
        "best": best,
        "best_by_cv": cv_best,
        "fits": fits,
        "verdict": verdict,
        "n_points": int(n),
    }
