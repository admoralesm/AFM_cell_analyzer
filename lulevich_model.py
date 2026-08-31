"""
Lulevich et al. 2006 cell-compression model.

Reference
---------
Lulevich, V., Zink, T., Chen, H.-Y., Liu, F.-T., Liu, G.-Y. (2006).
"Cell Mechanics Using Atomic Force Microscopy-Based Single-Cell Compression."
Langmuir 22(19), 8151-8155.

Physical model
--------------
Total force during whole-cell compression is the sum of a membrane
(balloon) term and an interior/cytoskeleton (Hertzian) term:

    F(e) = Am * Em * e^3  +  Ai * Ei * e^(3/2)

with the geometry prefactors

    Am = 2 * pi * h_m * R0 / (1 - nu_m)
    Ai = sqrt(2) * R0^(1/2) * h0^(3/2) / (3 * (1 - nu_i^2))

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
F is LINEAR in Em and Ei. There is no need for a non-linear optimiser, an
initial guess, or a convergence check: the fit is a bounded linear least
squares problem with a closed-form normal-equation solution. This makes the
result deterministic, guess-independent, and impossible to "fail to
converge". Uncertainties come from the analytic covariance matrix.
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


class LulevichModel:
    """Two-term Lulevich fit for single-cell compression curves."""

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

    def balloon_model_cubic(self, epsilon, Em):
        """Membrane (balloon) term, force in newtons."""
        return self.Am * Em * np.asarray(epsilon, dtype=float) ** 3

    def hertzian_contact_model(self, epsilon, Ei):
        """Interior (Hertzian) term, force in newtons."""
        eps = np.clip(np.asarray(epsilon, dtype=float), 0.0, None)
        return self.Ai * Ei * eps ** 1.5

    def combined_model(self, epsilon, Em, Ei, force_offset=0.0):
        """Full two-term model, force in newtons."""
        return (
            self.balloon_model_cubic(epsilon, Em)
            + self.hertzian_contact_model(epsilon, Ei)
            + force_offset
        )

    # ---------------------------------------------------------------- fitting

    def _select(self, epsilon_min, epsilon_max):
        mask = (self.epsilon >= epsilon_min) & (self.epsilon <= epsilon_max)
        return self.epsilon[mask], self.force[mask], mask

    def _design_matrix(self, eps, terms, fit_offset):
        cols, names = [], []
        if "membrane" in terms:
            cols.append(self.Am * eps ** 3)
            names.append("Em")
        if "interior" in terms:
            cols.append(self.Ai * np.clip(eps, 0.0, None) ** 1.5)
            names.append("Ei")
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

        Returns
        -------
        dict
            Fit results, diagnostics, and uncertainties. ``success`` is False
            only when the input data is unusable (too few points, degenerate).
        """
        eps, force, mask = self._select(epsilon_min, epsilon_max)
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

        Em = float(p.get("Em", 0.0))
        Ei = float(p.get("Ei", 0.0))
        F0 = float(p.get("F0", 0.0))

        # Predictions and goodness of fit (unweighted, so R^2 is comparable
        # across weighting choices).
        pred = X @ params
        residuals = force - pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((force - force.mean()) ** 2))
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
        f_mem = self.balloon_model_cubic(e_top, Em)
        f_int = self.hertzian_contact_model(e_top, Ei)
        f_tot = f_mem + f_int
        membrane_fraction = float(f_mem / f_tot) if f_tot > 0 else float("nan")

        Km = Em * self.h_membrane ** 3 / (12.0 * (1.0 - self.nu_m ** 2))

        warnings_list = self._warnings(Em, Ei, names, cond, r_squared, eps.size, membrane_fraction)

        out = {
            "success": True,
            "Em": Em,
            "Ei": Ei,
            "Em_MPa": Em / 1e6,
            "Ei_kPa": Ei / 1e3,
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
            "weighting": weighting,
            "fit_offset": bool(fit_offset),
            "condition_number": cond,
            "corr_Em_Ei": corr_EmEi,
            "membrane_fraction_at_max": membrane_fraction,
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

    def _warnings(self, Em, Ei, names, cond, r2, n_points, membrane_fraction):
        w = []
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
        if Em > 0 and not (PLAUSIBLE_EM_PA[0] <= Em <= PLAUSIBLE_EM_PA[1]):
            w.append(
                f"Em = {Em/1e6:.3g} MPa is outside the usual 0.001-1000 MPa "
                f"range. Check cell height, radius and force units."
            )
        if Ei > 0 and not (PLAUSIBLE_EI_PA[0] <= Ei <= PLAUSIBLE_EI_PA[1]):
            w.append(
                f"Ei = {Ei/1e3:.3g} kPa is outside the usual 0.001-10000 kPa "
                f"range. Check cell height, radius and force units."
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
