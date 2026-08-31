"""
Lulevich et al. 2006 Cell Mechanics Model
Implementation for C2C12 muscle cell compression analysis
Advanced fitting with automatic range detection and multi-method validation
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class LulevichModel:
    """
    Lulevich 2006 model for single-cell compression analysis with advanced fitting.

    References:
    Lulevich et al. (2006). Cell Mechanics Using Atomic Force Microscopy-Based
    Single-Cell Compression. Langmuir, 22(19), 8151-8155.
    """

    def __init__(self, force, relative_deformation, cell_height, cell_radius=None):
        """
        Initialize model with experimental data.

        Parameters:
        -----------
        force : array-like
            Force in Newtons
        relative_deformation : array-like
            Relative deformation (epsilon), dimensionless (0-1)
        cell_height : float
            Initial cell height in meters
        cell_radius : float, optional
            Cell radius in meters. If None, estimated from cell_height
        """
        self.force = np.array(force, dtype=float)
        self.epsilon = np.array(relative_deformation, dtype=float)
        self.cell_height = cell_height

        # Estimate cell radius from height if not provided
        if cell_radius is None:
            # C2C12 cells are typically slightly wider than tall
            self.R0 = cell_height * 0.55
        else:
            self.R0 = cell_radius

        # Physical constants
        self.h_membrane = 4e-9  # Membrane thickness (4 nm)
        self.poisson_ratio = 0.5  # Incompressible for living cells

        self.results = {}

    def balloon_model_cubic(self, epsilon, Em):
        """
        Equation 3: Lulevich balloon model for living cells (small deformation)

        F ≈ (2π Em h R0 ε³) / (1 - νm)

        Parameters:
        -----------
        epsilon : float or array
            Relative deformation
        Em : float
            Membrane Young's modulus (Pa)

        Returns:
        --------
        Force in Newtons
        """
        numerator = 2 * np.pi * Em * self.h_membrane * self.R0
        denominator = 1 - self.poisson_ratio
        return (numerator / denominator) * epsilon**3

    def hertzian_contact_model(self, epsilon, Ei):
        """
        Equation 6: Hertzian contact model for dead/fixed cells or cytoskeleton

        Fi = (√2 Ei R0^(1/2) ε^(3/2)) / (3(1 - νi²))

        Parameters:
        -----------
        epsilon : float or array
            Relative deformation
        Ei : float
            Interior (cytoskeleton) Young's modulus (Pa)

        Returns:
        --------
        Force in Newtons
        """
        numerator = np.sqrt(2) * Ei * self.R0**0.5
        denominator = 3 * (1 - self.poisson_ratio**2)
        return (numerator / denominator) * epsilon**1.5

    def _estimate_initial_guess_membrane(self, eps_fit, force_fit):
        """Estimate good initial guess for Em based on data."""
        if len(eps_fit) < 3:
            return 1e6
        # Use a data point to estimate
        idx = len(eps_fit) // 2
        eps_val = eps_fit[idx]
        force_val = force_fit[idx]
        if eps_val > 0:
            denominator = 1 - self.poisson_ratio
            numerator = 2 * np.pi * self.h_membrane * self.R0
            Em_guess = (force_val * denominator) / (numerator * eps_val**3)
            return np.clip(Em_guess, 1e3, 1e9)
        return 1e6

    def _estimate_initial_guess_cyto(self, eps_fit, force_fit):
        """Estimate good initial guess for Ei based on data."""
        if len(eps_fit) < 3:
            return 1e3
        idx = len(eps_fit) // 2
        eps_val = eps_fit[idx]
        force_val = force_fit[idx]
        if eps_val > 0:
            denominator = 3 * (1 - self.poisson_ratio**2)
            numerator = np.sqrt(2) * self.R0**0.5
            Ei_guess = (force_val * denominator) / (numerator * eps_val**1.5)
            return np.clip(Ei_guess, 1, 1e9)
        return 1e3

    def fit_membrane_elasticity(self, epsilon_max=0.3, epsilon_min=0.02):
        """
        Fit balloon model to elastic region with advanced range detection.

        Parameters:
        -----------
        epsilon_max : float
            Upper limit for fitting (default 0.3)
        epsilon_min : float
            Lower limit for fitting (avoid noise at zero)

        Returns:
        --------
        dict with fitting results
        """
        # Select data in fitting range
        mask = (self.epsilon >= epsilon_min) & (self.epsilon <= epsilon_max)
        eps_fit = self.epsilon[mask]
        force_fit = self.force[mask]

        if len(eps_fit) < 3:
            return {'success': False, 'error': 'Not enough data points in selected range', 'Em_MPa': 0}

        try:
            # Estimate better initial guess
            p0_guess = self._estimate_initial_guess_membrane(eps_fit, force_fit)

            # Try fitting with estimated bounds first
            try:
                popt, pcov = curve_fit(
                    self.balloon_model_cubic,
                    eps_fit,
                    force_fit,
                    p0=[p0_guess],
                    bounds=(1e3, 1e9),  # 1 kPa to 1 GPa
                    maxfev=5000
                )
                Em = popt[0]
            except:
                # If bounds don't work, try without bounds
                popt, pcov = curve_fit(
                    self.balloon_model_cubic,
                    eps_fit,
                    force_fit,
                    p0=[p0_guess],
                    maxfev=5000
                )
                Em = popt[0]
                # Ensure reasonable value
                if Em <= 0:
                    return {'success': False, 'error': 'Fitting failed to converge', 'Em_MPa': 0}

            # Calculate residuals and R²
            force_pred = self.balloon_model_cubic(eps_fit, Em)
            residuals = force_fit - force_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((force_fit - np.mean(force_fit))**2)

            # Prevent division by zero
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = 0

            # Calculate bending constant
            Km = (Em * self.h_membrane**3) / (12 * (1 - self.poisson_ratio**2))
            Km_kT = Km / (1.38e-23 * 300)

            self.results['membrane'] = {
                'Em': Em,
                'Em_MPa': Em / 1e6,
                'Km': Km,
                'Km_kT': Km_kT,
                'epsilon_range': [epsilon_min, epsilon_max],
                'r_squared': r_squared,
                'n_points': len(eps_fit),
                'residual_std': np.std(residuals),
                'success': True
            }

            return self.results['membrane']

        except Exception as e:
            return {'success': False, 'error': str(e), 'Em_MPa': 0}

    def fit_cytoskeleton_elasticity(self, epsilon_max=0.3, epsilon_min=0.05):
        """
        Fit Hertzian contact model with advanced range detection.

        Parameters:
        -----------
        epsilon_max : float
            Upper limit for fitting
        epsilon_min : float
            Lower limit for fitting

        Returns:
        --------
        dict with fitting results
        """
        # Select data in fitting range
        mask = (self.epsilon >= epsilon_min) & (self.epsilon <= epsilon_max)
        eps_fit = self.epsilon[mask]
        force_fit = self.force[mask]

        if len(eps_fit) < 3:
            return {'success': False, 'error': 'Not enough data points in selected range', 'Ei_kPa': 0}

        try:
            # Estimate better initial guess
            p0_guess = self._estimate_initial_guess_cyto(eps_fit, force_fit)

            # Try fitting with estimated bounds first
            try:
                popt, pcov = curve_fit(
                    self.hertzian_contact_model,
                    eps_fit,
                    force_fit,
                    p0=[p0_guess],
                    bounds=(1, 1e9),  # 1 Pa to 1 GPa
                    maxfev=5000
                )
                Ei = popt[0]
            except:
                # If bounds don't work, try without bounds
                popt, pcov = curve_fit(
                    self.hertzian_contact_model,
                    eps_fit,
                    force_fit,
                    p0=[p0_guess],
                    maxfev=5000
                )
                Ei = popt[0]
                # Ensure reasonable value
                if Ei <= 0:
                    return {'success': False, 'error': 'Fitting failed to converge', 'Ei_kPa': 0}

            # Calculate residuals and R²
            force_pred = self.hertzian_contact_model(eps_fit, Ei)
            residuals = force_fit - force_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((force_fit - np.mean(force_fit))**2)

            # Prevent division by zero
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = 0

            self.results['cytoskeleton'] = {
                'Ei': Ei,
                'Ei_kPa': Ei / 1e3,
                'Ei_Pa': Ei,
                'epsilon_range': [epsilon_min, epsilon_max],
                'r_squared': r_squared,
                'n_points': len(eps_fit),
                'residual_std': np.std(residuals),
                'success': True
            }

            return self.results['cytoskeleton']

        except Exception as e:
            return {'success': False, 'error': str(e), 'Ei_kPa': 0}

    def auto_detect_elastic_range(self):
        """
        Automatically detect optimal elastic fitting range with improved logic.
        Uses multiple detection methods and selects best range.

        Returns:
        --------
        dict with suggested ranges
        """
        # Method 1: Rupture detection
        rupture = self.detect_rupture_point()
        rupture_eps = rupture['epsilon']

        # Method 2: Linear region detection (where cubic model should work)
        # Look for where the force curve is most cubic-like
        dF_deps = np.gradient(self.force, self.epsilon)

        # Find the region with most consistent curvature
        # Start from small deformation and go up
        max_eps = min(0.3, rupture_eps * 0.8)
        min_eps = 0.01

        # Find good upper bound by looking for where linearity breaks
        # Use the point where second derivative becomes too large
        d2F_deps2 = np.gradient(dF_deps, self.epsilon)

        # Find indices in reasonable range
        valid_mask = (self.epsilon >= min_eps) & (self.epsilon <= max_eps)
        valid_eps = self.epsilon[valid_mask]
        valid_d2 = d2F_deps2[valid_mask]

        if len(valid_d2) > 0:
            # Use 75% point as upper bound (good balance)
            idx_75 = int(len(valid_eps) * 0.75)
            if idx_75 < len(valid_eps):
                epsilon_max = valid_eps[idx_75]
            else:
                epsilon_max = max_eps
        else:
            epsilon_max = max_eps

        # Ensure minimum spacing
        epsilon_max = max(0.1, epsilon_max)
        epsilon_min = 0.01

        self.results['auto_range'] = {
            'elastic_epsilon_min': epsilon_min,
            'elastic_epsilon_max': epsilon_max,
            'rupture_point': rupture_eps,
            'recommendation': f'Fit membrane model for ε ∈ [{epsilon_min:.4f}, {epsilon_max:.4f}]'
        }

        return self.results['auto_range']

    def detect_rupture_point(self):
        """
        Automatically detect cell membrane rupture point by analyzing
        force curve for stress peaks and discontinuities.

        Returns:
        --------
        dict with rupture analysis
        """
        # Calculate force derivative (dF/dε)
        dF_deps = np.gradient(self.force, self.epsilon)

        # Find peaks in second derivative (inflection points = stiffness changes)
        d2F_deps2 = np.gradient(dF_deps, self.epsilon)

        # Smooth to reduce noise
        window_size = max(3, len(self.epsilon) // 20)
        if window_size % 2 == 0:
            window_size += 1

        from scipy.ndimage import uniform_filter1d
        d2F_smooth = uniform_filter1d(d2F_deps2, size=window_size, mode='nearest')

        # Find peaks (stress points)
        max_height = np.max(np.abs(d2F_smooth))
        if max_height > 0:
            peaks, properties = find_peaks(np.abs(d2F_smooth), height=max_height * 0.1)
        else:
            peaks = []

        if len(peaks) > 0:
            rupture_idx = peaks[0]  # First major peak
            rupture_epsilon = self.epsilon[rupture_idx]
            rupture_force = self.force[rupture_idx]
        else:
            # If no peaks found, use point where slope changes significantly
            slopes = dF_deps
            rupture_idx = np.argmax(np.abs(np.gradient(slopes)))
            rupture_epsilon = self.epsilon[rupture_idx]
            rupture_force = self.force[rupture_idx]

        # Ensure we have a valid rupture point
        if rupture_epsilon == 0 or rupture_epsilon < 0.05:
            rupture_epsilon = min(0.25, np.percentile(self.epsilon, 75))
            rupture_idx = np.argmin(np.abs(self.epsilon - rupture_epsilon))
            rupture_force = self.force[rupture_idx]

        self.results['rupture'] = {
            'epsilon': rupture_epsilon,
            'force': rupture_force,
            'index': rupture_idx,
            'n_peaks_detected': len(peaks)
        }

        return self.results['rupture']

    def combined_model(self, epsilon, Em, Ei):
        """
        Combined Lulevich model: membrane (cubic) + cytoskeleton (Hertzian)

        F_total = (2π Em h R0 ε³) / (1 - νm) + (√2 Ei R0^(1/2) ε^(3/2)) / (3(1 - νi²))

        Parameters:
        -----------
        epsilon : float or array
            Relative deformation
        Em : float
            Membrane Young's modulus (Pa)
        Ei : float
            Cytoskeleton Young's modulus (Pa)

        Returns:
        --------
        Total force in Newtons
        """
        membrane_term = self.balloon_model_cubic(epsilon, Em)
        cytoskeleton_term = self.hertzian_contact_model(epsilon, Ei)
        return membrane_term + cytoskeleton_term

    def fit_combined_elasticity(self, epsilon_max=0.3, epsilon_min=0.01):
        """
        Fit COMBINED membrane + cytoskeleton model (two-term Lulevich fit).
        This is the proper two-parameter fit that extracts both Em and Ei simultaneously.

        Parameters:
        -----------
        epsilon_max : float
            Upper limit for fitting (default 0.3)
        epsilon_min : float
            Lower limit for fitting (default 0.01)

        Returns:
        --------
        dict with fitting results for both Em and Ei
        """
        # Select data in fitting range
        mask = (self.epsilon >= epsilon_min) & (self.epsilon <= epsilon_max)
        eps_fit = self.epsilon[mask]
        force_fit = self.force[mask]

        if len(eps_fit) < 4:  # Need at least 4 points for 2 parameters
            return {'success': False, 'error': 'Not enough data points for 2-parameter fit', 'Em_MPa': 0, 'Ei_kPa': 0}

        try:
            # Estimate initial guesses
            p0_Em = self._estimate_initial_guess_membrane(eps_fit, force_fit)
            p0_Ei = self._estimate_initial_guess_cyto(eps_fit, force_fit)

            # Try fitting with bounds
            try:
                popt, pcov = curve_fit(
                    self.combined_model,
                    eps_fit,
                    force_fit,
                    p0=[p0_Em, p0_Ei],
                    bounds=([1e3, 1], [1e9, 1e9]),  # Em: 1kPa-1GPa, Ei: 1Pa-1GPa
                    maxfev=5000
                )
                Em, Ei = popt[0], popt[1]
            except:
                # Try without bounds
                popt, pcov = curve_fit(
                    self.combined_model,
                    eps_fit,
                    force_fit,
                    p0=[p0_Em, p0_Ei],
                    maxfev=5000
                )
                Em, Ei = popt[0], popt[1]

            # Ensure reasonable values
            if Em <= 0 or Ei <= 0:
                return {'success': False, 'error': 'Fitting resulted in non-positive moduli', 'Em_MPa': 0, 'Ei_kPa': 0}

            # Calculate residuals and R²
            force_pred = self.combined_model(eps_fit, Em, Ei)
            residuals = force_fit - force_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((force_fit - np.mean(force_fit))**2)

            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = 0

            # Calculate bending constant
            Km = (Em * self.h_membrane**3) / (12 * (1 - self.poisson_ratio**2))
            Km_kT = Km / (1.38e-23 * 300)

            self.results['combined'] = {
                'Em': Em,
                'Em_MPa': Em / 1e6,
                'Ei': Ei,
                'Ei_kPa': Ei / 1e3,
                'Km': Km,
                'Km_kT': Km_kT,
                'epsilon_range': [epsilon_min, epsilon_max],
                'r_squared': r_squared,
                'n_points': len(eps_fit),
                'residual_std': np.std(residuals),
                'success': True
            }

            return self.results['combined']

        except Exception as e:
            return {'success': False, 'error': str(e), 'Em_MPa': 0, 'Ei_kPa': 0}

    def get_summary(self):
        """
        Get summary of all analysis results.

        Returns:
        --------
        dict with all results
        """
        return self.results
