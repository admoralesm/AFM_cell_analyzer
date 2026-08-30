"""
Lulevich et al. 2006 Cell Mechanics Model
Implementation for C2C12 muscle cell compression analysis
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class LulevichModel:
    """
    Lulevich 2006 model for single-cell compression analysis.

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

    def fit_membrane_elasticity(self, epsilon_max=0.3, eps_min=0.02):
        """
        Fit balloon model to elastic region to extract membrane Young's modulus.
        Typically valid for small deformations (ε < 0.3) before membrane rupture.

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
            return {'success': False, 'error': 'Not enough data points in selected range'}

        try:
            # Fit to cubic model
            popt, pcov = curve_fit(
                self.balloon_model_cubic,
                eps_fit,
                force_fit,
                p0=[1e6],  # Initial guess: 1 MPa
                bounds=(1e3, 1e9),  # 1 kPa to 1 GPa
                maxfev=10000
            )

            Em = popt[0]

            # Calculate residuals and R²
            force_pred = self.balloon_model_cubic(eps_fit, Em)
            residuals = force_fit - force_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((force_fit - np.mean(force_fit))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Calculate bending constant
            Km = (Em * self.h_membrane**3) / (12 * (1 - self.poisson_ratio**2))
            Km_kT = Km / (1.38e-23 * 300)  # Convert to kT units

            self.results['membrane'] = {
                'Em': Em,
                'Em_MPa': Em / 1e6,
                'Km': Km,
                'Km_kT': Km_kT,
                'epsilon_range': [epsilon_min, epsilon_max],
                'r_squared': r_squared,
                'n_points': len(eps_fit),
                'residual_std': np.std(residuals)
            }

            return self.results['membrane']

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def fit_cytoskeleton_elasticity(self, epsilon_max=0.3, eps_min=0.05):
        """
        Fit Hertzian contact model to extract cytoskeleton Young's modulus.
        Useful for analyzing post-rupture behavior or dead cell data.

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
            return {'success': False, 'error': 'Not enough data points in selected range'}

        try:
            # Fit to Hertzian model
            popt, pcov = curve_fit(
                self.hertzian_contact_model,
                eps_fit,
                force_fit,
                p0=[1e3],  # Initial guess: 1 kPa
                bounds=(1, 1e9),  # 1 Pa to 1 GPa
                maxfev=10000
            )

            Ei = popt[0]

            # Calculate residuals and R²
            force_pred = self.hertzian_contact_model(eps_fit, Ei)
            residuals = force_fit - force_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((force_fit - np.mean(force_fit))**2)
            r_squared = 1 - (ss_res / ss_tot)

            self.results['cytoskeleton'] = {
                'Ei': Ei,
                'Ei_kPa': Ei / 1e3,
                'Ei_Pa': Ei,
                'epsilon_range': [epsilon_min, epsilon_max],
                'r_squared': r_squared,
                'n_points': len(eps_fit),
                'residual_std': np.std(residuals)
            }

            return self.results['cytoskeleton']

        except Exception as e:
            return {'success': False, 'error': str(e)}

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
        d2F_smooth = uniform_filter1d(
            d2F_deps2, size=window_size, mode='nearest')

        # Find peaks (stress points)
        peaks, properties = find_peaks(
            np.abs(d2F_smooth), height=np.max(np.abs(d2F_smooth)) * 0.1)

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

        self.results['rupture'] = {
            'epsilon': rupture_epsilon,
            'force': rupture_force,
            'index': rupture_idx,
            'n_peaks_detected': len(peaks)
        }

        return self.results['rupture']

    def auto_detect_elastic_range(self):
        """
        Automatically detect optimal elastic fitting range.
        Uses curvature analysis and rupture detection.

        Returns:
        --------
        dict with suggested ranges
        """
        # Detect rupture point first
        rupture = self.detect_rupture_point()
        rupture_eps = rupture['epsilon']

        # Suggest elastic range as 0-80% of rupture point
        epsilon_max = max(0.15, rupture_eps * 0.8)
        epsilon_max = min(0.3, epsilon_max)  # Cap at 0.3

        # Minimum should avoid noise
        epsilon_min = 0.02

        self.results['auto_range'] = {
            'elastic_epsilon_min': epsilon_min,
            'elastic_epsilon_max': epsilon_max,
            'rupture_point': rupture_eps,
            'recommendation': f'Fit membrane model for ε ∈ [{epsilon_min:.4f}, {epsilon_max:.4f}]'
        }

        return self.results['auto_range']

    def get_summary(self):
        """
        Get summary of all analysis results.

        Returns:
        --------
        dict with all results
        """
        return self.results
