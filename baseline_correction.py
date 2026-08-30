"""
Baseline detection and correction for AFM force curves
Supports flat baselines and linear (slanted) baseline fitting
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from typing import Tuple, Dict


class BaselineCorrector:
    """Detect and correct baselines in force curves"""

    def __init__(self, deflection_data: np.ndarray, z_data: np.ndarray):
        """
        Initialize with raw AFM data

        Parameters:
        -----------
        deflection_data : array
            Raw cantilever deflection values
        z_data : array
            Z-piezo position values
        """
        self.deflection = np.array(deflection_data, dtype=float)
        self.z_position = np.array(z_data, dtype=float)
        self.baseline_start = 0
        self.baseline_end = len(deflection_data)
        self.baseline_offset = 0
        self.baseline_slope = 0

    def auto_detect_baseline(self, method='flat') -> Dict:
        """
        Automatically detect baseline region

        Parameters:
        -----------
        method : str
            'flat' - assumes flat baseline (constant)
            'linear' - fit linear baseline (slope + offset)

        Returns:
        --------
        dict with baseline parameters
        """
        if method == 'flat':
            return self._detect_flat_baseline()
        elif method == 'linear':
            return self._detect_linear_baseline()
        else:
            return self._detect_flat_baseline()

    def _detect_flat_baseline(self) -> Dict:
        """
        Detect flat baseline by finding regions with minimal variation
        Usually the beginning (before contact) or end (after retraction)
        """
        # Calculate local variation (rolling standard deviation)
        window_size = max(5, len(self.deflection) // 50)
        variation = np.array([
            np.std(self.deflection[max(0, i-window_size):i+window_size])
            for i in range(len(self.deflection))
        ])

        # Find regions with low variation (baseline candidates)
        low_var_indices = np.where(variation < np.percentile(variation, 20))[0]

        if len(low_var_indices) > 0:
            # Use the beginning region as baseline
            baseline_region = low_var_indices[:min(len(low_var_indices) // 3, 50)]
            self.baseline_offset = np.mean(self.deflection[baseline_region])
            self.baseline_slope = 0

            return {
                'method': 'flat',
                'offset': self.baseline_offset,
                'slope': 0,
                'baseline_region': baseline_region,
                'quality': 'auto-detected'
            }
        else:
            # Fallback: use first 10% as baseline
            baseline_pts = len(self.deflection) // 10
            self.baseline_offset = np.mean(self.deflection[:baseline_pts])
            self.baseline_slope = 0
            return {
                'method': 'flat',
                'offset': self.baseline_offset,
                'slope': 0,
                'quality': 'fallback'
            }

    def _detect_linear_baseline(self) -> Dict:
        """
        Detect linear (slanted) baseline by fitting line to flat regions
        """
        # First find flat region
        window_size = max(5, len(self.deflection) // 50)
        variation = np.array([
            np.std(self.deflection[max(0, i-window_size):i+window_size])
            for i in range(len(self.deflection))
        ])

        # Get low-variation points (likely baseline)
        low_var_threshold = np.percentile(variation, 30)
        baseline_indices = np.where(variation < low_var_threshold)[0]

        if len(baseline_indices) > 5:
            # Fit line to these points
            z_baseline = self.z_position[baseline_indices]
            def_baseline = self.deflection[baseline_indices]

            # Linear fit: def = slope * z + offset
            coeffs = np.polyfit(z_baseline, def_baseline, 1)
            self.baseline_slope = coeffs[0]
            self.baseline_offset = coeffs[1]

            return {
                'method': 'linear',
                'offset': self.baseline_offset,
                'slope': self.baseline_slope,
                'baseline_indices': baseline_indices,
                'quality': 'auto-detected'
            }
        else:
            # Fallback to flat
            return self._detect_flat_baseline()

    def correct_baseline(self, manual_offset: float = 0,
                        manual_slope: float = 0) -> np.ndarray:
        """
        Apply baseline correction

        Parameters:
        -----------
        manual_offset : float
            Manual offset adjustment (overrides auto-detected)
        manual_slope : float
            Manual slope adjustment (for linear baseline)

        Returns:
        --------
        corrected_deflection : array
        """
        # Use manual values if provided (non-zero), otherwise use auto-detected
        offset = manual_offset if manual_offset != 0 else self.baseline_offset
        slope = manual_slope if manual_slope != 0 else self.baseline_slope

        # Apply correction
        corrected = self.deflection - (offset + slope * self.z_position)

        return corrected

    def get_baseline_curve(self) -> np.ndarray:
        """Get the baseline curve for visualization"""
        baseline = self.baseline_offset + self.baseline_slope * self.z_position
        return baseline

    def estimate_contact_point(self, deflection_corrected: np.ndarray) -> int:
        """
        Estimate contact point (where deflection becomes significantly non-zero)

        Returns:
        --------
        contact_index : int
        """
        # Find where deflection deviates from baseline
        threshold = np.percentile(np.abs(deflection_corrected), 10)

        contact_indices = np.where(np.abs(deflection_corrected) > threshold)[0]

        if len(contact_indices) > 0:
            return contact_indices[0]
        else:
            return len(deflection_corrected) // 2


def calculate_relative_deformation(z_position: np.ndarray,
                                   contact_index: int,
                                   cell_height: float) -> np.ndarray:
    """
    Calculate relative deformation from Z position

    Parameters:
    -----------
    z_position : array
        Z-piezo position (in same units as cell_height)
    contact_index : int
        Index where contact occurs
    cell_height : float
        Initial cell height (in same units as z_position)

    Returns:
    --------
    relative_deformation : array
        Dimensionless deformation (0-1)
    """
    # Z displacement from contact point
    z_contact = z_position[contact_index]
    z_displacement = z_contact - z_position

    # Relative deformation
    rel_def = z_displacement / cell_height

    # Clip to reasonable range
    rel_def = np.maximum(rel_def, 0)

    return rel_def
