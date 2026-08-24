import numpy as np
from scipy.interpolate import RegularGridInterpolator

class HydrogenEffectiveRates:
    def __init__(self):
        # ---------------------------------------------------------
        # Grid definitions (Internal units: Te in eV, ne in cm^-3)
        # ---------------------------------------------------------
        self.Te_grid = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        self.ne_grid = np.array([1e8, 1e10, 1e12, 1e14, 1e16])
        
        # Logarithmic grids for interpolation
        self.log_Te = np.log10(self.Te_grid)
        self.log_ne = np.log10(self.ne_grid)

        # ---------------------------------------------------------
        # Reaction 13: e + H2+ --> H + H+ 
        # Approximate log10(rate in cm^3/s) digitized from graph
        # Axis 0: Te, Axis 1: ne
        # ---------------------------------------------------------
        rate13_log_data = np.array([
            [-8.7, -8.6, -8.5, -7.7, -7.4],  # Te = 0.1
            [-8.3, -8.0, -7.6, -7.2, -7.1],  # Te = 1.0
            [-6.9, -6.9, -6.9, -6.8, -6.8],  # Te = 10.0
            [-7.0, -6.9, -6.9, -6.8, -6.8],  # Te = 100.0
            [-7.1, -7.1, -7.0, -7.0, -7.0]   # Te = 1000.0
        ])
        
        # ---------------------------------------------------------
        # Reaction 14: e + H2+ --> H+ + H+ + 2e
        # Approximate log10(rate in cm^3/s) digitized from graph
        # Note: Drops off sharply below 2 eV. Padded with low values.
        # ---------------------------------------------------------
        rate14_log_data = np.array([
            [-15.0, -15.0, -15.0, -15.0, -15.0], # Te = 0.1 (Extrapolated cutoff)
            [-12.0, -12.0, -12.0, -12.0, -11.0], # Te = 1.0
            [-9.0,  -8.8,  -8.7,  -8.4,  -8.1],  # Te = 10.0
            [-8.1,  -8.0,  -8.0,  -7.7,  -7.5],  # Te = 100.0
            [-8.1,  -8.1,  -8.0,  -7.8,  -7.6]   # Te = 1000.0
        ])

        # Create the 2D interpolators
        self._interp13 = RegularGridInterpolator((self.log_Te, self.log_ne), rate13_log_data, 
                                                 bounds_error=False, fill_value=None)
        self._interp14 = RegularGridInterpolator((self.log_Te, self.log_ne), rate14_log_data, 
                                                 bounds_error=False, fill_value=None)

    def _get_rate(self, interpolator, ne_m3, Te_eV):
        """Internal method to handle units and log interpolation."""
        # Convert input density from m^-3 to cm^-3
        ne_cm3 = ne_m3 * 1e-6
        
        # Prevent log(0) or negative inputs
        ne_cm3 = np.maximum(ne_cm3, 1e8) 
        Te_eV = np.maximum(Te_eV, 0.1)
        
        pts = np.array([np.log10(Te_eV), np.log10(ne_cm3)]).T
        log_rate_cm3_s = interpolator(pts)
        
        # Convert rate from cm^3/s to m^3/s
        rate_m3_s = (10**log_rate_cm3_s) * 1e-6
        return rate_m3_s

    def reaction_13(self, ne, Te):
        """
        Effective Dissociative Exc. Rate Coeff. e + H2+ --> H + H+
        Inputs: ne [m^-3], Te [eV]
        Returns: Reaction rate [m^3/s]
        """
        return self._get_rate(self._interp13, ne, Te)

    def reaction_14(self, ne, Te):
        """
        Effective Dissociation Rate H2+ --> H+ + H+ + 2e
        Inputs: ne [m^-3], Te [eV]
        Returns: Reaction rate [m^3/s]
        """
        return self._get_rate(self._interp14, ne, Te)

# Example usage:
rates = HydrogenEffectiveRates()
