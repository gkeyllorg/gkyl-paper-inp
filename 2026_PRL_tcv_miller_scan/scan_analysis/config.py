"""
Central configuration for scan analysis.

All hardcoded values, default parameters, scan arrays, locations, fields,
filters, and normalization settings live here. To add a new field or scan
parameter, update the relevant dictionary in this module.
"""

import numpy as np

# =====================================================================
# Geometry constants
# =====================================================================
RAXIS = 0.87       # Major radius [m]
AMID = 0.24        # Minor radius [m]

# =====================================================================
# Default scan arrays  (override via CLI or direct assignment)
# =====================================================================
DEFAULT_SCAN_ARRAYS = {
    "kappa": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    "delta": [-0.6, -0.45, -0.3, -0.15, 0.15, 0.3, 0.45, 0.6],
    "energy_srcCORE": [0.1e6, 0.5e6, 1.0e6, 5.0e6],
}

# Known scan parameters (used for auto-detection from metadata)
KNOWN_SCAN_PARAMS = ['kappa', 'delta', 'energy_srcCORE']

# LaTeX symbols for scan parameters
SCAN_PARAM_SYMBOLS = {
    'kappa': r'\kappa',
    'delta': r'\delta',
    'energy_srcCORE': r'P_{\text{in}}',
}

# Plot-axis labels for scan parameters
SCAN_PARAM_LABELS = {
    'kappa': r'$\kappa$',
    'delta': r'$\delta$',
    'energy_srcCORE': r'Power [W] (log scale)',
    'nu': r'$\nu$',
    'beta': r'$\beta$',
}

# =====================================================================
# Default scan / gather settings
# =====================================================================
DEFAULT_SCANDIR = 'tcv_miller_scan_big'
DEFAULT_FRAME_IDX = 500
DEFAULT_FRAME_NAVG = 25

# =====================================================================
# Spatial locations  (name -> r/a value)
# =====================================================================
LOCATIONS = {
    'core': 0.85,
    'edge': 0.9,
    'lcfs': 1.0,
    'sol': 1.2,
    'limup': 1.2,
    'limlo': 1.2,
}

LOCATION_SYMBOLS = {
    'core': r'\text{core}',
    'edge': r'\text{edge}',
    'lcfs': r'\text{sep}',
    'sol': r'\text{SOL}',
    'limlo': r'\text{lo}',
    'limup': r'\text{lu}',
}

# =====================================================================
# Fields to extract during gather
# =====================================================================
GATHER_FIELDS = [
    'Ti', 'Te', 'ne', 'phi',
    'hflux_xi', 'hflux_xe',
    'pflux_xi', 'pflux_xe',
]

# =====================================================================
# Field value filters  (used during gather to replace out-of-range with NaN)
# =====================================================================
FIELD_FILTERS = {
    'Ti':       {'min': 0.0,    'max': 2000.0},
    'Te':       {'min': 0.0,    'max': 2000.0},
    'ne':       {'min': 0.0,    'max': 1e25},
    'phi':      {'min': -2000.0, 'max': 2000.0},
    'hflux_xi': {'min': -np.inf, 'max': np.inf},
    'hflux_xe': {'min': -np.inf, 'max': np.inf},
    'pflux_xi': {'min': -np.inf, 'max': np.inf},
    'pflux_xe': {'min': -np.inf, 'max': np.inf},
}

# =====================================================================
# Integrated moments to gather
# =====================================================================
INTMOM_NAMES = [
    'bflux_x_l_ne', 'bflux_x_l_ni', 'bflux_x_l_He', 'bflux_x_l_Hi',
    'bflux_x_u_ne', 'bflux_x_u_ni', 'bflux_x_u_He', 'bflux_x_u_Hi',
    'bflux_z_l_ne', 'bflux_z_l_ni', 'bflux_z_l_He', 'bflux_z_l_Hi',
    'bflux_z_u_ne', 'bflux_z_u_ni', 'bflux_z_u_He', 'bflux_z_u_Hi',
    'ne', 'ni', 'We', 'Wi',
]

# =====================================================================
# Normalization settings for pygkyl simulations
# =====================================================================
NORM_SETTINGS = {
    't': 'mus',
    'x': 'minor radius',
    'y': 'Larmor radius',
    'z': 'pi',
    'fluid velocities': 'thermal velocity',
    'temperatures': 'eV',
    'pressures': 'Pa',
    'energies': 'MJ',
    'current': 'kA',
    'gradients': 'major radius',
}

# Default simulation config name
SIM_CONFIG_NAME = 'tcv_nt'

# Metadata keys that are NOT scan parameters or field values
METADATA_KEYS = {
    'simdir', 'scanidx', 'tend', 'avg_dt', 'frame', 'navg',
    'intmom', 'vol_frac', 'lambda_q',
}
