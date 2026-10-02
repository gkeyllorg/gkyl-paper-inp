"""
Gather sub-package: simulation setup, field extraction, diagnostics, and
parallel runner for collecting scan metadata.
"""

from .simulation import setup_simulation, get_field_values, get_integrated_mom
from .diagnostics import get_average_dt, get_heatflux_width
from .runner import process_scan, run_gather

__all__ = [
    'setup_simulation', 'get_field_values', 'get_integrated_mom',
    'get_average_dt', 'get_heatflux_width',
    'process_scan', 'run_gather',
]
