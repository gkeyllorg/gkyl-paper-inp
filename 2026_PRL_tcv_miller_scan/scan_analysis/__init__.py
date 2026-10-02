"""
scan_analysis — modular Gkeyll scan metadata analysis toolkit.

Quick start::

    from scan_analysis import ScanMetadata

    scan = ScanMetadata('data/tcv_miller_scan_big_metadata_frame_500_navg_25.h5')
    scan.info()
    scan.plot_contour_grid(['Ti_core'], fixed_params={'energy_srcCORE': 1e6})

To gather new metadata::

    from scan_analysis.gather import run_gather
    run_gather(scandir='tcv_miller_scan_big', ncpu=4)

Or via CLI::

    python -m scan_analysis.gather.runner --ncpu 4
"""

from .scan_metadata import ScanMetadata
from .utils.calculate_arc_length import calculate_arc_length, find_bounce_angle, get_theta_from_R

__all__ = ['ScanMetadata', 'calculate_arc_length', 'find_bounce_angle', 'get_theta_from_R']
