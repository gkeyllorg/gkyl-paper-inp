"""
Post-processing diagnostics extracted from simulation output files.
"""

import re
import numpy as np
from scipy.optimize import curve_fit


def get_average_dt(logfile):
    """Compute the mean time-step from a Gkeyll log file.

    Parameters
    ----------
    logfile : str
        Path to the simulation log file.

    Returns
    -------
    float
        Average dt (0.0 if the file is missing or contains no dt lines).
    """
    dt_values = []
    try:
        with open(logfile, 'r') as f:
            for line in f:
                if 'dt = ' in line:
                    match = re.search(r'dt = ([\d.eE+-]+)', line)
                    if match:
                        dt_values.append(float(match.group(1)))
        return sum(dt_values) / len(dt_values) if dt_values else 0.0
    except FileNotFoundError:
        print(f"Log file {logfile} not found. Returning dt=0.0")
        return 0.0


def get_heatflux_width(simulation):
    """Estimate SOL heat-flux decay length lambda_q via exponential fit.

    The parallel heat flux at the last available frame is averaged in the
    binormal direction and evaluated at the upper limiter.  An exponential
    ``A * exp(-x / lambda_q)`` is fitted to the first quarter of the SOL
    domain to obtain the decay length.

    Parameters
    ----------
    simulation : pygkyl simulation

    Returns
    -------
    float
        lambda_q in metres.
    """
    simulation.normalization.reset('x')

    qframe = simulation.get_frame(
        fieldName='qpar',
        timeFrame=simulation.frame_list[-1],
        load=True,
    )
    qframe.slice(axs='x', ccoord=['avg', -1])

    x = np.squeeze(qframe.new_grids[0])
    q = np.abs(np.squeeze(qframe.values))
    lcfs_idx = np.argmin(np.abs(x - 0.04))
    xsol = x[lcfs_idx:] - 0.04
    qsol = q[lcfs_idx:]

    n = len(xsol)

    def _exp_decay(x, A, lam):
        return A * np.exp(-x / lam)

    popt, _ = curve_fit(_exp_decay, xsol[:n // 4], qsol[:n // 4],
                        p0=(qsol[0], 0.005))
    lambda_q = popt[1]
    print(f"Estimated heat flux length scale (lambda_q) = {lambda_q:.4f} m")
    return lambda_q
