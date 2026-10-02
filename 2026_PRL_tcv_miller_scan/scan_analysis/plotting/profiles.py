"""
Profile comparison and 1-D profile extraction plots.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

from ..fields import base_field, FIELD_SYMBOLS, FIELD_UNITS
from .. import config


def compare_profiles(scan, vary_param, vary_vals,
                     fixed_params=None, field='Ti',
                     frame_idx=500, cut_coords=None,
                     figname=None, cmap='viridis'):
    """Compare 1-D profiles across a single varying scan parameter.

    Parameters
    ----------
    scan : ScanMetadata
    vary_param : str
        Parameter to vary (e.g. ``'kappa'``).
    vary_vals : list
        Values of the varying parameter.
    fixed_params : dict, optional
    field : str
    frame_idx : int
    cut_coords : list, optional
    figname : str, optional
    cmap : str
    """
    if cut_coords is None:
        cut_coords = ['avg', 0.0]
    if fixed_params is None:
        fixed_params = {}

    vmin, vmax = np.min(vary_vals), np.max(vary_vals)
    param_label = config.SCAN_PARAM_SYMBOLS.get(vary_param, vary_param)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for val in vary_vals:
        sim_params = {**fixed_params, vary_param: val}
        field_pgkyl = field.replace('P', 'p')
        x, y, _, scanidx = scan.get_profile(sim_params, field_pgkyl,
                                             frame_idx=frame_idx,
                                             cut_coords=cut_coords)
        norm_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
        clr = plt.get_cmap(cmap)(norm_val)
        ax.plot(x, y, label=f"{scanidx}", color=clr)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r'$' + param_label + r'$')

    ax.set_xlabel(r'$r/a$')

    bf = base_field(field)
    ylabel = FIELD_SYMBOLS.get(bf, bf)
    if cut_coords[0] == 'avg' and cut_coords[1] == 0.0:
        ylabel = r'\langle ' + ylabel + r'\rangle_y'
    elif cut_coords[0] == 'avg' and cut_coords[1] == 'avg':
        ylabel = r'\langle ' + ylabel + r'\rangle_{y,z}'
    unit = FIELD_UNITS.get(bf, '')
    ax.set_ylabel(r'$' + ylabel + r'$ ' + unit)

    plt.tight_layout()
    if figname is not None:
        fig.savefig(figname, dpi=300)
        print(f"Figure saved to {figname}")
    plt.show()
