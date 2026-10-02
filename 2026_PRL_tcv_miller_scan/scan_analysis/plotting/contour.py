"""
Contour-grid plotting for 2-D parameter scans.
"""

import numpy as np
import matplotlib.pyplot as plt

from .. import config
from ..fields import base_field, FIELD_SCALING, FIELD_UNITS


def plot_contour_grid(data, field_list,
                      all_field_symbols, field_units=None, field_scaling=None,
                      suptitle='', method='contourf',
                      cmap='coolwarm', clim=None,
                      deviation=False, show_fig=True,
                      figfilename=None, dpi=300):
    """Plot contour grids for multiple fields on a 2-D parameter plane.

    Parameters
    ----------
    data : dict
        Output of ``extraction.extract_field_data`` (contains field arrays,
        ``x_param``, ``y_param``, ``x_vals``, ``y_vals``).
    field_list : list[str]
        Fields to plot (must be keys in *data*).
    all_field_symbols : dict
        ``{field: LaTeX_label}``
    field_units : dict, optional
    field_scaling : dict, optional
    suptitle : str
    method : str
        ``'contourf'``, ``'imshow'``, ``'scatter'``, or ``'pcolormesh'``.
    cmap : str
    clim : tuple, optional
    deviation : bool
        Plot percentage deviation from mean instead of raw values.
    show_fig : bool
    figfilename : str, optional
    dpi : int
    """
    if field_units is None:
        field_units = FIELD_UNITS
    if field_scaling is None:
        field_scaling = FIELD_SCALING

    x_param = data.get('x_param', 'delta')
    y_param = data.get('y_param', 'kappa')
    x_vals = data.get('x_vals', data.get('deltas'))
    y_vals = data.get('y_vals', data.get('kappas'))

    xlabel = config.SCAN_PARAM_LABELS.get(x_param, x_param)
    ylabel = config.SCAN_PARAM_LABELS.get(y_param, y_param)

    ncols = min(2, len(field_list))
    nrows = (len(field_list) + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axs = np.atleast_1d(axs).flatten()

    for idx, field in enumerate(field_list):
        ax = axs[idx]
        fs = all_field_symbols.get(field, field)

        if field not in data:
            print(f"Warning: Field '{field}' not found in data, skipping")
            ax.set_visible(False)
            continue

        toplot = data[field].copy()

        if deviation:
            ref = np.mean(toplot)
            toplot = (toplot - ref) / ref * 100
            label = r'$(' + fs + r' - v_0)/v_0$ [%]'
            plot_clim = (-50, 50)
        else:
            bf = base_field(field)
            unit = field_units.get(bf, '')
            label = fs + ' ' + unit
            plot_clim = clim
            scale = field_scaling.get(bf, 1.0)
            toplot *= scale

        use_contourf = ('energy_srcCORE' in (x_param, y_param))

        if method == 'contourf' or use_contourf:
            cf = ax.contourf(x_vals, y_vals, toplot, levels=20, cmap=cmap)
            if idx == 0:
                ax.scatter(x_vals, y_vals, s=2, c='w', marker='o')
        elif method == 'imshow':
            extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]
            cf = ax.imshow(toplot, extent=extent, origin='lower',
                           aspect='auto', cmap=cmap)
        elif method == 'scatter':
            cf = ax.scatter(x_vals, y_vals, c=toplot, cmap=cmap, s=600, marker='s')
        else:
            cf = ax.pcolormesh(x_vals, y_vals, toplot, cmap=cmap, shading='auto')

        if plot_clim:
            cf.set_clim(plot_clim)

        plt.colorbar(cf, ax=ax, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if x_param == 'energy_srcCORE':
            ax.set_xscale('log')
        else:
            ax.set_xticks(np.unique(x_vals))
            ax.tick_params(axis='x', rotation=45)

        if y_param == 'energy_srcCORE':
            ax.set_yscale('log')
        else:
            ax.set_yticks(np.unique(y_vals))

    for idx in range(len(field_list), len(axs)):
        axs[idx].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle)

    plt.tight_layout()

    if figfilename is not None:
        fig.savefig(figfilename, dpi=dpi)
        print(f"Figure saved to {figfilename}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)
