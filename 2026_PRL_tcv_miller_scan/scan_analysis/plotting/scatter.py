"""
Field-vs-field scatter plot with shape-encoded markers.

Each scan point is drawn with a shape that encodes the power level (triangle,
square, hexagon, ellipse), colour that encodes triangularity (delta), and
aspect ratio that encodes elongation (kappa).
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon
from matplotlib.colors import Normalize

from .. import config
from ..fields import base_field, FIELD_UNITS


MARKER_STYLES = ['triangle_up', 'rectangle', 'hexagon', 'ellipse']
LEGEND_MARKERS = ['^', 's', 'H', 'o']


def _make_patch(style, cx, cy, w, h, color, edgecolor, alpha):
    """Create a matplotlib patch of the given style."""
    if style == 'ellipse':
        return Ellipse((cx, cy), width=w, height=h,
                       facecolor=color, edgecolor=edgecolor,
                       linewidth=0.5, alpha=alpha, zorder=3)
    elif style == 'rectangle':
        return FancyBboxPatch((cx - w / 2, cy - h / 2), width=w, height=h,
                              boxstyle='square,pad=0',
                              facecolor=color, edgecolor=edgecolor,
                              linewidth=0.5, alpha=alpha, zorder=3)
    elif style == 'triangle_up':
        verts = np.array([[cx - w / 2, cy - h / 2],
                          [cx + w / 2, cy - h / 2],
                          [cx, cy + h / 2]])
        return Polygon(verts, closed=True,
                       facecolor=color, edgecolor=edgecolor,
                       linewidth=0.5, alpha=alpha, zorder=3)
    elif style == 'triangle_down':
        verts = np.array([[cx - w / 2, cy + h / 2],
                          [cx + w / 2, cy + h / 2],
                          [cx, cy - h / 2]])
        return Polygon(verts, closed=True,
                       facecolor=color, edgecolor=edgecolor,
                       linewidth=0.5, alpha=alpha, zorder=3)
    elif style == 'pentagon':
        verts = np.array([[cx, cy + h / 2],
                          [cx + w / 2, cy + h / 4],
                          [cx + w / 3, cy - h / 2],
                          [cx - w / 3, cy - h / 2],
                          [cx - w / 2, cy + h / 4]])
        return Polygon(verts, closed=True,
                       facecolor=color, edgecolor=edgecolor,
                       linewidth=0.5, alpha=alpha, zorder=3)
    elif style == 'hexagon':
        verts = np.array([[cx - w / 2, cy],
                          [cx - w / 4, cy + h / 2],
                          [cx + w / 4, cy + h / 2],
                          [cx + w / 2, cy],
                          [cx + w / 4, cy - h / 2],
                          [cx - w / 4, cy - h / 2]])
        return Polygon(verts, closed=True,
                       facecolor=color, edgecolor=edgecolor,
                       linewidth=0.5, alpha=alpha, zorder=3)
    else:
        raise ValueError(f"Unknown marker style: {style}")


def plot_field_vs_field(scan, field_x, field_y,
                        powers=None, alpha=0.8, cmap='coolwarm',
                        marker_size=None, figfilename=None, dpi=300,
                        xlim=None, ylim=None, axis_equal=False,
                        show_fig=True, annotate=False,
                        lines=None, shadows=None, edgecolor='None'):
    """Scatter-plot of one field against another across power levels.

    Parameters
    ----------
    scan : ScanMetadata
        Must expose ``.data``, ``.scan_params``, ``.scan_keys``,
        ``.all_field_symbols``, ``.field_units``.
    field_x, field_y : str
    powers : list[float], optional
    alpha, cmap, marker_size, figfilename, dpi, xlim, ylim : various
    axis_equal, show_fig, annotate : bool
    lines : list[dict], optional
        ``[{'x': [...], 'y': [...], 'kwargs': {...}}, ...]``
    shadows : list[dict], optional
        ``[{'x': [...], 'y': [lo, hi], 'kwargs': {...}}, ...]``
    edgecolor : str, optional
        Edge color for markers. Default 'None' (no edge).
    """
    if lines is None:
        lines = []
    if shadows is None:
        shadows = []

    for f in (field_x, field_y):
        if f not in scan.data:
            raise ValueError(f"Field '{f}' not found. Available: {sorted(scan.data.keys())}")

    if powers is None:
        powers = scan.scan_params.get('energy_srcCORE', [])
    if len(powers) > 4:
        raise ValueError(f"'powers' supports at most 4 values, got {len(powers)}.")

    all_powers = scan.scan_params.get('energy_srcCORE', [])
    for p in powers:
        if p not in all_powers:
            raise ValueError(f"Power {p} not in scan. Available: {all_powers}")

    delta_vals = np.array(scan.scan_params['delta'])
    kappa_vals = np.array(scan.scan_params['kappa'])
    kappa_min = kappa_vals.min()

    delta_norm = Normalize(vmin=delta_vals.min(), vmax=delta_vals.max())
    colormap = plt.get_cmap(cmap)

    # Compute data ranges
    all_x, all_y = [], []
    for p in powers:
        sl = scan._get_slices(p)
        all_x.append(scan.data[field_x][sl].flatten())
        all_y.append(scan.data[field_y][sl].flatten())
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_range = np.ptp(all_x) if np.ptp(all_x) > 0 else 1.0
    y_range = np.ptp(all_y) if np.ptp(all_y) > 0 else 1.0
    if xlim is not None:
        x_range = xlim[1] - xlim[0]
    if ylim is not None:
        y_range = ylim[1] - ylim[0]
    if marker_size is None:
        marker_size = x_range / (len(delta_vals) * 3.5)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_facecolor('white')
    ax.grid(False)

    free_keys = [k for k in scan.scan_keys if k != 'energy_srcCORE']
    free_vals = [scan.scan_params[k] for k in free_keys]
    free_combos = list(itertools.product(*free_vals))

    for p_idx, power in enumerate(powers):
        sl = scan._get_slices(power)
        flat_x = scan.data[field_x][sl].flatten()
        flat_y = scan.data[field_y][sl].flatten()
        style = MARKER_STYLES[p_idx]

        for i, combo in enumerate(free_combos):
            combo_dict = dict(zip(free_keys, combo))
            kappa = combo_dict['kappa']
            delta = combo_dict['delta']
            color = colormap(delta_norm(delta))
            aspect = (kappa / kappa_min) ** 2
            w = marker_size
            h = marker_size * aspect * (y_range / x_range)
            cx, cy = flat_x[i], flat_y[i]

            ax.add_patch(
                _make_patch(style, cx, cy, w, h, color, edgecolor, alpha)
            )
            if annotate:
                ax.annotate(f'{delta:.2f}/{kappa:.2f}', (cx, cy),
                            fontsize=6, ha='center', va='center', zorder=4)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=delta_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=r'$\delta$')

    # Legend
    handles = [
        plt.Line2D([0], [0], marker=LEGEND_MARKERS[i], linestyle='None',
                   markerfacecolor='lightgray', markeredgecolor='k',
                   markersize=10, label=f'{p / 1e6:.2f} MW')
        for i, p in enumerate(powers)
    ]
    ax.legend(handles=handles, loc='best', fontsize=10)

    # Axis labels
    fx_sym = scan.all_field_symbols.get(field_x, field_x)
    fy_sym = scan.all_field_symbols.get(field_y, field_y)
    ax.set_xlabel(fx_sym + ' ' + scan.field_units.get(base_field(field_x), ''))
    ax.set_ylabel(fy_sym + ' ' + scan.field_units.get(base_field(field_y), ''))

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(all_x.min() - x_range * 0.1, all_x.max() + x_range * 0.1)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(all_y.min() - y_range * 0.1, all_y.max() + y_range * 0.1)

    if axis_equal:
        ax.set_aspect('equal', adjustable='datalim')

    for line in lines:
        ax.plot(line['x'], line['y'], **line['kwargs'])
    for shadow in shadows:
        ax.fill_between(shadow['x'], shadow['y'][0], shadow['y'][1],
                        **shadow['kwargs'])

    plt.tight_layout()

    if figfilename is not None:
        fig.savefig(figfilename, dpi=dpi)
        print(f"Figure saved to {figfilename}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)
