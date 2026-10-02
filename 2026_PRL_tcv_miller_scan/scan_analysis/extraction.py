"""
Data extraction helpers: slicing pre-loaded arrays along scan parameter axes.
"""

import numpy as np
from typing import Dict, List, Optional, Any

from . import config


def get_sim_index(params, scan_keys, combinations):
    """Map a parameter dict to its index in the scan combination list.

    Parameters
    ----------
    params : dict
        e.g. ``{'kappa': 1.5, 'delta': 0.3, 'energy_srcCORE': 1e6}``
    scan_keys : list[str]
    combinations : list[tuple]

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If the parameter combination is not found.
    """
    key = tuple(params[k] for k in scan_keys)
    try:
        return combinations.index(key)
    except ValueError:
        raise ValueError(
            f"Parameters {params} not found in scan combinations.\n"
            f"Available scan keys: {scan_keys}"
        )

def get_multi_dim_index(params, scan_params, scan_keys):
    """Get the multi-dimensional index tuple for the given parameter combination.

    Parameters
    ----------
    params : dict
        e.g. ``{'kappa': 1.5, 'delta': 0.3, 'energy_srcCORE': 1e6}``
    scan_params : dict[str, list]
        e.g. ``{'kappa': [1.4, 1.5, 1.6], 'delta': [0.3, 0.45], 'energy_srcCORE': [1e5, 1e6]}``
    scan_keys : list[str]
        e.g. ``['kappa', 'delta', 'energy_srcCORE']``

    Returns
    -------
    tuple of ints

    Raises
    ------
    ValueError
        If any parameter value is not found in its corresponding scan axis.
    """
    try:
        return tuple(scan_params[k].index(params[k]) for k in scan_keys)
    except ValueError as e:
        raise ValueError(
            f"Parameter value not found: {e}\n"
            f"Parameters: {params}\n"
            f"Scan keys: {scan_keys}\n"
            f"Scan params: {scan_params}"
        ) from e

def extract_field_data(data, field_names, scan_params, scan_keys,
                       fixed_params=None, vary_params=None):
    """Extract 2-D slices of pre-loaded data for the requested fields.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        The pre-loaded data arrays (``scan.data``).
    field_names : list[str]
        Fields to extract (e.g. ``['Ti_core', 'Ti_core_lcfs']``).
    scan_params : dict[str, list]
        ``{param: sorted_values}`` describing the scan grid.
    scan_keys : list[str]
        Ordered parameter names matching array axes.
    fixed_params : dict, optional
        Parameters to hold constant (e.g. ``{'energy_srcCORE': 1e6}``).
    vary_params : list[str], optional
        The two parameters forming the 2-D slice (auto-detected if None).

    Returns
    -------
    dict
        - For each field: 2-D array (y × x)
        - ``x_param``, ``y_param``: parameter names
        - ``x_vals``, ``y_vals``: 2-D meshgrids
    """
    if fixed_params is None:
        fixed_params = {}
    if vary_params is None:
        vary_params = [p for p in scan_keys if p not in fixed_params]
    if len(vary_params) != 2:
        raise ValueError(
            f"Need exactly 2 varying parameters, got {len(vary_params)}: {vary_params}. "
            f"Available: {scan_keys}"
        )

    # Prefer delta as x-axis
    if 'delta' in vary_params:
        x_param = 'delta'
        y_param = [p for p in vary_params if p != 'delta'][0]
    else:
        x_param, y_param = vary_params

    # Build index slices
    slices = []
    for param in scan_keys:
        if param in fixed_params:
            slices.append(scan_params[param].index(fixed_params[param]))
        else:
            slices.append(slice(None))

    result = {}
    for field in field_names:
        if field not in data:
            print(f"Warning: Field '{field}' not found in data, skipping")
            continue

        arr = data[field][tuple(slices)]

        x_axis = scan_keys.index(x_param)
        y_axis = scan_keys.index(y_param)
        axes_before_x = sum(1 for i, s in enumerate(slices[:x_axis]) if isinstance(s, int))
        axes_before_y = sum(1 for i, s in enumerate(slices[:y_axis]) if isinstance(s, int))
        nx = x_axis - axes_before_x
        ny = y_axis - axes_before_y

        if nx < ny:
            arr = np.moveaxis(arr, [nx, ny], [1, 0])
        else:
            arr = np.moveaxis(arr, [ny, nx], [0, 1])

        result[field] = arr

    x_vals = scan_params[x_param]
    y_vals = scan_params[y_param]
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='xy')

    result['x_param'] = x_param
    result['y_param'] = y_param
    result['x_vals'] = x_mesh
    result['y_vals'] = y_mesh
    result[x_param + 's'] = x_mesh
    result[y_param + 's'] = y_mesh
    return result


def get_power_slices(scan_params, scan_keys, power_val):
    """Return index tuple that selects a fixed power and all other axes."""
    return tuple(
        scan_params[k].index(power_val) if k == 'energy_srcCORE' else slice(None)
        for k in scan_keys
    )
