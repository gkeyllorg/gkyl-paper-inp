"""
Simulation setup and field-value extraction for the gather pipeline.
"""

import copy
import numpy as np
import pygkyl

from .. import config


def setup_simulation(simdir, fileprefix, config_name=None):
    """Create a pygkyl simulation object with normalisation settings.

    Parameters
    ----------
    simdir : str
        Path to the simulation directory.
    fileprefix : str
        File prefix for the simulation output files.
    config_name : str, optional
        pygkyl configuration name (default: ``config.SIM_CONFIG_NAME``).

    Returns
    -------
    simulation
        Configured pygkyl simulation object.
    """
    if config_name is None:
        config_name = config.SIM_CONFIG_NAME
    simulation = pygkyl.load_sim_config(
        configName=config_name, simDir=simdir, filePrefix=fileprefix
    )
    for key, value in config.NORM_SETTINGS.items():
        simulation.normalization.set(key, value)
    return simulation


def get_field_values(simulation, frame_array, locations=None, fields=None,
                     filters=None):
    """Extract time-averaged field values at the requested spatial locations.

    Parameters
    ----------
    simulation : pygkyl simulation
    frame_array : list[int]
        Frames to average over.
    locations : dict, optional
        ``{name: r_over_a}`` mapping  (default: ``config.LOCATIONS``).
    fields : list[str], optional
        Field names to extract  (default: ``config.GATHER_FIELDS``).
    filters : dict, optional
        ``{field: {min, max}}`` for NaN-clamping  (default: ``config.FIELD_FILTERS``).

    Returns
    -------
    results : dict
        ``{field_loc: averaged_value}``
    tend : float
        Simulation time of the last loaded frame.
    """
    if locations is None:
        locations = config.LOCATIONS
    if fields is None:
        fields = config.GATHER_FIELDS
    if filters is None:
        filters = config.FIELD_FILTERS

    results = {f'{field}_{loc}': 0.0 for field in fields for loc in locations}
    ntake = 0

    for fidx in frame_array:
        frames = {field: simulation.get_frame(field, fidx) for field in fields}
        for field in fields:
            for loc_name, x_val in locations.items():
                key = f'{field}_{loc_name}'
                frame_copy = copy.deepcopy(frames[field])
                if loc_name == 'limup':
                    frame_copy.slice('scalar', [x_val, 'avg', -1])
                elif loc_name == 'limlo':
                    frame_copy.slice('scalar', [x_val, 'avg', 0])
                else:
                    frame_copy.slice('scalar', [x_val, 'avg', 0.0])
                results[key] += frame_copy.values[0, 0, 0]
        ntake += 1

    # Average
    for key in results:
        results[key] /= ntake

    # Filter out-of-range values
    for field in fields:
        filt = filters.get(field, {})
        vmin = filt.get('min', -np.inf)
        vmax = filt.get('max', np.inf)
        for loc_name in locations:
            key = f'{field}_{loc_name}'
            if results[key] < vmin or results[key] > vmax:
                results[key] = float('nan')

    return results, frames[fields[0]].time


def get_integrated_mom(simulation, flux_names=None, dt_samp=2.0):
    """Extract and subsample integrated-moment time series.

    Parameters
    ----------
    simulation : pygkyl simulation
    flux_names : list[str], optional
        Names of moment diagnostics (default: ``config.INTMOM_NAMES``).
    dt_samp : float
        Subsampling interval in microseconds (default: 2.0).

    Returns
    -------
    dict
        ``{name: {time, values, tunits, vunits, name}}``
    """
    if flux_names is None:
        flux_names = config.INTMOM_NAMES

    mom_dict = {}
    for flux in flux_names:
        intmom = pygkyl.IntegratedMoment(
            simulation=simulation, name=flux, load=True, ddt=False
        )
        tmax = intmom.time[-1]
        time_sub = np.arange(0, tmax + dt_samp, dt_samp)
        vals_sub = np.interp(time_sub, intmom.time, intmom.values)
        mom_dict[flux] = {
            'time': time_sub,
            'values': vals_sub,
            'tunits': intmom.tunits,
            'vunits': intmom.vunits,
            'name': intmom.fluxname,
        }
    return mom_dict
