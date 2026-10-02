"""
Wrappers for pygkyl-level simulation visualisation (2-D, 1-D, poloidal).
"""

import numpy as np


def plot_2D(scan, delta=None, kappa=None, energy_srcCORE=None,
            field='phi', frame_idx=500, cut_coords=None, cut_dir='xy',
            **kwargs):
    """Plot a 2-D field cut from a specific simulation.

    Parameters
    ----------
    scan : ScanMetadata
    delta, kappa, energy_srcCORE : float
    field : str
    frame_idx : int
    cut_coords : list, optional
    cut_dir : str
    **kwargs : passed to ``simulation.plot_2D``
    """
    simulation, frames, _, _ = scan.setup_simulation(
        delta=delta, kappa=kappa, energy_srcCORE=energy_srcCORE
    )
    fidx = np.argmin(np.abs(np.array(frames) - frame_idx))
    simulation.plot_2D(field_name=field, cut_coords=cut_coords,
                       cut_dir=cut_dir, frame_idx=fidx, **kwargs)


def plot_1D(scan, delta=None, kappa=None, energy_srcCORE=None,
            field='phi', frame_indices=None, cut_dir='x',
            cut_coords=None, space_time=False, **kwargs):
    """Plot 1-D field evolution from a specific simulation."""
    if frame_indices is None:
        frame_indices = [500]
    if cut_coords is None:
        cut_coords = [0.0, 0.0]
    simulation, frames, _, _ = scan.setup_simulation(
        delta=delta, kappa=kappa, energy_srcCORE=energy_srcCORE
    )
    simulation.plot_1D_time_evolution(cut_dir=cut_dir, cut_coords=cut_coords,
                                     field_name=field,
                                     frame_indices=frame_indices,
                                     space_time=space_time, **kwargs)


def plot_poloidal_projection(scan, delta=None, kappa=None, energy_srcCORE=None,
                             field_name='phi', frame_idx=None, out_file_name='',
                             nzInterp=32, colorMap='inferno', colorScale='lin',
                             showInset=True, showLCFS=True, xlim=None, ylim=None,
                             clim=None, logScaleFloor=1e-3, figout=None,
                             close_fig=False, fig_dpi=300, showAxis=True,
                             cutoutLimiter=False):
    """Plot poloidal cross-section from a specific simulation."""
    if xlim is None:
        xlim = []
    if ylim is None:
        ylim = []
    if clim is None:
        clim = []
    if figout is None:
        figout = []
    simulation, frames, _, _ = scan.setup_simulation(
        delta=delta, kappa=kappa, energy_srcCORE=energy_srcCORE
    )
    if frame_idx is None:
        frame_idx = len(frames) - 1
    simulation.plot_poloidal_projection(
        field_name=field_name, frame_idx=frame_idx, out_file_name=out_file_name,
        nzInterp=nzInterp, colorMap=colorMap, colorScale=colorScale,
        showInset=showInset, showLCFS=showLCFS, xlim=xlim, ylim=ylim, clim=clim,
        logScaleFloor=logScaleFloor, figout=figout, close_fig=close_fig,
        fig_dpi=fig_dpi, showAxis=showAxis, cutoutLimiter=cutoutLimiter)


def get_volume_integral(scan, delta=None, kappa=None, energy_srcCORE=None,
                        fieldName='WkinM2', frame_idx=500,
                        jacob_squared=False, average=False,
                        integral_bounds=None):
    """Compute a volume integral of a field."""
    if integral_bounds is None:
        integral_bounds = [None, None, None]
    simulation, frames, _, _ = scan.setup_simulation(
        delta=delta, kappa=kappa, energy_srcCORE=energy_srcCORE
    )
    frame = simulation.get_frame(fieldName, frame_idx, load=True)
    return frame.compute_volume_integral(
        jacob_squared=jacob_squared, average=average,
        integral_bounds=integral_bounds)
