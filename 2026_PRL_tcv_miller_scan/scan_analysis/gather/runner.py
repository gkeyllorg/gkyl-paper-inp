"""
Parallel gather runner and CLI entry point.

Usage (as a script)::

    python -m scan_analysis.gather.runner --ncpu 4 --scandir tcv_miller_scan_big

Or programmatically::

    from scan_analysis.gather.runner import run_gather
    metadata = run_gather(scan_arrays=..., scandir='...', frame_idx=500)
"""

import itertools
import time
import argparse
from multiprocessing import Pool

import pygkyl
import numpy as np

from .. import config
from ..loaders import save_metadata_h5
from .simulation import setup_simulation, get_field_values, get_integrated_mom
from .diagnostics import get_average_dt, get_heatflux_width


def _build_output_filename(scandir, frame_idx, frame_navg, filename_ext=None):
    """Construct the output filename from scan parameters."""
    name = f'{scandir}_metadata'
    if frame_idx >= 0:
        name += f'_frame_{frame_idx}'
    else:
        name += '_lastframes'
    if frame_navg > 1:
        name += f'_navg_{frame_navg}'
    if filename_ext:
        name += f'_{filename_ext}'
    return name


def process_scan(scanidx, *,
                 combinations, scandir, scanname,
                 scan_arrays, nscan,
                 frame_idx, frame_navg,
                 locations=None, fields=None, filters=None,
                 intmom_names=None, process_id=0):
    """Process one simulation index and return its metadata dict (or None).

    Parameters
    ----------
    scanidx : int
    combinations : list[tuple]
    scandir, scanname : str
    scan_arrays : dict
    nscan : int
    frame_idx, frame_navg : int
    locations, fields, filters : dict/list/dict, optional
    intmom_names : list[str], optional
    process_id : int
        Used to stagger I/O starts.

    Returns
    -------
    dict or None
    """
    if locations is None:
        locations = config.LOCATIONS
    if fields is None:
        fields = config.GATHER_FIELDS
    if filters is None:
        filters = config.FIELD_FILTERS
    if intmom_names is None:
        intmom_names = config.INTMOM_NAMES

    time.sleep(process_id * 0.1)

    simdir = f'{scandir}/{scanname}_{scanidx:05d}/'
    fileprefix = f'{scanname}_{scanidx:05d}'
    combo = combinations[scanidx]
    scan_keys = list(scan_arrays.keys())

    print(f"Processing scan {scanidx+1}/{nscan} "
          f"({', '.join(f'{k}={combo[i]}' for i, k in enumerate(scan_keys))})...")

    try:
        simulation = setup_simulation(simdir, fileprefix)

        sim_frames = pygkyl.file_utils.find_available_frames(simulation, 'field')
        if len(sim_frames) == 0:
            raise ValueError(f"No frames found for simulation {scanidx}")

        last_frame = frame_idx if frame_idx >= 0 else sim_frames[-1]
        first_frame = max(0, last_frame - frame_navg)

        if last_frame not in sim_frames:
            raise ValueError(
                f"Requested frame {last_frame} not available. "
                f"Available: {sim_frames[0]}–{sim_frames[-1]}"
            )

        frame_array = [f for f in range(first_frame, last_frame + 1) if f in sim_frames]
        if not frame_array:
            raise ValueError(f"No valid frames in range {first_frame}–{last_frame}")

        field_values, tend = get_field_values(simulation, frame_array,
                                              locations=locations,
                                              fields=fields,
                                              filters=filters)

        intmom_dict = get_integrated_mom(simulation, flux_names=intmom_names)

        avg_dt = get_average_dt(
            f'{scandir}/logs/std-{scanname}_{scanidx:05d}.log'
        )

        lambda_q = get_heatflux_width(simulation)

        data = {
            'simdir': simdir,
            'scanidx': scanidx,
            'tend': tend,
            'avg_dt': avg_dt,
            'vol_frac': simulation.geom_param.vol_frac,
            'intmom': intmom_dict,
            'lambda_q': lambda_q,
            **{k: combo[i] for i, k in enumerate(scan_keys)},
            **field_values,
        }

        print(
            f"{scanidx+1}/{nscan}: "
            + ", ".join(f"{k}={combo[i]}" for i, k in enumerate(scan_keys))
            + f", Ti={field_values.get('Ti_lcfs', 0):.1f}"
            + f", Te={field_values.get('Te_lcfs', 0):.1f}"
            + f", ne={field_values.get('ne_lcfs', 0):.1e}"
            + f", t={tend:.1f}, dt={avg_dt:.3e}"
        )
        return data

    except Exception as e:
        print(f"Error processing scan {scanidx}: {e}")
        return None


def _process_scan_wrapper(args_tuple):
    """Wrapper for ``multiprocessing.Pool.map`` (takes a single tuple)."""
    scanidx, kwargs = args_tuple
    return process_scan(scanidx, **kwargs)


def run_gather(scan_arrays=None, scandir=None, scanname=None,
               frame_idx=None, frame_navg=None,
               ncpu=4, chunksize=1, filename_ext=None):
    """Run the full gather pipeline (sequential or parallel).

    Parameters
    ----------
    scan_arrays : dict, optional
        Override ``config.DEFAULT_SCAN_ARRAYS``.
    scandir : str, optional
    scanname : str, optional
    frame_idx : int, optional
    frame_navg : int, optional
    ncpu : int
    chunksize : int
    filename_ext : str, optional

    Returns
    -------
    list[dict]
        Successfully gathered metadata entries.
    """
    if scan_arrays is None:
        scan_arrays = config.DEFAULT_SCAN_ARRAYS
    if scandir is None:
        scandir = config.DEFAULT_SCANDIR
    if scanname is None:
        scanname = scandir
    if frame_idx is None:
        frame_idx = config.DEFAULT_FRAME_IDX
    if frame_navg is None:
        frame_navg = config.DEFAULT_FRAME_NAVG

    combinations = list(itertools.product(*scan_arrays.values()))
    nscan = len(combinations)
    num_processes = min(ncpu, nscan)

    shared_kwargs = dict(
        combinations=combinations,
        scandir=scandir,
        scanname=scanname,
        scan_arrays=scan_arrays,
        nscan=nscan,
        frame_idx=frame_idx,
        frame_navg=frame_navg,
    )

    print(f"Processing {nscan} simulations using {num_processes} process(es) "
          f"(chunksize={chunksize})...")

    start = time.time()

    if num_processes == 1:
        metadata = []
        for i in range(nscan):
            result = process_scan(i, process_id=0, **shared_kwargs)
            if result is not None:
                metadata.append(result)
    else:
        args = [(i, {**shared_kwargs, 'process_id': i % num_processes})
                for i in range(nscan)]
        with Pool(processes=num_processes) as pool:
            raw = pool.map(_process_scan_wrapper, args, chunksize=chunksize)
        metadata = [m for m in raw if m is not None]

    elapsed = time.time() - start

    # Save
    outname = _build_output_filename(scandir, frame_idx, frame_navg, filename_ext)
    save_metadata_h5(outname + '.h5', metadata)

    print(f"\nMetadata saved to {outname}.h5")
    print(f"Successfully processed {len(metadata)}/{nscan} simulations")
    print(f"Total time: {elapsed:.1f}s "
          f"({elapsed / max(len(metadata), 1):.2f}s per simulation)")

    return metadata


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gather Gkeyll scan metadata in parallel'
    )
    parser.add_argument('--ncpu', type=int, default=4)
    parser.add_argument('--chunksize', type=int, default=1)
    parser.add_argument('--scandir', type=str, default=None)
    parser.add_argument('--frame-idx', type=int, default=None)
    parser.add_argument('--frame-navg', type=int, default=None)
    parser.add_argument('--ext', type=str, default=None,
                        help='Extra suffix for output filename')
    args = parser.parse_args()

    run_gather(
        scandir=args.scandir,
        frame_idx=args.frame_idx,
        frame_navg=args.frame_navg,
        ncpu=args.ncpu,
        chunksize=args.chunksize,
        filename_ext=args.ext,
    )


if __name__ == '__main__':
    main()
