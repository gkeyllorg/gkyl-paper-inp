import pygkyl
import itertools
import re
from multiprocessing import Pool, cpu_count
import os
import argparse
import copy
import time
import numpy as np
import h5py
from scipy.optimize import curve_fit

filename_ext = None # Add to the end of the filename.
scan_arrays = {
    "kappa": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    "delta": [-0.6, -0.45, -0.3, -0.15, 0.15, 0.3, 0.45, 0.6],
    "energy_srcCORE": [0.1e6, 0.5e6, 1.0e6, 5.0e6],
}
# scan_arrays = {"kappa": [1.1],"delta": [-0.6], "energy_srcCORE": [0.1e6]}
scandir = 'tcv_miller_scan_big'
# scandir = 'tcv_miller_scan_18x12x12x10x6'
scanname = scandir
frame_idx = -1 # e.g. we take the 100th frame (t=200mus), set to -1 to take the last frame
frame_navg = 50 # number of frames we average the data on.
# Define locations and fields to extract
locations = {'core': 0.85, 'edge': 0.9, 'lcfs': 1.0, 'sol': 1.2, 'limup': 1.2, 'limlo': 1.2}
fields = ['Ti', 'Te', 'ne', 'phi', 'hflux_xi', 'hflux_xe', 'pflux_xi', 'pflux_xe', 'betae']
filter_dict = {
    'Ti': {'min': 0.0, 'max': 2000.0},
    'Te': {'min': 0.0, 'max': 2000.0},
    'ne': {'min': 0.0, 'max': 1e25},
    'phi': {'min': -2000.0, 'max': 2000.0},
    'hflux_xi': {'min': -np.inf, 'max': np.inf},
    'hflux_xe': {'min': -np.inf, 'max': np.inf},
    'pflux_xi': {'min': -np.inf, 'max': np.inf},
    'pflux_xe': {'min': -np.inf, 'max': np.inf},
    'betae': {'min': 0.0, 'max': 1.0},
}
# ==================== end of user input ====================

def setup_simulation(simdir, fileprefix):
    """Setup simulation with normalization settings."""
    simulation = pygkyl.load_sim_config(configName='tcv_nt', simDir=simdir, filePrefix=fileprefix)
    norm_settings = {
        't': 'mus', 'x': 'minor radius', 'y': 'Larmor radius', 'z': 'pi',
        'fluid velocities': 'thermal velocity', 'temperatures': 'eV',
        'pressures': 'Pa', 'energies': 'MJ', 'current': 'kA',
        'gradients': 'major radius'
    }
    for key, value in norm_settings.items():
        simulation.normalization.set(key, value)
    return simulation

def get_field_values(simulation, frame_array, locations, fields):
    """Extract average field values at specified locations."""
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
                results[key] += frame_copy.values[0,0,0]
        ntake += 1
        
    # Average all values
    for key in results:
        results[key] /= ntake    
        
    # Apply filtering
    for field in fields:
        vmin = filter_dict[field].get('min', -np.inf)
        vmax = filter_dict[field].get('max', np.inf)
        for loc_name in locations:
            key = f'{field}_{loc_name}'
            if results[key] < vmin or results[key] > vmax:
                results[key] = float('nan')
                    
    return results, frames[fields[0]].time

def get_integrated_mom(simulation, flux_names):
    mom_dict = {}
    for flux in flux_names:
        intmom = pygkyl.IntegratedMoment(simulation=simulation, name=flux, load=True, ddt=False)
        # take a subset of the time points to reduce data size, 2 point per mus
        tmax = intmom.time[-1]
        dt_samp = 2.0  # microseconds
        time_subsampled = np.arange(0, tmax+dt_samp, dt_samp)
        values_subsampled = np.interp(time_subsampled, intmom.time, intmom.values)
        mom_dict[flux] = {'time': time_subsampled, 'values': values_subsampled, 
                            'tunits': intmom.tunits, 'vunits': intmom.vunits,
                            'name': intmom.fluxname}
    return mom_dict

def get_average_dt(logfile):
    """Calculate average dt from log file."""
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

    simulation.normalization.reset('x')

    qframe = simulation.get_frame(fieldName='qpar', timeFrame=simulation.frame_list[-1], load=True)

    qframe.slice(axs='x', ccoord=['avg', -1])

    x = np.squeeze(qframe.new_grids[0])
    q = np.abs(np.squeeze(qframe.values))
    lcfs_idx = np.argmin(np.abs(x - 0.04))
    xsol = x[lcfs_idx:] - 0.04
    qsol = q[lcfs_idx:]

    # Method 1: Fit an exponential decay to compute the heat flux length scale
    # n = len(xsol)
    # def exp_decay(x, A, lambda_):
    #     return A * np.exp(-x / lambda_)
    # popt, pcov = curve_fit(exp_decay, xsol[:n//4], qsol[:n//4], p0=(qsol[0], 0.005))
    # lambda_q = popt[1]
    
    # # Estimate the error in the fit by looking at the covariance matrix
    # perr = np.sqrt(np.diag(pcov))
    # delta_lambda_q = perr[1]
    
    # print(f"Estimated heat flux length scale (lambda_q) = {lambda_q:.4f} m")
    
    # Method 2: Find the first peak from the left and then find the point where the heat flux drops to 1/e of that peak value
    peak_idx = np.argmax(qsol)
    peak_value = qsol[peak_idx]
    threshold = peak_value / np.e
    below_threshold_indices = np.where(qsol[peak_idx:] < threshold)[0]
    if len(below_threshold_indices) > 0:
        lambda_q = xsol[peak_idx + below_threshold_indices[0]]
    else:
        lambda_q = xsol[-1]  # If it never drops below threshold, take the last point as an estimate
    # Estimate error by looking at the range of x values where q is between threshold and peak value
    above_threshold_indices = np.where(qsol[peak_idx:] > threshold)[0]
    if len(above_threshold_indices) > 0 and len(below_threshold_indices) > 0:
        delta_lambda_q = xsol[peak_idx + above_threshold_indices[-1]] - xsol[peak_idx + below_threshold_indices[0]]
    else:
        delta_lambda_q = 0.0  # If we can't find a valid range, set error to zero

    return lambda_q, delta_lambda_q

filename = f'{scandir}_metadata'
if frame_idx >= 0:
    filename += f'_frame_{frame_idx}'
else:
    filename += '_lastframes'
    
if frame_navg > 1:
    filename += f'_navg_{frame_navg}'

if filename_ext:
    filename += f'_{filename_ext}'
    
values = list(scan_arrays.values())
combinations = list(itertools.product(*values))
nscan = len(combinations)

def process_scan(args_tuple):
    """Process a single scan index - designed for parallel execution."""
    scanidx, process_id = args_tuple
    
    # Stagger process starts to reduce I/O contention
    time.sleep(process_id * 0.1)
    
    simdir = f'{scandir}/{scanname}_{scanidx:05d}/'
    fileprefix = f'{scanname}_{scanidx:05d}'
    
    print(f"Processing scan {scanidx+1}/{nscan} (kappa={combinations[scanidx][0]}, delta={combinations[scanidx][1]}, energy_srcCORE={combinations[scanidx][-1]})...")
    
    try:
        simulation = setup_simulation(simdir, fileprefix)
        
        sim_frames = pygkyl.file_utils.find_available_frames(simulation, 'field')
        if len(sim_frames) == 0:
            raise ValueError(f"No frames found for simulation {scanidx}")
        
        last_frame = frame_idx if frame_idx >= 0 else sim_frames[-1]
        first_frame = max(0, last_frame - frame_navg)  # Ensure non-negative
        
        # Validate frame range
        if last_frame not in sim_frames:
            raise ValueError(f"Requested frame {last_frame} not available. Available frames: {sim_frames[0]} to {sim_frames[-1]}")
        
        # Only include frames that actually exist
        frame_array = [f for f in range(first_frame, last_frame + 1) if f in sim_frames]
        if len(frame_array) == 0:
            raise ValueError(f"No valid frames in range {first_frame} to {last_frame}")
        
        # get field values
        field_values, tend = get_field_values(simulation, frame_array, locations, fields)
        
        # Get integrated bflux moments
        intmom_dict = get_integrated_mom(simulation, 
            ['bflux_x_l_ne', 'bflux_x_l_ni', 'bflux_x_l_He', 'bflux_x_l_Hi',
             'bflux_x_u_ne', 'bflux_x_u_ni', 'bflux_x_u_He', 'bflux_x_u_Hi',
             'bflux_z_l_ne', 'bflux_z_l_ni', 'bflux_z_l_He', 'bflux_z_l_Hi',
             'bflux_z_u_ne', 'bflux_z_u_ni', 'bflux_z_u_He', 'bflux_z_u_Hi',
             'ne', 'ni', 'We', 'Wi'])
        
        # Get simulation average dt from log file
        avg_dt = get_average_dt(f'{scandir}/logs/std-{scanname}_{scanidx:05d}.log')
        
        # Get heat flux width
        lambda_q, delta_lambda_q = get_heatflux_width(simulation)

        # Gather data
        data = {
            'simdir': simdir,
            'scanidx': scanidx,
            'kappa': combinations[scanidx][0],
            'delta': combinations[scanidx][1],
            'energy_srcCORE': combinations[scanidx][-1],
            'tend': tend,
            'avg_dt': avg_dt,
            'vol_frac': simulation.geom_param.vol_frac,
            'intmom': intmom_dict,
            'lambda_q': lambda_q,
            'delta_lambda_q': delta_lambda_q,
            **field_values,  # Unpack all field values
        }
        
        print("%d/%d: k=%.2f, d=%.2f, P=%.1e, Ti=%.1f, Te=%.1f, ne=%.1e, phi=%.1f, t=%.1f, dt=%.3e" % 
              (scanidx+1, nscan, data['kappa'], data['delta'], data['energy_srcCORE'], 
               field_values['Ti_lcfs'], field_values['Te_lcfs'], field_values['ne_lcfs'], 
               field_values['phi_lcfs'], tend, avg_dt))
        
        return data
    except Exception as e:
        print(f"Error processing scan {scanidx}: {e}")
        return None

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process simulation metadata in parallel')
parser.add_argument('--ncpu', type=int, default=4, 
                    help=f'Number of CPU cores to use (default: 4, recommended for I/O-bound tasks)')
parser.add_argument('--chunksize', type=int, default=1,
                    help='Number of tasks per worker process (default: 1, larger values reduce overhead)')
args = parser.parse_args()

# Determine number of parallel processes
num_processes = min(args.ncpu, nscan)  # Don't use more processes than scans
print(f"Processing {nscan} simulations using {num_processes} parallel processes (chunksize={args.chunksize})...")

# Run parallel processing
if __name__ == '__main__':
    start_time = time.time()
    
    # Check if we are sequential or parallel
    if num_processes == 1:
        metadata = []
        for i in range(nscan):
            result = process_scan((i, 0))
            if result is not None:
                metadata.append(result)
    else:
        # Create args with process IDs for staggered starts
        scan_args = [(i, i % num_processes) for i in range(nscan)]
        
        with Pool(processes=num_processes) as pool:
            metadata = pool.map(process_scan, scan_args, chunksize=args.chunksize)
    
    # Filter out any None results from failed scans
    metadata = [m for m in metadata if m is not None]
    
    # Save the metadata to an h5 file
    with h5py.File(filename+'.h5', 'w') as f:
        for m in metadata:
            grp = f.create_group(f"scan_{m['scanidx']:05d}")
            
            # Save scalar params and field values as attributes
            for key, val in m.items():
                if key == 'intmom':
                    continue
                grp.attrs[key] = val
            
            # Save integrated moment time series as datasets
            intmom_grp = grp.create_group('intmom')
            for mom_name, mom_data in m['intmom'].items():
                mom_grp = intmom_grp.create_group(mom_name)
                mom_grp.create_dataset('time', data=np.asarray(mom_data['time']))
                mom_grp.create_dataset('values', data=np.asarray(mom_data['values']))
                mom_grp.attrs['tunits'] = mom_data['tunits']
                mom_grp.attrs['vunits'] = mom_data['vunits']
                mom_grp.attrs['name'] = mom_data['name']
    
    elapsed_time = time.time() - start_time
    print(f"\nMetadata saved to {filename+'.h5'}")
    print(f"Successfully processed {len(metadata)}/{nscan} simulations")
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/len(metadata):.2f}s per simulation)")