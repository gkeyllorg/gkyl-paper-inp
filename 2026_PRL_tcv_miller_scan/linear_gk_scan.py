import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from gyacomo_interface.gyacomo_simulation import GyacomoSimulation
from scan_analysis.scan_metadata import ScanMetadata

path = '/pscratch/sd/a/ah1032/gkeyll_main/tcv_miller_scan/linear_gk_analysis/'
h5file = '/pscratch/sd/a/ah1032/gkeyll_main/tcv_miller_scan/tcv_miller_scan_big_metadata_frame_500_navg_25.h5'

# Load the Gkeyll scan metadata
miller_scan = ScanMetadata(h5file)

root_run_dir = f'{path}gyacomo_run_data/'
std_param_file = f'{path}gyacomo_input_template.F90'
gyac_exe = '/global/homes/a/ah1032/gyacomo/bin/gyacomo23_dp'
mpi_cmd = None  # None = run executable directly (serial). Use 'srun' or 'mpirun' for MPI runs.

verbose = False

gyac_params = {
    'q0': 2.6,
    'shear': 1.9,
    'eps': 0.28,
    'Lx': 200.0,
    'Nx': 16,
    'Ny': 2,
    'Nz': 24,
    'pmax': 4,
    'jmax': 2,
    'tmax': 10.0,
    'dt': 1e-3,
    'nu': 1.0
}

# get data from the Gkeyll simulations
kappa_vals = miller_scan.scan_params['kappa']
delta_vals = miller_scan.scan_params['delta']
energy_srcCORE_vals = miller_scan.scan_params['energy_srcCORE']

gyac_linear_results = {}

def run_gyac_kernel(gkeyll_params, ky=0.5, verbose=False):
    # Get parameters from Gkeyll scan
    p_ = {'kappa': gkeyll_params[0], 'delta': gkeyll_params[1], 'energy_srcCORE': gkeyll_params[2]}
    tau = miller_scan.get_field_value('tau_lcfs', params=p_)
    kTe = miller_scan.get_field_value('kTe', params=p_)
    kTi = miller_scan.get_field_value('kTi', params=p_)
    kne = miller_scan.get_field_value('kne', params=p_)
    tau = miller_scan.get_field_value('tau_lcfs', params=p_)
    
    # Initialize simulation
    run_dir = f"{root_run_dir}/kappa_{p_['kappa']:.2f}_delta_{p_['delta']:.2f}_energy_{p_['energy_srcCORE']:.1e}"
    sim = GyacomoSimulation(run_dir=run_dir,jobnum=0,executable=gyac_exe, mpi_cmd=mpi_cmd)
    sim.load_params_from_file(std_param_file)
    
    # Set up the constant parameters
    sim.set_param(gyac_params)
    
    # Set up the scan parameters
    sim.set_param({'Ly': 2*np.pi/ky})
    sim.set_param({'delta': p_['delta']})
    sim.set_param({'kappa': p_['kappa']})
    sim.set_param({'k_N_e': kne})
    sim.set_param({'k_T_e': kTe})
    sim.set_param({'k_N_i': kne})
    sim.set_param({'k_T_i': kTi})
    sim.set_param({'tau_e': tau})

    # Setup directory and write params
    sim.setup_directory(clean=True)
    
    sidx = miller_scan.get_sim_index(p_)
    print(f"i={sidx:05d}, d={p_['delta']:1.1f}, k={p_['kappa']:1.1f}, e={p_['energy_srcCORE']:1.1e}, tau={tau:1.1f}, kTe={kTe:2.1f}, kTi={kTi:2.1f}, kne={kne:2.1f}")

    # Run the simulation
    _ = sim.run(nproc=1, proc_distr=(1, 1, 1), blocking=True, verbose=verbose)

    # Compute growth rate
    gr_dict = sim.compute_growth_rate('phi', kx_val=0, ky_index=1)
    ky = gr_dict['ky'][0]
    omega = gr_dict['omega'][0]
    
    if verbose:
        print(f"    ky={ky:2.1f}, Ɣ={np.real(omega):2.1f}, ω={np.imag(omega):2.1f}")
    
    return ky, omega, sim.get_params_content()

def _run_with_index(args):
    i, params, ky = args
    ky, omega, param_content = run_gyac_kernel(params, ky=ky, verbose=verbose)
    return i, ky, omega, param_content


n_workers = 128  # number of parallel simulations

parameter_list = miller_scan.combinations[:]

ky_array = np.zeros(len(parameter_list))
omega_array = np.zeros(len(parameter_list), dtype=complex)
params_in_array = np.empty(len(parameter_list), dtype=object)

ky_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for ky in ky_list:
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_with_index, (i, params, ky)): i
            for i, params in enumerate(parameter_list)
        }
        for future in as_completed(futures):
            i, ky, omega, param_content = future.result()
            ky_array[i] = ky
            omega_array[i] = omega
            params_in_array[i] = param_content


    for i in range(len(parameter_list)):
        miller_scan.add_gk_linear_data(
            kappa=parameter_list[i][0], 
            delta=parameter_list[i][1], 
            energy_srcCORE=parameter_list[i][2], 
            ky=ky_array[i], 
            omega=omega_array[i],
            params_in=params_in_array[i])

    miller_scan.save_gk_linear_data()