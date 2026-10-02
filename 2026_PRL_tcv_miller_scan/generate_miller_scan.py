import os
import sys
import itertools
import stat
import argparse
scripts_path = os.path.expanduser('~/personal_gkyl_scripts/scripts')
sys.path.insert(0, scripts_path)
from slurm_auto_scripting import ScanConfig

def run(generate=True):
    config = ScanConfig(
    # --- Slurm Options ---
    job_name="tcv_miller_scan_big",
    account="m4564",
    time="06:00:00",
    qos="regular", # regular, debug, etc.
    constraint="gpu",
    email="ahoffman@pppl.gov",              # Leave empty to disable
    email_type="ALL", # BEGIN, END, FAIL, ALL
    # --- Resource Specifications ---
    total_gpus_per_node=4,
    total_cores_per_node=64,
    gpu_per_instance=2,
    num_jobs=1,
    # --- Scan Parameters ---
    scan_arrays={
        "kappa": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        "delta": [-0.6, -0.45, -0.3, -0.15, 0.15, 0.3, 0.45, 0.6],
        "energy_srcCORE": [0.1e6, 0.5e6, 1.0e6, 5.0e6], # in Watts,
    },
    # --- File and Directory Names ---
    work_dir="tcv_miller_scan_big",
    output_script="submit_scan.sh",
    gkeyll_input_c="gkeyll.c",
    gkyl_additional_options="",
    # --- Execution Control ---
    execution_mode="ALL", # "ALL", "FIRST", "FAIL"
    restart_from_last_frame=True,
    last_frame_detector_script= "~/personal_gkyl_scripts/simulation_scripts/utilities/gkyl_find_last_frame.sh",
    max_sim_time=1000.0  # Maximum simulation time in microseconds (this is just for monitoring purposes)
    )
    if generate:
        config.generate_script()
    return config  # Return config for monitoring

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate and monitor TCV Miller scan simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate SLURM submission scripts
  python generate_miller_scan.py
  
  # Monitor simulation progress (one-time snapshot)
  python generate_miller_scan.py -p
  
  # Monitor with custom max simulation time (in microseconds)
  python generate_miller_scan.py -p -m 500.0
  
  # Continuous monitoring (updates every 5 seconds)
  python generate_miller_scan.py -w
  
  # Continuous monitoring with custom refresh interval (10 seconds)
  python generate_miller_scan.py -w --interval 10
        """
    )
    
    parser.add_argument('-p', '--progress', action='store_true',
                        help='Show one-time progress summary')
    parser.add_argument('-m', '--max-time', type=float,
                        help='Override maximum simulation time in microseconds (default: 1000.0)')
    parser.add_argument('--interval', type=int, default=5,
                        help='Refresh interval for continuous monitoring in seconds (default: 5)')
    
    args = parser.parse_args()
    
    # Determine if we're monitoring or generating
    if args.progress:
        # Load config without generating scripts
        config = run(generate=False)
        
        # Override max_sim_time if specified
        if args.max_time is not None:
            config.max_sim_time = args.max_time
        
        # One-time snapshot
        config.print_summary()
        
    else:
        # Generate scripts
        run()
