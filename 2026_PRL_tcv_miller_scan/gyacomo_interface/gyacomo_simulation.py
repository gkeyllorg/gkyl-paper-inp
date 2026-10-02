"""
GyacomoSimulation class for managing Gyacomo gyrokinetic simulations.

This module provides a Python interface to:
- Configure simulation parameters via Fortran namelists
- Execute simulations with MPI
- Load and analyze HDF5 output data
- Visualize results (fluxes, fields, growth rates)
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import shutil

# Import existing Gyacomo utilities
from gyacomo_interface.gyacomo_load_data import load_data_0D, load_data_3D_frame, load_grids, load_h5path, load_h5path, load_params
from gyacomo_interface.gyacomo_tools import closest_index, zkxky_to_xy_const_z, compute_omega_t


class GyacomoSimulation:
    """
    A Python wrapper for Gyacomo gyrokinetic simulations.
    
    Manages simulation setup, execution, and analysis within a specified
    run directory. Handles parameter files, MPI execution, and data loading.
    
    Attributes
    ----------
    run_dir : str or Path
        Directory where simulation runs and outputs are stored
    jobnum : int
        Job index for restart capability (default: 0)
    params : dict
        Dictionary of all namelist parameters
    executable : str
        Path to gyacomo executable
    """
    
    def __init__(self, run_dir, jobnum=0, executable=None, mpi_cmd='mpirun'):
        """
        Initialize a Gyacomo simulation instance.
        
        Parameters
        ----------
        run_dir : str or Path
            Directory for simulation (created if doesn't exist)
        jobnum : int, optional
            Job index for restarts (default: 0)
        executable : str, optional
            Path to gyacomo executable (auto-detected if None)
        """
        self.run_dir = Path(run_dir)
        self.jobnum = jobnum
        self.params_file = self.run_dir / "params.in"
        self.output_file = self.run_dir / f"outputs_{jobnum:02d}.h5"
        self.stdout_file = self.run_dir / f"std_{jobnum:02d}.out"
        self.mpi_cmd = mpi_cmd
        
        # Auto-detect executable if not provided
        if executable is None:
            self.executable = self._find_executable()
        else:
            self.executable = Path(executable)
            
        # Initialize parameter dictionary structure
        self.params_nml = self._init_params_structure()
        
        # Initialize parameter table for easy access like 
        # ['BASIC']['dt'] becomes params_table['dt'] or
        # ['SPECIES'][0]['tau_'] becomes params_table['tau_i']
        self.params = self._init_params_table()
        
        # Load existing params if file exists
        if self.params_file.exists():
            self.load_params_from_file(self.params_file)
    
    def _find_executable(self):
        """Auto-detect gyacomo executable in bin/ directory."""
        # Look for executable relative to this script
        script_dir = Path(__file__).parent
        gyacomo_root = script_dir.parent.parent
        bin_dir = gyacomo_root / "bin"
        
        # Prefer double precision version
        for exe_name in ["gyacomo23_dp", "gyacomo23_sp", "gyacomo.exe"]:
            exe_path = bin_dir / exe_name
            if exe_path.exists():
                return exe_path
        
        # If not found, return default and let user override
        return bin_dir / "gyacomo23_dp"
    
    def _init_params_structure(self):
        """Initialize default parameter dictionary structure."""
        params = {
            'BASIC': {
                'nrun': 99999999,
                'dt': 0.002,
                'tmax': 5.0,
                'maxruntime': 72000,
                'job2load': -1
            },
            'GRID': {
                'pmax': 4,
                'jmax': 2,
                'Nx': 2,
                'Lx': 400.0,
                'Ny': 24,
                'Ly': 75.0,
                'Nz': 24,
                'SG': False,
                'Nexc': 0
            },
            'GEOMETRY': {
                'geom': 'miller',
                'q0': 2.8,
                'shear': 2.7,
                'eps': 0.26,
                'kappa': 1.45,
                's_kappa': 0.0,
                'delta': 0.35,
                's_delta': 0.0,
                'zeta': 0.0,
                's_zeta': 0.0,
                'parallel_bc': 'dirichlet',
                'shift_y': 0.0,
                'Npol': 1.0,
                'PB_PHASE': False
            },
            'DIAGNOSTICS': {
                'write_doubleprecision': True,
                'dtsave_0d': 0.05,
                'dtsave_1d': -1,
                'dtsave_2d': -1,
                'dtsave_3d': 0.05,
                'dtsave_5d': 2.0
            },
            'MODEL': {
                'LINEARITY': 'linear',
                'Na': 2,
                'mu_x': 0.0,
                'mu_y': 0.0,
                'N_HD': 4,
                'mu_z': 0.0,
                'HYP_V': 'hypcoll',
                'mu_p': 0.0,
                'mu_j': 0.0,
                'nu': 0.1,
                'beta': 0.0,
                'ADIAB_E': False
            },
            'CLOSURE': {
                'hierarchy_closure': 'truncation',
                'dmax': -1,
                'nonlinear_closure': 'anti_laguerre_aliasing',
                'nmax': -1
            },
            'SPECIES': [
                {
                    'name_': 'ions',
                    'tau_': 1.0,
                    'sigma_': 1.0,
                    'q_': 1.0,
                    'k_N_': 50.0,
                    'k_T_': 20.0
                },
                {
                    'name_': 'electrons',
                    'tau_': 1.0,
                    'sigma_': 0.023338,
                    'q_': -1.0,
                    'k_N_': 50.0,
                    'k_T_': 30.0
                }
            ],
            'COLLISION': {
                'collision_model': 'DG',
                'GK_CO': True
            },
            'INITIAL': {
                'INIT_OPT': 'blob'
            },
            'TIME_INTEGRATION': {
                'numerical_scheme': 'RK4'
            },
            'UNITS': {
                'n_ref': 5.0,
                'T_ref': 2.5,
                'R_ref': 1.7,
                'B_ref': 1.5,
                'm_ref': 1,
                'q_ref': 1,
                'WRITE_MW': False
            }
        }
        return params
    
    def _init_params_table(self):
        """
        Initialize parameter table for easy access.
        e.g.
        - ['BASIC']['dt'] becomes params_table['dt']
        - ['SPECIES'][0]['tau_'] becomes params_table['tau_i'].
        """
        table = {}
        for namelist, values in self.params_nml.items():
            if namelist == 'SPECIES':
                for i, species in enumerate(values):
                    prefix = 'i' if species['name_'].lower() == 'ions' else 'e'
                    for key, val in species.items():
                        if key.endswith('_'):
                            param_name = f"{key[:-1]}_{prefix}"
                            table[param_name] = val
            else:
                for key, val in values.items():
                    table[key] = val
        return table
    
    def _update_params_structure_from_table(self):
        """
        Update the nested params structure based on the flat params_table.
        This allows users to set parameters via params_table and have it reflected
        in the generated params.in file.
        """
        for namelist, values in self.params_nml.items():
            if namelist == 'SPECIES':
                for i, species in enumerate(values):
                    prefix = 'i' if species['name_'].lower() == 'ions' else 'e'
                    for key in species.keys():
                        if key.endswith('_'):
                            param_name = f"{key[:-1]}_{prefix}"
                            if param_name in self.params:
                                species[key] = self.params[param_name]
            else:
                for key in values.keys():
                    if key in self.params:
                        values[key] = self.params[key]
    
    def load_params_from_file(self, filename):
        """
        Load parameters from existing params.in file.
        
        Parameters
        ----------
        filename : str or Path
            Path to Fortran namelist parameter file
        """
        with open(filename, 'r') as f:
            content = f.read()
        
        # Parse namelists
        parsed = self._parse_namelist_file(content)
        
        # Update params dictionary
        for namelist, values in parsed.items():
            if namelist == 'SPECIES':
                # Handle multiple species
                self.params_nml['SPECIES'] = values
            elif namelist in self.params_nml:
                self.params_nml[namelist].update(values)
            else:
                self.params_nml[namelist] = values
                
    def set_param(self, dict_key_value):
        """
        Set a parameter value in the flat params_table.
        """
        for key, value in dict_key_value.items():
            self.params[key] = value
        self._update_params_structure_from_table()

    def _parse_namelist_file(self, content):
        """
        Parse Fortran namelist file content.
        
        Parameters
        ----------
        content : str
            File content as string
            
        Returns
        -------
        dict
            Dictionary of namelists and their parameters
        """
        namelists = {}
        current_namelist = None
        species_list = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('!'):
                continue
            
            # Remove inline comments
            if '!' in line:
                line = line.split('!')[0].strip()
            
            # Check for namelist start
            if line.startswith('&'):
                current_namelist = line[1:].strip()
                if current_namelist == 'SPECIES':
                    species_list.append({})
                elif current_namelist not in namelists:
                    namelists[current_namelist] = {}
                continue
            
            # Check for namelist end
            if line.startswith('/'):
                current_namelist = None
                continue
            
            # Parse parameter line
            if '=' in line and current_namelist:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert value to appropriate type
                parsed_value = self._parse_value(value)
                
                if current_namelist == 'SPECIES':
                    species_list[-1][key] = parsed_value
                else:
                    namelists[current_namelist][key] = parsed_value
        
        if species_list:
            namelists['SPECIES'] = species_list
        
        return namelists
    
    def _parse_value(self, value_str):
        """Convert string value to appropriate Python type."""
        value_str = value_str.strip()
        
        # Boolean
        if value_str.lower() in ['.t.', '.true.']:
            return True
        if value_str.lower() in ['.f.', '.false.']:
            return False
        
        # String (quoted)
        if value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]
        
        # Number
        try:
            if '.' in value_str or 'e' in value_str.lower():
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            # Return as string if can't parse
            return value_str
        
    def get_params_content(self):
        """
        Generate the Fortran namelist file content from current parameters.
        
        Returns
        -------
        str
            File content as a string
        """
        lines = []
        for namelist in ['BASIC', 'GRID', 'GEOMETRY', 'DIAGNOSTICS',
                        'MODEL', 'CLOSURE', 'SPECIES', 'COLLISION',
                        'INITIAL', 'TIME_INTEGRATION', 'UNITS']:

            if namelist not in self.params_nml:
                continue

            if namelist == 'SPECIES':
                # Write each species separately
                for species in self.params_nml['SPECIES']:
                    lines.append(f"&{namelist}\n")
                    for key, value in species.items():
                        lines.append(f" {key} = {self._format_value(value)}\n")
                    lines.append("/\n")
            else:
                # Write regular namelist
                lines.append(f"&{namelist}\n")
                for key, value in self.params_nml[namelist].items():
                    lines.append(f"  {key:15} = {self._format_value(value):15}")
                    lines.append(f" ! {self._get_param_comment(namelist, key)}\n")
                lines.append("/\n")
        return ''.join(lines)

    def write_params_file(self, filename=None):
        """
        Write parameters to Fortran namelist file.
        
        Parameters
        ----------
        filename : str or Path, optional
            Output file path (default: self.params_file)
        """
        if filename is None:
            filename = self.params_file
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(self.get_params_content())
    
    def _format_value(self, value):
        """Format Python value as Fortran namelist value."""
        if isinstance(value, bool):
            return '.t.' if value else '.f.'
        elif isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, float):
            return f"{value:.10g}"
        else:
            return str(value)
    
    def _get_param_comment(self, namelist, key):
        """Get descriptive comment for parameter."""
        comments = {
            'BASIC': {
                'nrun': 'number of time steps to perform',
                'dt': 'time step (not adaptive)',
                'tmax': 'maximal time [c_s/R]',
                'maxruntime': 'maximal wallclock runtime [sec]',
                'job2load': 'index of previous run to restart (-1:new)'
            },
            'GRID': {
                'pmax': 'maximal degree of Hermite basis (parallel velocity)',
                'jmax': 'maximal degree of Laguerre basis (magnetic moment)',
                'Nx': 'number of points in radial direction',
                'Lx': 'size of box in radial direction [rho_s]',
                'Ny': 'number of points in binormal direction',
                'Ly': 'size of box in binormal direction [rho_s]',
                'Nz': 'number of points in magnetic field direction',
                'SG': 'use staggered grid in z (experimental)',
                'Nexc': 'fulfill sheared boundary condition (-1:auto)'
            }
        }
        return comments.get(namelist, {}).get(key, '')
    
    def setup_directory(self, clean=False):
        """
        Create run directory and setup simulation environment.
        
        Parameters
        ----------
        clean : bool, optional
            If True, remove existing directory first (default: False)
        """
        if clean and self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Write params file
        self.write_params_file()
    
    def run(self, nproc=8, proc_distr = (2, 4, 1), blocking=True, verbose=True, stdout=None):
        """
        Execute Gyacomo simulation with MPI.
        
        Parameters
        ----------
        nproc : int, optional
            Number of MPI processes (default: 8)
        proc_distr : tuple of ints, optional
            Process distribution in (p, ky, z) directions (default: (2, 4, 1))
        blocking : bool, optional
            Wait for completion (default: True)
        verbose : bool, optional
            Print output to console (default: True)
        stdout : file-like or None, optional
            If provided, redirect stdout to this file-like object (default: None)
            
        Returns
        -------
        subprocess.CompletedProcess or subprocess.Popen
            Process object
        """
        # Ensure directory and params exist
        self.setup_directory()
        
        # Run command
        cmd_opt = '-np' if 'mpirun' in self.mpi_cmd else '-n'
        cmd = [
            self.mpi_cmd, cmd_opt, str(nproc),
            str(self.executable),
            str(proc_distr[0]), str(proc_distr[1]), str(proc_distr[2])
        ]
        
        if verbose:
            print(f"Running:")
            print(f"{' '.join(cmd)}")
            print(f"Working directory: {self.run_dir}")
        
        # Execute
        with open(self.params_file, 'r') as stdin_file:
            with open(self.stdout_file, 'w') as stdout_file:
                if blocking:
                    result = subprocess.run(cmd, stdin=stdin_file, stdout=stdout_file, cwd=self.run_dir)
                    if verbose:
                        print(f"Simulation completed with exit code: {result.returncode}")
                    return result
                else:
                    process = subprocess.Popen(cmd, stdin=stdin_file, stdout=stdout_file, cwd=self.run_dir)
                    return process
    
    def check_status(self):
        """
        Check simulation status and progress.
        
        Returns
        -------
        dict
            Status information (exists, current_time, max_time, etc.)
        """
        status = {
            'output_exists': self.output_file.exists(),
            'stdout_exists': self.stdout_file.exists(),
            'current_time': None,
            'max_time': None,
            'progress': None
        }
        
        if status['output_exists']:
            try:
                # Load time data
                t0d = load_h5path(str(self.output_file), 'data/var0d/time')
                status['current_time'] = float(t0d[-1])
                status['max_time'] = self.params_nml['BASIC']['tmax']
                status['progress'] = status['current_time'] / status['max_time'] * 100
            except:
                pass
        
        return status
    
    # ==================== Analysis Methods ====================
    
    def load_params(self):
        """
        Load simulation parameters from HDF5 output.
        
        Returns
        -------
        dict
            Parameter dictionary from HDF5 file
        """
        if not self.output_file.exists():
            raise FileNotFoundError(f"Output file not found: {self.output_file}")
        return load_params(str(self.output_file))
    
    def load_grids(self):
        """
        Load simulation grids.
        
        Returns
        -------
        tuple
            (x, kx, y, ky, z, p, j) coordinate arrays
        """
        if not self.output_file.exists():
            raise FileNotFoundError(f"Output file not found: {self.output_file}")
        return load_grids(str(self.output_file))
    
    def load_time_trace(self, varname):
        """
        Load 0D time trace data.
        
        Parameters
        ----------
        varname : str
            Variable name ('hflux_x', 'pflux_x', 'energy', etc.)
            
        Returns
        -------
        tuple
            (time, data) arrays
        """
        if not self.output_file.exists():
            raise FileNotFoundError(f"Output file not found: {self.output_file}")
        return load_data_0D(str(self.output_file), varname)
    
    def load_field_3d(self, varname, time=None):
        """
        Load 3D field data at specific time.
        
        Parameters
        ----------
        varname : str
            Field name ('phi', 'Na00', 'dens', 'upar', etc.)
        time : float, optional
            Requested time (finds closest frame)
            If None, loads last frame
            
        Returns
        -------
        tuple
            (time_array, field_3d, actual_time)
        """
        if not self.output_file.exists():
            raise FileNotFoundError(f"Output file not found: {self.output_file}")
        
        if time is None:
            # Load last available frame
            t3d = load_h5path(str(self.output_file), 'data/var3d/time')
            time = t3d[-1]
        
        return load_data_3D_frame(str(self.output_file), varname, time)
    
    def field_to_realspace(self, field_kxkyz, iz=-1):
        """
        Transform 3D spectral field to real space at constant z.
        
        Parameters
        ----------
        field_kxkyz : ndarray
            3D complex field (Nz, Nkx, Nky)
        iz : int, optional
            z-index (-1 for outboard midplane)
            
        Returns
        -------
        ndarray
            2D real space field (Ny, Nx)
        """
        return zkxky_to_xy_const_z(field_kxkyz, iz)
    
    def compute_growth_rate(self, field_name='phi', kx_val=0.0, ky_index=None, 
                           time_range=None, iz=None):
        """
        Compute complex growth rate from field evolution.
        
        Parameters
        ----------
        field_name : str, optional
            Field to analyze (default: 'phi')
        kx_val : float, optional
            kx value to analyze (default: 0.0)
        ky_index : int or list, optional
            ky index/indices to analyze (default: all)
        time_range : tuple, optional
            (t_start, t_end) for analysis window
        iz : int, optional
            z-index (default: midplane)
            
        Returns
        -------
        dict
            Dictionary with 'ky', 'omega' (complex growth rates), 'time'
        """
        # Load grids
        x, kx, y, ky, z, p, j = self.load_grids()
        
        if iz is None:
            iz = len(z) // 2  # Midplane
        
        # Find kx index
        ikx = closest_index(kx, kx_val)
        
        # Load 3D time data
        t3d = load_h5path(str(self.output_file), 'data/var3d/time')
        
        # Apply time range
        if time_range is not None:
            t_start, t_end = time_range
            mask = (t3d >= t_start) & (t3d <= t_end)
            t3d = t3d[mask]
            time_indices = np.where(mask)[0]
        else:
            time_indices = np.arange(len(t3d))
        
        # Collect field values
        values = []
        for it in time_indices:
            _, field, _ = load_data_3D_frame(
                str(self.output_file), field_name, t3d[it - time_indices[0]]
            )
            field_slice = field[iz, ikx, :].squeeze()
            values.append(field_slice)
        values = np.array(values)  # shape (nt, nky)
        
        # Determine which ky to analyze
        if ky_index is None:
            ky_indices = range(len(ky))
        elif isinstance(ky_index, int):
            ky_indices = [ky_index]
        else:
            ky_indices = ky_index
        
        # Compute omega for each ky
        omega_list = []
        ky_list = []
        for iky in ky_indices:
            omega_t, time_out = compute_omega_t(t3d, values[:, iky])
            omega_list.append(omega_t[-1])  # Take final value
            ky_list.append(ky[iky])
        
        return {
            'ky': np.array(ky_list),
            'omega': np.array(omega_list),
            'time': t3d
        }
    
    # ==================== Visualization Methods ====================
    
    def plot_fluxes(self, ax=None, species_labels=None):
        """
        Plot heat and particle flux time traces.
        
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes for plotting (creates new if None)
        species_labels : list of str, optional
            Labels for species (default: ['ions', 'electrons'])
            
        Returns
        -------
        tuple
            (fig, axes) matplotlib objects
        """
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        else:
            axes = ax
            fig = axes[0].figure
        
        # Load data
        t0d, hflux_x = self.load_time_trace('hflux_x')
        t0d, pflux_x = self.load_time_trace('pflux_x')
        
        if species_labels is None:
            species_labels = ['ions', 'electrons']
        
        # Plot heat flux
        nt0d = t0d.size
        if hflux_x.size > nt0d:
            for i, label in enumerate(species_labels):
                axes[0].plot(t0d, hflux_x[i::len(species_labels)], 
                           label=f'{label} heat flux')
        else:
            axes[0].plot(t0d, hflux_x, label='heat flux')
        
        axes[0].set_title('Radial Heat Flux')
        axes[0].set_xlabel(r'$t c_s/R$')
        axes[0].set_ylabel(r'$Q_x$')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot particle flux
        if pflux_x.size > nt0d:
            for i, label in enumerate(species_labels):
                axes[1].plot(t0d, pflux_x[i::len(species_labels)], 
                           label=f'{label} particle flux')
        else:
            axes[1].plot(t0d, pflux_x, label='particle flux')
        
        axes[1].set_title('Radial Particle Flux')
        axes[1].set_xlabel(r'$t c_s/R$')
        axes[1].set_ylabel(r'$P_x$')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_field_slice(self, field_name, time=None, iz=-1, ax=None, 
                        cmap='seismic', **kwargs):
        """
        Plot 2D slice of field in real space.
        
        Parameters
        ----------
        field_name : str
            Field to plot ('phi', 'Na00', 'dens', etc.)
        time : float, optional
            Time to plot (default: last frame)
        iz : int, optional
            z-index (default: -1 for midplane)
        ax : matplotlib axes, optional
            Axes for plotting
        cmap : str, optional
            Colormap (default: 'seismic')
        **kwargs
            Additional arguments for imshow
            
        Returns
        -------
        tuple
            (fig, ax, im) matplotlib objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure
        
        # Load data
        x, kx, y, ky, z, p, j = self.load_grids()
        t3d, field_3d, tf = self.load_field_3d(field_name, time)
        
        # Transform to real space
        field_xy = self.field_to_realspace(field_3d, iz)
        
        # Plot
        im = ax.imshow(field_xy, extent=[x[0], x[-1], y[0], y[-1]],
                      cmap=cmap, interpolation='nearest', **kwargs)
        ax.set_title(f'{field_name} (z=0, t={tf:.2f})')
        ax.set_xlabel(r'$x/\rho_s$')
        ax.set_ylabel(r'$y/\rho_s$')
        plt.colorbar(im, ax=ax, label=field_name)
        
        return fig, ax, im
    
    def plot_summary(self, time=None, figsize=(12, 8)):
        """
        Create comprehensive 4-panel summary plot.
        
        Parameters
        ----------
        time : float, optional
            Time for field plots (default: last frame)
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        tuple
            (fig, axes) matplotlib objects
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Top left: heat flux
        t0d, hflux_x = self.load_time_trace('hflux_x')
        nt0d = t0d.size
        if hflux_x.size > nt0d:
            axes[0, 0].plot(t0d, hflux_x[::2], label='ion')
            axes[0, 0].plot(t0d, hflux_x[1::2], label='electron')
        else:
            axes[0, 0].plot(t0d, hflux_x)
        axes[0, 0].set_title('Radial Heat Flux')
        axes[0, 0].set_xlabel(r'$t c_s/R$')
        axes[0, 0].set_ylabel(r'$Q_x$')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Bottom left: particle flux
        t0d, pflux_x = self.load_time_trace('pflux_x')
        if pflux_x.size > nt0d:
            axes[1, 0].plot(t0d, pflux_x[::2], label='ion')
            axes[1, 0].plot(t0d, pflux_x[1::2], label='electron')
        else:
            axes[1, 0].plot(t0d, pflux_x)
        axes[1, 0].set_title('Radial Particle Flux')
        axes[1, 0].set_xlabel(r'$t c_s/R$')
        axes[1, 0].set_ylabel(r'$P_x$')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Top right: density field
        x, kx, y, ky, z, p, j = self.load_grids()
        t3d, Ni00, tf = self.load_field_3d('Na00', time)
        Ni00_xy = self.field_to_realspace(Ni00, -1)
        im1 = axes[0, 1].imshow(Ni00_xy, extent=[x[0], x[-1], y[0], y[-1]],
                               cmap='seismic', interpolation='nearest')
        axes[0, 1].set_title(f'Na00 (z=0, t={tf:.2f})')
        axes[0, 1].set_xlabel(r'$x/\rho_s$')
        axes[0, 1].set_ylabel(r'$y/\rho_s$')
        
        # Bottom right: potential field
        t3d, phi, tf = self.load_field_3d('phi', time)
        phi_xy = self.field_to_realspace(phi, -1)
        im2 = axes[1, 1].imshow(phi_xy, extent=[x[0], x[-1], y[0], y[-1]],
                               cmap='seismic', interpolation='nearest')
        axes[1, 1].set_title(f'phi (z=0, t={tf:.2f})')
        axes[1, 1].set_xlabel(r'$x/\rho_s$')
        axes[1, 1].set_ylabel(r'$y/\rho_s$')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_dispersion(self, field_name='phi', kx_val=0.0, ky_range=None,
                       time_range=None, figsize=(5, 3.5)):
        """
        Plot growth rate and frequency vs wavenumber.
        
        Parameters
        ----------
        field_name : str, optional
            Field to analyze (default: 'phi')
        kx_val : float, optional
            kx value (default: 0.0)
        ky_range : tuple, optional
            (ky_min, ky_max) range to plot
        time_range : tuple, optional
            (t_start, t_end) for growth rate calculation
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        tuple
            (fig, ax) matplotlib objects
        """
        # Compute growth rates
        result = self.compute_growth_rate(
            field_name=field_name,
            kx_val=kx_val,
            time_range=time_range
        )
        
        ky = result['ky']
        omega = result['omega']
        
        # Apply ky range filter
        if ky_range is not None:
            mask = (ky >= ky_range[0]) & (ky <= ky_range[1])
            ky = ky[mask]
            omega = omega[mask]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(ky, np.real(omega), 'o-', label=r'$\gamma$ (growth rate)')
        ax.plot(ky, np.imag(omega), 's--', label=r'$\omega$ (frequency)')
        ax.set_xlabel(r'$k_y \rho_s$')
        ax.set_ylabel(r'$\omega$ [$c_s/R$]')
        ax.set_title(f'Dispersion Relation ({field_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def __repr__(self):
        """String representation."""
        status = "exists" if self.output_file.exists() else "not run"
        return f"GyacomoSimulation(dir={self.run_dir}, job={self.jobnum}, status={status})"
