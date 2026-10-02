"""
ScanMetadata: high-level orchestrator for scan analysis.

This is the main user-facing class. It delegates to the specialised modules
for loading, field computation, extraction, and plotting.

Example usage::

    from scan_analysis import ScanMetadata

    scan = ScanMetadata('data/tcv_miller_scan_big_metadata_frame_500_navg_25.h5')
    scan.info()

    # Contour grid on kappa-delta plane
    scan.plot_contour_grid(['Ti_core', 'Ti_core_lcfs'],
                           fixed_params={'energy_srcCORE': 1e6})

    # Field-vs-field scatter
    scan.plot_field_vs_field('ne_core', 'Te_core', powers=[1e5, 1e6, 5e6])

    # Profile comparison
    scan.compare_profiles('kappa', [1.1, 1.2, 1.3, 1.4],
                          fixed_params={'delta': 0.3, 'energy_srcCORE': 1e6})
"""

import itertools
import numpy as np
import pygkyl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

from . import config
from . import fields as fmod
from .loaders import load_metadata, save_linear_gk_data as _save_linear_gk, \
    clear_linear_gk_data as _clear_linear_gk
from .extraction import extract_field_data as _extract, \
    get_sim_index as _get_idx, get_power_slices, \
    get_multi_dim_index as _get_multi_dim_index
from .plotting import style as _style  # auto-applies style on import
from .plotting.contour import plot_contour_grid as _plot_contour
from .plotting.scatter import plot_field_vs_field as _plot_scatter
from .plotting.profiles import compare_profiles as _compare_profiles
from .plotting import simulation as sim_plots


class ScanMetadata:
    """Unified interface for loading, analysing, and visualising scan results.

    Attributes
    ----------
    metadata_file : Path
    metadata : list[dict]
    fields : list[str]
    locations : list[str]
    scan_params : dict[str, list]
    scan_keys : list[str]
    combinations : list[tuple]
    all_fields : list[str]
    all_field_symbols : dict[str, str]
    field_symbols : dict
    field_units : dict
    field_scaling : dict
    location_symbols : dict
    data : dict[str, np.ndarray]
    """

    def __init__(self, metadata_file: Union[str, Path], bflux_tavg: float = 25.0):
        self.metadata_file = Path(metadata_file)
        self.bflux_tavg = bflux_tavg

        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        #  Load raw metadata
        self.metadata = load_metadata(self.metadata_file)
        if not self.metadata:
            raise ValueError("Metadata file is empty")

        # Detect fields, locations
        self.fields, self.locations, self.available_field_keys = (
            fmod.detect_fields_and_locations(self.metadata)
        )

        # Scan parameters
        self._extract_scan_parameters()

        # Field properties
        self.field_symbols, self.field_units, self.field_scaling, self.location_symbols = (
            fmod.build_field_properties(self.fields, self.locations)
        )
        self.field_refvals = dict(fmod.FIELD_REFVALS)

        # All-field symbols (including composites)
        self.all_fields, self.all_field_symbols = fmod.build_all_field_symbols(
            self.fields, self.locations, self.field_symbols, self.location_symbols
        )

        # Sim directory pattern
        self._detect_sim_pattern()

        # Pre-load & compute composite fields
        self._preload_all_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_scan_parameters(self):
        self.scan_params = {}
        for param in config.KNOWN_SCAN_PARAMS:
            if param in self.metadata[0]:
                vals = sorted(set(entry[param] for entry in self.metadata))
                self.scan_params[param] = vals
        self.scan_keys = list(self.scan_params.keys())
        values = [self.scan_params[k] for k in self.scan_keys]
        self.combinations = list(itertools.product(*values))

    def _detect_sim_pattern(self):
        if 'simdir' in self.metadata[0]:
            simdir = self.metadata[0]['simdir']
            parts = simdir.rstrip('/').rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                self.sim_base_dir = parts[0].rsplit('/', 1)[0] if '/' in parts[0] else '.'
                self.sim_prefix = parts[0].rsplit('/', 1)[-1] if '/' in parts[0] else parts[0]
            else:
                self.sim_base_dir = "."
                self.sim_prefix = "sim"
        else:
            self.sim_base_dir = "."
            self.sim_prefix = "sim"

    def _preload_all_data(self):
        """Load base field data and compute all composite fields."""
        param_shapes = [len(self.scan_params[k]) for k in self.scan_keys]

        self.data = {}

        # Allocate base fields
        all_keys = list(self.available_field_keys)
        for loc in self.locations:
            for pf in ('Pi', 'Pe'):
                key = f'{pf}_{loc}'
                if key not in all_keys:
                    all_keys.append(key)
        for key in all_keys:
            self.data[key] = np.zeros(param_shapes)

        # Fill from metadata
        for entry in self.metadata:
            indices = tuple(self.scan_params[k].index(entry[k]) for k in self.scan_keys)
            for key in self.available_field_keys:
                if key in entry and not isinstance(entry[key], (dict, list)):
                    self.data[key][indices] = entry[key]

        # Store scan parameters as data arrays too
        for param in self.scan_keys:
            self.data[param] = np.zeros(param_shapes)
            for entry in self.metadata:
                indices = tuple(self.scan_params[k].index(entry[k]) for k in self.scan_keys)
                self.data[param][indices] = entry[param]

        # Zero-fill missing flux data
        for flux in ('hflux_xi', 'hflux_xe', 'pflux_xi', 'pflux_xe'):
            for loc in self.locations:
                key = f'{flux}_{loc}'
                if key not in self.data:
                    self.data[key] = np.zeros(param_shapes)

        # Compute all composite fields via the fields module
        fmod.register_all_composite_fields(
            self.data, self.metadata,
            self.fields, self.locations,
            self.scan_params, self.scan_keys,
            self.all_field_symbols, self.field_units,
            self.location_symbols, self.bflux_tavg,
        )

    def _get_slices(self, power_val):
        return get_power_slices(self.scan_params, self.scan_keys, power_val)

    def _ensure_ky(self, ky):
        """Recompute linear GK fields if *ky* changed (or first call)."""
        if ky is None:
            return
        if getattr(self, '_current_ky', None) == ky:
            return
        self._current_ky = ky
        fmod.compute_linear_gk_fields(
            self.data, self.metadata,
            self.scan_params, self.scan_keys,
            self.all_field_symbols, self.field_units, ky,
        )

    # ------------------------------------------------------------------
    # Public API: info
    # ------------------------------------------------------------------

    def info(self):
        """Display summary information about the scan."""
        print("=" * 80)
        print("ScanMetadata Information")
        print("=" * 80)
        print(f"\nMetadata file: {self.metadata_file}")
        print(f"Total simulations: {len(self.metadata)}")
        print(f"Pre-loaded fields: {len(self.data)} (including composites)")
        total_mb = sum(a.nbytes for a in self.data.values()) / (1024 * 1024)
        print(f"Memory usage: {total_mb:.2f} MB")

        print("\n" + "-" * 80)
        print("Detected Fields:")
        print("-" * 80)
        for f in self.fields:
            sym = self.field_symbols.get(f, f)
            unit = self.field_units.get(f, '')
            print(f"  {f:10s} - {sym:10s} {unit}")

        print("\n" + "-" * 80)
        print("Detected Locations:")
        print("-" * 80)
        for loc in self.locations:
            sym = self.location_symbols.get(loc, loc)
            print(f"  {loc:10s} - r/a = {sym}")

        print("\n" + "-" * 80)
        print("Scan Parameters:")
        print("-" * 80)
        for param, vals in self.scan_params.items():
            print(f"  {param:20s}: {len(vals)} values")
            print(f"    Range: [{min(vals)}, {max(vals)}]")
            if len(vals) <= 10:
                print(f"    Values: {vals}")

        print("\n" + "-" * 80)
        print("Available Field Keys (field_location):")
        print("-" * 80)
        for f in self.fields:
            keys = [k for k in self.available_field_keys if k.startswith(f'{f}_')]
            print(f"  {f}: {', '.join(keys)}")

        print("\n" + "-" * 80)
        print("Composite Fields (differences between locations):")
        print("-" * 80)
        diff_fields = [f for f in self.data if f.count('_') == 2]
        print(f"  Total: {len(diff_fields)} difference fields pre-computed")
        if diff_fields:
            print(f"  Examples: {', '.join(diff_fields[:5])}")

        print("\n" + "-" * 80)
        print("Sample Data Ranges (pre-loaded):")
        print("-" * 80)
        for key in self.available_field_keys[:4]:
            if key in self.data:
                v = self.data[key]
                print(f"  {key:15s}: [{v.min():.3e}, {v.max():.3e}]")

        print("=" * 80)

    def __repr__(self):
        return (f"ScanMetadata(file='{self.metadata_file.name}', "
                f"n_sims={len(self.metadata)}, "
                f"n_fields={len(self.data)}, "
                f"params={list(self.scan_params.keys())})")

    # ------------------------------------------------------------------
    # Public API: data access
    # ------------------------------------------------------------------

    def get_sim_index(self, params: Dict[str, Any]) -> int:
        return _get_idx(params, self.scan_keys, self.combinations)

    def get_multi_dim_index(self, params: Dict[str, Any]) -> Tuple:
        return _get_multi_dim_index(params, self.scan_params, self.scan_keys)
    
    def extract_field_data(self, field_names: List[str],
                           fixed_params: Optional[Dict[str, Any]] = None,
                           vary_params: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        return _extract(self.data, field_names, self.scan_params, self.scan_keys,
                        fixed_params=fixed_params, vary_params=vary_params)
        
    # ------------------------------------------------------------------
    # Public API: get a value of a field at a specific index or parameter combination
    # ------------------------------------------------------------------
    
    def get_field_value(self, field_name: str, params: Dict[str, Any] = None, scan_idx: int = None) -> float:
        if field_name not in self.data:
            raise ValueError(f"Field '{field_name}' not found in data")
        if scan_idx is not None:
            return self.data[field_name][scan_idx]
        else:
            idx = self.get_multi_dim_index(params)
            return self.data[field_name][idx]

    # ------------------------------------------------------------------
    # Public API: plotting
    # ------------------------------------------------------------------

    def plot_contour_grid(self, fields_or_data, fields=None,
                          fixed_params=None, suptitle='', method='contourf',
                          cmap='coolwarm', clim=None, deviation=False,
                          show_fig=True, figfilename=None, dpi=300, ky=None):
        self._ensure_ky(ky)
        if isinstance(fields_or_data, list):
            field_list = fields_or_data
            data = self.extract_field_data(field_list, fixed_params=fixed_params)
        elif isinstance(fields_or_data, dict):
            data = fields_or_data
            if fields is None:
                raise ValueError("When passing data dict, must provide 'fields'")
            field_list = fields
        else:
            raise TypeError(f"Expected list or dict, got {type(fields_or_data)}")

        _plot_contour(data, field_list,
                      all_field_symbols=self.all_field_symbols,
                      field_units=self.field_units,
                      field_scaling=self.field_scaling,
                      suptitle=suptitle, method=method,
                      cmap=cmap, clim=clim, deviation=deviation,
                      show_fig=show_fig, figfilename=figfilename, dpi=dpi)

    def plot_field_vs_field(self, field_x: str, field_y: str, **kwargs):
        self._ensure_ky(kwargs.pop('ky', None))
        _plot_scatter(self, field_x, field_y, **kwargs)

    def compare_profiles(self, vary_param, vary_vals,
                         fixed_params=None, field='Ti',
                         frame_idx=500, cut_coords=None,
                         figname=None, cmap='viridis'):
        _compare_profiles(self, vary_param, vary_vals,
                          fixed_params=fixed_params, field=field,
                          frame_idx=frame_idx, cut_coords=cut_coords,
                          figname=figname, cmap=cmap)

    # ------------------------------------------------------------------
    # Public API: simulation-level access
    # ------------------------------------------------------------------

    def setup_simulation(self, delta=None, kappa=None, energy_srcCORE=None,
                         scanidx=None) -> Tuple:
        if scanidx is None:
            if delta is None or kappa is None or energy_srcCORE is None:
                raise ValueError(
                    "Provide either scanidx or all of delta, kappa, energy_srcCORE"
                )
            params = {'delta': delta, 'kappa': kappa, 'energy_srcCORE': energy_srcCORE}
            scanidx = self.get_sim_index(params)

        simdir = f"{self.sim_prefix}/{self.sim_prefix}_{scanidx:05d}/"
        fileprefix = f"{self.sim_prefix}_{scanidx:05d}"

        simulation = pygkyl.load_sim_config(
            config.SIM_CONFIG_NAME, simDir=simdir, filePrefix=fileprefix
        )
        if kappa is not None:
            simulation.geom_param.kappa = kappa
        if delta is not None:
            simulation.geom_param.delta = delta
        simulation.geom_param.qaxis = 1.2
        simulation.geom_param.qlcfs = 2.6
        simulation.geom_param.qprofile_R = 'quadratic'
        simulation.geom_param.update_geom_params()

        sim_frames = pygkyl.file_utils.find_available_frames(simulation, 'field')
        return simulation, sim_frames, simdir, fileprefix

    def plot_2D(self, **kwargs):
        sim_plots.plot_2D(self, **kwargs)

    def plot_1D(self, **kwargs):
        sim_plots.plot_1D(self, **kwargs)

    def plot_poloidal_projection(self, **kwargs):
        sim_plots.plot_poloidal_projection(self, **kwargs)

    def get_volume_integral(self, **kwargs):
        return sim_plots.get_volume_integral(self, **kwargs)

    def get_profile(self, params: Dict[str, Any], field: str,
                    frame_idx: int = 500,
                    cut_coords: Optional[List] = None) -> Tuple:
        if cut_coords is None:
            cut_coords = ['avg', 0.0]

        figout = []
        scanidx = self.get_sim_index(params)
        simulation, sim_frames, _, _ = self.setup_simulation(scanidx=scanidx)
        simulation.plot_1D_time_evolution('x', cut_coords, field,
                                         sim_frames[frame_idx],
                                         close_fig=True, figout=figout)
        fig = figout[0]
        ax = fig.axes[0]
        lines = ax.get_lines()
        xdata = lines[0].get_xdata()
        ydata = lines[0].get_ydata()
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        return xdata, ydata, label, scanidx

    # ------------------------------------------------------------------
    # Public API: linear GK data
    # ------------------------------------------------------------------

    def add_gk_linear_data(self, delta, kappa, energy_srcCORE, ky, omega, params_in=''):
        """
        Merge linear GK data for a specific parameter combination into the internal dictionary.

        Parameters
        ----------
        delta : float
            Triangularity
        kappa : float
            Elongation
        energy_srcCORE : float
            Core power
        ky : array-like
            Binormal wave numbers
        omega : array-like
            Complex frequencies from linear GK calculation
        params_in : str, optional
            Input parameters string
        """
        new_entry = {
            'ky': np.atleast_1d(np.asarray(ky, dtype=np.float64)),
            'omega': np.atleast_1d(np.asarray(omega, dtype=np.complex128)),
            'params_in': params_in
        }
        sim_idx = self.get_sim_index({'delta': delta, 'kappa': kappa, 'energy_srcCORE': energy_srcCORE})
        current_entry = self.metadata[sim_idx].get('linear_gk', {})
        # Check if we already have this ky
        if current_entry == {}:
            self.metadata[sim_idx]['linear_gk'] = new_entry
        else:
            existing_ky = self.metadata[sim_idx]['linear_gk']['ky']
            existing_omega = self.metadata[sim_idx]['linear_gk']['omega']

            # Split new entries into duplicates (overwrite) and truly new (append)
            is_duplicate = np.isin(new_entry['ky'], existing_ky)
            new_ky    = new_entry['ky'][~is_duplicate]
            new_omega = new_entry['omega'][~is_duplicate]

            # Overwrite omega for duplicate ky values
            for ky_val, om_val in zip(new_entry['ky'][is_duplicate], new_entry['omega'][is_duplicate]):
                mask = existing_ky == ky_val
                existing_omega[mask] = om_val

            # Append genuinely new ky values and sort
            if len(new_ky) > 0:
                merged_ky    = np.concatenate((existing_ky, new_ky))
                merged_omega = np.concatenate((existing_omega, new_omega))
                sort_indices = np.argsort(merged_ky)
                self.metadata[sim_idx]['linear_gk']['ky']    = merged_ky[sort_indices]
                self.metadata[sim_idx]['linear_gk']['omega'] = merged_omega[sort_indices]
            else:
                self.metadata[sim_idx]['linear_gk']['omega'] = existing_omega

    def save_gk_linear_data(self):
        """
        Save the GK linear data into the original HDF5 metadata file.
        """
        _save_linear_gk(self.metadata_file, self.metadata)

    def clear_gk_linear_data(self):
        """
        Clear all GK linear data from the metadata and save the cleaned metadata back to the file.
        """
        _clear_linear_gk(self.metadata_file)