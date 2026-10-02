"""
Field registry: symbols, units, scaling factors, and composite field builders.

To add a new field, update the relevant dictionary here and, if it requires
a derived computation, add a builder function in ``register_composite_fields``.
"""

import copy
import numpy as np
from . import config


# =====================================================================
# LaTeX symbols for base fields
# =====================================================================
FIELD_SYMBOLS = {
    'Ti': r'T_i',       'Te': r'T_e',
    'ne': r'n_e',       'phi': r'\phi',
    'Pi': r'P_i',       'Pe': r'P_e',
    'ni': r'n_i',       'upar': r'u_\parallel',
    'betae': r'\beta_e',
    'hflux_xi': r'Q_{xi}',   'hflux_xe': r'Q_{xe}',
    'pflux_xi': r'\Gamma_{xi}', 'pflux_xe': r'\Gamma_{xe}',
    'kn': r'k_n',       'kTi': r'k_{T_i}', 'kTe': r'k_{T_e}',
    'avg_dt': r'\Delta t', 'lambda_q': r'\lambda_q',
}

# =====================================================================
# Units for fields
# =====================================================================
FIELD_UNITS = {
    'Ti': r'[eV]',      'Te': r'[eV]',
    'ne': r'[$m^{-3}$]', 'phi': r'[V]',
    'Pi': r'[kPa]',     'Pe': r'[kPa]',
    'ni': r'[$m^{-3}$]', 'upar': r'[m/s]',
    'betae': r'[$\%$]',
    'hflux_xi': r'[MW/m$^2$]', 'hflux_xe': r'[MW/m$^2$]',
    'pflux_xi': r'[m$^{-2}$ s$^{-1}$]', 'pflux_xe': r'[m$^{-2}$ s$^{-1}$]',
    'T': r'[eV]',       'n': r'[$m^{-3}$]',
    'p': r'[V]',        'P': r'[kPa]',
    'Dxi': r'[m$^2$/s]', 'Dxe': r'[m$^2$/s]',
    'chixi': r'[m$^2$/s]', 'chixe': r'[m$^2$/s]',
    'kn': r'',           'kTi': r'',           'kTe': r'',
    'avg_dt': r'[ns]',   'lambda_q': r'[mm]',
}

# =====================================================================
# Scaling factors  (multiply raw value before plotting)
# =====================================================================
FIELD_SCALING = {
    'hflux_xi': 1e-6,  'hflux_xe': 1e-6,
    'pflux_xi': 1e-6,  'pflux_xe': 1e-6,
    'avg_dt': 1e9,     'lambda_q': 1e3,
}

# =====================================================================
# Reference values  (for optional normalisation)
# =====================================================================
FIELD_REFVALS = {
    'T': 200,                         # eV
    'n': 1e19,                        # m^-3
    'p': 200,                         # V
    'P': 200 * 1.602e-19 * 1e19,      # Pa
}

# =====================================================================
# Integrated-moment symbol helpers
# =====================================================================
INTMOM_SPECIES_SYM  = {'ne': r'n_e', 'ni': r'n_i', 'He': r'e', 'Hi': r'i'}
INTMOM_SPECIES_BASE = {'ne': r'\Gamma', 'ni': r'\Gamma', 'He': r'Q', 'Hi': r'Q'}
INTMOM_DIR_SYM  = {'x': 'x', 'z': 'z'}
INTMOM_SIDE_SYM = {'l': r'\ell', 'u': 'u'}

# =====================================================================
# Helper utilities
# =====================================================================

def base_field(key):
    """Strip location suffixes from a field key to get the base field name."""
    result = copy.copy(key)
    for loc in config.LOCATIONS:
        result = result.replace(f'_{loc}', '')
    return result


def detect_fields_and_locations(metadata):
    """Auto-detect field names and location names from metadata keys.

    Returns (fields, locations, available_field_keys).
    """
    sample_keys = set(metadata[0].keys())
    field_location_pairs = []
    for key in sample_keys:
        if key in config.METADATA_KEYS or key in config.KNOWN_SCAN_PARAMS:
            continue
        parts = key.split('_')
        if len(parts) == 2:
            field_location_pairs.append((parts[0], parts[1]))
        elif len(parts) == 3:
            field_location_pairs.append((f'{parts[0]}_{parts[1]}', parts[2]))

    fields = sorted(set(f for f, _ in field_location_pairs))
    locations = sorted(set(l for _, l in field_location_pairs))
    available_field_keys = [f'{f}_{l}' for f, l in field_location_pairs]
    # Always include scalar metadata fields
    for extra in ('avg_dt', 'lambda_q', 'vol_frac'):
        if extra not in available_field_keys:
            available_field_keys.append(extra)
    return fields, locations, available_field_keys


def build_field_properties(fields, locations):
    """Build field_symbols, field_units, field_scaling, location_symbols dicts."""
    fs = {f: FIELD_SYMBOLS.get(f, f) for f in fields}
    fs['Pi'] = FIELD_SYMBOLS.get('Pi', r'P_i')
    fs['Pe'] = FIELD_SYMBOLS.get('Pe', r'P_e')

    fu = dict(FIELD_UNITS)
    for f in fields:
        if f not in fu:
            fu[f] = ''

    fsc = dict(FIELD_SCALING)

    ls = {loc: config.LOCATION_SYMBOLS.get(loc, loc) for loc in locations}

    return fs, fu, fsc, ls


def build_all_field_symbols(fields, locations, field_symbols, location_symbols):
    """Build the complete ``all_field_symbols`` dict including composites.

    Returns (all_fields, all_field_symbols).
    """
    all_fields = []
    afs = {}

    # Base field_location combinations
    for field in fields:
        for loc in locations:
            key = f'{field}_{loc}'
            all_fields.append(key)
            fs = field_symbols.get(field, field)
            ls = location_symbols.get(loc, loc)
            afs[key] = r'${' + fs + r'}^{' + ls + r'}$'

    # Pressure fields
    for loc in locations:
        for pfield in ['Pi', 'Pe']:
            key = f'{pfield}_{loc}'
            if key not in all_fields:
                all_fields.append(key)
            fs = field_symbols.get(pfield, pfield)
            ls = location_symbols.get(loc, loc)
            afs[key] = r'${' + fs + r'}^{' + ls + r'}$'

    # Difference and normalised fields
    all_base = list(set(fields + ['Pi', 'Pe']))
    for field in all_base:
        for i, loc1 in enumerate(locations):
            # LCFS-normalised
            key = f'{field}_norm_{loc1}'
            fs = field_symbols.get(field, field)
            ls1 = location_symbols.get(loc1, loc1)
            afs[key] = (r'${' + fs + r'}^{' + ls1 + r'} / {' + fs + r'}^{\text{sep}}$')
            # Pairwise differences
            for j, loc2 in enumerate(locations):
                if i >= j:
                    continue
                key = f'{field}_{loc1}_{loc2}'
                if key not in all_fields:
                    all_fields.append(key)
                ls2 = location_symbols.get(loc2, loc2)
                afs[key] = (r'${' + fs + r'}^{' + ls1 + r'} - {' + fs + r'}^{' + ls2 + r'}$')

    # Scalar metadata fields
    for extra in ('avg_dt', 'lambda_q'):
        if extra not in all_fields:
            all_fields.append(extra)
        afs[extra] = r'${' + FIELD_SYMBOLS.get(extra, extra) + r'}$'

    # Diffusion coefficient symbols
    for flux in ('hflux_xi', 'hflux_xe', 'pflux_xi', 'pflux_xe'):
        for suffix in ('Dx', 'chix'):
            symbol = 'D' if suffix == 'Dx' else r'\chi'
            key = f'{suffix}{flux[-1]}'
            if key not in afs:
                afs[key] = r'${' + symbol + r'}_{x' + flux[-1] + r'}$'

    # General metadata keys
    for key in config.METADATA_KEYS:
        if key not in afs:
            afs[key] = r'${' + FIELD_SYMBOLS.get(key, key) + r'}$'

    return all_fields, afs


# =====================================================================
# Composite field builders
# =====================================================================
# Each function receives ``(data, fields, locations, scan_params, scan_keys)``
# and modifies ``data`` / ``all_field_symbols`` in place.

def compute_pressure_fields(data, fields, locations, metadata, scan_params, scan_keys):
    """Compute Pi and Pe from Ti, Te, ne at each location."""
    param_shapes = [len(scan_params[k]) for k in scan_keys]
    for loc in locations:
        for pfield, tfield in [('Pi', 'Ti'), ('Pe', 'Te')]:
            key = f'{pfield}_{loc}'
            if key not in data:
                data[key] = np.zeros(param_shapes)
    for entry in metadata:
        indices = tuple(scan_params[k].index(entry[k]) for k in scan_keys)
        for loc in locations:
            if f'Ti_{loc}' in entry and f'ne_{loc}' in entry:
                data[f'Pi_{loc}'][indices] = (
                    1.5 * entry[f'Ti_{loc}'] * entry[f'ne_{loc}'] * 1.602e-19 * 1e-3
                )
            if f'Te_{loc}' in entry and f'ne_{loc}' in entry:
                data[f'Pe_{loc}'][indices] = (
                    1.5 * entry[f'Te_{loc}'] * entry[f'ne_{loc}'] * 1.602e-19 * 1e-3
                )


def compute_difference_fields(data, fields, locations, all_field_symbols):
    """Compute pairwise location differences and LCFS-normalised fields."""
    all_base = list(set(fields + ['Pi', 'Pe']))
    for field in all_base:
        for i, loc1 in enumerate(locations):
            # LCFS-normalised
            field1 = f'{field}_{loc1}'
            fieldlcfs = f'{field}_lcfs'
            if field1 in data and fieldlcfs in data:
                key = f'{field}_norm_{loc1}'
                data[key] = data[field1] / data[fieldlcfs]
            # Pairwise differences
            for j, loc2 in enumerate(locations):
                if i >= j:
                    continue
                field2 = f'{field}_{loc2}'
                if field1 in data and field2 in data:
                    key = f'{field}_{loc1}_{loc2}'
                    data[key] = data[field1] - data[field2]


def compute_temperature_ratio(data, locations, all_field_symbols, location_symbols):
    """Ion-to-electron temperature ratio at each location."""
    for loc in locations:
        key = f'tau_{loc}'
        if f'Ti_{loc}' in data and f'Te_{loc}' in data:
            data[key] = data[f'Ti_{loc}'] / data[f'Te_{loc}']
            all_field_symbols[key] = (
                r'$T_i/T_e|_{' + location_symbols.get(loc, loc) + r'}$'
            )


def compute_diffusivity(data, all_field_symbols):
    """Particle and heat diffusivity coefficients from LCFS fluxes and gradients."""
    loc_vals = config.LOCATIONS
    amid = config.AMID
    delta_x = loc_vals['edge'] - loc_vals['lcfs']

    # Particle diffusivity  D_x{s}
    for flux in ('pflux_xi', 'pflux_xe'):
        s = flux[-1]
        pflux_lcfs = data.get(f'{flux}_lcfs')
        ne_edge = data.get('ne_edge')
        ne_lcfs = data.get('ne_lcfs')
        if pflux_lcfs is None or ne_edge is None or ne_lcfs is None:
            continue
        grad_n = (ne_edge - ne_lcfs) / (amid * delta_x)
        key = f'Dx{s}'
        data[key] = -pflux_lcfs / grad_n
        all_field_symbols[key] = r'$D_{x' + s + r'}^{\text{sep}}$'

    # Heat diffusivity  chi_x{s}
    for flux in ('hflux_xi', 'hflux_xe'):
        s = flux[-1]
        hflux_lcfs = data.get(f'{flux}_lcfs')
        pflux_lcfs = data.get(f'pflux_x{s}_lcfs')
        T_lcfs = data.get(f'T{s}_lcfs')
        ne_edge = data.get('ne_edge')
        ne_lcfs = data.get('ne_lcfs')
        T_edge = data.get(f'T{s}_edge')
        if any(v is None for v in (hflux_lcfs, pflux_lcfs, T_lcfs, ne_edge, ne_lcfs, T_edge)):
            continue
        delta_n = ne_edge - ne_lcfs
        delta_T = T_edge - T_lcfs
        grad_T = delta_T / (amid * delta_x)
        key = f'chix{s}'
        data[key] = -(hflux_lcfs - 1.5 * T_lcfs * pflux_lcfs) / (grad_T * delta_n)
        all_field_symbols[key] = r'$\chi_{' + s + r'}^{\text{sep}}$'

    # Ratios
    if 'chixe' in data and 'chixi' in data:
        data['chixe_over_chixi'] = data['chixe'] / data['chixi']
        all_field_symbols['chixe_over_chixi'] = (
            r'$\chi_{e} / \chi_{i}|_{\text{sep}}$'
        )
    if 'Dxe' in data and 'chixe' in data and 'chixi' in data:
        data['De_over_chitot'] = data['Dxe'] / (data['chixe'] + data['chixi'])
        all_field_symbols['De_over_chitot'] = (
            r'$D_{e} / (\chi_{e} + \chi_{i})|_{\text{sep}}$'
        )
        data['De_over_chie'] = data['Dxe'] / (data['chixe'])
        all_field_symbols['De_over_chie'] = (
            r'$D_{e} / (\chi_{e})|_{\text{sep}}$'
        )

def compute_relative_variations(data, all_field_symbols):
    """Relative edge-to-LCFS drops and gradient scale lengths."""
    raxis = config.RAXIS
    amid = config.AMID
    loc_vals = config.LOCATIONS
    Roverdr = raxis / ((loc_vals['lcfs'] - loc_vals['edge']) * amid)

    for name, field, sym_delta, sym_rl in [
        ('dne_rel', 'ne', r'\Delta n_e / n_{e}', r'R/L_{n_e}|_{\text{sep}}'),
        ('dTe_rel', 'Te', r'\Delta T_e / T_{e}', r'R/L_{T_e}|_{\text{sep}}'),
        ('dTi_rel', 'Ti', r'\Delta T_i / T_{i}', r'R/L_{T_i}|_{\text{sep}}'),
    ]:
        edge_key = f'{field}_edge'
        lcfs_key = f'{field}_lcfs'
        if edge_key in data and lcfs_key in data:
            data[name] = (data[edge_key] - data[lcfs_key]) / data[edge_key] * 100
            all_field_symbols[name] = r'$' + sym_delta + r'$ [%]'
            kname = f'k{field}'
            data[kname] = (data[edge_key] - data[lcfs_key]) / data[edge_key] * Roverdr
            all_field_symbols[kname] = r'$' + sym_rl + r'$'


def compute_energy_density(data, all_field_symbols):
    """Ion energy density at core."""
    if 'Ti_core' in data and 'ne_core' in data:
        data['Edens_i'] = 1.5 * data['Ti_core'] * data['ne_core'] * 1.602e-19
        all_field_symbols['Edens_i'] = r'$E_{dens,i}$ [J/m$^3$]'


def compute_intmom_averages(data, metadata, scan_params, scan_keys,
                            all_field_symbols, field_units, bflux_tavg):
    """Average integrated-moment time series and build aggregate fields.

    Populates ``data`` with time-averaged boundary-flux scalars, directional
    sums, wall aggregates, and confinement-time estimates.
    """
    # Find integrated-moment names from first valid entry
    intmom_names = None
    for entry in metadata:
        if 'intmom' in entry and entry['intmom']:
            intmom_names = list(entry['intmom'].keys())
            break
    if intmom_names is None:
        return

    param_shapes = [len(scan_params[k]) for k in scan_keys]

    for name in intmom_names:
        data[name] = np.full(param_shapes, np.nan)

    def _find_idx(param, value):
        vals = scan_params[param]
        try:
            return vals.index(value)
        except ValueError:
            fval = float(value)
            dists = [abs(float(v) - fval) for v in vals]
            best = int(np.argmin(dists))
            if dists[best] / (abs(float(vals[best])) + 1e-300) < 1e-6:
                return best
            raise

    for entry in metadata:
        if 'intmom' not in entry or not entry['intmom']:
            continue
        try:
            indices = tuple(_find_idx(k, entry[k]) for k in scan_keys)
        except (ValueError, KeyError):
            continue
        for name in intmom_names:
            if name not in entry['intmom']:
                continue
            t = np.asarray(entry['intmom'][name]['time'], dtype=float)
            v = np.asarray(entry['intmom'][name]['values'], dtype=float)
            if len(t) == 0:
                continue
            mask = t >= (t[-1] - bflux_tavg)
            data[name][indices] = np.nanmean(v[mask]) if mask.any() else np.nan

    # Build symbols for individual intmom fields
    for name in intmom_names:
        parts = name.split('_')
        if len(parts) == 4:
            _, d, s, sp = parts
            base = INTMOM_SPECIES_BASE.get(sp, r'\Gamma')
            sp_sym = INTMOM_SPECIES_SYM.get(sp, sp)
            d_sym = INTMOM_DIR_SYM.get(d, d)
            s_sym = INTMOM_SIDE_SYM.get(s, s)
            sym = r'$' + base + r'_{' + d_sym + ',' + s_sym + r'}^{' + sp_sym + r'}$'
        else:
            sym = name
        all_field_symbols[name] = sym
        if name not in field_units:
            field_units[name] = r'[s$^{-1}$]'

    # --- Directional totals ---
    bflux_names = [n for n in intmom_names if 'flux' in n]
    by_dirside = {}
    for name in bflux_names:
        parts = name.split('_')
        if len(parts) == 4:
            _, d, s, _ = parts
            by_dirside.setdefault((d, s), []).append(name)
    for (d, s), members in by_dirside.items():
        tot_key = f'bflux_{d}_{s}_tot'
        data[tot_key] = np.nansum(np.stack([data[m] for m in members], axis=0), axis=0)
        d_sym = INTMOM_DIR_SYM.get(d, d)
        s_sym = INTMOM_SIDE_SYM.get(s, s)
        all_field_symbols[tot_key] = r'$(\Gamma+Q)_{' + d_sym + r',' + s_sym + r'}$'
        field_units[tot_key] = r'[s$^{-1}$]'

    # --- Species-summed per direction/side ---
    for d in INTMOM_DIR_SYM:
        for side in INTMOM_SIDE_SYM:
            n_key = f'bflux_{d}_{side}_n'
            h_key = f'bflux_{d}_{side}_H'
            n1 = f'bflux_{d}_{side}_ne'
            n2 = f'bflux_{d}_{side}_ni'
            h1 = f'bflux_{d}_{side}_He'
            h2 = f'bflux_{d}_{side}_Hi'
            if n1 in data and n2 in data:
                data[n_key] = data[n1] + data[n2]
                d_sym = INTMOM_DIR_SYM[d]
                s_sym = INTMOM_SIDE_SYM[side]
                all_field_symbols[n_key] = (
                    r'$\Gamma_{' + d_sym + r',' + s_sym + r'}$ [s$^{-1}$]'
                )
            if h1 in data and h2 in data:
                data[h_key] = data[h1] + data[h2]
                d_sym = INTMOM_DIR_SYM[d]
                s_sym = INTMOM_SIDE_SYM[side]
                all_field_symbols[h_key] = (
                    r'$Q_{' + d_sym + r',' + s_sym + r'}$ [MW]'
                )

    # --- Wall aggregates  (z_l, z_u, x_u) ---
    _wall_sides = {('z', 'l'), ('z', 'u'), ('x', 'u')}
    _particle_sp = {'ne', 'ni'}
    _heat_sp = {'He', 'Hi'}
    wall_n = [n for n in bflux_names
              if len(n.split('_')) == 4
              and (n.split('_')[1], n.split('_')[2]) in _wall_sides
              and n.split('_')[3] in _particle_sp]
    wall_H = [n for n in bflux_names
              if len(n.split('_')) == 4
              and (n.split('_')[1], n.split('_')[2]) in _wall_sides
              and n.split('_')[3] in _heat_sp]
    if wall_n:
        data['bflux_wall_n'] = np.nansum(np.stack([data[m] for m in wall_n], axis=0), axis=0)
        all_field_symbols['bflux_wall_n'] = r'$\Gamma_\mathrm{wall}$'
        field_units['bflux_wall_n'] = r'[s$^{-1}$]'
    if wall_H:
        data['bflux_wall_H'] = np.nansum(np.stack([data[m] for m in wall_H], axis=0), axis=0)
        all_field_symbols['bflux_wall_H'] = r'$Q_\mathrm{wall}$'
        field_units['bflux_wall_H'] = r'[MW]'


def compute_confinement_time(data, all_field_symbols):
    """Estimate confinement time tau_E = W / P_sol."""
    p, w = 0, 0
    try:
        for s in ('i', 'e'):
            w = w + data[f'W{s}']
            for l in ('x_u', 'z_u', 'z_l'):
                p = p + data[f'bflux_{l}_H{s}']
        data['tau_E'] = w / p
        all_field_symbols['tau_E'] = r'$\tau_E$ [s]'
    except KeyError:
        pass


def compute_linear_gk_fields(data, metadata, scan_params, scan_keys,
                             all_field_symbols, field_units, ky):
    """Interpolate linear GK growth rate and frequency at a given *ky*.

    Populates ``data['gamma']``, ``data['omega']``, and
    ``data['omega_over_gamma']``.
    """
    param_shapes = [len(scan_params[k]) for k in scan_keys]
    gamma_arr = np.full(param_shapes, np.nan)
    omega_arr = np.full(param_shapes, np.nan)

    for entry in metadata:
        lgk = entry.get('linear_gk', {})
        ky_arr = lgk.get('ky')
        omega_complex = lgk.get('omega')
        if ky_arr is None or omega_complex is None:
            continue
        ky_arr = np.asarray(ky_arr, dtype=float)
        omega_complex = np.asarray(omega_complex, dtype=complex)
        if len(ky_arr) == 0 or ky < ky_arr.min() or ky > ky_arr.max():
            continue
        indices = tuple(scan_params[k].index(entry[k]) for k in scan_keys)
        gamma_arr[indices] = np.interp(ky, ky_arr, omega_complex.real)
        omega_arr[indices] = np.interp(ky, ky_arr, omega_complex.imag)

    data['gamma'] = gamma_arr
    data['omega'] = omega_arr
    with np.errstate(divide='ignore', invalid='ignore'):
        data['omega_over_gamma'] = omega_arr / gamma_arr

    all_field_symbols['gamma'] = r'$\gamma\, R/c_s$'
    all_field_symbols['omega'] = r'$\omega\, R/c_s$'
    all_field_symbols['omega_over_gamma'] = r'$\omega / \gamma$'
    field_units['gamma'] = ''
    field_units['omega'] = ''
    field_units['omega_over_gamma'] = ''


# =====================================================================
# Master builder
# =====================================================================

def register_all_composite_fields(data, metadata, fields, locations,
                                  scan_params, scan_keys,
                                  all_field_symbols, field_units,
                                  location_symbols, bflux_tavg=25.0):
    """Run all composite-field builders in the correct order."""
    compute_pressure_fields(data, fields, locations, metadata, scan_params, scan_keys)
    compute_difference_fields(data, fields, locations, all_field_symbols)
    compute_temperature_ratio(data, locations, all_field_symbols, location_symbols)
    compute_diffusivity(data, all_field_symbols)
    compute_relative_variations(data, all_field_symbols)
    compute_energy_density(data, all_field_symbols)
    compute_intmom_averages(data, metadata, scan_params, scan_keys,
                            all_field_symbols, field_units, bflux_tavg)
    compute_confinement_time(data, all_field_symbols)
