"""
Metadata I/O: loading from JSON / HDF5 and saving to HDF5.
"""

import json
import h5py
import numpy as np
from pathlib import Path


def load_metadata(filepath):
    """Load metadata from a JSON or HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to the metadata file.

    Returns
    -------
    list[dict]
        One dictionary per simulation entry.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix in ('.h5', '.hdf5'):
        return _load_h5(filepath)
    return _load_json(filepath)


def _load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def _load_h5(filepath):
    """Load metadata from HDF5 written by ``save_metadata_h5``.

    Expected layout::

        scan_NNNNN/
            attrs: simdir, scanidx, kappa, delta, ...
            intmom/
                <name>/
                    datasets: time, values
                    attrs: tunits, vunits, name
            linear_gk/
                    ky: array (double)
                    omega: array (complex128)
                    params_in: string (bytes)
    """
    metadata = []
    with h5py.File(filepath, 'r') as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            entry = {}
            for key, val in grp.attrs.items():
                if isinstance(val, bytes):
                    entry[key] = val.decode('utf-8')
                elif hasattr(val, 'item'):
                    entry[key] = val.item()
                else:
                    entry[key] = val
            if 'intmom' in grp:
                intmom_dict = {}
                for intmom_name, flux_grp in grp['intmom'].items():
                    def _s(v):
                        if isinstance(v, (bytes, np.bytes_)):
                            return v.decode('utf-8')
                        if hasattr(v, 'item'):
                            v = v.item()
                        return str(v) if not isinstance(v, str) else v
                    try:
                        intmom_dict[intmom_name] = {
                            'time':   np.asarray(flux_grp['time'][:], dtype=float),
                            'values': np.asarray(flux_grp['values'][:], dtype=float),
                            'tunits': _s(flux_grp.attrs.get('tunits', 'mus')),
                            'vunits': _s(flux_grp.attrs.get('vunits', '')),
                            'name':   _s(flux_grp.attrs.get('name', intmom_name)),
                        }
                    except Exception as exc:
                        print(f'Warning: could not load intmom dataset {intmom_name}: {exc}')
                entry['intmom'] = intmom_dict
            if 'linear_gk' in grp:
                lin_gk_dict = {}
                try:
                    lgk_grp = grp['linear_gk']
                    lin_gk_dict = {
                        'ky': np.asarray(lgk_grp['ky'][:], dtype=np.float64),
                        'omega': np.asarray(lgk_grp['omega'][:], dtype=np.complex128)
                        }
                except Exception as exc:
                    print(f'Warning: could not load linear_gk dataset for {grp_name}: {exc}')
                entry['linear_gk'] = lin_gk_dict
            metadata.append(entry)
    return metadata


def save_metadata_h5(filepath, metadata):
    """Save a list of metadata dicts to HDF5.

    Parameters
    ----------
    filepath : str or Path
        Output HDF5 path.
    metadata : list[dict]
        Each dict must contain ``scanidx`` and optionally ``intmom``.
    """
    with h5py.File(filepath, 'w') as f:
        for m in metadata:
            grp = f.create_group(f"scan_{m['scanidx']:05d}")
            for key, val in m.items():
                if key == 'intmom':
                    continue
                grp.attrs[key] = val
            if 'intmom' in m:
                intmom_grp = grp.create_group('intmom')
                for mom_name, mom_data in m['intmom'].items():
                    mg = intmom_grp.create_group(mom_name)
                    mg.create_dataset('time', data=np.asarray(mom_data['time']))
                    mg.create_dataset('values', data=np.asarray(mom_data['values']))
                    mg.attrs['tunits'] = mom_data['tunits']
                    mg.attrs['vunits'] = mom_data['vunits']
                    mg.attrs['name'] = mom_data['name']

def save_linear_gk_data(filepath, metadata):
    """Save linear GK data into an existing HDF5 metadata file.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 metadata file (opened in append mode).
    metadata : list[dict]
        List of metadata dictionaries, each optionally containing a
        ``'linear_gk'`` sub-dict with keys ``'ky'``, ``'omega'``, and
        ``'params_in'``.
    """
    with h5py.File(filepath, 'a') as f:
        for entry in metadata:
            if 'linear_gk' not in entry:
                continue
            lgk = entry['linear_gk']
            scanidx = int(entry['scanidx'])
            omega_array = np.atleast_1d(np.asarray(lgk['omega'], dtype=np.complex128))
            ky_array = np.atleast_1d(np.asarray(lgk['ky'], dtype=np.float64))
            group_name = f"scan_{scanidx:05d}/linear_gk"
            if group_name in f:
                del f[group_name]
            grp = f.create_group(group_name)
            grp.create_dataset('omega', data=omega_array, dtype=np.complex128)
            grp.create_dataset('ky', data=ky_array, dtype=np.float64)
            grp.create_dataset('params_in', data=np.bytes_(lgk.get('params_in', '')))

def clear_linear_gk_data(filepath):
    """Clear linear GK data from an existing HDF5 metadata file.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 metadata file (opened in append mode).
    """
    with h5py.File(filepath, 'a') as f:
        for grp_name in f.keys():
            group_name = f"{grp_name}/linear_gk"
            if group_name in f:
                del f[group_name]