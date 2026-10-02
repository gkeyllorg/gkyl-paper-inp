# scan_analysis

Modular Python toolkit for analysing Gkeyll parameter-scan simulations. It gathers metadata from simulation outputs, pre-computes derived quantities (pressures, diffusivities, gradient scale lengths, confinement times, …), and provides plotting routines for contour grids, field-vs-field scatter plots, and profile comparisons.

## Package structure

```
scan_analysis/
├── config.py                # Central config: geometry, scan arrays, locations, fields, filters
├── fields.py                # Field registry: symbols, units, scaling, composite-field builders
├── loaders.py               # Read/write metadata (JSON, HDF5)
├── extraction.py            # Slice pre-loaded arrays along scan-parameter axes
├── scan_metadata.py         # ScanMetadata class (main user-facing API)
├── gather/
│   ├── simulation.py        # pygkyl simulation setup & field extraction
│   ├── diagnostics.py       # Log-file dt, SOL heat-flux width (λ_q)
│   └── runner.py            # Parallel gather pipeline & CLI
└── plotting/
    ├── style.py             # Matplotlib style defaults
    ├── contour.py           # 2-D contour grids on parameter planes
    ├── scatter.py           # Shape-encoded field-vs-field scatter plots
    ├── profiles.py          # 1-D profile comparison across parameter values
    └── simulation.py        # Wrappers for pygkyl 2-D/1-D/poloidal plots
```

## Quick start

### Analyse existing metadata

```python
from scan_analysis import ScanMetadata

scan = ScanMetadata('data/tcv_miller_scan_big_metadata_frame_500_navg_25.h5')
scan.info()

# Contour grid on the κ–δ plane at fixed power
scan.plot_contour_grid(['Ti_core', 'Te_core'],
                       fixed_params={'energy_srcCORE': 1e6})

# Field-vs-field scatter
scan.plot_field_vs_field('ne_core', 'Te_core', powers=[0.5e6, 1e6, 5e6])

# Compare radial profiles
scan.compare_profiles('kappa', [1.1, 1.3, 1.5],
                      fixed_params={'delta': 0.3, 'energy_srcCORE': 1e6})
```

### Gather new metadata

```python
from scan_analysis.gather import run_gather

run_gather(scandir='tcv_miller_scan_big', ncpu=4)
```

Or from the command line:

```bash
python -m scan_analysis.gather.runner --ncpu 4 --scandir tcv_miller_scan_big
```

## Adding a new derived field

1. Add its symbol, unit, and scaling factor to the dictionaries in `fields.py`.
2. Write a small builder function (receives the `data` dict and modifies it in-place).
3. Register the builder in `register_all_composite_fields()` at the bottom of `fields.py`.

## Adding a new scan parameter

Add the parameter name to `KNOWN_SCAN_PARAMS` in `config.py`. It will be auto-detected from the metadata keys. Optionally add a LaTeX symbol in `SCAN_PARAM_SYMBOLS` and an axis label in `SCAN_PARAM_LABELS`.
