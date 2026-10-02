# Gkeyll TCV Miller Geometry Parameter Scan Dataset

This dataset contains results from a comprehensive three-parameter scan of gyrokinetic turbulence simulations for TCV tokamak-like conditions using Miller geometry. The dataset consists of 256 Gkeyll simulations exploring the effects of plasma elongation (κ), triangularity (δ), and core energy source on turbulent transport. The simulation setup is similar to A.C.D. Hoffmann et al. (2025) arXiv:2510.11874.

## Authors and Usage Terms

**Authors**: Antoine C.D. Hoffmann, Manaure Francisquez, Tess N. Bernard, Ammar Hakim, Gregory W. Hammett, and the Gkeyll team.

**Affiliation**: Princeton Plasma Physics Laboratory (PPPL), Princeton, NJ, USA. & General Atomics, San Diego, CA, USA.

This dataset is freely available for scientific and non-commercial use. **Any publication, presentation, or derivative work using this dataset must include at least Antoine C.D. Hoffmann as co-author.**

## Simulation Code

All simulations were performed using **Gkeyll** full-F long-wavelength electrostatic gyrokinetics solver (https://github.com/ammarhakim/gkeyll).

- **Gkeyll Version**: Git commit hash `75c25360e3de`

## Dataset Structure

```
tcv_miller_scan_big/
├── README.md                         # This file
├── parameter_table.txt               # Parameter mapping for all simulations
├── gkeyll.c                          # Gkeyll input file/source
├── submit_scan.sh                    # Main submission script
├── submit_scan_job_*.sh              # Individual job submission scripts
├── std-tcv_miller_scan_big_*.log     # Standard output logs
├── err-tcv_miller_scan_big_*.log     # Error logs
├── slurm_out/                        # SLURM output files
└── tcv_miller_scan_big_XXXXX/        # Simulation directories (256 total)
    ├── *-elc_M0_*.gkyl               # Electron density frame
    ├── *-ion_M3par_*.gkyl            # Ion parallel 3rd moment frame
    ├── *-ion_source_*.gkyl           # Ion source frame
    ├── *-elc_source_Moments_*.gkyl   # Hamiltonian moments
    ├── *-ion_integrated_moms.gkyl    # Integrated ion moments time series
    ├── ...                           # Additional species related files
    ├── *-jacobgeo*.gkyl              # Geometric Jacobians
    ├── *-mapc2p*.gkyl                # Coordinate mappings
    ├── ...                           # Additional geometric and mapping files 
```
Moment time frames are output every 2 microseconds, phase space frames every 20 microseconds, and integrated diagnostics every 0.02 microseconds.

### Handling The Files

The output files follow the pattern: `tcv_miller_scan_big_XXXXX-<quantity>_<frame>.gkyl`

- `XXXXX`: Simulation ID (00000-00255)
- `<quantity>`: Physical quantity (e.g., ion_M3par, ion_M3perp, jacobgeo)
- `<frame>`: Time frame number

All data files use the `.gkyl` format, which can be read using:
- **postgkyl**: Python-based Gkeyll post-processing tool (https://github.com/ammarhakim/postgkyl)
- Load with: `import postgkyl as pg; data = pg.data.GData('filename.gkyl')`
- Use the `parameter_table.txt` file to map simulation IDs to parameter values.

#### Main Output Types

1. **Distribution functions** (`*-ion[elc]_*.gkyl`)
   - 5D frame of distribution functions for ions and electrons
   - Output every 10 microseconds
   
2. **Species Moments** (`*-ion_M0_*.gkyl`, `*-elc_M3perp_*.gkyl`, `*-ion[elc]_HamiltonianMoments_*.gkyl` etc.)
   - 3D frame of moments derived from distribution functions
   - Density, flow velocity, temperature, heat flux, etc.
   - Output every 2 microseconds

4. **Integrated Diagnostics** (`*-ion[elc]_integrated_moms.gkyl`)
   - Volume-integrated quantities
   - Energy, particle number, boundary fluxes, source intensity, etc.
   - Output every 0.02 microseconds

5. **Geometric Data** (`*-jacobgeo*.gkyl`, `*-mapc2p*.gkyl`)
   - Coordinate system Jacobians
   - Mapping between computational and physical coordinates

## Additional Information


### Parameter Space

The dataset explores a 3-dimensional parameter space with 256 simulations:

| Parameter | Name | Values | Count | Description |
|-----------|--------|--------|-------|-------------|
| Elongation | kappa | 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8 | 8 | Plasma cross-section elongation |
| Triangularity | delta | -0.6, -0.45, -0.3, -0.15, 0.15, 0.3, 0.45, 0.6 | 8 | Plasma cross-section triangularity |
| Core Energy Source | energy_srcCORE | 0.1, 0.5, 1.0, 5.0 | 4 | Power injection (MW) |

**Total Simulations**: 8 × 8 × 4 = 256

Each simulation is identified by a five-digit ID (00000-00255). The mapping between simulation IDs and parameter values is provided in [parameter_table.txt](parameter_table.txt).

### Computational Grid

5D phase space grid (2 spatial + 1 toroidal + 2 velocity):

| Dimension | Cells | Range | Description |
|-----------|-------|-------|-------------|
| x (radial) | 36 | [0.0, 0.12] | Radial coordinate |
| y (binormal) | 24 | [-0.092, 0.092] | Binormal coordinate |
| z (toroidal) | 16 | [-π, π] | Toroidal angle |
| vpar | 12 | [-0.707, 0.707] | Parallel velocity |
| mu | 8 | [0.0, 1.0] | Magnetic moment |

**Grid Type**: linear + quadratic in vpar, quadratic in mu
**Total Degrees of Freedom**: 2 x 36 × 24 × 16 × 12 × 8 × 48 (nodes/cell) = ~ 127 million DOF per simulation

### Physics Model
- **Model**: Full-F long-wavelength electrostatic gyrokinetics
- **Geometry**: Limited TCV plasma (domain goes from r/a ~ 0.85 to 1.35)
- **Species**: Kinetic ions and electrons
- **Collisions**: Dougherty operator
- **Boundaries**: Absorbing radial, periodic binormal, twist-and-shift + conducting sheath parallel
- **Sources**: Adaptive particle/energy sources with wall recycling
- **Initial Conditions**: All simulations start from a hyperbolic tangent density and temperature profile

### Computing Resources
- **Computing Facility**: National Energy Research Scientific Computing Center (NERSC), Perlmutter GPU cluster (4 NVIDIA A100 GPUs per node)
- **Account**: m5053
- **Funding**: PPPL and the U.S. Department of Energy (DOE)
- **Total Simulations**: 256
- **Simulated Time Range**: 1.04 - 2.11 milliseconds per simulation
- **Slurm job decomposition**: Batches of 256 Gkeyll instances, distributed on 128 nodes (2 GPUs per instance) with a wall time of 6 hours per batch.
- **Number of Restarts**: 21
- **Total Wall Clock Time**: ~5.4 days
- **Total Node-Hours**: ~16,900 node-hours
- **Total GPU-Hours**: ~67,500 GPU-hours

## Contact

For questions about this dataset, please contact:
- **Antoine C.D. Hoffmann** (ahoffman@pppl.gov)

## Citation

When using this dataset, please cite at least https://iopscience.iop.org/article/10.1088/1741-4326/ae4eff and include, after approval, Antoine C.D. Hoffmann as co-author in any publications or presentations.
A dedicated publication on the data set is in preparation. Please contact the authors for more information.

## Acknowledgments

The authors thank Jimmy Juno for his assistance with code development, the rest of the Gkeyll team for helpful conversations, the TCV team for providing experimental data and support, and Felix Parra and the PPPL theory group for their valuable discussions. This work is supported by a DOE Distinguished Scientist award, the CEDA SciDAC project and other PPPL projects via DOE Contract Number DE-AC02-09CH11466 for the Princeton Plasma Physics Laboratory.

---

**Dataset Location**: `/pscratch/sd/a/ah1032/gkeyll_main/tcv_miller_scan/tcv_miller_scan_big`  
**Last Updated**: March 2026
