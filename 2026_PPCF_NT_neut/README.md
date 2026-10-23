# DIII-D inner-wall-limited gyrokinetic inputs

This directory contains the C input files used for the positive- and
negative-triangularity DIII-D inner-wall-limited (IWL) simulations. The inputs
use three configuration-space dimensions and two gyrokinetic velocity-space
dimensions (`3x2v`) with first-order serendipity basis functions.

## Input files

| Input | Triangularity | Neutral species | Plasma sources | Default end time |
| --- | ---: | --- | --- | ---: |
| `gk_d3d_iwl_posD_base_3x2v.c` | `+0.4` | No | Core and recycling | `0.50 ms` |
| `gk_d3d_iwl_posD_neut_3x2v.c` | `+0.4` | Fluid `D0` | Core | `0.45 ms` |
| `gk_d3d_iwl_negD_neut_3x2v.c` | `-0.4` | Fluid `D0` | Core | `0.50 ms` |

The neutral cases include electron-impact ionization, recombination,
charge exchange, and wall-recycling normalization. The base case omits the
evolved neutral species and represents recycling through an adaptive plasma
source.

All cases currently use the following default resolution:

```text
configuration space: 60 x 60 x 16
velocity space:      12 x 8
polynomial order:    1
```

The LCFS is located one third of the way across the radial domain, so radial
resolutions should remain divisible by three.

## Required Gkeyll revision

These inputs target Gkeyll commit:

```text
d13698dc89cca4d1d5600d46b60f3353d9ba5d2d
```

Using a substantially newer Gkeyll installation may produce errors for the
IWL boundary-condition, biased-plane, geometry, or neutral-recycling APIs.
Check out and install the pinned revision before compiling the inputs:

```bash
cd /path/to/gkeyll
git checkout d13698dc89cca4d1d5600d46b60f3353d9ba5d2d

# Install dependencies and configure using the appropriate machine scripts
# or your site-specific configuration. For example, on macOS:
./machines/mkdeps.macos.sh
./machines/configure.macos.sh
make install -j 8
```

The exact configuration command depends on the machine. MPI and GPU support
must be enabled when Gkeyll is configured if they will be used at run time.
The default installation prefix is `$HOME/gkylsoft`. For a custom
configuration that builds only this solver, pass `--app=gyrokinetic` directly
to `./configure` along with the compiler and dependency options appropriate to
the machine.

## Compiling the inputs

An installed Gkeyll tree provides a makefile for external C inputs. From this
directory, build all three executables with:

```bash
make -f "$HOME/gkylsoft/gkeyll/share/Makefile"
```

To build only one input:

```bash
make -f "$HOME/gkylsoft/gkeyll/share/Makefile" \
  FILE=gk_d3d_iwl_posD_neut_3x2v.c
```

This produces an executable beside each source file, with the `.c` suffix
removed. If Gkeyll was installed under another prefix, replace
`$HOME/gkylsoft` accordingly or pass the matching `PREFIX` to `make`.

For a CUDA-enabled installation, the external input may need to be compiled
with the same CUDA compiler and architecture used for Gkeyll, for example:

```bash
make -f "$HOME/gkylsoft/gkeyll/share/Makefile" \
  CC=nvcc CUDA_ARCH=80 FILE=gk_d3d_iwl_posD_neut_3x2v.c
```

## Running

Each input uses a distinct internal Gkeyll application name and therefore a
distinct output prefix:

| Input | Output prefix |
| --- | --- |
| `gk_d3d_iwl_posD_base_3x2v.c` | `gk_d3d_iwl_posD_base_3x2v` |
| `gk_d3d_iwl_posD_neut_3x2v.c` | `gk_d3d_iwl_posD_neut_3x2v` |
| `gk_d3d_iwl_negD_neut_3x2v.c` | `gk_d3d_iwl_negD_neut_3x2v` |

Separate run directories remain convenient for
keeping logs, restart files, and post-processing products isolated.

Example serial runs:

```bash
mkdir -p runs/posD_base runs/posD_neut runs/negD_neut

cd runs/posD_base
../../gk_d3d_iwl_posD_base_3x2v

cd ../posD_neut
../../gk_d3d_iwl_posD_neut_3x2v

cd ../negD_neut
../../gk_d3d_iwl_negD_neut_3x2v
```

For a short smoke test, use step mode:

```bash
./gk_d3d_iwl_posD_neut_3x2v -s 10
```

For an MPI run, enable MPI with `-M` and specify a domain decomposition whose
product equals the number of MPI ranks. For example, an eight-rank `2 x 2 x 2`
decomposition is:

```bash
mpirun -n 8 ./gk_d3d_iwl_posD_neut_3x2v -M -c 2 -d 2 -e 2
```

For a GPU-enabled build, add `-g`:

```bash
./gk_d3d_iwl_posD_neut_3x2v -g
```

For multi-GPU execution, combine the MPI and GPU options using the launcher
and rank-to-device binding required by the local system.

## Useful command-line options

All options are optional. Run an executable with `-h` to display the complete
list supported by the pinned revision.

| Option | Meaning |
| --- | --- |
| `-s N` | Stop after `N` update steps; useful for smoke tests |
| `-M` | Enable MPI |
| `-g` | Enable GPU execution |
| `-c PX -d PY -e PZ` | Set the configuration-space MPI decomposition |
| `-x NX -y NY -z NZ` | Override the configuration-space resolution |
| `-u NVPAR -v NMU` | Override the two velocity-space resolutions |
| `-b ms` | Use the modal-serendipity basis used by these inputs |
| `-r N` | Restart from output frame `N` |
| `-m` | Enable memory-allocation tracing |

Command-line resolution values override the defaults in `create_ctx()`.

## Restarting

To continue a run at its existing resolution, launch from the directory that
contains its output files and pass the desired frame number:

```bash
./gk_d3d_iwl_posD_neut_3x2v -r 200
```

For MPI restarts, use the same MPI options and a compatible decomposition:

```bash
mpirun -n 8 ./gk_d3d_iwl_posD_neut_3x2v \
  -M -c 2 -d 2 -e 2 -r 200
```

The commented `init_from_file` blocks inside the species definitions import 
distribution functions and an inverse Jacobian from explicitly named files, 
for example when changing resolution. Edit and uncomment those blocks only 
when performing such an import; they are not needed for an ordinary `-r` 
restart.

## Output cadence and files

The tests write 500 frames over `0.50 ms`. Thus, all bthree currently 
write configuration-space data every `1 microsecond`.

`write_phase_freq = 0.1` gives a phase-space output interval ten times the
configuration-space interval. Integrated diagnostics are evaluated more
frequently than field output.

Gkeyll writes prefixed `.gkyl` data files and a statistics JSON file in the
working directory. The distinct prefixes listed above identify the case that
produced each file.

## Parameters commonly edited

Most simulation parameters are collected in `create_ctx()` in each input.
The most commonly adjusted groups are:

- magnetic geometry: `a_shift`, `R_axis`, `R_LCFSmid`, `kappa`, and `delta`;
- plasma reference values: `n0`, `Te0`, and `Ti0`;
- neutral density and recycling controls in the neutral cases;
- source powers, widths, locations, and adaptive-source settings;
- `num_cell_*`, `poly_order`, and velocity-space bounds;
- `final_time`, `num_frames`, `write_phase_freq`, and time-step failure limits.

When changing the resolution, preserve the ordering of `ctx.cells`:

```text
{x, y, z, v_parallel, mu}
```

Before a production run, compile the edited input and perform a short serial
or single-node smoke test with `-s 10`.
