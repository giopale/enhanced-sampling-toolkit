# enhanced-sampling-toolkit

Tools for analysing MD + enhanced sampling simulations (LAMMPS + PLUMED/OPES): reweighting to
free-energy surfaces, minimum-energy paths, barrier convergence and structure extraction.

The toolkit is being rebuilt from the scripts in `to_refactor/`; see [REFACTOR.md](REFACTOR.md)
for the plan and the status of each part. At this stage the configuration, reading the simulation
settings, loading CV data, reweighting, combining runs, minimum energy paths, figures, barrier
convergence and the OPES / model-deviation checks are in place.

## Install

```bash
pip install -e '.[dev]'      # from a clone of this repository
```

This installs the `estk` command and the `estk` Python package.

## Quick start

In the folder that contains your run folders (`000/`, `001/`, ...):

```bash
estk init --pattern '00*'    # writes estk.yaml, with an entry for each CV found in COLVAR
estk config show             # settings in effect and where each one comes from
estk config validate         # check estk.yaml after editing it
estk runs                    # temperature, timestep, bias, walls... read for each run
estk cvs d0 cn --time 0 20   # rows kept, CV ranges and frames past walls, per run
estk reweight d0 cn -j 4     # per-run surfaces + all runs combined -> reweight-d0-cn/
estk reweight d0             # 1D profile
estk combine reweight-d0-cn/0*.dat -m equal -o equal.dat   # recombine with another mode
estk mep reweight-d0-cn/combined.dat   # minimum energy path and barriers (+ each run's)
estk info reweight-d0-cn/combined.dat  # grid, units and how a surface or path was made
estk plot surface reweight-d0-cn/combined.dat --path reweight-d0-cn/combined.mep.dat --units eV
estk plot surface reweight-d0-cn/*.dat --units eV       # all runs side by side
estk plot path reweight-d0-cn/combined.mep.dat --units eV -o profile.pdf
estk convergence time d0 cn --until 20 --step 1 --units eV          # barrier vs time
estk convergence bandwidth d0 cn --cv d0 --range 0.005 0.08 --units eV  # vs bandwidth
estk check opes                                # c(t), zed, neff, nker of each run
estk check model-deviation --levels 0.2,0.35   # ML potential trust levels
```

Axis labels come from `estk.yaml` (`cv.d0.label: "d(Ti–O)"`, `cv.d0.units: Å`). Figures use the
`paper` style (single-column journal figures); `--style slides` or your own `.mplstyle` file
change it.

Surfaces and paths are text files (PLUMED grid layout) whose header records how they were made:
temperature, bandwidths, time range, runs and weights, estk version and the command.
`--compress` (or any output name ending in `.gz`) writes them gzip-compressed.

From Python:

```python
from estk import Project

project = Project()                      # uses the nearest estk.yaml
tables = project.load_cvs(["d0", "cn"], time=(0, 20))
tables[0]["d0"], tables[0].bias, tables[0].kbt

result = project.reweight(["d0", "cn"], time=(0, 20))
fes = result.surfaces["000"]           # .grid, .values (kJ/mol, min 0), .metadata
both = result.combined                 # all runs; .error = spread between runs

from estk.paths import minimum_energy_path
path = minimum_energy_path(both)       # .points, .energies, .barrier, .reverse_barrier
path.save("combined.mep.dat")

import estk
fes = estk.load("reweight-d0-cn/combined.dat")   # a Surface, or a path for a .mep.dat file

from estk.plotting import plot_style, plot_surface, plot_path_profile
with plot_style("paper"):
    ax = plot_surface(both, path=path, units="eV")   # returns the matplotlib Axes
```

Coming from the old Hydra configuration:

```bash
estk config migrate to_refactor/config/case/atio2-h2o.yaml -o estk.yaml
```

This keeps the facts about the system (CV limits and bandwidths, atom types, atom groups) and
lists the settings it left out.

## Development

```bash
pytest
ruff check . && ruff format --check .
```
