# enhanced-sampling-toolkit

Tools for analysing MD + enhanced sampling simulations (LAMMPS + PLUMED/OPES): reweighting to
free-energy surfaces, minimum-energy paths, barrier convergence and structure extraction.

The toolkit is being rebuilt from the scripts in `to_refactor/`; see [REFACTOR.md](REFACTOR.md)
for the plan and the status of each part. At this stage the configuration, reading the simulation
settings, loading CV data, reweighting, combining runs and minimum energy paths are in place.

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
```

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
