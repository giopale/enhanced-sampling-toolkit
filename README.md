# enhanced-sampling-toolkit

Tools for analysing MD + enhanced sampling simulations (LAMMPS + PLUMED/OPES): reweighting to
free-energy surfaces, minimum-energy paths, barrier convergence and structure extraction.

The toolkit is being rebuilt from the scripts in `to_refactor/`; see [REFACTOR.md](REFACTOR.md)
for the plan and the status of each part. At this stage only the project configuration is in
place.

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
