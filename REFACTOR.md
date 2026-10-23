# Refactor plan

This document tracks the refactor of the code in `to_refactor/` into an installable,
documented toolkit. There is one section per functionality, each with:

- **Current state**: what the code does today and where it lives
- **Problems**: what makes it hard to use, maintain or trust
- **Target design**: where it goes and what the API/CLI looks like
- **Status**: `todo` / `in progress` / `done`

Update the status line of a section when work on it starts or finishes, and the
[status overview](#status-overview) table with it.

**Naming:** the pip package is `enhanced-sampling-toolkit`; the import package, the command and
the project file are `estk` / `estk.toml`. (`est` is already taken on PyPI.)

---

## Design overview (A + C)

The toolkit becomes a **Python library first** (usable from scripts and notebooks), with a
**thin command-line interface** on top. Hydra is removed.

### Principles

1. **Infer, don't ask.** Anything the simulation files already contain (temperature, timestep,
   bias factor, PLUMED walls, available CVs) is read from them. The config only overrides it.
2. **Explicit data flow.** Every step returns an object (`FreeEnergySurface`, `Path`, ...) and
   writes one self-describing file. The next step takes that object or file as input. No step
   finds its input by globbing for a file-name convention.
3. **Each kind of setting has one home:**

   | Kind of setting | Lives in | Example |
   |---|---|---|
   | Facts about the system | `estk.toml` (project file) | CV labels, limits, bandwidths, atom types |
   | Choices for this run | CLI flags / function arguments | which CVs, time window, number of minima |
   | Cosmetics | plot styles (`--style paper`, `.mplstyle`) | colours, font sizes, figure size |
   | Defaults | code | grid size 100×100, string method `dt` |

4. **No global state.** Functions take arguments and return values. Nothing reads or rewrites a
   shared config object.
5. **Errors are exceptions with clear messages.** No `sys.exit()` in library code; the CLI turns
   exceptions into a clean error message and a non-zero exit code.

### Package layout

```
pyproject.toml
src/estk/
├── __init__.py          # public API: Project, FreeEnergySurface, Path, ...
├── config.py            # estk.toml schema (pydantic) + loader
├── errors.py            # EstError and subclasses
├── logging.py           # get_logger, CLI logging set-up
├── migrate.py           # old Hydra case YAML -> estk.toml
├── project.py           # Project: run discovery + high-level methods
├── io/
│   ├── lammps.py        # log.lammps parsing
│   ├── plumed.py        # PLUMED input / state / COLVAR parsing
│   ├── colvar.py        # time-aligned CV table assembly
│   ├── grids.py         # FES / property grid read/write (native + PLUMED text)
│   └── trajectory.py    # H5MD / lammpstrj / xyz frame access
├── reweight/
│   ├── weights.py       # bias -> frame log-weights
│   ├── kde.py           # KDE FES and KDE-averaged properties (log-space)
│   ├── bandwidth.py     # Silverman, SCV
│   └── average.py       # combining several runs
├── surface.py           # FreeEnergySurface / PropertySurface classes
├── paths/
│   ├── string2d.py      # string method
│   └── mep.py           # minima picking, MEP, barrier
├── analysis/
│   ├── convergence.py   # barrier vs time / vs parameter
│   ├── structures.py    # frame selection near points / in basins
│   ├── along_path.py    # CV averages along a path
│   └── proton_transfer.py
├── plotting/
│   ├── styles/          # bundled .mplstyle files
│   ├── surface.py
│   ├── path.py
│   └── diagnostics.py
├── cli/
│   ├── main.py          # Typer app: `estk ...`
│   └── ...              # one module per command group
└── logging.py
tests/
docs/
```

### User-facing shape

```toml
# estk.toml
[runs]
root = "."
pattern = "00*"

[files]                 # optional; these are the defaults
lammps_log = "log.lammps"
plumed_input = "plumed.plumed"
colvar = "COLVAR"

[cv.d0]
label = "d(Ti–O) (Å)"
limits = [1.5, 2.5]
bandwidth = 0.0216

[cv.cn]
label = "CN(O–H)"
limits = [0.75, 3.0]

[atoms]
type_map = { 1 = "H", 2 = "O", 3 = "Ti" }
```

```python
from estk import Project
proj = Project("estk.toml")
fes = proj.reweight(cvs=("d0", "cn"), time=(0, 20))
path = fes.mep(minima=2, split="vertical")
print(path.barrier)
```

```bash
estk reweight --cvs d0 cn --time 0 20            # -> d0-cn.fes.h5
estk mep d0-cn.fes.h5 --minima 2 --split vertical # -> d0-cn.mep.h5 + plot
```

---

## Status overview

| # | Functionality | Status |
|---|---|---|
| 1 | [Packaging and entry point](#1-packaging-and-entry-point) | in progress |
| 2 | [Configuration](#2-configuration) | done |
| 3 | [Logging and error handling](#3-logging-and-error-handling) | done |
| 4 | [Run discovery and simulation metadata](#4-run-discovery-and-simulation-metadata) | todo |
| 5 | [CV data loading](#5-cv-data-loading) | todo |
| 6 | [Frame weights and KDE reweighting](#6-frame-weights-and-kde-reweighting) | todo |
| 7 | [Combining runs](#7-combining-runs) | todo |
| 8 | [Bandwidth selection](#8-bandwidth-selection) | todo |
| 9 | [Surface file format](#9-surface-file-format) | todo |
| 10 | [Minimum-energy path (string method)](#10-minimum-energy-path-string-method) | todo |
| 11 | [Barrier convergence](#11-barrier-convergence) | todo |
| 12 | [Surface plotting](#12-surface-plotting) | todo |
| 13 | [CVs along the path](#13-cvs-along-the-path) | todo |
| 14 | [Structure extraction](#14-structure-extraction) | todo |
| 15 | [Proton-transfer CVs](#15-proton-transfer-cvs) | todo |
| 16 | [Diagnostic plots](#16-diagnostic-plots) | todo |
| 17 | [Standalone tools](#17-standalone-tools) | todo |
| 18 | [Tests, CI and docs](#18-tests-ci-and-docs) | todo |

---

## 1. Packaging and entry point

**Status:** in progress

**Done:** `pyproject.toml` (setuptools, `src/estk/`, Python ≥ 3.10, `estk` console script, extras
`migrate` and `dev`, ruff and pytest settings), package skeleton with `estk --version`.
**Left:** extras `structures`/`pt` and the scientific dependencies (added with the sections that
need them); delete the dead files and `to_refactor/` once their logic has moved.

### Current state
- `to_refactor/fes.py` is a Hydra app. It reads `cfg._target_` (e.g. `commands.mep.run`),
  imports the module and calls `run(cfg)`.
- Modules import each other as top-level names (`from utils import ...`,
  `from logger_setup import logger`), so the code only works when run from inside `to_refactor/`.
- No `pyproject.toml` or requirements file. Dependencies are implicit: hydra-core, omegaconf,
  rich, numpy, scipy, matplotlib, pandas, h5py, ase, MDAnalysis, tqdm, `ectoolkits`.
- Dead files: `commands/reweight_property_old.py`, `commands/find_structures_bin.py`,
  `commands/plot_cvs_along_mep.py.bak`.

### Problems
- Can't be installed, imported from elsewhere, or tested.
- `parser.py` has the same name as a former standard-library module, and `utils.py` is a
  catch-all.
- Each command module re-implements its own startup (style, job discovery, logging banner).

### Target design
- `pyproject.toml` with `src/estk/` layout and a console script `estk = estk.cli.main:app`.
- Core dependencies: numpy, scipy, matplotlib, pydantic, typer, rich, h5py.
- Optional extras: `estk[structures]` (ase), `estk[pt]` (MDAnalysis, ectoolkits).
- Code modules arranged as in the [package layout](#package-layout); relative imports inside the
  package.
- Delete the dead files once their remaining useful logic has been moved (see sections 6, 13, 14).
- Keep `to_refactor/` read-only during the migration and delete it at the end.

---

## 2. Configuration

**Status:** done

**Done:** `src/estk/config.py` (schema, `find_config`, `load_config`, `explain`, template),
`src/estk/migrate.py`, and the commands `estk init`, `estk config show|validate|migrate`.
Tested in `tests/test_config.py`, `tests/test_migrate.py`, `tests/test_cli.py`.
**Handled elsewhere:** `config show` lists only `estk.toml` and default values for now; values
from CLI flags and from the simulation files are added with sections 4 and 6. Plot styles are
part of section 12.

### Current state
- `config/common.yaml` holds defaults. `config/command/*.yaml` only set `_target_` and a dummy
  `param`. `config/case/*.yaml` (up to ~600 lines) hold everything else.
- The case file mixes system facts (`cvs.<name>.limits/sigma`, atom indices), command settings
  (`convergence_barrier.*`, `find_structures*`, `cvs_along_mep.*`), plot cosmetics
  (`surface_data.<prop>.plot.*`) and per-run choices (`time_interval`, `property`,
  `just_average`, `cvs.select`).
- The `mep:` block is repeated 4 times and the `plot:` block 7 times under `surface_data`.
- Hard-coded absolute paths (`mplstyle.dir: /users/pgiorgio/...`, cluster scratch paths).
- `temp: 350` is in the config but the temperature is actually parsed from `log.lammps`.

### Problems
- A new user can't tell which keys matter for which command.
- Global keys change the behaviour of every command, and commands rewrite the config
  (`convergence_barrier_time` sets `cfg.time_interval` then calls `reweight_property`).
- No validation: typos are silently ignored or fail deep inside a run.
- Hydra override syntax (`cvs.select=[d0,cn]`, `???`, `+key=`) is a hurdle for newcomers.

### Target design
- One small, optional `estk.toml` per project, validated with pydantic models in `estk/config.py`:
  - `[runs]`: `root`, `pattern`
  - `[files]`: names of the LAMMPS log, PLUMED input and COLVAR files
  - `[cv.<name>]`: `label`, `units`, `limits`, `bandwidth`, `scale`, `source` (for CVs stored
    in separate `CV-<name>` files). TOML has no `null`, so a one-sided range is written as
    `min = 0.75` or `max = ...`; `limits = [lo, hi]` is shorthand for both.
  - `[atoms]`: `type_map`, named groups of 1-based atom indices (e.g. `water_oxygens`, surface
    sites)
  - `[plot]`: default `style`
- Every command setting becomes a CLI flag / function argument with a default in code.
- Plot cosmetics move to bundled styles (`estk/plotting/styles/*.mplstyle`) chosen with `--style`.
  A user can pass a path to their own `.mplstyle`.
- Precedence: CLI flag > `estk.toml` > value read from the simulation files > code default.
- `estk init` writes a commented `estk.toml` with the CVs found in the first run's COLVAR.
- `estk config show` prints the merged, effective settings and where each value came from.
- Migration: a one-off `estk config migrate case.yaml` that converts the CV block and atom
  groups of an old case file and lists the keys it dropped.
- The config file is searched for in the current folder and its parents; `--config` selects
  another file. Relative paths in it are resolved against the file's folder.

---

## 3. Logging and error handling

**Status:** done

**Done:** `src/estk/logging.py`, `src/estk/errors.py`, and error handling in the CLI.
**Handled elsewhere:** `--skip-failed` is added with the multi-run reweighting (section 6).

### Current state
- `logger_setup.py` clears the root logger's handlers at import and configures a logger named
  `"opes-metad-fes"`.
- Output mixes `logger`, `rich.print` and bare `print`.
- Errors are handled with `sys.exit()` (often with no message) and `assert`.
- `ProcessPoolExecutor` failures in `reweight_property` are logged and the run continues, so a
  failed run silently drops out of the average.

### Problems
- Importing the package changes global logging for anyone who uses it as a library.
- Library code that calls `sys.exit()` can't be used from a notebook.

### Target design
- `estk/logging.py`: library modules use `logging.getLogger(__name__)` and never configure
  handlers. The CLI installs a `rich` handler and maps `-v/-q` to log levels.
- Custom exceptions (`EstError`, `ConfigError`, `MissingFileError`, `MissingLimitsError`) with
  clear messages. The CLI catches `EstError` and prints it without a traceback (unless `-v`).
- A failure in a run stops the combined calculation unless `--skip-failed` is given, and
  skipped runs are listed in the summary and the output metadata.

---

## 4. Run discovery and simulation metadata

**Status:** todo

### Current state
- `utils.find_job_dirs(dir_md, dir_job)` globs run folders.
- `parser.py` reads:
  - temperature from the `fix ... nvt temp` line of `log.lammps` (fixed column `parts[5]`)
  - timestep and units
  - `STATE_WFILE` from the PLUMED input, then the bias factor from that state file
  - `LOWER_WALLS`/`UPPER_WALLS` `AT=` values for a CV
  - COLVAR header fields
- Only the first 1000 (or 200) lines are scanned.
- The run index comes from the folder name if it contains `job-00`, otherwise from the position
  in the glob list.
- Each command repeats the same sequence: parse temperature, state file, bias factor, then
  compute `kbT = T*0.0083144621`.

### Problems
- Fixed column positions break on any variation of the LAMMPS input (`fix 1 all nvt temp ...`
  vs `fix nvt all nvt temp ...`, variables like `${T}`).
- The run index depends on glob order, so outputs can be renamed silently when folders change.
- Units are assumed (kJ/mol for energies, ps for PLUMED time) and not recorded.

### Target design
- `estk.io.lammps.read_log_metadata(path) -> LammpsMetadata(temperature, timestep, units)`,
  parsing by keyword (`temp` followed by values) rather than column number, resolving simple
  `variable` definitions, and scanning the whole file.
- `estk.io.plumed.read_input(path) -> PlumedInput` with `state_file`, `walls: dict[cv, (lo, hi)]`,
  `bias_factor`, `energy_units`, `time_units`.
- `Run` dataclass: `path`, `name` (the folder name, used as the stable identifier),
  `metadata` (temperature, kbT, timestep, bias factor, walls, units).
- `Project.runs` discovers runs from `[runs]` and builds `Run` objects once.
  The CLI shows a table with `estk runs`.
- Physical constants and unit conversions live in one place (`estk/units.py`).

---

## 5. CV data loading

**Status:** todo

### Current state
- `utils.read_colvar_in_time_interval` reads COLVAR rows in a time window (config in ns,
  PLUMED time assumed to be ps).
- CVs missing from COLVAR are read from `CV-<name>` files (computed afterwards with the PLUMED
  driver) and merged with `safe_stack_colvar`.
- Optional property files (e.g. `model_devi.out`, indexed by step) are merged by converting
  steps to time.
- NaN masking and `cvs.filter` range filtering.
- `cvs.<name>.scale` (e.g. radians to degrees) is only applied in `plot_cvs_along_mep`.
- The whole block (≈100 lines) is copied in `reweight_property`, `find_structures`,
  `find_structures_level_set` and `plot_cvs_along_mep`.

### Problems
- Four copies that have already diverged (scaling exists in one, NaN warnings differ).
- `masks[0] & masks[1]` assumes exactly two CVs.
- Rows are merged by position (`safe_stack_colvar`), not by time, so a missing or duplicated
  row in a `CV-*` file shifts every following row. (`estimation_bandwidth/minimize_scv_approx.py`
  already has de-duplication and time alignment that could be reused.)
- The time-unit assumption (ns in config, ps in COLVAR) is implicit.

### Target design
- `estk.io.colvar.load_cv_table(run, cvs, *, time=None, extra=None) -> CVTable`.
  - `CVTable` holds `time`, `bias`, and a column per CV (a small wrapper around a structured
    array or `pandas.DataFrame`).
  - Columns come from COLVAR, `CV-<name>` files (from `cv.<name>.source` or the default naming
    rule), or property files (`extra={"max_devi_f": "model_devi.out"}`).
  - Rows are joined **on time**, with duplicate times removed (keep last, as the SCV script
    does) and a warning if the files don't overlap.
  - `scale` from the config is applied at load time for every command.
  - NaN rows and rows outside `filter` ranges are dropped, with the counts logged and stored.
- Works for any number of CVs.
- Time windows are given in ns at the interface. Conversion uses the units read from PLUMED.

---

## 6. Frame weights and KDE reweighting

**Status:** todo

### Current state
- Two implementations exist:
  - `ectoolkits_reweight.py` (≈1100 lines): grid-point-by-grid-point `calc_FES`,
    `calc_property_surface`, `calc_c_of_t`, parallelised with `concurrent.futures`; plus file I/O.
  - `utils_reweighting.py` (≈490 lines): newer, vectorised, log-space versions:
    `compute_frame_weights`, `make_grid_2d`, `fes_kde_2d(_logsum)`, `kde_avg_2d_logsum_signed`,
    `point_kernel_weights`.
- `commands/reweight_property.py` uses the new functions for the FES and properties, and still
  imports the old ones.
- Weights are `exp(+V/kbT)` from the `opes.bias` column, which is correct for OPES.
- The kernel is a diagonal 2×2 Gaussian; the grid is fixed to 2D.
- The output name is `potNNN-<cv0>-<cv1>-rew-<prop>.dat` in each run folder.

### Problems
- Two code paths for the same thing, and it's unclear which is authoritative.
- The bias column name `opes.bias` is hard-coded, so metadynamics or other OPES labels don't work.
- 2D only.
- `fes_shift_to_zero` default is `False` per run, but `read_fes` shifts the minimum to zero
  anyway when reading.

### Target design
- `estk.reweight.weights.log_weights(bias, kbT) -> np.ndarray` (log-space only, no `exp` until
  needed).
- `estk.reweight.kde`:
  - `fes(cvs: CVTable, grid: Grid, bandwidth, *, kbT) -> FreeEnergySurface`
  - `average_property(cvs, quantity, grid, bandwidth) -> PropertySurface`
  - Built on the log-sum-exp kernels from `utils_reweighting.py`, generalised to N dimensions
    with a full or diagonal bandwidth matrix. Memory is controlled by chunking over frames.
- `Grid` object (limits + number of points per CV). Limits are resolved in this order:
  CLI > `estk.toml` > PLUMED walls > data range.
- Bias column is detected from the PLUMED input (`*.bias` of the biasing action) or set with
  `--bias-column`.
- `Project.reweight(cvs, *, property=None, time=None, grid=None, bandwidth=None)` runs each
  run in parallel and returns per-run surfaces plus the combined one (section 7).
- CLI: `estk reweight --cvs d0 cn [--property max_devi_f --source model_devi.out] [--time 0 20]
  [--grid 100 100] [--bandwidth 0.02 0.03] [-o d0-cn.fes.h5]`.
- Port `calc_c_of_t` only if it's still needed; otherwise drop `ectoolkits_reweight.py`.
- Validate the new code against the current outputs on one real dataset before deleting the old
  modules (regression test, section 18).

---

## 7. Combining runs

**Status:** todo

### Current state
- `reweight_property.run` reads all per-run files back from disk, weights each run by
  `exp(log_weight_total)` and writes `<cv0>-<cv1>-rew-<prop>-AVG.dat` with a standard deviation
  column.

### Problems
- **Probable correctness bug:** `compute_frame_weights` subtracts each run's own maximum
  log-weight before `log_weight_total` is computed. The totals are therefore not on a common
  scale, and the weight given to each run in the average is not what was intended.
- Free energies are averaged directly. Averaging probabilities (then taking `-kT log`) is the
  standard choice and is consistent with weighting by total bias weight.
- The standard deviation column is unweighted while the mean is weighted.
- The comment says the factor `0.0103636` converts "to J/mol"; dividing by it converts eV to
  kJ/mol.
- Results go through text files and are read back instead of being kept in memory.

### Target design
- `estk.reweight.average.combine(surfaces, *, mode="probability") -> FreeEnergySurface`.
  - `mode="probability"`: sum unnormalised log-densities across runs (log-sum-exp), then
    convert to free energy. This is equivalent to reweighting all frames together and needs
    each run's absolute log-weight total.
  - `mode="equal"`: each run counts equally (useful as a check).
  - Spread between runs stored as a separate `error` field, weighted consistently with the mean.
- Per-run log-weight totals are computed **before** any shift, and stored in each surface's
  metadata.
- Write a unit test on synthetic data: two runs with known bias offsets must give the same
  result as reweighting the concatenated frames.
- Document the convention in `docs/`.

---

## 8. Bandwidth selection

**Status:** todo

### Current state
- `compute_bw_silverman` (in `ectoolkits_reweight.py`) uses Silverman's rule with the effective
  sample size from weights, divided by `sqrt(bias_factor)`. Called with `dim=2` hard-coded.
- Bandwidths can be set per CV with `cvs.<name>.sigma`.
- `estimation_bandwidth/minimize_scv_exact.py` and `minimize_scv_approx.py` are separate
  scripts implementing smoothed cross-validation (SCV) for a diagonal bandwidth matrix; the
  approximate one uses Monte Carlo pairs and has its own argparse CLI.

### Problems
- Silverman uses the **unweighted** standard deviation of the biased data (`cv_array.std()`),
  while the sample size is weighted. Whether that is intended (OPES heuristic) should be
  written down.
- The SCV scripts aren't connected to the reweighting, so their output is copied by hand into
  the config.

### Target design
- `estk.reweight.bandwidth`:
  - `silverman(cvs, log_weights, *, bias_factor=1.0, weighted_std=False)`
  - `scv(cvs, log_weights, *, method="approx"|"exact", ...)`
- `bandwidth=` accepts a number or list per CV, `"silverman"` (default) or `"scv"`.
- `estk bandwidth --cvs d0 cn [--method scv]` prints the values and can write them into
  `estk.toml` (`--save`).
- The chosen bandwidths are stored in the output surface's metadata.

---

## 9. Surface file format

**Status:** todo

### Current state
- `write_fes` writes a PLUMED-like text grid: a `FIELDS` line, `SET min_/max_/num_grid_points_`
  lines, `SET log_weight_total`, then `x y value [extra]` columns.
- `read_fes` / `read_property` read it back; `read_fes` converts kJ/mol to eV and shifts the
  minimum to zero. There are also commented-out and `_old` readers.
- The CV order is assumed from the order of the `SET` lines.
- `write_fes` defaults `max_cv_y` to `-1` (looks like a typo for `1`).
- MEP files (`*-MEP.dat`) store extra columns whose meaning is encoded in the header
  (`job-0...`), and `plot_cvs_along_mep` takes the last 4 columns.

### Problems
- Metadata needed downstream (temperature, units, bandwidths, time window, runs used) isn't
  stored, so later commands must re-derive it or re-parse the runs.
- Units change between writing (kJ/mol) and reading (eV).

### Target design
- `FreeEnergySurface` / `PropertySurface` (in `estk/surface.py`) hold: `grid`, `values`,
  `error`, `units`, `cvs` (names + labels), and `metadata` (temperature, bandwidths, time window,
  runs and their weights, estk version, the command that produced it).
- Native format: one HDF5 file (`*.fes.h5`, `*.prop.h5`, `*.mep.h5`) with datasets plus
  metadata as attributes. `Surface.save(path)` / `estk.load(path)`.
- Export / import of the PLUMED text grid (`estk export --plumed`, `estk.load("fes.dat")`) so
  existing files and PLUMED tools keep working.
- Energies stored in one unit (kJ/mol), converted only for display (`--units eV`).

---

## 10. Minimum-energy path (string method)

**Status:** todo

### Current state
- `stringmethod_custom.py`: `String2D` class (zero-temperature string method on a 2D grid,
  gradients by finite differences, Euler steps, reparameterisation, optional fixed or flexible
  endpoints, convergence history).
- `utils.compute_mep` picks 2 or 3 minima by splitting the grid into halves/thirds
  (`split: horizontal|vertical`), with an "ugly fix" to reverse the path order, then runs the
  string method. `save_mep`/`load_mep` write text files.
- `commands/mep.py` (≈400 lines) globs the per-run and average FES files, runs the MEP for each,
  and draws a multi-panel figure with the paths and energy profiles.

### Problems
- Minima are found by splitting the map in fixed parts, which fails when two basins are in the
  same half.
- Computation, file discovery and plotting are mixed in one function.
- The barrier isn't returned as a value; it's read off plots or files.

### Target design
- `estk.paths.string2d.String2D`: keep the algorithm, clean up the API, add tests on analytic
  surfaces (e.g. Müller–Brown).
- `estk.paths.mep`:
  - `find_minima(fes, n=2, *, method="basins"|"split", split=None)`. The default uses the basin
    labelling already written for `find_structures_level_set` (section 14), with the current
    split method as an option.
  - `mep(fes, *, start=None, end=None, via=None, npts=15, ...) -> Path`
  - `Path` holds the points, the arc length, the energy profile, and `barrier`,
    `reverse_barrier`, `saddle`.
- `FreeEnergySurface.mep(...)` as a shortcut.
- CLI: `estk mep d0-cn.fes.h5 [--minima 2] [--start x y] [--end x y] [--via x y] [--plot]`
  prints the barrier and writes `d0-cn.mep.h5`.

---

## 11. Barrier convergence

**Status:** todo

### Current state
- `convergence_barrier_time.py`: for growing time windows, rewrites `cfg.time_interval`, calls
  `reweight_property(cfg)` and `mep(cfg)`, copies files into
  `convergence_barrier/<time>/` (which the user must create first), then plots barrier vs time.
- `convergence_barrier_parameter.py`: the same over a sweep of a CV bandwidth
  (`sigma_d0`/`sigma_cn` with a `geomspace` mesh).
- `convergence_barrier.py`: an older version of the time scan.

### Problems
- Works by mutating the global config and moving files around.
- In the per-run mode, the expected file name is built as `pot{jj}-...` from the folder path,
  which doesn't match the `potNNN-...` names that `reweight_property` writes.
- Requires a precomputed MEP file in a fixed location.
- The output directory must be created by hand.

### Target design
- `estk.analysis.convergence`:
  - `barrier_vs_time(project, cvs, times, **mep_kwargs) -> pandas.DataFrame`
  - `barrier_vs_bandwidth(project, cvs, cv, values, **mep_kwargs) -> pandas.DataFrame`
  - Each step is a plain call to `project.reweight(...)` and `fes.mep(...)`. Intermediate
    surfaces are kept in memory, or cached on disk with `--save-steps DIR`.
- CLI: `estk convergence time --cvs d0 cn --until 20 --step 0.5` and
  `estk convergence bandwidth --cvs d0 cn --cv d0 --range 0.001 0.5 --num 20 --log`.
  Both write a CSV and a plot.
- Delete the old `convergence_barrier.py`.

---

## 12. Surface plotting

**Status:** todo

### Current state
- `commands/plot_surface_data.py` (≈400 lines) draws multi-panel contour plots of the FES or of
  property maps, with optional MEP overlay, marked points and special points.
- Every visual setting comes from `surface_data.<prop>.plot` (size, row/col, scale, fontsize,
  vmin/vmax, levels, colour maps per panel, labels, `subplots_adjust`, save formats).
- `utils.apply_mpl_style` loads a style from a hard-coded absolute path.

### Problems
- Most of the config file is plot cosmetics, repeated per property.
- The layout (rows/columns) is guessed from the number of files.

### Target design
- `estk.plotting.surface.plot_surface(surface, ax=None, *, path=None, levels=None, cmap=None,
  vmin=None, vmax=None, points=None) -> Axes`. One surface, one axes; returns the axes so
  users can compose their own figures.
- `plot_surfaces(surfaces, ncols=2)` for the per-run + average grid.
- Labels come from `cv.<name>.label` in `estk.toml`; the colour-bar label from the surface units.
- Bundled styles (`paper`, `slides`, `notebook`), plus any user `.mplstyle` path.
- CLI: `estk plot surface d0-cn.fes.h5 [--path d0-cn.mep.h5] [--levels 0 0.8 0.1]
  [--style paper] [-o fes.png]`.

---

## 13. CVs along the path

**Status:** todo

### Current state
- `commands/plot_cvs_along_mep.py` (≈700 lines) reloads all CV data (copy of section 5), then
  for each point of the average MEP computes kernel-weighted averages of extra CVs
  (`cvs.add`), and plots them along the path with per-CV colours and axis limits from
  `cvs_along_mep.cvs.<name>.plot`.
- A `.bak` copy of an earlier version is committed.

### Problems
- Finds the MEP by file name (`<cv0>-<cv1>-fes-AVG-MEP.dat`) and reads the per-run paths from
  "the last 4 columns".
- Most of the file is data loading and plotting configuration.

### Target design
- `estk.analysis.along_path.average_along_path(project, path, quantities) -> pandas.DataFrame`
  using `point_kernel_weights` from section 6 at each path point, with the error between runs.
- `estk.plotting.path.plot_along_path(df, quantities, ...)`.
- CLI: `estk along-path d0-cn.mep.h5 --quantities dOOliq_min theta1 theta2 [-o cvs.png]`.
- Delete `plot_cvs_along_mep.py.bak`.

---

## 14. Structure extraction

**Status:** todo

### Current state
- `commands/find_structures.py`: for each chosen point on the FES, computes per-frame kernel
  weights, takes the highest-weight frames up to a cumulative probability threshold
  (`probability_thr`), and optionally writes the matching atomic structures.
- `commands/find_structures_level_set.py`: labels basins below an isovalue of the FES
  (`label_basins`, flood fill) and selects frames inside the chosen basin; supports random
  sub-sampling.
- `commands/find_structures_bin.py`: older binning version (no config file).
- Both active modules contain copies of: H5MD readers (`build_step_index`,
  `h5md_to_ase_atoms`), a LAMMPS dump indexer (`build_lammpstrj_index`,
  `read_atoms_at_timestep_sorted`), and the CV-loading block.
- `find_structures.py` reads `cfg.find_structures.probability_thr` even when called through
  another config section.

### Problems
- Duplicated I/O code; atom type map hard-coded or in each config section.
- The FES grid is found by globbing `*<cv0>-<cv1>-rew-fes.dat` in each run folder.

### Target design
- `estk.io.trajectory`: one `Trajectory` interface over H5MD, LAMMPS dump (with a cached step
  index) and XYZ, returning `ase.Atoms` for given steps. Type map from `[atoms]`.
- `estk.analysis.structures`:
  - `frames_near(project, point, *, probability=0.9, max_frames=None) -> FrameSelection`
  - `frames_in_basin(project, fes, *, isovalue, point=None, sample=None, seed=None)
    -> FrameSelection`
  - `FrameSelection` lists `(run, step, time, weight)` and can `.write(path)` the structures
    (xyz/extxyz/lammps-data via ase) or `.to_csv()`.
- Basin labelling (`label_basins`) moves to `estk.paths.mep` / a shared module so the MEP minima
  search can use it.
- CLI: `estk structures near d0-cn.fes.h5 --point 1.99 1.33 [--probability 0.9] [--save out.xyz]`
  and `estk structures basin d0-cn.fes.h5 --isovalue 0.1 --point 2.0 1.0 [--sample 400]`.
- Delete `find_structures_bin.py`.

---

## 15. Proton-transfer CVs

**Status:** todo

### Current state
- `commands/proton_transfer_cv.py` computes direct/indirect proton-transfer CVs (`d`, `delta`)
  with `ectoolkits.analysis.proton_transfer_cv.ProtonTransferCV` and MDAnalysis, per run in
  parallel, and writes `CV-<name>` files.
- Atom indices (1-based) and surface sites A–D/liquid are listed in the case YAML; the CV names
  must follow `direct-d-<site>` etc.

### Problems
- System-specific indices live in a generic config section.
- Timestep is inferred from the first two COLVAR rows.

### Target design
- `estk.analysis.proton_transfer.compute(run, donor, acceptors, water, kind) -> CVTable`, with
  index groups referenced by name from `[atoms]` in `estk.toml`
  (e.g. `[atoms.groups] site_A = [311]`).
- Output written in the `CV-<name>` format so it's picked up by section 5 automatically.
- CLI: `estk cv proton-transfer --kind direct --donor ads_O --acceptor site_A --water water_O`.
- Optional extra (`estk[pt]`), since it depends on MDAnalysis and ectoolkits.

---

## 16. Diagnostic plots

**Status:** todo

### Current state
- `commands/plot_opes_c.py`: time series and histograms of `opes.rct`, `opes.neff`, `opes.zed`
  from each run's COLVAR.
- `commands/plot_model_deviation.py`: time series of the ML model deviation from
  `model_devi.out`.
- Both re-implement figure layout (2×2 when more than one file).

### Problems
- Fixed layouts; the OPES column names are hard-coded.

### Target design
- `estk.plotting.diagnostics.plot_timeseries(project, columns, source="COLVAR")` as the shared
  implementation, with `opes_summary(project)` and `model_deviation(project)` as presets.
- CLI: `estk check opes` and `estk check model-deviation [--threshold 0.2 0.35]`; the second also
  prints the fraction of frames above the thresholds.

---

## 17. Standalone tools

**Status:** todo

### Current state
- `toolkit/h5md_to_xyz.py`: writes one H5MD frame to XYZ with a hard-coded type map and an
  orthorhombic cell.
- `estimation_bandwidth/*.py`: see section 8.

### Target design
- `estk convert h5md-to-xyz TRAJ.h5 --frame N -o out.xyz` built on `estk.io.trajectory`
  (section 14), using the `[atoms]` type map and handling general cells.
- The SCV scripts become `estk bandwidth --method scv` (section 8).

---

## 18. Tests, CI and docs

**Status:** todo

### Current state
- No tests, no CI, one-line README.

### Target design
- **Unit tests** (pytest) for each module, on synthetic data:
  - parsers on small fixture files (several LAMMPS/PLUMED input variants)
  - reweighting: a 2D double-well sampled with a known bias must give back the unbiased FES
    within tolerance; the multi-run combination test from section 7
  - string method on the Müller–Brown surface (known saddle)
  - config validation errors
- **Regression test:** a small, trimmed copy of a real run set (a few thousand frames) in
  `tests/data/`. Compare the new `estk reweight` + `estk mep` with the outputs of the current code
  (saved once before the old code is removed).
- **CI:** GitHub Actions with ruff (lint + format) and pytest on supported Python versions.
- **Docs:** README with install + a 5-minute tutorial (`estk init`, `estk reweight`, `estk mep`,
  `estk plot surface`); `docs/` for the reweighting conventions (sections 6–8) and a page per
  command; docstrings on the public API.

---

## Suggested order of work

1. Packaging skeleton, logging, exceptions, config schema (sections 1–3).
2. Metadata parsing and CV loading (4–5) with their tests.
3. Save reference outputs from the current code on the regression dataset.
4. Reweighting, combining runs and bandwidths (6–8), checked against the reference and the
   synthetic tests; settle the run-weighting question in section 7.
5. Surface objects and file format (9).
6. MEP and convergence (10–11).
7. Plotting (12), then the remaining analyses (13–17).
8. Remove `to_refactor/`, finish docs and CI (18).
