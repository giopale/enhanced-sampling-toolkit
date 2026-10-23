import math

import numpy as np
import pytest

from estk.errors import EstError
from estk.grid import Grid
from estk.project import Project
from estk.reweight import combine, kde_on_grid
from estk.surface import Surface

from .conftest import LAMMPS_LOG, PLUMED_INPUT
from .test_reweight import KBT_350, fx, fy, sample


def write_run(folder, x, y, bias, t0=0.0):
    folder.mkdir(parents=True)
    (folder / "log.lammps").write_text(LAMMPS_LOG)
    (folder / "plumed.plumed").write_text(PLUMED_INPUT)
    t = t0 + np.arange(len(x)) * 0.05
    np.savetxt(
        folder / "COLVAR",
        np.column_stack([t, x, y, bias]),
        fmt="%.8f",
        header="FIELDS time d0 cn opes.bias",
        comments="#! ",
    )


def frames(seed, n=4000, offset=0.0):
    rng = np.random.default_rng(seed)
    ax, ay = np.linspace(1.3, 2.7, 2001), np.linspace(0.0, 3.0, 2001)
    x, y = sample(fx(ax), ax, rng, n, 10.0), sample(fy(ay), ay, rng, n, 10.0)
    bias = -0.9 * (fx(x) + fy(y)) + offset
    return x, y, bias


CONFIG = (
    "runs:\n  pattern: '{pattern}'\n"
    "cv:\n  d0:\n    limits: [1.5, 2.5]\n    bandwidth: 0.03\n"
    "  cn:\n    limits: [0.75, 2.25]\n    bandwidth: 0.05\n"
)


@pytest.fixture(scope="module")
def two_runs_and_merged(tmp_path_factory):
    """Two runs whose biases differ by a constant, and one run with all their frames."""
    root = tmp_path_factory.mktemp("combine")
    a = frames(0, offset=-65.0)
    b = frames(1, n=3000, offset=-60.0)  # 5 kJ/mol higher bias: more weight per frame
    write_run(root / "runs" / "a", *a)
    write_run(root / "runs" / "b", *b)
    merged = [np.concatenate([p, q]) for p, q in zip(a, b, strict=True)]
    write_run(root / "merged" / "all", *merged)
    for sub in ("runs", "merged"):
        (root / sub / "estk.yaml").write_text(CONFIG.format(pattern="*"))
    return Project(root / "runs" / "estk.yaml"), Project(root / "merged" / "estk.yaml")


def test_probability_mode_equals_reweighting_all_frames_together(two_runs_and_merged):
    runs, merged = two_runs_and_merged
    result = runs.reweight(["d0", "cn"], points=(21, 15))
    single = merged.reweight(["d0", "cn"], points=(21, 15), combine=None).surfaces["all"]
    combined = result.combined
    assert combined.metadata["combined"] == "probability"
    np.testing.assert_allclose(combined.values, single.values, atol=1e-8)
    # combined: -kT log of the normalised density (weights sum 1, kernels integrate to 1);
    # single run: kernels not normalised, so it differs by kT log(2 pi h_x h_y)
    assert combined.metadata["fes_offset"] == pytest.approx(
        single.metadata["fes_offset"] + KBT_350 * math.log(2 * math.pi * 0.03 * 0.05), abs=1e-8
    )
    # run weights are the shares of sum exp(V/kT)
    lw = {n: s.metadata["log_weight_total"] for n, s in result.surfaces.items()}
    share_b = 1 / (1 + math.exp(lw["a"] - lw["b"]))
    assert combined.metadata["run_weights"]["b"] == pytest.approx(share_b)
    assert combined.metadata["frames"] == 7000


def test_property_combination_equals_all_frames_together(two_runs_and_merged):
    runs, merged = two_runs_and_merged
    for project in (runs, merged):
        for run in project.runs:
            table = run.load_cvs(["d0", "cn"])
            q = np.sin(3 * table["d0"]) + table["cn"]
            text = "\n".join(f"{t:.8f} {v:.10f}" for t, v in zip(table.time, q, strict=True))
            (run.path / "CV-q").write_text("#! FIELDS time q\n" + text + "\n")
    combined = runs.reweight(["d0", "cn"], quantity="q", points=(11, 9)).combined
    single = merged.reweight(["d0", "cn"], quantity="q", points=(11, 9)).surfaces["all"]
    np.testing.assert_allclose(combined.values, single.values, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(combined.fes, single.fes, atol=1e-8)
    assert np.all(combined.error[np.isfinite(combined.error)] >= 0)


def test_equal_mode_ignores_bias_offsets(two_runs_and_merged):
    runs, _ = two_runs_and_merged
    result = runs.reweight(["d0", "cn"], points=(11, 9), combine="equal")
    combined = result.combined
    assert combined.metadata["run_weights"] == {"a": 0.5, "b": 0.5}
    kbt = KBT_350
    density = (
        sum(np.exp(-(s.values + s.metadata["fes_offset"]) / kbt) for s in result.surfaces.values())
        / 2
    )
    expected = -kbt * np.log(density)
    np.testing.assert_allclose(combined.values, expected - expected.min(), atol=1e-8)
    # the error is the spread of the two runs around the combined surface
    per_run = np.array(
        [s.values + s.metadata["fes_offset"] for s in result.surfaces.values()]
    ) + kbt * math.log(2 * math.pi * 0.03 * 0.05)
    center = combined.values + combined.metadata["fes_offset"]
    np.testing.assert_allclose(
        combined.error, np.sqrt(np.mean((per_run - center) ** 2, axis=0)), rtol=1e-8
    )


def test_probability_mode_with_different_bandwidths():
    """Each run's kernels are normalised: the result is the sum over all frames of
    exp(V/kT) times a normalised Gaussian of that run's bandwidth."""
    rng = np.random.default_rng(5)
    grid = Grid(("x", "y"), (0.0, 0.0), (1.0, 1.0), (6, 5))
    kbt = 2.5
    runs, expected = {}, np.zeros(grid.shape)
    for name, h in (("a", (0.1, 0.2)), ("b", (0.05, 0.3))):
        pts = rng.random((50, 2))
        raw = rng.normal(0, 1, 50)
        log_total = np.log(np.sum(np.exp(raw)))
        log_p, _ = kde_on_grid(pts, raw - log_total, grid, np.array(h))
        offset = float((-kbt * log_p).min())
        runs[name] = Surface(
            grid,
            -kbt * log_p - offset,
            metadata={
                "run": name,
                "kbt": kbt,
                "log_weight_total": log_total,
                "fes_offset": offset,
                "bandwidth": {"x": h[0], "y": h[1]},
            },
        )
        g = grid.coordinates()
        d2 = ((g[:, None, :] - pts[None]) / np.array(h)) ** 2
        normal = np.exp(-0.5 * d2.sum(axis=2)) / (2 * np.pi * h[0] * h[1])
        expected += (normal @ np.exp(raw)).reshape(grid.shape)
    combined = combine(runs)
    f = -kbt * np.log(expected)
    np.testing.assert_allclose(combined.values, f - f.min(), atol=1e-9)


def test_combine_checks():
    grid = Grid(("x",), (0.0,), (1.0,), (3,))
    meta = {"kbt": 2.5, "log_weight_total": 0.0, "fes_offset": 0.0, "bandwidth": {"x": 0.1}}
    a = Surface(grid, np.zeros(3), metadata={**meta, "run": "a"})
    other_grid = Surface(Grid(("x",), (0.0,), (2.0,), (3,)), np.zeros(3), metadata=meta)
    hot = Surface(grid, np.zeros(3), metadata={**meta, "kbt": 3.0})
    with pytest.raises(EstError, match="different grids"):
        combine({"a": a, "b": other_grid})
    with pytest.raises(EstError, match="different temperatures"):
        combine({"a": a, "b": hot})
    with pytest.raises(EstError, match="unknown combination mode"):
        combine({"a": a}, mode="mean")
    no_meta = Surface(grid, np.zeros(3), metadata={"kbt": 2.5})
    with pytest.raises(EstError, match="log_weight_total"):
        combine({"x": no_meta})


def test_combined_file_roundtrip(two_runs_and_merged, tmp_path):
    runs, _ = two_runs_and_merged
    result = runs.reweight(["d0", "cn"], points=(5, 4))
    paths = result.save(tmp_path)
    assert [p.name for p in paths] == ["a.dat", "b.dat", "combined.dat"]
    back = Surface.load(tmp_path / "combined.dat")
    np.testing.assert_allclose(back.values, result.combined.values)
    np.testing.assert_allclose(back.error, result.combined.error, equal_nan=True)
    assert back.metadata["run_weights"] == result.combined.metadata["run_weights"]
    # surfaces read back from files can be combined again
    again = combine({n: Surface.load(tmp_path / f"{n}.dat") for n in ("a", "b")})
    np.testing.assert_allclose(again.values, result.combined.values, atol=1e-7)
