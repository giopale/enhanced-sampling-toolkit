import numpy as np
import pytest

from estk.errors import EstError, MissingLimitsError
from estk.grid import Grid, resolve_limits
from estk.io.colvar import CVTable
from estk.project import Project
from estk.reweight import kde_on_grid, reweight_table, silverman
from estk.surface import Surface

from .conftest import LAMMPS_LOG, PLUMED_INPUT

KBT_350 = 350 * 8.314462618e-3


# --- the old algorithm, copied from to_refactor/utils_reweighting.py (fes_kde_2d_logsum) -----


def old_fes_kde_2d_logsum(cv_x, cv_y, grid_X, grid_Y, H, logw, kbT, frame_chunk=7):
    Hinv = np.linalg.inv(0.5 * (H + H.T))
    a, b, c = Hinv[0, 0], Hinv[0, 1], Hinv[1, 1]
    nx, ny = grid_X.shape
    gx, gy = grid_X.reshape(-1), grid_Y.reshape(-1)
    logp = np.full(gx.size, -np.inf)
    for start in range(0, cv_x.size, frame_chunk):
        end = min(start + frame_chunk, cv_x.size)
        dX = gx[:, None] - cv_x[None, start:end]
        dY = gy[:, None] - cv_y[None, start:end]
        dist2 = a * dX * dX + 2.0 * b * dX * dY + c * dY * dY
        arg = logw[None, start:end] - 0.5 * dist2
        m = np.max(arg, axis=1)
        chunk_logp = m + np.log(np.sum(np.exp(arg - m[:, None]), axis=1))
        logp = np.logaddexp(logp, chunk_logp)
    return (-kbT * logp).reshape(nx, ny)


def old_silverman(cv_array, dim, weights, bias_factor):
    neff = np.power(np.sum(weights), 2) / np.sum(np.power(weights, 2))
    return cv_array.std() / np.sqrt(bias_factor) * np.power(neff * (dim + 2) / 4.0, -1 / (dim + 4))


# --- kernel sums -------------------------------------------------------------------------------


@pytest.fixture
def frames():
    rng = np.random.default_rng(1)
    points = rng.normal([2.0, 1.5], [0.2, 0.4], size=(60, 2))
    logw = rng.normal(0, 2, size=60)
    return points, logw


def test_kde_matches_brute_force(frames):
    points, logw = frames
    grid = Grid(("x", "y"), (1.5, 0.5), (2.5, 2.5), (7, 5))
    H = np.array([[0.04, 0.01], [0.01, 0.09]])
    q = points[:, 0] * 3 - points[:, 1]
    # tiny memory budget: one frame per chunk
    log_p, avg = kde_on_grid(points, logw, grid, H, quantities={"q": q}, memory_bytes=1)

    hinv = np.linalg.inv(H)
    g = grid.coordinates()
    diff = g[:, None, :] - points[None, :, :]
    dist2 = np.einsum("gni,ij,gnj->gn", diff, hinv, diff)
    k = np.exp(logw[None, :] - 0.5 * dist2)
    np.testing.assert_allclose(log_p.reshape(-1), np.log(k.sum(axis=1)), rtol=1e-12)
    np.testing.assert_allclose(avg["q"].reshape(-1), (k @ q) / k.sum(axis=1), rtol=1e-10)


def test_kde_matches_old_code(frames):
    points, logw = frames
    grid = Grid(("x", "y"), (1.5, 0.5), (2.5, 2.5), (9, 6))
    H = np.diag([0.03**2, 0.05**2])
    log_p, _ = kde_on_grid(points, logw, grid, np.array([0.03, 0.05]))
    X, Y = grid.mesh()
    old = old_fes_kde_2d_logsum(points[:, 0], points[:, 1], X, Y, H, logw, KBT_350)
    new = -KBT_350 * log_p
    finite = np.isfinite(old)
    assert np.array_equal(finite, np.isfinite(new))
    np.testing.assert_allclose(new[finite], old[finite], rtol=1e-10)


def test_kde_far_from_data_is_minus_inf():
    grid = Grid(("x",), (0.0,), (100.0,), (3,))
    log_p, avg = kde_on_grid(
        np.array([[0.0]]), np.zeros(1), grid, np.array([0.1]), quantities={"q": np.ones(1)}
    )
    # log space: far from the data log p is very negative but finite, as in the old code
    assert log_p[0] == pytest.approx(0.0)
    assert log_p[2] == pytest.approx(-0.5 * (100 / 0.1) ** 2)
    assert avg["q"][2] == pytest.approx(1.0)


def test_fast_path_matches_direct_sums_far_from_data(frames):
    """The factorised 2D path must agree with the log-space sums, also where it underflows."""
    points, logw = frames
    grid = Grid(("x", "y"), (0.0, -20.0), (40.0, 20.0), (9, 7))  # mostly far from the frames
    fast, fast_avg = kde_on_grid(
        points, logw, grid, np.array([0.03, 0.05]), quantities={"q": points[:, 0]}
    )
    H = np.diag([0.03**2, 0.05**2])
    H[0, 1] = H[1, 0] = 1e-30  # not diagonal: forces the direct log-space sums
    direct, direct_avg = kde_on_grid(points, logw, grid, H, quantities={"q": points[:, 0]})
    assert np.all(np.isfinite(fast))
    assert fast.min() < -1e5  # really far: these points needed the fallback
    np.testing.assert_allclose(fast, direct, rtol=1e-9)
    np.testing.assert_allclose(fast_avg["q"], direct_avg["q"], rtol=1e-9)


def test_kde_without_frames_in_reach_is_minus_inf():
    grid = Grid(("x",), (0.0,), (1.0,), (2,))
    log_p, avg = kde_on_grid(
        np.array([[0.0]]), np.array([-np.inf]), grid, np.array([0.1]), quantities={"q": [1.0]}
    )
    assert np.all(np.isneginf(log_p))
    assert np.all(np.isnan(avg["q"]))


def test_kde_input_checks(frames):
    points, logw = frames
    grid = Grid(("x", "y"), (0, 0), (1, 1), (2, 2))
    with pytest.raises(EstError, match="positive"):
        kde_on_grid(points, logw, grid, np.array([0.1, -1]))
    with pytest.raises(EstError, match="positive definite"):
        kde_on_grid(points, logw, grid, np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(EstError, match="shape"):
        kde_on_grid(points[:, :1], logw, grid, np.array([0.1, 0.1]))


def test_silverman_matches_old_code(frames):
    points, logw = frames
    w = np.exp(logw)
    assert silverman(points[:, 0], w, ndim=2, bias_factor=10) == pytest.approx(
        old_silverman(points[:, 0], 2, w, 10), rel=1e-12
    )
    assert silverman(points[:, 0], ndim=1) == pytest.approx(
        np.std(points[:, 0]) * (60 * 3 / 4) ** (-1 / 5)
    )


# --- grid and limits ---------------------------------------------------------------------------


def test_grid():
    grid = Grid(("a", "b"), (0.0, 1.0), (1.0, 3.0), (3, 2))
    assert grid.shape == (3, 2)
    np.testing.assert_allclose(grid.axes()[0], [0, 0.5, 1])
    assert grid.coordinates()[:2].tolist() == [[0.0, 1.0], [0.0, 3.0]]
    with pytest.raises(EstError, match="below"):
        Grid(("a",), (1.0,), (0.0,), (3,))


def test_resolve_limits_precedence():
    from estk.config import ProjectConfig

    config = ProjectConfig.model_validate({"cv": {"cn": {"limits": [0.75, None]}}})
    limits, sources = resolve_limits(
        ["d0", "cn"],
        explicit={"d0": (None, 2.4)},
        config=config,
        walls=[{"d0": (1.5, 2.5), "cn": (None, 3.0)}, {"d0": (1.4, 2.5)}],
        data={"d0": (1.0, 3.0), "cn": (0.0, 4.0)},
    )
    assert limits == {"d0": (1.4, 2.4), "cn": (0.75, 3.0)}
    assert sources["d0"] == ("PLUMED walls", "command line")
    assert sources["cn"] == ("estk.yaml", "PLUMED walls")
    with pytest.raises(MissingLimitsError, match="lower limit for z"):
        resolve_limits(["z"])


# --- physics: recover a known free energy -----------------------------------------------------


def sample(f_values, axis, rng, n, gamma):
    """Sample from p ∝ exp(-F/(gamma kT)) on a fine 1D axis by inverse CDF."""
    p = np.exp(-f_values / (gamma * KBT_350))
    cdf = np.cumsum(p) / p.sum()
    return np.interp(rng.random(n), cdf, axis)


def fx(x):
    return 5 * KBT_350 * (((x - 2.0) / 0.25) ** 2 - 1) ** 2  # double well, barrier 5 kT


def fy(y):
    return 0.5 * KBT_350 * ((y - 1.5) / 0.3) ** 2


def make_biased_run(folder, n=100_000, gamma=10.0, seed=0):
    """A run sampled from the well-tempered distribution of F(x, y) = fx(x) + fy(y)."""
    rng = np.random.default_rng(seed)
    axis_x, axis_y = np.linspace(1.3, 2.7, 4001), np.linspace(0.0, 3.0, 4001)
    x = sample(fx(axis_x), axis_x, rng, n, gamma)
    y = sample(fy(axis_y), axis_y, rng, n, gamma)
    # static well-tempered bias, added to the potential as in PLUMED, plus an offset
    bias = -(1 - 1 / gamma) * (fx(x) + fy(y)) - 65.0
    folder.mkdir(parents=True)
    (folder / "log.lammps").write_text(LAMMPS_LOG)
    (folder / "plumed.plumed").write_text(PLUMED_INPUT)
    data = np.column_stack([np.arange(n) * 0.05, x, y, bias])
    np.savetxt(
        folder / "COLVAR", data, fmt="%.6f", header="FIELDS time d0 cn opes.bias", comments="#! "
    )
    return folder


@pytest.fixture(scope="module")
def biased_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("biased")
    make_biased_run(root / "000", seed=0)
    make_biased_run(root / "001", seed=1)
    (root / "estk.yaml").write_text(
        "runs:\n  pattern: '0*'\n"
        "cv:\n  d0:\n    limits: [1.5, 2.5]\n    bandwidth: 0.01\n"
        "  cn:\n    limits: [0.75, 2.25]\n    bandwidth: 0.02\n"
    )
    return Project(root / "estk.yaml")


def test_recovers_known_free_energy(biased_project):
    result = biased_project.reweight(["d0", "cn"], points=(41, 31))
    X, Y = result.grid.mesh()
    true = fx(X) + fy(Y)
    true -= true.min()
    low = true < 3 * KBT_350  # well-sampled region
    for surface in result.surfaces.values():
        # Remove the constant offset (the zero is set by the noisy minimum); what is left is
        # sampling noise of about 0.15 kT with 1e5 frames (the two runs differ by ~0.2 kT rms).
        error = (surface.values - true)[low]
        error -= error.mean()
        assert np.sqrt(np.mean(error**2)) < 0.2 * KBT_350
        assert np.max(np.abs(error)) < 0.6 * KBT_350
        assert surface.values.min() == 0.0
        assert surface.metadata["bandwidth_source"] == {"d0": "estk.yaml", "cn": "estk.yaml"}


def test_property_average(biased_project):
    for run in biased_project.runs:
        table = run.load_cvs(["d0", "cn"])
        q = 2 * table["d0"] - table["cn"]
        lines = [f"{t:.6f} {v:.6f}" for t, v in zip(table.time, q, strict=True)]
        (run.path / "CV-q").write_text("#! FIELDS time q\n" + "\n".join(lines) + "\n")
    result = biased_project.reweight(["d0", "cn"], quantity="q", points=(11, 11))
    surface = result.surfaces["000"]
    X, Y = result.grid.mesh()
    assert surface.quantity == "q"
    # Kernel averages are biased by ~h^2 d(log p)/dx where the density is steep (grid edges);
    # in the well-sampled region they match the exact 2x - y.
    low = fx(X) + fy(Y) - (fx(X) + fy(Y)).min() < 3 * KBT_350
    assert np.max(np.abs(surface.values - (2 * X - Y))[low]) < 0.03


def test_reweight_metadata_and_bias_offset(biased_project):
    surface = biased_project.reweight(["d0", "cn"], points=11).surfaces["000"]
    meta = surface.metadata
    assert meta["frames"] == 100_000
    assert meta["kbt"] == pytest.approx(KBT_350)
    assert meta["temperature"] == 350.0
    assert meta["bias_factor"] == 10.0
    assert 0 < meta["effective_frames"] < 100_000
    # a constant offset in the bias changes log_weight_total but not the surface
    table = biased_project.runs[0].load_cvs(["d0", "cn"])
    shifted = CVTable(table.run_name, table.time, table.columns, table.bias + 100.0, table.kbt)
    other = reweight_table(shifted, surface.grid, configured_bandwidth={"d0": 0.01, "cn": 0.02})
    np.testing.assert_allclose(other.values, surface.values, atol=1e-9)
    assert other.metadata["log_weight_total"] == pytest.approx(
        meta["log_weight_total"] + 100.0 / KBT_350
    )


def test_silverman_used_without_bandwidth(biased_project):
    result = biased_project.reweight(["d0", "cn"], points=5, bandwidth={"d0": 0.05})
    meta = result.surfaces["000"].metadata
    assert meta["bandwidth_source"] == {"d0": "command line", "cn": "estk.yaml"}
    config = biased_project.config.model_copy(deep=True)
    config.cv["cn"].bandwidth = None
    meta = Project(config).reweight(["d0", "cn"], points=5).surfaces["000"].metadata
    assert meta["bandwidth_source"]["cn"] == "silverman"
    assert 0 < meta["bandwidth"]["cn"] < 0.5


def test_parallel_gives_same_result(biased_project):
    serial = biased_project.reweight(["d0", "cn"], points=9)
    parallel = biased_project.reweight(["d0", "cn"], points=9, jobs=2)
    for name in serial.surfaces:
        np.testing.assert_array_equal(serial.surfaces[name].values, parallel.surfaces[name].values)


# --- walls ------------------------------------------------------------------------------------


def test_walls_report_and_include(tmp_path):
    folder = tmp_path / "000"
    folder.mkdir()
    (folder / "log.lammps").write_text(LAMMPS_LOG)
    (folder / "plumed.plumed").write_text(PLUMED_INPUT)  # upper wall d0 at 2.5, KAPPA 1000
    rows = [f"{i * 0.05:.2f} {d0} 1.0 0.0" for i, d0 in enumerate([2.0, 2.4, 2.6, 2.7])]
    (folder / "COLVAR").write_text("#! FIELDS time d0 cn opes.bias\n" + "\n".join(rows) + "\n")
    run = Project(Project().config).config  # defaults
    project = Project(run)
    project.config.runs.root = tmp_path
    project.config.runs.pattern = "0*"

    table = project.runs[0].load_cvs(["d0"])
    n, max_kbt = table.report.walls["d0 upper"]
    assert n == 2
    assert max_kbt == pytest.approx(1000 * 0.2**2 / KBT_350)
    assert table.report.walls["d0 lower"] == (0, None)
    np.testing.assert_allclose(table.bias, 0.0)

    with_walls = project.runs[0].load_cvs(["cn"], include_walls=True)
    assert "d0" not in with_walls.names  # loaded only for the walls
    np.testing.assert_allclose(with_walls.bias, [0, 0, 1000 * 0.1**2, 1000 * 0.2**2])
    assert with_walls.report.walls_in_bias


# --- file format ------------------------------------------------------------------------------


def test_surface_roundtrip(tmp_path):
    grid = Grid(("d0", "cn"), (1.5, 0.75), (2.5, 3.0), (4, 3))
    values = np.arange(12.0).reshape(4, 3)
    values[0, 0] = np.inf
    surface = Surface(grid, values, metadata={"run": "000", "bandwidth": {"d0": 0.02}})
    path = surface.save(tmp_path / "out" / "000.dat")
    text = path.read_text()
    assert text.startswith("#! FIELDS d0 cn fes\n")
    assert '#! SET estk.run "000"' in text
    back = Surface.load(path)
    assert back.grid == grid
    np.testing.assert_array_equal(back.values, values)
    assert back.metadata == {"run": "000", "bandwidth": {"d0": 0.02}}
    assert back.units == "kj/mol"


def test_reads_old_format(tmp_path):
    # as written by to_refactor/ectoolkits_reweight.write_fes (np.savetxt, x slowest)
    X, Y = np.meshgrid(np.linspace(1.5, 2.5, 3), np.linspace(0.75, 3.0, 2), indexing="ij")
    Z = X + 10 * Y
    header = (
        "FIELDS d0 cn  fes\nSET min_d0      1.500\nSET max_d0      2.500\n"
        "SET num_grid_points_d0 3\nSET min_cn      0.750\nSET max_cn      3.000\n"
        "SET num_grid_points_cn 2\nSET log_weight_total 12.5"
    )
    path = tmp_path / "pot000-d0-cn-rew-fes.dat"
    np.savetxt(path, np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), fmt="%.6f", header=header)
    surface = Surface.load(path)
    assert surface.grid.names == ("d0", "cn")
    assert surface.grid.points == (3, 2)
    np.testing.assert_allclose(surface.values, Z)
    assert surface.metadata["log_weight_total"] == 12.5
