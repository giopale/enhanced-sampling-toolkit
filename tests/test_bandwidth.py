"""Bandwidth selection: SCV, the methods of the reweighting, estk bandwidth and --save."""

import numpy as np
import pytest
from typer.testing import CliRunner

from estk.cli.main import app
from estk.config import load_config, set_cv_bandwidths
from estk.errors import ConfigError, EstError
from estk.io.colvar import CVTable
from estk.project import Project
from estk.reweight import estimate_bandwidths, scv, silverman
from estk.reweight.core import choose_bandwidths
from estk.surface import Surface

from .test_combine import CONFIG, frames, write_run

# --- the SCV criterion ------------------------------------------------------------------------


def test_binned_matches_exact():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2)) * [0.1, 0.3]
    w = rng.uniform(0.2, 1.0, 500)
    binned, exact = scv(x, w, method="binned"), scv(x, w, method="exact")
    np.testing.assert_allclose(binned.bandwidth, exact.bandwidth, rtol=2e-3)
    np.testing.assert_allclose(binned.objective, exact.objective, rtol=2e-3)
    assert binned.converged and binned.weighted


def test_normal_data_gives_the_normal_reference():
    """For normal data the optimal bandwidth is Silverman's rule (its derivation assumes it)."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20000, 2)) * [0.1, 0.3]
    h = scv(x).bandwidth
    reference = [silverman(x[:, k], ndim=2) for k in range(2)]
    np.testing.assert_allclose(h, reference, rtol=0.1)


def test_bimodal_data_gets_narrower_kernels_than_silverman():
    rng = np.random.default_rng(3)
    n = 5000
    modes = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    x = np.column_stack([modes + 0.2 * rng.normal(size=n), rng.normal(size=n)])
    h = scv(x).bandwidth
    assert h[0] < 0.6 * silverman(x[:, 0], ndim=2)  # Silverman oversmooths two modes
    assert h[1] == pytest.approx(silverman(x[:, 1], ndim=2), rel=0.25)


def test_weights():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(800, 2))
    plain = scv(x).bandwidth
    np.testing.assert_allclose(scv(x, np.full(800, 3.0)).bandwidth, plain)  # equal weights
    w = np.exp(-2 * x[:, 0] ** 2)
    np.testing.assert_allclose(scv(x, w, weighted=False).bandwidth, plain)
    weighted = scv(x, w)
    assert weighted.effective_frames < 800 and weighted.weighted


def test_one_cv_pilot_and_errors():
    rng = np.random.default_rng(5)
    x = rng.normal(size=3000)
    result = scv(x)  # a 1-D array is one CV
    assert result.bandwidth.shape == (1,)
    assert result.bandwidth[0] == pytest.approx(silverman(x), rel=0.15)
    np.testing.assert_allclose(scv(x, pilot=[0.3]).pilot, [0.3])
    with pytest.raises(EstError, match="method"):
        scv(x, method="fast")
    with pytest.raises(EstError, match="more frames"):
        scv(x[:5])
    with pytest.raises(EstError, match="no spread"):
        scv(np.column_stack([x, np.ones_like(x)]))
    with pytest.raises(EstError, match="weights for"):
        scv(x, np.ones(10))
    with pytest.raises(EstError, match="pilot"):
        scv(x, pilot=[0.1, 0.2])


# --- reweighting ------------------------------------------------------------------------------


def test_choose_bandwidths_with_a_method():
    rng = np.random.default_rng(6)
    table = CVTable(
        "r", np.arange(1000.0), {"a": rng.normal(size=1000), "b": rng.normal(size=1000)}
    )
    weights = np.full(1000, 1e-3)
    configured = {"a": 0.5, "b": 0.5}
    h, sources = choose_bandwidths(table, ["a", "b"], weights, configured=configured)
    assert sources == {"a": "estk.yaml", "b": "estk.yaml"}
    h, sources = choose_bandwidths(table, ["a", "b"], weights, explicit={"a": 0.1},
                                   configured=configured, method="scv")  # fmt: skip
    assert sources == {"a": "command line", "b": "scv"}
    assert h[0] == 0.1 and h[1] != 0.5
    _, sources = choose_bandwidths(table, ["a", "b"], weights, method="scv-weighted")
    assert sources == {"a": "scv-weighted", "b": "scv-weighted"}
    with pytest.raises(EstError, match="unknown bandwidth method"):
        choose_bandwidths(table, ["a"], weights, method="plugin")


@pytest.fixture
def project_root(tmp_path):
    write_run(tmp_path / "a", *frames(0, n=1500))
    write_run(tmp_path / "b", *frames(1, n=1500))
    (tmp_path / "estk.yaml").write_text("# my project\n" + CONFIG.format(pattern="*"))
    return tmp_path


def test_estimate_bandwidths(project_root):
    estimates = estimate_bandwidths(Project(project_root / "estk.yaml"), ["d0", "cn"],
                                    methods=["silverman", "scv", "scv-weighted"])  # fmt: skip
    assert [e.run for e in estimates] == ["a", "b", "all"]
    assert estimates[-1].frames == 3000
    for est in estimates:
        assert set(est.bandwidths) == {"silverman", "scv", "scv-weighted"}
        assert all(h > 0 for values in est.bandwidths.values() for h in values.values())
    with pytest.raises(EstError, match="methods"):
        estimate_bandwidths(Project(project_root / "estk.yaml"), ["d0"], methods=["plugin"])


def test_bandwidth_command_and_save(project_root, monkeypatch):
    monkeypatch.chdir(project_root)
    runner = CliRunner()
    result = runner.invoke(app, ["bandwidth", "d0", "cn"])
    assert result.exit_code == 0, result.output
    for text in ("silverman", "scv", "all", "d0=0.03, cn=0.05"):
        assert text in result.output

    result = runner.invoke(app, ["bandwidth", "d0", "cn", "--save"])
    assert result.exit_code == 0, result.output
    saved = load_config(project_root / "estk.yaml").cv
    pooled = estimate_bandwidths(Project(project_root / "estk.yaml"), ["d0", "cn"],
                                 methods=["scv"])[-1].bandwidths["scv"]  # fmt: skip
    assert saved["d0"].bandwidth == pytest.approx(pooled["d0"], rel=1e-3)
    assert saved["d0"].limits == (1.5, 2.5)  # the rest is kept
    assert (project_root / "estk.yaml").read_text().startswith("# my project\n")

    result = runner.invoke(app, ["bandwidth", "d0", "--method", "silverman", "--save"])
    assert result.exit_code == 0, result.output
    assert "(silverman, all)" in result.output
    result = runner.invoke(app, ["bandwidth", "d0", "--method", "silverman",
                                 "--method", "scv-weighted", "--save"])  # fmt: skip
    assert result.exit_code == 1 and "one --method" in result.output
    result = runner.invoke(app, ["bandwidth", "d0", "--method", "plugin"])
    assert result.exit_code == 1


def test_reweight_with_a_method(project_root, monkeypatch):
    monkeypatch.chdir(project_root)
    runner = CliRunner()
    result = runner.invoke(app, ["reweight", "d0", "cn", "--points", "21", "--bandwidth", "scv",
                                 "--bandwidth", "cn=0.04", "-o", "out"])  # fmt: skip
    assert result.exit_code == 0, result.output
    meta = Surface.load(project_root / "out" / "a.dat").metadata
    assert meta["bandwidth_source"] == {"d0": "scv", "cn": "command line"}
    assert meta["bandwidth"]["cn"] == 0.04 and meta["bandwidth"]["d0"] != 0.03
    result = runner.invoke(app, ["reweight", "d0", "--bandwidth", "scv",
                                 "--bandwidth", "silverman"])  # fmt: skip
    assert result.exit_code == 1 and "one method" in result.output


# --- writing estk.yaml ------------------------------------------------------------------------

YAML = """\
# comment kept
runs:
  pattern: '*'

cv:
  d0:
    label: d(Ti–O)   # inline comment kept
    limits: &lim [1.5, 2.5]
    bandwidth: 0.03
  cn:
    limits: *lim
  q: {}
  "quoted": {limits: [0, 1]}

atoms:
  type_map: {1: H}
"""


def test_set_cv_bandwidths(tmp_path):
    path = tmp_path / "estk.yaml"
    path.write_text(YAML)
    set_cv_bandwidths(path, {"d0": 0.0123456, "cn": 0.05, "q": 0.1, "quoted": 0.2, "new": 0.3})
    text = path.read_text()
    assert text.startswith("# comment kept\n")
    assert "label: d(Ti–O)   # inline comment kept" in text and "&lim" in text
    assert "    bandwidth: 0.01235\n" in text
    cv = load_config(path).cv
    assert {n: c.bandwidth for n, c in cv.items()} == {
        "d0": 0.01235, "cn": 0.05, "q": 0.1, "quoted": 0.2, "new": 0.3,
    }  # fmt: skip
    assert cv["cn"].limits == (1.5, 2.5) and cv["quoted"].limits == (0, 1)
    assert load_config(path).atoms.type_map == {1: "H"}


def test_set_cv_bandwidths_without_a_cv_section(tmp_path):
    path = tmp_path / "estk.yaml"
    path.write_text("runs:\n  pattern: '*'\n")
    set_cv_bandwidths(path, {"d0": 0.02})
    assert load_config(path).cv["d0"].bandwidth == 0.02


def test_set_cv_bandwidths_leaves_the_file_on_error(tmp_path):
    path = tmp_path / "estk.yaml"
    text = "cv:\n  d0: {limits: *missing}\n"
    path.write_text(text)
    with pytest.raises(ConfigError):
        set_cv_bandwidths(path, {"d0": 0.02})
    assert path.read_text() == text
