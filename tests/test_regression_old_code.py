"""Compare the new reweighting with the output of the original scripts on real runs.

The reference data in tests/data/runs is made with tools/make_test_data.py; the
``<run>/pot*-<cv0>-<cv1>-rew-fes.dat`` files were written by
``to_refactor/fes.py command=reweight_property`` with the case file
to_refactor/config/case/atio2-h2o.yaml.
"""

import re

import numpy as np
import pytest

from estk.migrate import migrate_case_yaml
from estk.project import Project
from estk.surface import Surface

from .conftest import DATA_DIR, REPO_ROOT

RUNS = DATA_DIR / "runs"
CASE = REPO_ROOT / "to_refactor" / "config" / "case" / "atio2-h2o.yaml"
OLD_FILE = re.compile(r"pot(\d+)-(.+)-(.+)-rew-fes\.dat$")

old_files = sorted(RUNS.glob("*/pot*-rew-fes.dat")) if RUNS.is_dir() else []
pytestmark = pytest.mark.skipif(
    not old_files, reason="no reference output in tests/data/runs (see tools/make_test_data.py)"
)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    # The settings the old code used: the case file, with the runs in tests/data/runs.
    text, _ = migrate_case_yaml(CASE)
    config_path = tmp_path_factory.mktemp("config") / "estk.yaml"
    config_path.write_text(text)
    project = Project(config_path)
    project.config.runs.root = RUNS
    project.config.runs.pattern = "*"
    return project


@pytest.mark.parametrize("old_path", old_files, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_free_energy_matches_old_code(project, old_path):
    old = Surface.load(old_path)
    cvs = list(OLD_FILE.search(old_path.name).groups()[1:])
    assert list(old.grid.names) == cvs

    result = project.reweight(cvs, points=old.grid.points)
    new = result.surfaces[old_path.parent.name]

    # Same grid: the limits come from the case file and the PLUMED walls in both codes.
    np.testing.assert_allclose(new.grid.lower, old.grid.lower, atol=1e-3)
    np.testing.assert_allclose(new.grid.upper, old.grid.upper, atol=1e-3)

    # The old files hold -kT log p (weights normalised to 1, not shifted), written with 6
    # decimals; compare after shifting both minima to 0. The old code used k_B = 0.0083144621
    # kJ/(mol K), estk the CODATA value 0.008314462618: F differs by a factor 1 + 6e-8, which
    # is visible (2e-5 kJ/mol) only where F is several hundred kJ/mol.
    old_values = old.values - np.min(old.values[np.isfinite(old.values)])
    finite = np.isfinite(old_values)
    assert np.array_equal(finite, np.isfinite(new.values))
    np.testing.assert_allclose(new.values[finite], old_values[finite], atol=2e-5, rtol=1e-7)
    assert new.metadata["fes_offset"] == pytest.approx(np.min(old.values[finite]), abs=2e-5)
