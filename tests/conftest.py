from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run the test inside an empty temporary folder."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_colvar(folder: Path, fields: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "COLVAR"
    path.write_text("#! FIELDS " + " ".join(fields) + "\n" + " ".join("0" for _ in fields) + "\n")
    return path


LAMMPS_LOG = """LAMMPS (2 Aug 2023)
variable T equal 350
units metal
  timestep 0.0005
fix 1 all nvt temp ${T} ${T} $(100*dt)
run 1000
Step Temp PotEng
0 349.2 -1234.5
1000 351.0 -1230.1
Loop time of 12.3 on 4 procs
"""

PLUMED_INPUT = """# test input
UNITS LENGTH=A TIME=ps ENERGY=kj/mol
d0: DISTANCE ATOMS=1,2
cn: COORDINATION GROUPA=1 GROUPB=3 R_0=1.3
opes: OPES_METAD ...
  ARG=d0,cn
  PACE=500 BARRIER=60
  BIASFACTOR=10
  STATE_WFILE=STATE
...
uwall: UPPER_WALLS ARG=d0,cn AT=2.5,3.0 KAPPA=1000,1000
LOWER_WALLS ARG=d0 AT=1.5 KAPPA=1000 LABEL=lwall
PRINT FILE=COLVAR ARG=d0,cn,opes.bias STRIDE=10
"""


def make_run(folder: Path, n: int = 100, *, dt: float = 0.005) -> Path:
    """Write a fake run: COLVAR every ``dt`` ps with d0, cn and opes.bias."""
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "log.lammps").write_text(LAMMPS_LOG)
    (folder / "plumed.plumed").write_text(PLUMED_INPUT)
    t = np.arange(n) * dt
    rows = [
        f"{ti:.6f} {1.5 + 0.01 * i:.6f} {i % 7 * 0.5:.6f} {0.1 * i:.6f}" for i, ti in enumerate(t)
    ]
    (folder / "COLVAR").write_text(
        "#! FIELDS time d0 cn opes.bias\n#! SET min_d0 0\n" + "\n".join(rows) + "\n"
    )
    return folder
