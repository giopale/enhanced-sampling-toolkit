"""Run ECToolkits' ProtonTransferCV (as downloaded, logger stubbed) on traj.xyz.

Usage: python ectoolkits_reference.py ECTOOLKITS_SOURCE_DIR OUTPUT_DIR
(needs MDAnalysis and pandas; ECToolkits 0.4.4 and MDAnalysis 2.11 made the files here)
"""

import importlib.util
import logging
import sys
import types

import numpy as np
from MDAnalysis import Universe

WATER = list(range(1, 40))  # 0-based: water O 2-40
CASES = {
    "direct-sites": dict(idxs_type1_o=[0], idxs_type2_o=[40, 41, 42]),
    "indirect-sites": dict(idxs_type1_o=[0], idxs_type2_o=[40, 41, 42], num_bridge=1,
                           idxs_water_o=WATER),
    "direct-liquid": dict(idxs_type1_o=[0], idxs_type2_o=WATER),
    "indirect-liquid": dict(idxs_type1_o=[0], idxs_type2_o=WATER, num_bridge=1,
                            idxs_water_o=WATER),
}  # fmt: skip


def load_module(source):
    log_module = types.ModuleType("ectoolkits.log")
    log_module.get_logger = logging.getLogger
    sys.modules["ectoolkits"] = types.ModuleType("ectoolkits")
    sys.modules["ectoolkits.log"] = log_module
    path = f"{source}/ectoolkits/analysis/proton_transfer_cv.py"
    spec = importlib.util.spec_from_file_location("proton_transfer_cv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(source, out):
    module = load_module(source)
    universe = Universe(f"{out}/traj.xyz", format="XYZ")
    universe.atoms.dimensions = [9.0, 9.5, 10.0, 90, 90, 90]
    for name, options in CASES.items():
        analysis = module.ProtonTransferCV(universe.atoms, verbose=False, **options)
        analysis.run()
        np.savetxt(f"{out}/ref-{name}.txt", analysis.results[:, 1:3])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
