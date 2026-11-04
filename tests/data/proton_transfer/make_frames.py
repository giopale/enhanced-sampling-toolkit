"""Write random water-like frames as traj.xyz (for ECToolkits) and traj.lammpstrj (for estk).

Usage: python make_frames.py OUTPUT_DIR
"""

import sys

import numpy as np

N_WATER, N_SURFACE, FRAMES = 40, 6, 60
L = np.array([9.0, 9.5, 10.0])


def frame(rng):
    """Positions and elements: water O (ids 1-40), surface O (41-46), H (47-126)."""
    oxygens = rng.uniform(0, 1, (N_WATER, 3)) * L
    directions = rng.normal(size=(N_WATER, 2, 3))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    lengths = rng.uniform(0.9, 1.6, (N_WATER, 2, 1))
    hydrogens = (oxygens[:, None, :] + directions * lengths).reshape(-1, 3) % L
    surface = rng.uniform(0, 1, (N_SURFACE, 3)) * L
    elements = ["O"] * (N_WATER + N_SURFACE) + ["H"] * (2 * N_WATER)
    return np.vstack([oxygens, surface, hydrogens]), elements


def main(out):
    rng = np.random.default_rng(3)
    with open(f"{out}/traj.xyz", "w") as xyz, open(f"{out}/traj.lammpstrj", "w") as dump:
        for k in range(FRAMES):
            positions, elements = frame(rng)
            n = len(positions)
            xyz.write(f"{n}\n{L[0]} {L[1]} {L[2]}\n")
            for e, p in zip(elements, positions, strict=True):
                xyz.write(f"{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            dump.write(f"ITEM: TIMESTEP\n{k * 100}\nITEM: NUMBER OF ATOMS\n{n}\n"
                       f"ITEM: BOX BOUNDS pp pp pp\n0 {L[0]}\n0 {L[1]}\n0 {L[2]}\n"
                       "ITEM: ATOMS id type element x y z\n")  # fmt: skip
            for i in rng.permutation(n):  # atoms in a different order in every frame
                kind = 1 if elements[i] == "H" else 2
                x, y, z = positions[i]
                dump.write(f"{i + 1} {kind} C {x:.6f} {y:.6f} {z:.6f}\n")


if __name__ == "__main__":
    main(sys.argv[1])
