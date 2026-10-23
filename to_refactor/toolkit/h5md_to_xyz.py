#!/usr/bin/env python3
import h5py
import numpy as np
import sys


type_map={1:'H',2:'O',3:'Ti'}

def h5md_to_xyz(h5file, frame, outfile):
    with h5py.File(h5file, 'r') as f:

        # --- read atom names ---
        # H5MD convention: /particles/atoms/name or species
        if "particles" not in f:
            raise RuntimeError("H5MD does not contain /particles group.")

        # assume one particle group
        pgrp = list(f["particles"].keys())[0]
        g = f["particles"][pgrp]
        species=g['species']['value'][0]
        names=np.array([type_map[ii] for ii in species])

        edges=f['/particles/all/box/edges/value'][0]

        cell_par=' '.join([str(ii) for ii in edges]) +' 90.0 90.0 90.0'
        print(f'cell parameters: {edges}')


        # --- read positions ---
        pos = g["position"]["value"][frame]  # shape N×3

        # --- write xyz ---
        with open(outfile, "w") as out:
            out.write(f"{len(pos)}\n")
            out.write(f"{cell_par}\n")
            for name, r in zip(names, pos):
                out.write(f"{name} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: h5md_to_xyz.py TRAJ.h5md FRAME OUTPUT.xyz")
        sys.exit(1)

    print(f'WARNING!!! --- using type map: {type_map}')

    h5file = sys.argv[1]
    frame = int(sys.argv[2])
    outfile = sys.argv[3]

    h5md_to_xyz(h5file, frame, outfile)
    print(f"Wrote {outfile}")

