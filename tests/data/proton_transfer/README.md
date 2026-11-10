# Proton-transfer reference data

`traj.lammpstrj.gz`: 60 random water-like frames (40 waters, 6 surface oxygens without H, in a
9 × 9.5 × 10 Å periodic box; atom ids: water O 1–40, surface O 41–46, H 47–126; LAMMPS type
1 = H, 2 = O), written by `make_frames.py` (which also writes the same frames as `traj.xyz`,
not kept). Atoms are in a different order in every frame.

`ref-<case>.txt`: delta and d per frame from ECToolkits 0.4.4
(`ectoolkits.analysis.proton_transfer_cv.ProtonTransferCV`, with MDAnalysis 2.11), run by
`ectoolkits_reference.py` on `traj.xyz` (0-based indices there):

| case | type 1 | type 2 | bridges |
|---|---|---|---|
| direct-sites | O 1 | O 41–43 | |
| indirect-sites | O 1 | O 41–43 | O 2–40 |
| direct-liquid | O 1 | O 2–40 | |
| indirect-liquid | O 1 | O 2–40 | O 2–40 |

estk reproduces all four with `ectoolkits_compat=True`. Without it, indirect-liquid differs:
ECToolkits also accepts some chains whose bridge is the donor or the acceptor (see
`estk.cv.proton_transfer`).
