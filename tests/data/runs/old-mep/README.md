# Old MEP output (regression reference)

Written by the original scripts on the runs in `tests/data/runs`, from that folder:

    python ../../../to_refactor/fes.py command=mep dir_md=. 'dir_job=00*' 'cvs.select=[d0,cn]' property=fes

with the settings of `to_refactor/config/case/atio2-h2o.yaml` (`surface_data.fes.mep`: split
vertical, 2 minima, 15 points, dt 0.01, 2000 steps, fixed ends; free energies clipped to
`plot.vmax` = 1.5 eV). Energies are in eV, shifted so that each surface's minimum is 0.

- `d0-cn-fes-AVG-MEP.dat`: path on `d0-cn-rew-fes-AVG.dat` (columns d0, cn, energy, then
  min/max/std over the runs and each run's energy along this path).
- `00N-d0-cn-fes-MEP.dat`: path on run N's own surface (`00N/pot00N-d0-cn-rew-fes.dat`).
  Run 003 is missing: the old code wrote the average's path over `003/d0-cn-fes-MEP.dat`.
