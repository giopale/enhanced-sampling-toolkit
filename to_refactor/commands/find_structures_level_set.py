from __future__ import annotations
from rich import print
import sys
import numpy as np
from logger_setup import logger
from pathlib import Path
from parser import get_temperature_from_log, parse_state_wfile, parse_bias_factor, parse_colvar_header
from parser import parse_cv_wall
from ectoolkits_reweight import compute_bw_silverman, read_fes, write_fes
from omegaconf import ListConfig
from utils import extract_field_map, find_job_dirs,find_files, read_colvar_in_time_interval, safe_stack_colvar, apply_mpl_style
from utils_reweighting import point_kernel_weights, compute_frame_weights
import matplotlib.pyplot as plt 
import h5py
from ase import Atoms
from ase.io import write
from tqdm import tqdm
import json
from collections import deque


TYPE_MAP = np.array(["H", "O", "Ti"])

def label_basins(Z, L, *, connectivity=8):
    """
    Label connected components of mask = finite(Z) & (Z < L).
    Returns:
      labels: (nx, ny) int array, 0 = not in any basin, 1..nbasins basin ids
      nbasins: number of basins
      mask: boolean mask used
    """
    Z = np.asarray(Z, dtype=float)
    nx, ny = Z.shape

    mask = np.isfinite(Z) & (Z < L)
    labels = np.zeros((nx, ny), dtype=np.int32)

    if connectivity == 4:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    cur = 0
    q = deque()

    for i in range(nx):
        for j in range(ny):
            if not mask[i, j] or labels[i, j] != 0:
                continue

            cur += 1
            labels[i, j] = cur
            q.append((i, j))

            while q:
                ii, jj = q.popleft()
                for di, dj in neigh:
                    ni, nj = ii + di, jj + dj
                    if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj] and labels[ni, nj] == 0:
                        labels[ni, nj] = cur
                        q.append((ni, nj))

    return labels, cur, mask


def point_basin_id_nearest_cell(X, Y, Z, labels, x0, y0, *, L):
    """
    Returns basin id (int) or 0 if outside/unknown.
    Uses the grid cell index; requires Z(x0,y0) < L by bilinear interpolation.
    """
    xbins = X[:, 0]
    ybins = Y[0, :]

    # find cell
    i = np.searchsorted(xbins, x0) - 1
    j = np.searchsorted(ybins, y0) - 1
    if i < 0 or i >= xbins.size - 1 or j < 0 or j >= ybins.size - 1:
        return 0

    # quick reject if cell corners are not all finite (holes)
    z11 = Z[i, j]
    z21 = Z[i + 1, j]
    z12 = Z[i, j + 1]
    z22 = Z[i + 1, j + 1]
    if not (np.isfinite(z11) and np.isfinite(z21) and np.isfinite(z12) and np.isfinite(z22)):
        return 0

    # bilinear interpolate to enforce Z(x0,y0) < L
    x1, x2 = xbins[i], xbins[i + 1]
    y1, y2 = ybins[j], ybins[j + 1]
    tx = (x0 - x1) / (x2 - x1)
    ty = (y0 - y1) / (y2 - y1)

    z0 = ((1 - tx) * (1 - ty) * z11
          + tx * (1 - ty) * z21
          + (1 - tx) * ty * z12
          + tx * ty * z22)

    if not (np.isfinite(z0) and z0 < L):
        return 0

    # basin id from labels: use the most common label among the 4 corners
    # (more stable than just labels[i,j] if boundary runs through the cell)
    corner_ids = np.array([labels[i, j], labels[i + 1, j], labels[i, j + 1], labels[i + 1, j + 1]])
    corner_ids = corner_ids[corner_ids > 0]
    if corner_ids.size == 0:
        return 0
    vals, counts = np.unique(corner_ids, return_counts=True)
    return int(vals[np.argmax(counts)])



def build_step_index(f, step_path="/particles/all/species/step"):
    steps = f[step_path][...]  # int array, shape (nframes,)
    # If steps are strictly increasing and unique, this is perfect:
    return {int(s): i for i, s in enumerate(steps)}

def h5md_to_ase_atoms(fname, steps, type_map, pbc=True):
    steps = np.atleast_1d(steps).astype(int)

    atoms_list = []
    with h5py.File(fname, "r") as f:
        step_to_frame = build_step_index(f, "/particles/all/species/step")

        pos_ds  = f["/particles/all/position/value"]
        spec_ds = f["/particles/all/species/value"]
        box_ds  = f["/particles/all/box/edges/value"]

        for st in tqdm(steps,total=len(steps), ncols=80):
            if int(st) not in step_to_frame:
                raise KeyError(f"timestep {st} not found in {fname}")

            fr = step_to_frame[int(st)]

            symbols = [type_map[int(i)] for i in spec_ds[fr]]
            atoms = Atoms(
                symbols=symbols,
                positions=pos_ds[fr],
                cell=np.diag(box_ds[fr]),
                pbc=pbc,
            )
            atoms_list.append(atoms)

    return atoms_list



def build_lammpstrj_index(files, index_out="traj.index.json"):
    """
    Build index mapping timestep -> {"file": str, "offset": int}
    offset is the byte position of the 'ITEM: TIMESTEP' line.

    This scans each file once (streaming, low RAM).
    """
    index = {}
    for fn in files:
        fn = str(fn)
        with open(fn, "rb", buffering=1024 * 1024) as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith(b"ITEM: TIMESTEP"):
                    step_line = f.readline()
                    if not step_line:
                        break
                    step = int(step_line.strip())
                    # keep first occurrence; if you prefer "last wins", just overwrite
                    if step not in index:
                        index[step] = {"file": fn, "offset": offset}

    # store sorted (nice to inspect)
    steps_sorted = sorted(index.keys())
    out = {
        "steps": steps_sorted,
        "index": {str(s): index[s] for s in steps_sorted},
    }
    with open(index_out, "w", encoding="utf-8") as w:
        json.dump(out, w)
    return out


def _readline(f):
    line = f.readline()
    if not line:
        raise EOFError("Unexpected EOF while reading frame")
    return line


def read_atoms_at_timestep_sorted(
    index_json,
    timestep,
    pbc=True,
    prefer_element=False,
    type_map=None,
):
    """
    Read one LAMMPS frame by timestep and return ASE Atoms,
    always sorted by atom id.
    """

    # Load index
    with open(index_json, "r", encoding="utf-8") as r:
        meta = json.load(r)

    rec = meta["index"].get(str(int(timestep)))
    if rec is None:
        raise KeyError(f"Timestep {timestep} not found in index")

    fn = rec["file"]
    off = int(rec["offset"])

    with open(fn, "rb", buffering=1024 * 1024) as f:
        f.seek(off)

        # --- TIMESTEP ---
        if not _readline(f).startswith(b"ITEM: TIMESTEP"):
            raise ValueError("Corrupt index: not at TIMESTEP")

        step = int(_readline(f).strip())

        # --- NUMBER OF ATOMS ---
        if not _readline(f).startswith(b"ITEM: NUMBER OF ATOMS"):
            raise ValueError("Missing NUMBER OF ATOMS")

        natoms = int(_readline(f).strip())

        # --- BOX ---
        if not _readline(f).startswith(b"ITEM: BOX BOUNDS"):
            raise ValueError("Missing BOX BOUNDS")

        xlo, xhi = map(float, _readline(f).split())
        ylo, yhi = map(float, _readline(f).split())
        zlo, zhi = map(float, _readline(f).split())

        cell = np.diag([xhi - xlo, yhi - ylo, zhi - zlo])

        # --- ATOMS header ---
        line = _readline(f)
        if not line.startswith(b"ITEM: ATOMS"):
            raise ValueError("Missing ATOMS section")

        cols = line.decode().strip().split()[2:]
        col = {c: i for i, c in enumerate(cols)}

        # Mandatory
        for k in ("id", "x", "y", "z"):
            if k not in col:
                raise ValueError(f"Missing column '{k}'")

        has_element = "element" in col
        has_type = "type" in col

        if prefer_element and not has_element:
            raise ValueError("prefer_element=True but no element column")

        if (not prefer_element) and (not has_type or type_map is None):
            raise ValueError("Need type + type_map")

        # --- Read atoms ---
        ids = np.empty(natoms, dtype=np.int64)
        pos = np.empty((natoms, 3), dtype=np.float64)
        sym = [None] * natoms

        for i in range(natoms):
            parts = _readline(f).decode().split()

            ids[i] = int(parts[col["id"]])

            pos[i, 0] = float(parts[col["x"]]) #- xlo
            pos[i, 1] = float(parts[col["y"]]) #- ylo
            pos[i, 2] = float(parts[col["z"]]) #- zlo

            if prefer_element:
                sym[i] = parts[col["element"]]
            else:
                t = int(parts[col["type"]])
                sym[i] = type_map[t]

        # --- Sort by id ---
        order = np.argsort(ids)

        pos = pos[order]
        sym = [sym[i] for i in order]


        atoms = Atoms(
            symbols=sym,
            positions=pos,
            cell=cell,
            pbc=pbc,
        )

        atoms.info["timestep"] = step

        return atoms





def run(cfg):
    print(f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] {cfg.param}")
    apply_mpl_style(cfg.mplstyle.dir,cfg.mplstyle.style)

    # get relevant config
    #pp=cfg.surface_data.get(prop, {})
    cfg_find=cfg.find_structures_level_set

    prop=cfg.get('property', 'fes')
    cvs=cfg.cvs.get('select',None)
    assert cvs is not None


    
   ###################################### 
    # find directories
    jobdirs=find_job_dirs(Path(cfg.get('dir_md','.')), cfg.dir_job)

    logger.info('#'*100+f' find_structures_level_set')

    # loop over directories
    files_written=[]
    for idx, jj in enumerate(jobdirs): 

        logger.info('#'*60+f' {jj.name}')
    
        # get job index for output
        if 'job-00' in jj.name:
            jj_idx = int(str(jj.name).split('-')[-1])
        else:
            jj_idx = idx
    
        # get simulation parameters
        temp = get_temperature_from_log(jj/ cfg["file_lammps_log"])
        state_wfile = parse_state_wfile(jj/ cfg["file_plumed_input"])
        bias_factor = parse_bias_factor(jj/state_wfile)
        kbT = temp*0.0083144621  # kJ/mol
        ngrid = cfg.get("ngrid", [100, 100])
    
        logger.info(f'temp is {temp:.0f} K')
        logger.info(f'kbT is {kbT:.3g}')
        logger.info(f'ngrid: {ngrid}')
    
        cvs_dict = {}
    
        # parse cvs from COLVAR
        colvar_fields = extract_field_map(jj/'COLVAR')
        colvars_plumed = [str(ii) for ii in colvar_fields.keys()]
    
        cvs = list(cfg["cvs"]["select"])
        logger.info(f"collective variables: {', '.join(cvs)}")
    
        # get time interval (nanoseconds)
        timeint = cfg.get("time_interval", [0.0, np.inf])
        # move to picoseconds
        # in my setup PLUMED outputs picoseconds!
        timeint = [1000 * ii if ii is not None else None for ii in timeint]
        logger.info(f'selected time interval: {timeint}')
    
        # parse COLVAR
        data = read_colvar_in_time_interval(jj/'COLVAR', timeint[0], timeint[1])
    
        # parse CVs not in COLVAR
        # support for extra cvs calculated a posteriori using PLUMED-driver
        masks = []
        for cv_sel in cvs:
            if cv_sel not in colvar_fields.keys():
                # PLUMED can produce CVs with multiple components, such as z.min, opes.rct, ...
                # my convention is to save them into a file named without componenents
                cv_without_components = cv_sel.split('.')[0] if '.' in cv_sel else cv_sel
                logger.info(f'cv {cv_sel} is not in COLVAR. loading file CV-{cv_without_components}')
                # Note: time units in PLUMED-driver must match the simulation units!
                data_new_cv = read_colvar_in_time_interval(jj/f'CV-{cv_without_components}', timeint[0], timeint[1])
                # stack added colvars with the others
                data = safe_stack_colvar(data, data_new_cv)
                colvar_fields.update({cv_sel: np.shape(data)[-1]-1})
                # remove nans
                mask = ~np.isnan(data_new_cv[:, 1])
                n_false = np.count_nonzero(~mask)
                logger.warning(f'nan detected in colvar {cv_sel}: {n_false} cases')
            else:
                mask = np.array([True]*len(data))
            masks.append(mask)
    
        # filter data outside some given limits
        mask_new = np.array([True]*len(data))
        cvs_to_filter = cfg["cvs"].get("filter", None)
        if cvs_to_filter is not None:
            for name, limits_filter in cvs_to_filter.items():
                mask_new &= (data[:, colvar_fields[name]] >= limits_filter[0]) & \
                            (data[:, colvar_fields[name]] <= limits_filter[1])
    
        mask_and = mask_new & masks[0] & masks[1]
        if np.count_nonzero(mask_and) != len(data):
            logger.info(f'filtering nan/limits, retaining {np.count_nonzero(mask_and)} structures ({np.count_nonzero(mask_and)/len(mask_and)*100:.3f}%)')
            data = data[mask_and]
        logger.debug('DONE data filtering')
    
    
    
        # parse the limits of the cvs map to use for the analysis
        for ii in cvs:
            logger.debug(f'limits: {ii}')
            limits = []
            # get limits from plumed
            for wall in ['LOWER', 'UPPER']:
                limits.append(parse_cv_wall(file_path=jj/cfg["file_plumed_input"], cv=ii, wall=wall))
            logger.debug(f'limits: walls done')
            # merge with limits from config file
            limits_cfg = cfg.get("cvs", {}).get(ii, {}).get("limits", [None, None])
            for kk in [0, 1]:
                if limits_cfg[kk] is not None:
                    limits[kk] = limits_cfg[kk]
            cvs_dict[ii] = {"limits": limits}
            if None in limits:
                logger.info(f'you must set limits for cv \'{ii}\'. Currently {limits}. Exiting')
                sys.exit()
        logger.debug('DONE parsing cvs limits')
    
        # store the data of the selected CVs
        for idx_cv, ii in enumerate(cvs):
            cvs_dict[ii]["data"] = data[:, colvar_fields[ii]]
    
        # store the bias
        bias = data[:, colvar_fields["opes.bias"]]

        # compute weights
        weights = compute_frame_weights(bias, kbT, normalize=True)
    
        # compute bandwidths
        matrix_kernel = np.zeros((2,2))
        for idx_cv, ii in enumerate(cvs):
            sig = cfg.get('cvs',{}).get(ii,{}).get('sigma',None)
    
            msg = '(specified from config)'
            if sig is None:
                msg='(silverman)'
                bw = compute_bw_silverman(cvs_dict[ii]["data"], dim=2,
                                      weights=weights, bias_factor=bias_factor)
            else:
                bw = sig
            logger.info(f'bandwidth for \'{ii}\' is {bw:.5f} {msg}')
            matrix_kernel[idx_cv,idx_cv]  = bw**2
    
        file_fes = find_files(jj,f'*{cvs[0]}-{cvs[1]}-rew-fes.dat')
        assert(len(file_fes) == 1)
        file_fes = file_fes[0]


        # get grid
        X, Y, Z, _, = read_fes(file_fes, dim=2)

        isovalue = cfg_find.get('isovalue',None)
        logger.info(f'looking for basins, isovalue {isovalue:.3f}')
        labels, nbasins, mask = label_basins(Z, L=isovalue, connectivity=8)
        logger.info(f"grid points below isovalue: {mask.sum()} / {mask.size} ({100*mask.mean():.2f}%)")

        if nbasins > 0:
            logger.info(f'found {nbasins} basins')
        else:
            sys.exit(f'no basins found for isovalue {isovalue}. Goodbye')

        xpts = np.asarray(cvs_dict[cvs[0]]['data'], dtype=float)
        ypts = np.asarray(cvs_dict[cvs[1]]['data'], dtype=float)
        
        if xpts.shape != ypts.shape:
            raise ValueError(f"x and y must have same shape, got {xpts.shape} vs {ypts.shape}")
        
        points = np.column_stack((xpts, ypts))  # shape (npoints, 2)
        
        ids = np.array(
            [point_basin_id_nearest_cell(X, Y, Z, labels, x0, y0, L=isovalue) for x0, y0 in points],
            dtype=np.int32,
        )
        nin = np.count_nonzero(ids)
        logger.info(f"points assigned to basins: {nin} / {len(points)} ({100*nin/len(points):.2f}%)")

        basin_ids = np.unique(ids)
        basin_ids = basin_ids[basin_ids != 0]  # drop "not in any basin"

        idx = np.arange(points.shape[0])
        basin_indices = {bid: idx[ids == bid] for bid in basin_ids}



        data_basin = {}
        for bid, idx in basin_indices.items():
            xavg=np.mean(points[idx,0])
            yavg=np.mean(points[idx,1])

            w_tot = np.sum(weights)
            w_sub = weights[idx]
            frac = np.sum(w_sub) / w_tot


            logger.info(f'basin {bid}:')
            logger.info(f'\tcentered at {xavg:.2f} {yavg:.2f}')
            logger.info(f'\tcontains {len(idx)} structures ({len(idx)/len(points)*100:.2f}%)')
            logger.info(f'\tfes[basin {bid}] integrates to {frac*100:.2f}%')

            data_basin.update({bid:{'center':[xavg,yavg], 'idx_selected': idx}})



        #######################################################    
        #######################################################    

        cfg_choice = cfg_find.get("choose", {})
        seed = cfg_choice.get("random", {}).get("seed", None)
        
        for bid, bdata in data_basin.items():
            center_basin = bdata['center']
            logger.info('')
            logger.info('*'*100 + f' {cvs[0]} {center_basin[0]:.2f}, {cvs[1]} {center_basin[1]:.2f}')

            indices_basin = bdata["idx_selected"]         # GLOBAL indices
            w_sub = weights[indices_basin]
            p_sub = w_sub / np.sum(w_sub)
        
            thr = cfg_choice.get("probability_thr", None)
        
            if thr is not None:
                order = np.argsort(p_sub)[::-1]           # LOCAL
                cum = np.cumsum(p_sub[order])
                nstruct = np.searchsorted(cum, thr, side="left") + 1
                chosen_local = order[:nstruct]
                logger.info(f'{nstruct} structures found, {nstruct/len(w_sub):.2g}% of basin structures')
            else:
                nstruct = cfg_choice.get("random", {}).get("nstruct", None)
                assert seed is not None
                assert nstruct is not None
                assert nstruct <= len(indices_basin)
        
                rng = np.random.default_rng(seed + int(bid))  # or rng outside loop
                chosen_local = rng.choice(len(indices_basin), size=nstruct, replace=False, p=p_sub)
        
            indices_struct = indices_basin[chosen_local]   # GLOBAL
            weights_struct = weights[indices_struct]       # or w_sub[chosen_local]
        
            sample_freq = 100
            timesteps_struct = indices_struct * sample_freq

            logger.info(f'first 5 indices: {indices_struct[:5]}')
            logger.info(f'---------------> assuming sample_freq = {sample_freq:d}! <---------------')
            logger.info(f'first 5 TIMESTEPS: {timesteps_struct[:5]}')
            logger.info(f'first 5 WEIGHTS: {weights_struct[:5]}')


            fname_out=f'struct-indices-basin-{cvs[0]}{center_basin[0]:.2f}-{cvs[1]}{center_basin[1]:.2f}.dat'
            prob=np.sum(weights_struct)
            header=f"# TIMESTEPS of {nstruct} structures, summing up weight to {prob:.4g}"
            line_timesteps=",".join([str(ii) for ii in timesteps_struct])
            line_weights=",".join([f"{ii:.5g}" for ii in weights_struct])

            with open(jj/fname_out, "w") as f:
                f.write(header + "\n")
                f.write(line_weights + "\n")
                f.write(line_timesteps + "\n")
            logger.info(f'indices saved to {jj/fname_out}')

            cfg_save = cfg_find.get('save_structures',None)

            if cfg_save is None:
                continue

            type_map = cfg_save.get('type_map',None)
            file_index = jj/"traj.index.json"
            file_traj_lammps = [jj/kk for kk in cfg_save.file_lammps]
            fname_out_structures = jj/f'structures-basin-{cvs[0]}{center_basin[0]:.2f}-{cvs[1]}{center_basin[1]:.2f}.xyz'

            if not file_index.exists():
                idx = build_lammpstrj_index(file_traj_lammps, jj/"traj.index.json")


            atoms = []
#            for tt in tqdm(timesteps_struct,total=len(timesteps_struct),ncols=80):
#                atoms.append(read_atoms_at_timestep_sorted(file_index,tt,type_map=type_map))

            if True:

                file_traj = jj/Path(cfg_save.file_dump_h5)
                assert file_traj.exists()
                assert type_map is not None
                logger.info(f'Using the following type map (from config): {type_map}')

                fname_out_structures = jj/f'structures-basin-{cvs[0]}{center_basin[0]:.2f}-{cvs[1]}{center_basin[1]:.2f}.xyz'
        
                logger.info(f'loading atoms from {file_traj}')
                atoms = h5md_to_ase_atoms(file_traj, timesteps_struct, type_map)


                logger.info('writing...')
                write(fname_out_structures, atoms)

                logger.info(f'file written {fname_out_structures}')

            



