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
import matplotlib
matplotlib.use('Agg')
import h5py
from ase import Atoms
from ase.io import write
from tqdm import tqdm
import json

TYPE_MAP = np.array(["H", "O", "Ti"])

def closest_grid_point(X, Y, x0, y0):
    dist2 = (X - x0)**2 + (Y - y0)**2
    return np.unravel_index(np.argmin(dist2), dist2.shape)

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

import json
import numpy as np
from ase import Atoms


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


#def h5md_to_ase_atoms(fname, frames, type_map, pbc=True):
#    """
#    Read selected frames from H5MD file and return ASE Atoms list.
#
#    Parameters
#    ----------
#    fname : str
#        H5MD file
#    frames : int or sequence
#        Frame indices
#    type_map : dict[int, str]
#        Maps species id -> chemical symbol
#        Example: {1: "H", 2: "O", 3: "Ti"}
#    pbc : bool
#        Periodic boundary conditions
#
#    Returns
#    -------
#    atoms_list : list[ase.Atoms]
#    """
#
#    frames = np.atleast_1d(frames)
#
#    atoms_list = []
#
#    with h5py.File(fname, "r") as f:
#        pos_ds = f["/particles/all/position/value"]
#        spec_ds = f["/particles/all/species/value"]
#        box_ds = f["/particles/all/box/edges/value"]
#
#        for fr in frames:
#            # Positions (natoms, 3)
#            positions = pos_ds[fr]
#
#            # Species ids (natoms,)
#            species_id = spec_ds[fr]
#
#            # Convert to symbols using provided map
#            try:
#                symbols = [type_map[int(i)] for i in species_id]
#            except KeyError as e:
#                raise KeyError(f"Species id {e} not in type_map") from None
#
#            # Box (orthorhombic)
#            cell = np.diag(box_ds[fr])
#
#            atoms = Atoms(
#                symbols=symbols,
#                positions=positions,
#                cell=cell,
#                pbc=pbc,
#            )
#
#            atoms_list.append(atoms)
#
#    return atoms_list




def run(cfg):
    print(f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] {cfg.param}")
    apply_mpl_style(cfg.mplstyle.dir,cfg.mplstyle.style)

    # get relevant config
    #pp=cfg.surface_data.get(prop, {})
    cfg_find=cfg.find_structures

    prop=cfg.get('property', 'fes')
    cvs=cfg.cvs.get('select',None)
    assert cvs is not None

    points_fes=cfg_find.get('points',None)
    assert points_fes is not None
    
    points_fes_names=[f'{cvs[0]}-{pt[0]:.2f}-{cvs[1]}-{pt[1]:.2f}' for pt in points_fes]

    logger.info(f'selecting structures for the following points on the \'{"-".join(cvs)}\' fes: {points_fes}')

    
   ###################################### 
    # find directories
    jobdirs=find_job_dirs(Path(cfg.get('dir_md','.')), cfg.dir_job)

    logger.info('#'*100+f' find_structures')

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
        X, Y, _, _, = read_fes(file_fes, dim=2)
    

        # cycle over selected points in CV space
        for idx, point in enumerate(points_fes):
            logger.info('')
            logger.info(f'computing weights for point [{point[0]:.3f},{point[1]:.3f}]')

            i, j = closest_grid_point(X,Y,point[0], point[1])

            parameters = {
                        'cv_x': cvs_dict[cvs[0]]["data"],
                        'cv_y': cvs_dict[cvs[1]]["data"],
                        'logw': np.log(weights),
                        'H': matrix_kernel,
                        'xg': point[0],
                        'yg': point[1],
                        }
            ww, _ = point_kernel_weights(**parameters)


            # sort and sum
            indices_sorted=np.argsort(ww)[::-1]
            weights_sorted=ww[indices_sorted]
            cum_sum_weights=np.cumsum(weights_sorted)

            # select structures
            logger.info(f'selecting structures constituting {cfg_find.probability_thr*100:.2f}% of the weights, highest weights first')
            prob=cfg_find.probability_thr
            nstruct=np.searchsorted(cum_sum_weights,cfg.find_structures.probability_thr) 
            logger.info(f'{nstruct} structures found, {nstruct/len(weights):.2g}% of total structures')

            # cut the array if the structures are too many
            max_struct=cfg_find.get('max_structures',None)
            if max_struct is None:
                max_struct=np.inf
            if nstruct > max_struct:
                nstruct=max_struct
                logger.info(f'retaining the first {max_struct} structures ({nstruct/len(weights):.2g}% of total),summing up to {cum_sum_weights[nstruct-1]*100:.2f}% weight')
                prob=cum_sum_weights[nstruct-1]
            
    
            indices_struct=indices_sorted[:nstruct]
            weights_struct=weights_sorted[:nstruct]
            cutoff=0.003
            
            if '000' in jj.name:
                limit=None
                #limit=np.where(weights_sorted > cutoff)[-1][-1]

                end=nstruct

                fig, ax=plt.subplots()
                yy=weights_sorted[:end+1000]
                xx=np.arange(1,len(yy)+1,1)
                ax.plot(xx,yy,ls='-',label='weights', color='tab:blue')

                if limit is not None:
                    end=np.max((limit,nstruct))
                    ax.fill_between(xx[:limit+1],yy[:limit+1],color='tab:blue',alpha=0.3)
                #ax.plot([xx[limit+1],ax.get_xlim()[-1]+np.diff(ax.get_ylim())[0]*0.1], [cutoff,cutoff],color='r',lw=.5,label=f'cutoff @ {cutoff:.4f}')

                ax.fill_between(xx[:nstruct-1],yy[:nstruct-1],color='tab:green',alpha=0.1)
                #ax.plot([xx[nstruct-1],ax.get_xlim()[-1]+np.diff(ax.get_ylim())[0]*0.1], [yy[nstruct-1],yy[nstruct-1]],color='g',lw=.5,label=f'p={prob*100:.0f}% @ {yy[nstruct-1]:.4f}')

                #ax.text(0.1, 0.9, f"nstruct: cutoff/prob {limit+1:d} / {nstruct+1:d}", transform=ax.transAxes, va='bottom',ha='left')
                #ax.text(0.1, 0.9, f"$\sum$weights<cutoff: {np.cumsum(weights_sorted)[limit]:.2f}", transform=ax.transAxes, va='top',ha='left')

                ax.set_xlabel('number of structures')
                ax.set_ylabel('weight')
                ax.set_title(f'weights-{points_fes_names[idx]}')
#                ax.set_xlim([None,xx[-1]])
                plt.legend()
                fname_plot=f'weights-{points_fes_names[idx]}.png' 
                plt.tight_layout()
            
                plt.savefig(fname_plot)
                logger.info(f'file saved: {fname_plot}')
            
            logger.info(f'first 10 indices: {indices_struct[:10]}')
            sample_freq=100
            timesteps_struct=indices_struct*sample_freq
            logger.info(f'---------------> assuming sample_freq = {sample_freq:d}! <---------------')
            logger.info(f'first 10 TIMESTEPS: {timesteps_struct[:10]}')
            logger.info(f'first 10 WEIGHTS: {weights_struct[:10]}')


            fname_out=f'struct-indices-{points_fes_names[idx]}.dat'
            header=f"# TIMESTEPS of {nstruct} structures, summing up weight to {prob:.4g}, highest weight first"
            line_timesteps=",".join([str(ii) for ii in timesteps_struct])
            line_weights=",".join([f"{ii:.5g}" for ii in weights_struct])

            with open(jj/fname_out, "w") as f:
                f.write(header + "\n")
                f.write(line_weights + "\n")
                f.write(line_timesteps + "\n")
            logger.info(f'indices saved to {jj/fname_out}')

            cfg_save = cfg_find.get('save_structures',None)

            if cfg_save is None:
                return

            type_map = cfg_save.get('type_map',None)
            file_index = jj/"traj.index.json"
            file_traj_lammps = [jj/kk for kk in cfg_save.file_lammps]
            fname_out_structures = jj/f'structures-{points_fes_names[idx]}.xyz'

            if not file_index.exists():
                idx = build_lammpstrj_index(file_traj_lammps, jj/"traj.index.json")


            atoms = []
#            for tt in tqdm(timesteps_struct,total=len(timesteps_struct),ncols=80):
#                atoms.append(read_atoms_at_timestep_sorted(file_index,tt,type_map=type_map))

            if cfg_save.do_save:

                file_traj = jj/Path(cfg_save.file_dump_h5)
                assert file_traj.exists()
                assert type_map is not None
                logger.info(f'Using the following type map (from config): {type_map}')

                fname_out_structures = jj/f'structures-{points_fes_names[idx]}.xyz'
        
                logger.info(f'loading atoms from {file_traj}')
                atoms = h5md_to_ase_atoms(file_traj, timesteps_struct, type_map)



                write(fname_out_structures, atoms)

                logger.info(f'file written {fname_out_structures}')

            



