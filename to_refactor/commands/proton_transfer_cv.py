from rich import print
import sys
import numpy as np
from logger_setup import logger
from pathlib import Path
from parser import get_temperature_from_log, parse_state_wfile, parse_bias_factor, parse_colvar_header
from parser import parse_cv_wall
from omegaconf import ListConfig
from utils import extract_field_map, find_job_dirs,find_files, read_colvar_in_time_interval, safe_stack_colvar
from utils import ensure_list, write_pt_cv
from ectoolkits.analysis.proton_transfer_cv import ProtonTransferCV
from MDAnalysis import Universe
from MDAnalysis.coordinates.chain import ChainReader
import concurrent.futures


def process_jobdir(args):
    jj, idx, pp, cfg_target, cfg = args


    logger.info('#'*60+f' {jj.name}')
    # load colvar
    data_colvar = np.loadtxt(jj/'COLVAR')
    timestep = data_colvar[1, 0] - data_colvar[0, 0]

    jj_idx = int(str(jj.name).split('-')[-1]) if 'job-00' in jj.name else idx

    # find traj file
    file_traj = jj / pp.get('file_traj', 'dump.xyz')
    if not file_traj.exists():
        logger.info(f'file {file_traj} does not exist. exiting')
        return None

    # load box for pbc
    with open(file_traj, 'r') as f:
        _ = next(f)
        box = next(f).strip()
    box = [float(ii) for ii in box.split()] + [90.0, 90.0, 90.0]

    logger.info(f'box read from file {file_traj.name}: {box}')

    # init MDA
    u = Universe(file_traj, format="XYZ")
    logger.info(f'trajectory loaded: {len(u.trajectory)} steps')
    u.atoms.dimensions = box

    atomgroup = u.atoms

    # select indirect, direct
    case_selected = pp.get('case', 'default')
    if 'default' in case_selected:
        logger.info(f'you must select a set of target atoms to study. see cfg.proton_transfer.case. exiting')
        return -1
    
    # get selected cvs - needed to handle multiple targets
    cvs=cfg.get('cvs',{}).get('select',None)
    if cvs is None:
        logger.info(f'choose something like cvs.select=[direct-d-A,direct-delta-A]')
        sys.exit()

    choices=['direct-d', 'direct-delta','indirect-d','indirect-delta']

    for cv in cvs:
        test=[ii in cv for ii in choices]
        if not any(test):
            logger.info(f'cvs error: the format must be direct-d-<something>. Now {cv}.')    
            sys.exit()
   
    idx_delta = next((i for i, s in enumerate(cvs) if "delta" in s), None) 


    # deal with multiple presets for the surface sites
    new_vars = pp.get('surf_sites', {}).get(case_selected, {})
    for key, val in new_vars.items():
        pp.update({key: val})

    indices_type1 = ensure_list(pp.get('indices_type1', None))
    indices_type2 = ensure_list(pp.get('indices_type2', None))
    indices_water = ensure_list(pp.get('indices_water', None))
    
    logger.info(f'indices from config (1 BASED):')
    logger.info(f'---> type 1 {indices_type1}')
    logger.info(f'---> type 2 {indices_type2}')

    if pp.kind is None or pp.kind == 'indirect':
        logger.info(f'---> water {indices_water}')

    if indices_type1 is None or indices_type2 is None:
        logger.info('error while parsing type1 or type2 indices. exiting')
        return None

    # pass to 0-based indexing
    indices_type1 = [ii - 1 for ii in indices_type1]
    indices_type2 = [ii - 1 for ii in indices_type2]
    indices_water = [ii - 1 for ii in indices_water]


    # direct PT
    if pp.kind is None or pp.kind == 'direct':
        logger.info('#'*40 + ' computing direct PT')
        options_dir = {
            'atomgroup': atomgroup,
            'idxs_type1_o': indices_type1,
            'idxs_type2_o': indices_type2,
            ' extra_detail': True
        }

        direct = ProtonTransferCV(**options_dir)
        direct.run()

        mask = ~np.isnan(direct.results[:, 1])
        logger.info(f'proton transfers found: {np.count_nonzero(mask)} ({np.count_nonzero(mask) / len(u.trajectory) * 100:.3g}% of {len(u.trajectory)} frames)')

        direct.results[:, 0] *= timestep
        write_pt_cv(direct.results[:, 0:2], jj / f'CV-{cvs[idx_delta]}', **{'name_cv_x': f'{cvs[idx_delta]}'})
        write_pt_cv(direct.results[:, 0:4:2], jj / f'CV-{cvs[idx_delta-1]}', **{'name_cv_x': f'{cvs[idx_delta-1]}'})

    # indirect PT
    if pp.kind is None or pp.kind == 'indirect':
        logger.info('#'*20 + 'computing indirect PT')
        options_ind = {
            'atomgroup': atomgroup,
            'idxs_type1_o': indices_type1,
            'idxs_type2_o': indices_type2,
            'num_bridge': 1,
            'idxs_water_o': indices_water,
            ' extra_detail': True
        }

        indirect = ProtonTransferCV(**options_ind)
        indirect.run()

        mask = ~np.isnan(indirect.results[:, 1])
        logger.info(f'proton transfers found: {np.count_nonzero(mask)} ({np.count_nonzero(mask) / len(u.trajectory) * 100:.3g}% of {len(u.trajectory)} frames)')

        indirect.results[:, 0] *= timestep
        write_pt_cv(indirect.results[:, 0:2], jj / f'CV-{cvs[idx_delta]}', **{'name_cv_x': f'{cvs[idx_delta]}'})
        write_pt_cv(indirect.results[:, 0:4:2], jj / f'CV-{cvs[idx_delta-1]}', **{'name_cv_x': f'{cvs[idx_delta-1]}'})

    return jj

def run(cfg):
    print(f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] {cfg.param}")
    jobdirs = find_job_dirs(Path(cfg.get('dir_md', '.')), cfg.dir_job)
    logger.info('#'*100 + f' proton transfer')

    pp = cfg.get('proton_transfer', {})

    args = [(jj, idx, pp, cfg._target_, cfg) for idx, jj in enumerate(jobdirs)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_jobdir, args))

    logger.info('done')


#
