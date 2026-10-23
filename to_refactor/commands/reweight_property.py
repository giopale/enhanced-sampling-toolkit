from rich import print
import sys
import numpy as np
from logger_setup import logger
from pathlib import Path
from parser import get_temperature_from_log, get_timestep_from_log, parse_state_wfile, parse_bias_factor, parse_colvar_header
from parser import parse_cv_wall
from ectoolkits_reweight import calc_FES, calc_property_surface, compute_bw_silverman, read_fes, write_fes, read_property
from omegaconf import ListConfig
from utils import extract_field_map, find_job_dirs,find_files, read_colvar_in_time_interval, safe_stack_colvar
from concurrent.futures import ProcessPoolExecutor, as_completed
from omegaconf import OmegaConf


from utils_reweighting import compute_frame_weights, make_grid_2d, fes_kde_2d, fes_kde_2d_logsum, kde_avg_2d_logsum_signed

def process_jobdir(jj, idx, cfg_dict, pp):
    """Process a single jobdir. Returns list of written files."""
    files_written = []

    logger.info('#'*60+f' {jj.name}')

    # get job index for output
    if 'job-00' in jj.name:
        jj_idx = int(str(jj.name).split('-')[-1])
    else:
        jj_idx = idx

    # get simulation parameters
    temp = get_temperature_from_log(jj/ cfg_dict["file_lammps_log"])
    timestep  = get_timestep_from_log(jj/ cfg_dict["file_lammps_log"])
    state_wfile = parse_state_wfile(jj/ cfg_dict["file_plumed_input"])
    bias_factor = parse_bias_factor(jj/state_wfile)
    kbT = temp*0.0083144621  # kJ/mol
    ngrid = cfg_dict.get("ngrid", [100, 100])


    logger.info(f'timestep is {timestep:.6f} ps')
    logger.info(f'temp is {temp:.0f} K')
    logger.info(f'kbT is {kbT:.3g}')
    logger.info(f'ngrid: {ngrid}')

    # property to reweight
    prop = cfg_dict.get('property','fes')
    fprop = pp.get("file_prop", None)

    cvs_dict = {}

    # parse cvs from COLVAR
    colvar_fields = extract_field_map(jj/'COLVAR')
    colvars_plumed = [str(ii) for ii in colvar_fields.keys()]


    cvs = list(cfg_dict.get('cvs',{}).get('select',[]))
    logger.info(f"collective variables: {', '.join(cvs)}")

    # get time interval (nanoseconds)
    timeint = cfg_dict.get("time_interval", [0.0, np.inf])
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

    ###############################################
    ###############################################
    ###############################################
    ###############################################
    if fprop is not None:
        fields_prop = extract_field_map(jj/fprop)
        ts=None
        if 'step' in fields_prop.keys():
            ts=timestep
        data_new_cv_tmp = read_colvar_in_time_interval(jj/fprop, timeint[0], timeint[1], timestep=ts)
        data_new_cv = np.zeros((len(data_new_cv_tmp),2))
        data_new_cv[:,0] = data_new_cv_tmp[:,0]
        data_new_cv[:,1] = data_new_cv_tmp[:,fields_prop[prop]]
        data = safe_stack_colvar(data, data_new_cv)
        colvar_fields.update({prop: np.shape(data)[-1]-1})
        
    print('STACKING DONE')


    # filter data outside some given limits
    mask_new = np.array([True]*len(data))
    cvs_to_filter = cfg_dict["cvs"].get("filter", None)
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
            limits.append(parse_cv_wall(file_path=jj/cfg_dict["file_plumed_input"], cv=ii, wall=wall))
        logger.debug(f'limits: walls done')
        # merge with limits from config file
        limits_cfg = cfg_dict.get("cvs", {}).get(ii, {}).get("limits", [None, None])
        for kk in [0, 1]:
            if limits_cfg[kk] is not None:
                limits[kk] = limits_cfg[kk]
        cvs_dict[ii] = {"limits": limits}
        if None in limits:
            logger.info(f'you must set limits for cv \'{ii}\'. Currently {limits}. Exiting')
            sys.exit()
    logger.debug('DONE parsing cvs limits')

    for key, val in cvs_dict.items():
        logger.info(f'limits for CV {key} set to {val["limits"]}')

    # store the data of the selected CVs
    for idx_cv, ii in enumerate(cvs):
        cvs_dict[ii]["data"] = data[:, colvar_fields[ii]]

    # store the bias
    bias = data[:, colvar_fields["opes.bias"]]

    # compute bandwidths
    sigmas = cfg_dict.get("sigma", [None]*len(cvs))
    # get weights
    weights = compute_frame_weights(bias, kbT, normalize=False)
    logw = np.log(weights)

    # get the total to make an average across traj later
    m = np.max(logw)
    log_weight_total = m + np.log(np.sum(np.exp(logw - m)))
    # normalize to also get the single-trajectory fes correct
    weights /= np.sum(weights)
    logw = np.log(weights)

    ##########################3 
    
    matrix_kernel = np.zeros((2,2))
    for idx_cv, ii in enumerate(cvs):
        sig = cfg_dict.get('cvs',{}).get(ii,{}).get('sigma',None)

        msg = '(specified from config)'
        if sig is None:
            msg='(silverman)'
            bw = compute_bw_silverman(cvs_dict[ii]["data"], dim=2,
                                  weights=weights, bias_factor=bias_factor)
        else:
            bw = sig
        logger.info(f'bandwidth for \'{ii}\' is {bw:.5f} {msg}')
        matrix_kernel[idx_cv,idx_cv]  = bw**2



    # prepare grid
    grid_X, grid_Y = make_grid_2d(
                            xmin = cvs_dict[cvs[0]]["limits"][0],
                            xmax = cvs_dict[cvs[0]]["limits"][1],
                            n_grid_x = ngrid[0],
                            ymin = cvs_dict[cvs[1]]["limits"][0],
                            ymax = cvs_dict[cvs[1]]["limits"][1],
                            n_grid_y = ngrid[1]
                            )
    
    # set up reweighting run
    options_fes =  { 'cv_x': cvs_dict[cvs[0]]["data"],
                    'cv_y': cvs_dict[cvs[1]]["data"],
                    'grid_X': grid_X,
                    'grid_Y': grid_Y,
                    'H': matrix_kernel,
                    'logw': logw,
                    'frame_chunk': 100,
                    }

    if prop == 'fes':
        shift_to_zero = cfg_dict.get('fes_shift_to_zero',False)
        logger.info(f'shift fes to zero: {shift_to_zero}')
        options_fes.update({
            'shift_min_to_zero': shift_to_zero,
            'kbT': kbT,})

        F = fes_kde_2d_logsum( **options_fes )
    else:
        options_fes.update({
            'q': data[:,colvar_fields[prop]]
            })
        F = kde_avg_2d_logsum_signed(**options_fes)


    options = {
        'name_cv_x': cvs[0],
        'name_cv_y': cvs[1],
        'min_cv_x': cvs_dict[cvs[0]]['limits'][0],
        'max_cv_x': cvs_dict[cvs[0]]['limits'][1],
        'min_cv_y': cvs_dict[cvs[1]]['limits'][0],
        'max_cv_y': cvs_dict[cvs[1]]['limits'][1],
        'num_grid_points_x': ngrid[0],
        'num_grid_points_y': ngrid[1],
        'property_name': f'{cfg_dict["property"]}',
        'log_weight_total':log_weight_total,
    }

    fname_out = str(jj)+f"/pot{jj_idx:03d}-{cvs[0]}-{cvs[1]}-rew-{cfg_dict['property']}.dat"
    write_fes(grid_X.flatten(), grid_Y.flatten(), F.flatten(),
              fname_out,
              **options)

    files_written.append(fname_out)

    return files_written, cvs_dict


def run(cfg):
    print(f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] {cfg.param}")

    cvs_selected=cfg.cvs.get('select',None)
    if cvs_selected is None:
        logger.error('you must set cvs.select')
        sys.exit(f'bye')

    jobdirs = find_job_dirs(Path(cfg.get('dir_md', '.')), cfg.dir_job)

    prop = cfg.get('property', None)
    if prop is None:
        print('No property to analyze in config file. Exiting.')
        sys.exit()
    logger.info('#'*100+f' reweighting {prop}')

    pp = cfg.surface_data.get(prop, {})
    cvs = cfg.cvs['select']

    files_written_all = []
    cvs_dict_all = []

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_jobdir, jj, idx, cfg, pp): jj
            for idx, jj in enumerate(jobdirs)
        }
        for fut in as_completed(futures):
            try:
                files_written, cvs_dict = fut.result()
                files_written_all.extend(files_written)
                cvs_dict_all.append(cvs_dict)
            except Exception as e:
                jj = futures[fut]
                logger.error(f"Jobdir {jj} failed: {e}")

    logger.info(f'computing average {prop} over jobs')

    if str(prop).lower() == 'fes':
        reader = read_fes
    else:
        reader = read_property

    if len(files_written_all) < 1:
        logger.info(f'not enough files to make an average ({len(files_written_all)}). Exiting')
        sys.exit()

    Z_all = []
    logw_all = []
    for ff in files_written_all:
        X, Y, Z, _, logw_traj = reader(Path(ff), dim=2)
        Z_all.append(Z)
        logw_all.append(logw_traj)

    # weight each trajectory according to the applied bias
    weights_trajectories = np.exp(logw_all) / np.sum(np.exp(logw_all))
    for file_traj, ww_traj in zip(files_written_all, weights_trajectories):
        txt=f'trajectory {Path(file_traj).parent} has weight {ww_traj:.5g}'
        logger.info(txt)

    factor = 1
    if str(prop).lower() == 'fes':
        factor = 0.0103636  # to J/mol
    Z_all = np.array(Z_all) / factor
    Z_ave = np.average(Z_all, axis=0, weights=weights_trajectories)
    Z_std = np.std(Z_all, axis=0)

    # pick the first non-empty cvs_dict (or merge them if needed)
    cvs_dict = next((cd for cd in cvs_dict_all if cd), {})

    ngrid=cfg.get('ngrid',[100,100])

    options = {
        'name_cv_x': cvs[0],
        'name_cv_y': cvs[1],
        'min_cv_x': cvs_dict[cvs[0]]['limits'][0],
        'max_cv_x': cvs_dict[cvs[0]]['limits'][1],
        'min_cv_y': cvs_dict[cvs[1]]['limits'][0],
        'max_cv_y': cvs_dict[cvs[1]]['limits'][1],
        'num_grid_points_x': ngrid[0],
        'num_grid_points_y': ngrid[1],
        'property_name': f'{prop} stddev({prop})',
        'column_extra': Z_std,
    }

    write_fes(X, Y, Z_ave,
              f"{cvs[0]}-{cvs[1]}-rew-{prop}-AVG.dat",
              **options)
