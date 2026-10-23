from rich import print
import sys
import numpy as np
from logger_setup import logger
from pathlib import Path
from parser import get_temperature_from_log, parse_state_wfile, parse_bias_factor, parse_colvar_header
from parser import parse_cv_wall
from ectoolkits_reweight import calc_FES, calc_property_surface, compute_bw_silverman, read_fes, write_fes, read_property
from omegaconf import ListConfig
from utils import extract_field_map, find_job_dirs,find_files, read_colvar_in_time_interval, safe_stack_colvar
from concurrent.futures import ProcessPoolExecutor, as_completed
from omegaconf import OmegaConf

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
    state_wfile = parse_state_wfile(jj/ cfg_dict["file_plumed_input"])
    bias_factor = parse_bias_factor(jj/state_wfile)
    kbT = temp*0.0083144621  # kJ/mol
    nbins = cfg_dict.get("nbins", [101, 101])

    logger.info(f'temp is {temp:.0f} K')
    logger.info(f'kbT is {kbT:.3g}')
    logger.info(f'nbins: {nbins}')

    cvs_dict = {}

    # parse cvs from COLVAR
    colvar_fields = extract_field_map(jj/'COLVAR')
    colvars_plumed = [str(ii) for ii in colvar_fields.keys()]

    cvs = list(cfg_dict["cvs"]["select"])
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

    # store the data of the selected CVs
    for idx_cv, ii in enumerate(cvs):
        cvs_dict[ii]["data"] = data[:, colvar_fields[ii]]

    # store the bias
    bias = data[:, colvar_fields["opes.bias"]]

    # compute bandwidths
    sigmas = cfg_dict.get("sigma", [None]*len(cvs))
    weights = np.exp(bias/kbT)
    
    for idx_cv, ii in enumerate(cvs):
        sig = cfg_dict.get('cvs',{}).get(ii,{}).get('sigma',None)

        msg = '(specified from config)'
        if sig is None:
            msg='(silverman)'
            bw = compute_bw_silverman(cvs_dict[ii]["data"], dim=2,
                                  weights=weights, bias_factor=bias_factor)
            if "delta" in ii:
                bw /= 2
        else:
            bw = sig
        cvs_dict[ii]["bandwidth"] = bw
        logger.info(f'bandwidth for \'{ii}\' is {bw:.5f} {msg}')

    if cfg_dict.get("swap_coordinates", False):
        cvs = cvs.reverse()

    fprop = pp.get("file_prop", None)
   
    fname_out = str(jj)+f"/pot{jj_idx:03d}-{cvs[0]}-{cvs[1]}-rew-{cfg_dict['property']}.dat"

    special_points=cfg_dict.get('special_points',[])

    if cfg_dict["property"] == "fes":
        calc_FES(cv_x=cvs_dict[cvs[0]]["data"],
                 cv_y=cvs_dict[cvs[1]]["data"],
                 bias=bias,
                 bandwidth_x=cvs_dict[cvs[0]]["bandwidth"],
                 bandwidth_y=cvs_dict[cvs[1]]["bandwidth"],
                 nbins_x=nbins[0],
                 nbins_y=nbins[1],
                 kbT=kbT,
                 save_file=True,
                 name_file=fname_out,
                 min_cv_x=cvs_dict[cvs[0]]["limits"][0],
                 max_cv_x=cvs_dict[cvs[0]]["limits"][1],
                 min_cv_y=cvs_dict[cvs[1]]["limits"][0],
                 max_cv_y=cvs_dict[cvs[1]]["limits"][1],
                 name_cv_x=cvs[0],
                 name_cv_y=cvs[1],
                 dim=2,
                 property_name="fes",
                 special_points=special_points,
                )
        files_written.append(fname_out)

    elif fprop is not None:
        # reweight some other quantity that is not FES
        fprop = find_files(jj, fprop)
        if len(fprop) > 1:
            logger.info(f'too many data files for {cfg_dict["property"]}. Exiting.')
            sys.exit()
        fprop = fprop[0]
        propdata = np.loadtxt(fprop)
        prop_fields = extract_field_map(fprop)
        propdata = propdata[:, prop_fields[pp["prop_field"]]]

        if len(propdata) < len(bias):
            logger.error(f'property {cfg_dict["property"]} has less data than bias {len(propdata)}/{len(bias)}')
        if len(propdata) > len(bias):
            logger.info(f'property {cfg_dict["property"]} has MORE data than bias {len(propdata)}/{len(bias)}')
            logger.info(f'trimming {cfg_dict["property"]}')
            propdata=propdata[:len(bias)]

        calc_property_surface(cv_x=cvs_dict[cvs[0]]["data"],
                              cv_y=cvs_dict[cvs[1]]["data"],
                              bias=bias,
                              bandwidth_x=cvs_dict[cvs[0]]["bandwidth"],
                              bandwidth_y=cvs_dict[cvs[1]]["bandwidth"],
                              nbins_x=nbins[0],
                              nbins_y=nbins[1],
                              kbT=kbT,
                              prop=propdata,
                              save_file=True,
                              name_file=fname_out,
                              min_cv_x=cvs_dict[cvs[0]]["limits"][0],
                              max_cv_x=cvs_dict[cvs[0]]["limits"][1],
                              min_cv_y=cvs_dict[cvs[1]]["limits"][0],
                              max_cv_y=cvs_dict[cvs[1]]["limits"][1],
                              name_cv_x=cvs[0],
                              name_cv_y=cvs[1],
                              dim=2,
                              property_name=cfg_dict["property"])
        files_written.append(fname_out)

    return files_written, cvs_dict


def run(cfg):
    print(f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] {cfg.param}")

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
    for ff in files_written_all:
        X, Y, Z, _ = reader(Path(ff), dim=2)
        Z_all.append(Z)

    factor = 1
    if str(prop).lower() == 'fes':
        factor = 0.0103636  # to J/mol
    Z_all = np.array(Z_all) / factor
    Z_ave = np.mean(Z_all, axis=0)
    Z_std = np.std(Z_all, axis=0)

    # pick the first non-empty cvs_dict (or merge them if needed)
    cvs_dict = next((cd for cd in cvs_dict_all if cd), {})

    options = {
        'name_cv_x': cvs[0],
        'name_cv_y': cvs[1],
        'min_cv_x': cvs_dict[cvs[0]]['limits'][0],
        'max_cv_x': cvs_dict[cvs[0]]['limits'][1],
        'min_cv_y': cvs_dict[cvs[1]]['limits'][0],
        'max_cv_y': cvs_dict[cvs[1]]['limits'][1],
        'num_grid_points_x': cfg.get('nbins', [101, 101])[0] + 1,
        'num_grid_points_y': cfg.get('nbins', [101, 101])[1] + 1,
        'property_name': f'{prop} stddev({prop})',
        'column_extra': Z_std,
    }

    write_fes(X, Y, Z_ave,
              f"{prop}-{cvs[0]}-{cvs[1]}-rew-ave.dat",
              **options)
