from rich import print
from pathlib import Path
from logger_setup import logger
from omegaconf import ListConfig
from stringmethod_custom import String2D
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from string import Template
import pandas as pd
import sys
from matplotlib.colors import LinearSegmentedColormap, to_rgb


def apply_mpl_style(style_dir, style_file):
    style_file=Path(style_dir)/style_file
    if style_file.exists():
        plt.style.use(style_file)
        logger.info(f'using mpl style {style_file.name}')
    else:
        logger.info(f'mpl style file not found {style_file}')
    
def ensure_list(x):
    if isinstance(x, str) or not isinstance(x, Iterable):
        return [x]
    return list(x)

def shift_to_backup(cfg):
    if cfg.get('backup',{}).get('read',False):
        pass 

def join_if_list(item):
    if isinstance(item, ListConfig):
        return ''.join(item)
    else:
        return item

def config_to_list(item):
    if isinstance(item, ListConfig):
        return list(item)
    else:
        return [item]


def find_job_dirs(dirname, jobname):
    #logger.info(f"looking for directories: {dirname}")
    dd=Path(dirname)
    if isinstance(jobname, ListConfig):
        jobdirs=[dd/ii for ii in jobname]
    else:
        jobdirs = [p for p in dd.glob(jobname) if p.is_dir()]
    jobdirs=sorted(jobdirs)
    mm=', '.join([ii.name for ii in jobdirs])

    if len(jobdirs) < 1:
        logger.info('No job dirs found. Exiting')
        sys.exit()
    else:
        logger.info(f'looking for directories: {dirname} - {len(jobdirs)} found: {mm}')

    return jobdirs


def find_files(dirname, fname, allow_failure=False):
    #logger.info(f"looking for files: {dirname}")
    dd=Path(dirname)
    if isinstance(fname, ListConfig):
        files=[dd/ii for ii in fname if (dd/ii).is_file ]
    else:
        files = [p for p in dd.glob(f'**/{fname}') if p.is_file()]
    files=sorted(files)
    mm=', '.join([ii.name for ii in files])
    if len(files) < 1:
        logger.info(f'looking for files: {fname} - none found.')
        if not allow_failure:
            logger.info(f'Exiting')
            sys.exit()
    else:
        logger.info(f'looking for files: {fname} - {len(files)} found: {mm}')

    return files


def extract_field_map(filename):
    with open(filename, 'r') as file:
        first_line = file.readline().strip()

    if first_line.startswith("#! FIELDS"):
        fields = first_line.split()[2:]  # Skip "#! FIELDS"
        field_map = {field: idx  for idx, field in enumerate(fields)}
        logger.info(f'file {filename} has FIELDS: {" ".join(field_map.keys())}')
        return field_map
    elif first_line.startswith("#"):
        fields = first_line.split()[1:]  # Skip "#! FIELDS"
        field_map = {field: idx  for idx, field in enumerate(fields)}
        logger.info(f'file {filename} has FIELDS: {" ".join(field_map.keys())}')
        return field_map
    else:
        raise ValueError("Invalid format: first line must start with '#! FIELDS' or '#'")


def read_colvar_in_time_interval(filename, tmin, tmax, log=True, timestep=None):
    """
    Reads a COLVAR file into a numpy array and selects rows where
    time (first column) is between tmin and tmax (inclusive).

    it is your responsibility to match the units!

    Parameters:
        filename (str): Path to the COLVAR file.
        tmin (float): Minimum time.
        tmax (float): Maximum time.

    Returns:
        np.ndarray: Selected rows as a NumPy array.
    """
    logger.debug(f'reading {filename}')
    modified=False
    if tmin is not None or tmax is not None:
        modified=True
    if tmin is None:
        tmin=0.0
    if tmax is None:
        tmax=np.inf
    # Load data while skipping lines starting with '#'
    #data = np.loadtxt(filename, comments='#')
    df = pd.read_csv(filename, sep='\s+', comment='#', header=None)
    df = df.drop_duplicates(subset=0, keep='last') # this is to allow for multiple restarts
    data = df.to_numpy()

    if timestep is not None:
        logger.info(f'converting timesteps to time ({filename}, dt = {timestep} ps)')
        data[:,0] *= timestep
    
    # Select rows where the first column (time) is in the interval
    mask = (data[:, 0] >= tmin) & (data[:, 0] <= tmax)
    selected_data = data[mask]
    logger.info(f'reading {filename.name} file: the simulation time is {data[-1,0]/1000:.2f} ns')

    if data[0,0] > 0.0:
        if log:
            logger.warning(f'COLVAR data starts from {data[0,0]/1000:.3f} ns: check that this is what you want!')
    if modified:
        if log:
            logger.info(f'selecting time from {tmin/1000:.2f} ns to {selected_data[-1,0]/1000:.2f} ns')
    logger.debug(f'DONE reading {filename}')
    return selected_data
    

def safe_stack_colvar(arr1, arr2):
    if not np.allclose(arr1[:, 0], arr2[:, 0]):
        raise ValueError("Time columns do not match!")
    
    return  np.hstack([arr1, arr2[:, 1:]])

def _find_min_idx_in_slice(Z: np.ndarray, rs: slice, cs: slice) -> tuple[int, int]:
    a = Z[rs, cs]
    i0, j0 = np.unravel_index(np.argmin(a, axis=None), a.shape)
    i = i0 + (rs.start or 0)
    j = j0 + (cs.start or 0)
    return (i, j)

def _split_slices(n: int, k: int) -> list[slice]:
    base = n // k
    cuts = [t * base for t in range(k)] + [n]
    return [slice(cuts[t], cuts[t + 1]) for t in range(k)]

def _pick_minima_points(Z, X, Y, split: str, nminima: int, logger):
    split = str(split).lower()
    nminima = int(nminima)

    if nminima not in (2, 3):
        raise ValueError(f"nminima must be 2 or 3, got {nminima}")
    if split not in ("horizontal", "vertical"):
        raise ValueError(f"split must be 'horizontal' or 'vertical', got {split}")

    nrows, ncols = Z.shape
    if X.shape != Z.shape or Y.shape != Z.shape:
        raise ValueError(f"X,Y must have same shape as Z. Got Z{Z.shape}, X{X.shape}, Y{Y.shape}")

    if split == "horizontal":
        rs_list = _split_slices(nrows, nminima)
        cs_list = [slice(0, ncols)] * nminima
        labels = ["lower", "upper"] if nminima == 2 else ["lower", "middle", "upper"]
    else:  # vertical
        rs_list = [slice(0, nrows)] * nminima
        cs_list = _split_slices(ncols, nminima)
        labels = ["left", "right"] if nminima == 2 else ["left", "middle", "right"]

    inds = []
    pts = []
    for lab, rs, cs in zip(labels, rs_list, cs_list):
        ind = _find_min_idx_in_slice(Z, rs, cs)
        pt = [float(X[ind]), float(Y[ind])]
        inds.append(ind)
        pts.append(pt)
        logger.info(f"minimum in the {lab} part of fes ({split}) {ind} {pt}")

#    ind_fin = inds[0]
#    ind_ini = inds[-1]
#    ind_mid = inds[1] if nminima == 3 else None
#
#    pt_fin = pts[0]
#    pt_ini = pts[-1]
#    pt_mid = pts[1] if nminima == 3 else None

    # ugly fix to reverse the mep order
    if split == "horizontal":
        ind_ini = inds[0]
        ind_fin = inds[-1]
        pt_ini = pts[0]
        pt_fin = pts[-1]
    else:
        ind_fin = inds[0]
        ind_ini = inds[-1]
        pt_fin = pts[0]
        pt_ini = pts[-1]
    
    ind_mid = inds[1] if nminima == 3 else None
    pt_mid = pts[1] if nminima == 3 else None

    return (pt_ini, pt_mid, pt_fin, ind_ini, ind_mid, ind_fin)

def compute_mep(X, Y, Z, **options):
    maxsteps = options.get("maxsteps", 200)
    dt = options.get("dt", 0.1)
    traj_every = options.get("traj_every", 50)
    npts = options.get("npts", 15)
    flexible = options.get("flexible", False)

    xbins = X[:, 0]  
    ybins = Y[0, :]

    S = String2D(xbins, ybins, Z, indexing="ij")

    split = options.get("split", "horizontal")
    nminima = options.get("nminima", 2)

    pt_ini, pt_mid_auto, pt_fin, ind_ini, ind_mid, ind_fin = _pick_minima_points(
        Z, X, Y, split=split, nminima=nminima, logger=logger
    )

    mid_point = options.get("midpoint", None)
    if mid_point is None:
        if pt_mid_auto is not None:
            mid_point = pt_mid_auto
        else:
            mid_point = [0.5 * (pt_ini[0] + pt_fin[0]), 0.5 * (pt_ini[1] + pt_fin[1])]

    logger.info(f"initial point {pt_ini}, the free energy is {Z[ind_ini]:.5f}")
    if ind_mid is not None:
        logger.info(f"mid point     {mid_point}, the free energy is {Z[ind_mid]:.5f}")
    logger.info(f"final point   {pt_fin}, the free energy is {Z[ind_fin]:.5f}")

    # Keep your original ordering rule on x coordinate
    begin, end = (pt_fin, pt_ini) if (pt_ini[0] > pt_fin[0]) else (pt_ini, pt_fin)
    S.compute_mep(
        begin=begin,
        mid=[mid_point],
        end=end,
        maxsteps=maxsteps,
        dt=dt,
        traj_every=traj_every,
        npts=npts,
        flexible=flexible,
    )

    # Build output, preserving your flipping logic
    if pt_ini[0] > pt_fin[0]:
        mep_arr = np.array(
            [
                np.flip(S.mep.T[0]),
                np.flip(S.mep.T[1]),
                np.flip(S.get_mep_energy_profile()[1]),
            ]
        )
    else:
        mep_arr = np.array([S.mep.T[0], S.mep.T[1], S.get_mep_energy_profile()[1]])

    if options.get("print_convergence", False):
        logger.info("saving file: convergence-mep.dat")
        np.savetxt("convergence-mep.dat", np.array(S.convergence_history))

    return mep_arr.T

#def compute_mep(X, Y, Z, **options):
#
#    maxsteps = options.get('maxsteps',200)
#    dt = options.get('dt', 0.1)
#    traj_every = options.get('traj_every',50)
#    npts = options.get('npts',15)
#    flexible = options.get('flexible',False)
#
#    xbins = X[0]
#    ybins = Y[:, 0]
#
#    S = String2D(xbins, ybins, Z.T, indexing='ij')
#    #S.prepare_interpolation()
##    S.plot_V(clip_max=1.8, levels=20)
#    #plt.show()
#    
#    pt_mid=None
#    split=options.get('split','horizontal')
#    nminima=options.get('nminima','2')
#    if 'horizon' in split:
#        kk=np.shape(Z)[0]//2
#        a = Z[:kk, :]
#        ind_lo = np.unravel_index(np.argmin(a, axis=None), a.shape)
#        logger.info(f"minimum in the lower part of fes {ind_lo}")
#        a = Z[kk:, :]
#        ind_up = np.unravel_index(np.argmin(a, axis=None), a.shape)
#        ind_up = (ind_up[0]+50, ind_up[1])
#        logger.info(f"minimum in the upper part of fes {ind_up}")
#        pt_ini = [X[ind_up], Y[ind_up]]
#        pt_fin = [X[ind_lo], Y[ind_lo]]
#    if 'triple' in split:
#        nrows = np.shape(Z)[0]
#        one_third = nrows // 3
#    
#        # Upper third
#        a = Z[2*one_third:, :]
#        ind_up = np.unravel_index(np.argmin(a, axis=None), a.shape)
#        ind_up = (ind_up[0] + 2*one_third, ind_up[1])
#    
#        # Middle third
#        a = Z[one_third:2*one_third, :]
#        ind_mid = np.unravel_index(np.argmin(a, axis=None), a.shape)
#        ind_mid = (ind_mid[0] + one_third, ind_mid[1])
#    
#        # Lower third
#        a = Z[:one_third, :]
#        ind_lo = np.unravel_index(np.argmin(a, axis=None), a.shape)
#    
#        pt_ini = [X[ind_up], Y[ind_up]]
#        pt_mid = [X[ind_mid], Y[ind_mid]]
#        pt_fin = [X[ind_lo], Y[ind_lo]]
#
#        logger.info(f"minimum in the upper third of fes {ind_up} {pt_ini}")
#        logger.info(f"minimum in the middle third of fes {ind_mid} {pt_mid}")
#        logger.info(f"minimum in the lower third of fes {ind_lo} {pt_fin}")
#
#    elif 'verti' in split:
#        kk = np.shape(Z)[1] // 2  # Split along columns now
#        
#        # Left half
#        a = Z[:, :kk]
#        ind_lo = np.unravel_index(np.argmin(a, axis=None), a.shape)
#        logger.info(f"minimum in the left part of fes {ind_lo}")
#        
#        # Right half
#        a = Z[:, kk:]
#        ind_up = np.unravel_index(np.argmin(a, axis=None), a.shape)
#        ind_up = (ind_up[0], ind_up[1] + kk)  # adjust column index after slicing
#        logger.info(f"minimum in the right part of fes {ind_up}")
#        
#        pt_ini = [X[ind_lo], Y[ind_lo]]
#        pt_fin = [X[ind_up], Y[ind_up]]
#    else:
#        logger.info(f'provide a suitable value for mep.split: horizontal, vertical. now {split}')
#
#
#    # find two minima
#    mid_point=pt_mid
#    mid_point=options.get('midpoint',None)
#    if mid_point is None:
#        mid_point = [0.5*(pt_ini[0]+pt_fin[0]), 0.5*(pt_ini[1]+pt_fin[1])]
#    logger.info(f"initial point {pt_ini}, the free energy is {Z[ind_up]}")
#    logger.info(f"final point {pt_fin} the free energy is {Z[ind_lo]}")
#
#    if pt_ini[0] > pt_fin[0]:
#        S.compute_mep(begin=pt_fin, mid=[mid_point], end=pt_ini, maxsteps=maxsteps, dt=dt, traj_every=traj_every, npts=npts, flexible=flexible)
#    else:
#        S.compute_mep(begin=pt_ini, mid=[mid_point], end=pt_fin,  maxsteps=maxsteps, dt=dt, traj_every=traj_every, npts=npts, flexible=flexible)
#
#    if pt_ini[0] > pt_fin[0]:
#        mep_arr= np.array([np.flip(S.mep.T[0]), np.flip(S.mep.T[1]),  np.flip(S.get_mep_energy_profile()[1])])
#    else:
#        mep_arr=np.array( [S.mep.T[0], S.mep.T[1],  S.get_mep_energy_profile()[1]])
#
#    if options.get('print_convergence',False):
#        logger.info(f'saving file: convergence-mep.dat')
#        np.savetxt('convergence-mep.dat',np.array(S.convergence_history))
#
#
##    fig, ax, cbar= S.plot_mep()
##    plt.show()
#        
#    return mep_arr.T

def save_mep(mep_fes,fname_mep,header=None, more_columns=None):
    logger.info(f'saving file: {fname_mep}')
    if header is None:
        header=f'Minimum energy path {fname_mep}\ncolvar1, colvar2, fes (, sigma_fes)'
    if more_columns is not None:
        mep_fes=np.hstack((mep_fes,more_columns))
    np.savetxt(fname_mep, mep_fes, header=header,fmt='%.8e')

def load_mep(fname_mep):
    logger.info(f'loading file: {fname_mep}')
    mep_fes=np.loadtxt(fname_mep)
    return mep_fes

#def write_pt_cv(data,filename,**options):
#
#    name_cv_x=options.get('name_cv_x','cv_x')
#    name_cv_y=options.get('name_cv_y',None)
#    other_columns=options.get('other_columns','')
#
#    template_header = \
#"""! FIELDS time ${name_cv_x} ${name_cv_y} ${other_columns}"""
#    if name_cv_y is None:
#        template_header = \
#    """! FIELDS time ${name_cv_x} ${other_columns}"""
#
#    header = Template(template_header).substitute(name_cv_x=name_cv_x,
#                                                        name_cv_y=name_cv_y,
#                                                        other_columns=other_columns
#                                                        )
#
#    arr_to_save=data #np.array([X, Y]).T
#
#    logger.info(f'saving file: {filename}')
#    np.savetxt(filename,arr_to_save,fmt="%.6f",header=header, )

def write_pt_cv(data, filename, **options):
    name_cv_x = options.get('name_cv_x', 'cv_x')
    name_cv_y = options.get('name_cv_y', None)
    other_columns = options.get('other_columns', '')

    if name_cv_y is None:
        template_header = "! FIELDS time ${name_cv_x} ${other_columns}"
    else:
        template_header = "! FIELDS time ${name_cv_x} ${name_cv_y} ${other_columns}"

    header = Template(template_header).substitute(
        name_cv_x=name_cv_x,
        name_cv_y=name_cv_y or '',
        other_columns=other_columns
    ).strip()

    logger.info(f'saving file: {filename}')

    with open(filename, 'w') as f:
        f.write(f"#{header}\n")  # no leading "# "
        np.savetxt(f, data, fmt="%.6f")






def color_to_color_cmap(color_start, color_end, name=None, n=256):
    """
    Create a Matplotlib colormap going from one color to another.

    Parameters
    ----------
    color_start : str or tuple
        Starting color. Can be a hex string, e.g. "#ffffff",
        a named color, e.g. "white", or an RGB tuple in [0, 1].

    color_end : str or tuple
        Ending color. Can be a hex string, e.g. "#1f77b4",
        a named color, e.g. "blue", or an RGB tuple in [0, 1].

    name : str or None
        Name of the colormap. If None, a name is generated.

    n : int
        Number of color levels.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Colormap interpolating from color_start to color_end.
    """
    rgb_start = to_rgb(color_start)
    rgb_end = to_rgb(color_end)

    if name is None:
        name = (
            f"{color_start}_to_{color_end}"
            .replace("#", "")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
        )

    return LinearSegmentedColormap.from_list(
        name,
        [rgb_start, rgb_end],
        N=n,
    )


