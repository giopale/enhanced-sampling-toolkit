from string import Template
from typing import Tuple, Union, Optional
import sys
import numpy as np
import numpy.typing as npt
import concurrent.futures
from tqdm import tqdm
from logger_setup import logger
#from ectoolkits.log import get_logger

# adapted from FES_from_Reweighting.py that written by Michele Invernizzi
# then copied here from https://github.com/robinzyb/ECToolkits/blob/main/ectoolkits/analysis/reweight.py

##logger = get_logger(__name__)

def read_cv(colvar_file: str,
            usecols: int,
            dim:int,
            ):

    if dim == 1 and len(usecols) == 1:
        cv = np.loadtxt(colvar_file, usecols=usecols)
        return cv
    elif dim == 2 and len(usecols) == 2:
        cv = np.loadtxt(colvar_file, usecols=usecols)
        cv_x = cv[:, 0]
        cv_y = cv[:, 1]
        return cv_x, cv_y

def read_bias(colvar_file: str,
              usecols: int=3,
              ) -> npt.NDArray:
    bias = np.loadtxt(colvar_file, usecols=usecols)
    return bias

def compute_bw_silverman(cv_array: npt.NDArray,
                         dim: int,
                         **kwargs,
                         ) -> float:
    """
    Compute bandwidths using Silverman's rule of thumb.

    Parameters:
    cv_array (npt.NDArray): The input array for which the bandwidth is to be computed.
    dim (int): The dimensionality of the data.
    **kwargs: Additional keyword arguments.
        - weights (optional, array-like): Weights for the data points.
        - bias_factor (optional, float): Bias factor to adjust the bandwidth. Default is 1.0.

    Returns:
    float: The computed bandwidth.

    Notes:
    - If weights are provided, the effective sample size (neff) is computed.
    - The bandwidth is computed using the formula:
      bw = sigma_0 / (sqrt(bias_factor)) * ((neff * (dim + 2) / 4.0) ** (-1.0 / (dim + 4.0)))
    - The function prints intermediate values for sigma_0, number of configurations (nconf),
      effective sample size (neff), and the computed bandwidth (bw).
    """
    weights = kwargs.get('weights', None)
    bias_factor = kwargs.get('bias_factor', 1.0)
    if weights is not None:
        neff = np.power(np.sum(weights), 2) / np.sum(np.power(weights, 2))
    else:
        neff = cv_array.shape[0]

    sigma_0 = cv_array.std()
    #sigma_0 = sigma_0/
    bw = sigma_0/(np.sqrt(bias_factor))*np.power((neff *(dim+2)/4.0), -1.0/(dim+4.0))
    logger.info(f"SILVERMAN sigma_0 {sigma_0:.3f} nconf: {cv_array.shape[0]} neff: {neff:.3f} bw: {bw:.3f}")
    return bw

def parse_fes_header(file_fes):
    header = {}

    with open(file_fes, "r", errors="replace") as f:
        for line in f:
            line = line.strip()

            if not line.startswith("# SET"):
                continue

            parts = line.split(maxsplit=3)
            if len(parts) != 4:
                continue

            key = parts[2]
            value = parts[3]

            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

            header[key] = value

    return header

def parse_fes_nbins(file_fes):

    with open(file_fes,'r', errors="replace") as f:
        set_lines = [line.strip() for line in f if "SET" in line]
    filtered_lines = [line for line in set_lines if "num_grid_points" in line or "nbins" in line]
    if len(filtered_lines) == 0:
        logger.error(f'could not parse nbins from fes file {file_fes}')
        sys.exit()

    #nbins={}
    nbins=[]
    for ll in filtered_lines:
        parts=ll.split()
        value=int(parts[-1])
        cv=parts[-2].split('_')[-1]
        #nbins.update({cv:value})
        nbins.append(value)
    # WARNING: here I'm assuming that the CVs comments are in the same order as the columns.
    return nbins


def safe_load_column(filename, column_index):
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() and not line.strip().startswith('#'):
                n_cols = len(line.split())
                break
        else:
            raise ValueError("File is empty or only contains comments.")
    
    if column_index < n_cols:
        return np.loadtxt(filename, usecols=column_index)
    else:
        return None

def read_fes_old(file_fes: str,
             dim: int,
             **kwargs,
            ) -> Union[Tuple[npt.NDArray, npt.NDArray],
                      Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
                      ]:
    """
    Read the free energy surface (FES) data from PLUMED fes files.
    """
    #if dim == 2:
    #    if 'num_grid_points_x' not in kwargs or 'num_grid_points_y' not in kwargs:
    #        raise ValueError("For 2D FES, 'num_grid_points_x' and 'num_grid_points_y' must be provided in kwargs.")
    nbins=parse_fes_nbins(file_fes)
    num_grid_points_x = kwargs.get('num_grid_points_x', nbins[0])
    num_grid_points_y = kwargs.get('num_grid_points_y', nbins[1])
    if dim == 2:
        logger.debug(f'{file_fes.name}: loading fes')
        X = np.loadtxt(file_fes, usecols=0)
        Y = np.loadtxt(file_fes, usecols=1)
        Z = np.loadtxt(file_fes, usecols=2)
        Z_extra = safe_load_column(file_fes, 3)
        X = X.reshape(num_grid_points_x, num_grid_points_y)
        Y = Y.reshape(num_grid_points_x, num_grid_points_y)
        Z = Z.reshape(num_grid_points_x, num_grid_points_y)
        if Z_extra is not None:
            Z_extra = Z_extra.reshape(num_grid_points_x, num_grid_points_y)
        Z = Z*0.0103636  # to eV
        Z -= np.min(Z)
        if Z_extra is not None:
            Z_extra = Z_extra.reshape(num_grid_points_x, num_grid_points_y)
            Z_extra = Z_extra*0.0103636  # to eV
            Z_extra -= np.min(Z_extra)
        return X,Y,Z,Z_extra
    elif dim == 1:
        print("Reading 1D FES")
        x = np.loadtxt(file_fes, usecols=0)
        z = np.loadtxt(file_fes, usecols=1)
        z = z*0.0103636  # to eV
        z_min = np.min(z)
        z = z - z_min
        return x, z
    else:
        raise ValueError("dim should be either 1 or 2")

def _load_grid_2d(file_fes, fill, **kwargs):
    """Load columns 0..3 of a 2D PLUMED grid file onto an (nx, ny) grid.

    Rows are placed by their (x, y) values, so the file ordering does not matter.
    Bins absent from the file keep the value `fill`.
    Returns X, Y, Z, Z_extra (Z_extra is None if there is no 4th column).
    """
    nbins = parse_fes_nbins(file_fes)
    nx = int(kwargs.get("num_grid_points_x", nbins[0]))
    ny = int(kwargs.get("num_grid_points_y", nbins[1]))

    data = np.loadtxt(file_fes, usecols=(0, 1, 2))
    xcol, ycol, zcol = data[:, 0], data[:, 1], data[:, 2]
    zextra = safe_load_column(file_fes, 3)  # may be None

    xbins = np.unique(xcol)
    ybins = np.unique(ycol)
    if xbins.size != nx or ybins.size != ny:
        raise ValueError(f"{file_fes}: grid size mismatch: unique x={xbins.size} (expected {nx}), "
                         f"unique y={ybins.size} (expected {ny})")

    ix = np.searchsorted(xbins, xcol)
    iy = np.searchsorted(ybins, ycol)

    Z = np.full((nx, ny), fill, dtype=float)
    Z[ix, iy] = zcol

    Z_extra = None
    if zextra is not None:
        Z_extra = np.full((nx, ny), fill, dtype=float)
        Z_extra[ix, iy] = zextra

    X, Y = np.meshgrid(xbins, ybins, indexing="ij")
    return X, Y, Z, Z_extra


def read_fes(file_fes: str, dim: int, **kwargs):
    """Read a 2D PLUMED FES; returns X, Y, Z, Z_extra (eV, shifted to min 0), log_weight_total."""
    if dim != 2:
        raise ValueError("Only 2D FES files are supported.")

    header = parse_fes_header(file_fes)
    log_weight_total = header.get('log_weight_total', None)

    X, Y, Z, Z_extra = _load_grid_2d(file_fes, fill=np.inf, **kwargs)

    # Units + shift
    Z = Z * 0.0103636  # kJ/mol -> eV
    Z -= np.nanmin(Z)
    if Z_extra is not None:
        Z_extra = Z_extra * 0.0103636
        Z_extra -= np.nanmin(Z_extra)

    return X, Y, Z, Z_extra, log_weight_total


def read_property(file_fes: str, dim: int, **kwargs
                  ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[float]]:
    """Read a 2D property grid; returns X, Y, Z, Z_extra, log_weight_total (same as read_fes).

    Values are returned as stored (no unit conversion, no shift).
    Bins missing from the file are NaN.
    """
    if dim != 2:
        raise ValueError("Only 2D property files are supported.")

    logger.debug(f'{getattr(file_fes, "name", file_fes)}: loading')
    header = parse_fes_header(file_fes)
    log_weight_total = header.get('log_weight_total', None)

    X, Y, Z, Z_extra = _load_grid_2d(file_fes, fill=np.nan, **kwargs)
    return X, Y, Z, Z_extra, log_weight_total


#def read_fes(file_fes: str, dim: int, **kwargs):
#
#    header=parse_fes_header(file_fes)
#    nbins = parse_fes_nbins(file_fes)
#
#    log_weight_total=header.get('log_weight_total',None)
#    nx = int(kwargs.get("num_grid_points_x", nbins[0]))
#    ny = int(kwargs.get("num_grid_points_y", nbins[1]))
#
#    if dim != 2:
#        raise ValueError("Only showing the 2D robust version here.")
#
#    data = np.loadtxt(file_fes, usecols=(0, 1, 2))
#    xcol = data[:, 0]
#    ycol = data[:, 1]
#    zcol = data[:, 2]
#
#    zextra = safe_load_column(file_fes, 3)  # may be None
#
#    # Build bins robustly
#    xbins = np.unique(xcol)
#    ybins = np.unique(ycol)
#
#    if xbins.size != nx or ybins.size != ny:
#        raise ValueError(f"Grid size mismatch: unique x={xbins.size} (expected {nx}), "
#                         f"unique y={ybins.size} (expected {ny})")
#
#    # Map each (x,y) to indices, independent of file ordering
#    ix = np.searchsorted(xbins, xcol)
#    iy = np.searchsorted(ybins, ycol)
#
#    Z = np.full((nx, ny), np.inf, dtype=float)
#    Z[ix, iy] = zcol
#
#    X, Y = np.meshgrid(xbins, ybins, indexing="ij")
#
#    # Units + shift
#    Z = Z * 0.0103636 # to eV
#    Z -= np.nanmin(Z)
#
#    Z_extra = None
#    if zextra is not None:
#        Z_extra = np.full((nx, ny), np.inf, dtype=float)
#        Z_extra[ix, iy] = zextra
#        Z_extra = Z_extra * 0.0103636 # to eV
#        Z_extra -= np.nanmin(Z_extra)
#
#    return X, Y, Z, Z_extra, log_weight_total
#
#
#
#
#def read_property(file_fes: str,
#             dim: int,
#             **kwargs,
#            ) -> Union[Tuple[npt.NDArray, npt.NDArray],
#                      Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
#                      ]:
#    """
#    Read property data from PLUMED fes files.
#    """
#    #if dim == 2:
#    #    if 'num_grid_points_x' not in kwargs or 'num_grid_points_y' not in kwargs:
#    #        raise ValueError("For 2D FES, 'num_grid_points_x' and 'num_grid_points_y' must be provided in kwargs.")
#    nbins=parse_fes_nbins(file_fes)
#    num_grid_points_x = kwargs.get('num_grid_points_x', nbins[0])
#    num_grid_points_y = kwargs.get('num_grid_points_y', nbins[1])
#    if dim == 2:
#        logger.debug(f'{file_fes.name}: loading')
#        X = np.loadtxt(file_fes, usecols=0)
#        Y = np.loadtxt(file_fes, usecols=1)
#        Z = np.loadtxt(file_fes, usecols=2)
#        Z_extra = safe_load_column(file_fes, 3)
#        X = X.reshape(num_grid_points_x, num_grid_points_y)
#        Y = Y.reshape(num_grid_points_x, num_grid_points_y)
#        Z = Z.reshape(num_grid_points_x, num_grid_points_y)
#        if Z_extra is not None:
#            Z_extra = Z_extra.reshape(num_grid_points_x, num_grid_points_y)
#        return X,Y,Z, Z_extra
#    elif dim == 1:
#        logger.debug(f'{file_fes.name}: loading')
#        x = np.loadtxt(file_fes, usecols=0)
#        z = np.loadtxt(file_fes, usecols=1)
#        return x, z
#    else:
#        raise ValueError("dim should be either 1 or 2")



def make_grid(grid_bin_cv_x: int,
              cv_x_min: float,
              cv_x_max: float,
              dim: int,
              **kwargs
              ) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    """
    Generates a grid of coordinates based on the specified parameters.
    Parameters:
    grid_bin_cv_x (int): Number of bins for the x-axis.
    grid_bin_cv_y (int): Number of bins for the y-axis.
    cv_x_min (float): Minimum value for the x-axis.
    cv_x_max (float): Maximum value for the x-axis.
    cv_y_min (float): Minimum value for the y-axis.
    cv_y_max (float): Maximum value for the y-axis.
    Returns:
    tuple: one 1D array representing the x coordinates of the grid.
    tuple: Two 2D arrays representing the x and y coordinates of the grid.
    """
    grid_bin_cv_x += 1
    gird_cv_x_min = cv_x_min
    gird_cv_x_max = cv_x_max
    grid_cv_x = np.linspace(gird_cv_x_min, gird_cv_x_max, grid_bin_cv_x)
    if dim == 1:
        return grid_cv_x
    elif dim == 2:
        gird_cv_y_min = kwargs.get("cv_y_min", None)
        gird_cv_y_max = kwargs.get("cv_y_max", None)
        grid_bin_cv_y = kwargs.get("grid_bin_cv_y", None)
        grid_bin_cv_y += 1
        grid_cv_y = np.linspace(gird_cv_y_min, gird_cv_y_max, grid_bin_cv_y)
        X,Y=np.meshgrid(grid_cv_x,grid_cv_y)
        return X, Y


def calc_one_FES_point(cv_x: npt.NDArray,
                       bias: npt.NDArray,
                       point_x: float,
                       sigma_x: float,
                       kbT: float,
                       dim: int,
                       **kwargs
                       ) -> float:
    """
    Calculate the free energy surface (FES) value for a given point.

    Parameters:
    - cv_x: float, x-coordinate of the collective variable (CV)
    - cv_y: float, y-coordinate of the collective variable (CV)
    - bias: float, bias potential value
    - point_x: float, x-coordinate of the point
    - point_y: float, y-coordinate of the point
    - sigma_x: float, standard deviation of the x-coordinate
    - sigma_y: float, standard deviation of the y-coordinate
    - kbT: float, product of Boltzmann constant and temperature

    Returns:
    - fes: float, free energy surface value for the given point
    """
    cv_y = kwargs.get("cv_y", None)
    point_y = kwargs.get("point_y", None)
    sigma_y = kwargs.get("sigma_y", None)

    bias = bias / kbT

    dist_x = (point_x - cv_x) / sigma_x
    if dim == 1:
        arg = bias - 0.5*dist_x*dist_x
    elif dim == 2:
        dist_y = (point_y - cv_y) / sigma_y
        arg = bias - 0.5*dist_x*dist_x - 0.5*dist_y*dist_y
    else:
        raise ValueError("dim should be either 1 or 2")

    fes = -kbT*np.logaddexp.reduce(arg)
    weights = np.exp(arg)

    return fes, weights 


def calc_one_FES_point_weights(cv_x: npt.NDArray,
                       bias: npt.NDArray,
                       point_x: float,
                       sigma_x: float,
                       kbT: float,
                       dim: int,
                       **kwargs
                       ) -> float:
    """
    Calculate the weights for a given point on the CV grid.

    Parameters:
    - cv_x: float, x-coordinate of the collective variable (CV)
    - cv_y: float, y-coordinate of the collective variable (CV)
    - bias: float, bias potential value
    - point_x: float, x-coordinate of the point
    - point_y: float, y-coordinate of the point
    - sigma_x: float, standard deviation of the x-coordinate
    - sigma_y: float, standard deviation of the y-coordinate
    - kbT: float, product of Boltzmann constant and temperature

    Returns:
    - fes: float, free energy surface value for the given point
    """
    cv_y = kwargs.get("cv_y", None)
    point_y = kwargs.get("point_y", None)
    sigma_y = kwargs.get("sigma_y", None)

    bias = bias / kbT

    dist_x = (point_x - cv_x) / sigma_x
    if dim == 1:
        arg = bias - 0.5*dist_x*dist_x
    elif dim == 2:
        dist_y = (point_y - cv_y) / sigma_y
        arg = bias - 0.5*dist_x*dist_x - 0.5*dist_y*dist_y
    else:
        raise ValueError("dim should be either 1 or 2")


    return np.exp(arg)


def calc_one_FES_point_wrapper(args):
        """ Wrapper function to unpack arguments for parallel execution. """
        i, j, fes_x, fes_y, cv_x, cv_y, bias, bandwidth_x, bandwidth_y, kbT, dim = args
        fes, weights = calc_one_FES_point(cv_x=cv_x,
                                        cv_y=cv_y,
                                        bias=bias,
                                        point_x=fes_x[i, j],
                                        point_y=fes_y[i, j],
                                        sigma_x=bandwidth_x,
                                        sigma_y=bandwidth_y,
                                        kbT=kbT,
                                        dim=dim)
        return i, j, fes, weights



#TODO: add documentation in jupyter book
def calc_FES(cv_x: npt.NDArray,
             bandwidth_x: float,
             nbins_x: int,
             bias: npt.NDArray,
             kbT: float,
             dim: int=2,
             **kwargs,
             ) -> npt.NDArray:
    """
    Calculate the Free Energy Surface (FES) using the provided collective variables (CVs) and bias.

    Parameters:
    cv_x (npt.NDArray): Collective variable values along the x-axis.
    bandwidth_x (float): Bandwidth for the Gaussian kernel along the x-axis.
    nbins_x (int): Number of bins along the x-axis for the grid.
    bias (npt.NDArray): Bias values corresponding to the CVs. must be divied by kbT to make it dimensionless before passing to this function.
    kbT (float): Thermal energy (Boltzmann constant times temperature).
    dim (int, optional): Dimensionality of the FES calculation (1 or 2). Default is 2.
    **kwargs: Additional keyword arguments:
        - cv_y (npt.NDArray, optional): Collective variable values along the y-axis (for 2D FES).
        - bandwidth_y (float, optional): Bandwidth for the Gaussian kernel along the y-axis (for 2D FES).
        - nbins_y (int, optional): Number of bins along the y-axis for the grid (for 2D FES).
        - min_cv_x (float, optional): Minimum value of the CV along the x-axis.
        - max_cv_x (float, optional): Maximum value of the CV along the x-axis.
        - min_cv_y (float, optional): Minimum value of the CV along the y-axis (for 2D FES).
        - max_cv_y (float, optional): Maximum value of the CV along the y-axis (for 2D FES).
        - save_file (bool, optional): Whether to save the FES data to a file. Default is True.
        - name_file (str, optional): Filename to save the FES data. Default is "fes.dat".
        - name_cv_x (str, optional): Name of the CV along the x-axis. Default is "cv_x".
        - name_cv_y (str, optional): Name of the CV along the y-axis. Default is "cv_y".

    Returns:
    np.ndarray: Calculated FES values on the grid.
    """
    # Extract the keyword arguments
    save_file = kwargs.get("save_file", True)
    name_file = kwargs.get("name_file", "fes.dat")

    name_cv_x = kwargs.get("name_cv_x", "cv_x")

    min_cv_x = kwargs.get("min_cv_x", cv_x.min())
    max_cv_x = kwargs.get("max_cv_x", cv_x.max())


    special_points=kwargs.get("special_points",[])

    # the syntax is ugly.. any better way to do this?
    template_header_1d = \
"""Free energy surface generated by ECToolkits
FIELDS ${name_cv_x} () free_energy (kJ/mol)
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}"""

    template_header_2d = \
"""Free energy surface generated by ECToolkits
FIELDS ${name_cv_x} () ${name_cv_y} () free_energy (kJ/mol)
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET sigma_${name_cv_x} ${sigma_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}
SET min_${name_cv_y} ${min_cv_y}
SET max_${name_cv_y} ${max_cv_y}
SET sigma_${name_cv_y} ${sigma_y}
SET num_grid_points_${name_cv_y} ${num_grid_points_y}"""

    if dim == 1:
        logger.info("Calculating 1D FES")

        fes_x = make_grid(grid_bin_cv_x=nbins_x,
                          cv_x_min=min_cv_x,
                          cv_x_max=max_cv_x,
                          dim=dim
                          )
        # Note the num_grid_points_x are nbins_x+1
        num_grid_points_x = len(fes_x)
        fes = np.zeros(num_grid_points_x)

        for i in range(num_grid_points_x):
            logger.info(f'   working...  {(i/num_grid_points_x):.0%}')
            fes[i] = calc_one_FES_point(cv_x=cv_x,
                                        bias=bias,
                                        point_x=fes_x[i],
                                        sigma_x=bandwidth_x,
                                        kbT=kbT,
                                        dim=dim)

        if save_file:
            header = Template(template_header_1d).substitute(name_cv_x=name_cv_x,
                                                             min_cv_x=f"{min_cv_x:10.3f}",
                                                             max_cv_x=f"{max_cv_x:10.3f}",
                                                             num_grid_points_x=num_grid_points_x,
                                                             )

            np.savetxt(name_file,
                       np.array([fes_x.flatten(), fes.flatten()]).T,
                       fmt="%.6f",
                       header=header,
                       )
            logger.info(f'file saved: {name_file}')
        return fes

    elif dim == 2:
        logger.info("Calculating 2D FES")

        name_cv_y = kwargs.get("name_cv_y", "cv_y")
        cv_y = kwargs.get("cv_y", None)
        bandwidth_y = kwargs.get("bandwidth_y", None)
        nbins_y = kwargs.get("nbins_y", None)
        min_cv_y = kwargs.get("min_cv_y", cv_y.min())
        max_cv_y = kwargs.get("max_cv_y", cv_y.max())

        fes_x, fes_y = make_grid(grid_bin_cv_x=nbins_x,
                                grid_bin_cv_y=nbins_y,
                                cv_x_min=min_cv_x,
                                cv_x_max=max_cv_x,
                                cv_y_min=min_cv_y,
                                cv_y_max=max_cv_y,
                                dim=dim,
                                )
    
        special_points_weights=[]
        bandwidths=[bandwidth_x,bandwidth_y]
        for point in special_points:
            for kk, wholegrid in enumerate([fes_x,fes_y]):
                if kk==1:
                    grid=wholegrid[:,0]
                else:
                    grid=wholegrid[0]
                spacing=grid[1]-grid[0]
                special_bin=np.argmin(np.abs(grid-point[kk]))
                logger.info(f'sigma/spacing is {bandwidths[kk]/spacing:.3g} ({bandwidths[kk]:.3g}/{spacing:.3g})')
            weight_point=np.zeros(len(bias))
            for padx in [0]:
                for pady in [0]:
                    value = calc_one_FES_point_weights(cv_x=cv_x,
                                                   cv_y=cv_y,
                                                   bias=bias,
                                                   point_x=point[0+padx],
                                                   point_y=point[1+pady],
                                                   sigma_x=bandwidth_x,
                                                   sigma_y=bandwidth_y,
                                                   kbT=kbT,
                                                   dim=dim,
                                                   )
                    weight_point+=value
            special_points_weights.append(weight_point)


        # Note the num_grid_points_x and num_grid_points_y are nbins_x+1 and nbins_y+1
        num_grid_points_x = len(fes_x)
        num_grid_points_y = len(fes_y)
        fes = np.zeros((num_grid_points_x, num_grid_points_y))

#        for i in range(num_grid_points_x):
#            logger.info(f'   working...  {(i/num_grid_points_x):.0%}')
#            for j in range(num_grid_points_y):
#                fes[i, j] = calc_one_FES_point(cv_x=cv_x,
#                                               cv_y=cv_y,
#                                               bias=bias,
#                                               point_x=fes_x[i, j],
#                                               point_y=fes_y[i, j],
#                                               sigma_x=bandwidth_x,
#                                               sigma_y=bandwidth_y,
#                                               kbT=kbT,
#                                               dim=dim,
#                                               )

# parallel version

        tasks = [(i, j, fes_x, fes_y, cv_x, cv_y, bias, bandwidth_x, bandwidth_y, kbT, dim)
                        for i in range(num_grid_points_x)
                        for j in range(num_grid_points_y)]

        allweights=np.zeros(len(bias))
        with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
                for idx, (i, j, value, weights) in enumerate(tqdm(executor.map(calc_one_FES_point_wrapper, tasks), 
                                                                  desc='working', total=len(tasks), ncols=80)):
                        fes[i, j] = value
                        allweights += weights
        if len(special_points)>0:
            for kk, pt in enumerate(special_points):
                
                logger.info(f'special point [{pt[0]:.3f},{pt[1]:.3f}] has weight {np.sum(special_points_weights[kk])/np.sum(allweights):.4g}')
                thr=.9
                count=np.sum(
                    np.cumsum(
                                np.sort(special_points_weights[kk])[::-1] / np.sum(special_points_weights[kk])
                            ) < thr )
                to_see=np.sort(special_points_weights[kk])[::-1]/np.sum(special_points_weights[kk])
                
                logger.info(f'{count} structures constitute {100*thr:.1f}% of the bin')

        if save_file:
            header = Template(template_header_2d).substitute(name_cv_x=name_cv_x,
                                                             name_cv_y=name_cv_y,
                                                             min_cv_x=f"{min_cv_x:10.3f}",
                                                             max_cv_x=f"{max_cv_x:10.3f}",
                                                             sigma_x=f"{bandwidth_x:14.6f}",
                                                             num_grid_points_x=num_grid_points_x,
                                                             min_cv_y=f"{min_cv_y:10.3f}",
                                                             max_cv_y=f"{max_cv_y:10.3f}",
                                                             sigma_y=f"{bandwidth_y:14.6f}",
                                                             num_grid_points_y=num_grid_points_y,
                                                             )
            np.savetxt(name_file,
                       np.array([fes_x.flatten(), fes_y.flatten(), fes.flatten()]).T,
                       fmt="%.6f",
                       header=header,
                       )
            logger.info(f'file saved: {name_file}')
        return fes

    else:
        raise ValueError("dim should be either 1 or 2")


#TODO: add pytest for this function
def calc_one_property_point(cv_x: npt.NDArray,
                            bias: npt.NDArray,
                            point_x: float,
                            bandwidth_x: float,
                            kbT: float,
                            dim: int,
                            prop: npt.NDArray,
                            **kwargs,
                            ) -> float:


    cv_y = kwargs.get("cv_y", None)
    point_y = kwargs.get("point_y", None)
    bandwidth_y = kwargs.get("bandwidth_y", None)

    bias = bias/kbT

    dist_x = (point_x - cv_x) / bandwidth_x
    if dim == 1:
        arg = bias - 0.5*dist_x*dist_x
    elif dim == 2:
        dist_y = (point_y - cv_y) / bandwidth_y
        arg = bias - 0.5*dist_x*dist_x - 0.5*dist_y*dist_y 

    arg = np.exp(arg) # G(s(R)-sk) * exp(beta*V)
    denominator = np.sum(arg) # < G(s(R)-sk) * exp(beta*V) >_V
    numerator = np.sum(prop*arg) # < prop * G(s(R)-sk) * exp(beta*V) >_V
    reweighted_prop = numerator / denominator  # 

    return reweighted_prop


def calc_one_c_of_t_point(cv_x: npt.NDArray,
                            bias: npt.NDArray,
                            point_x: float,
                            bandwidth_x: float,
                            kbT: float,
                            dim: int,
                            **kwargs,
                            ) -> float:


    cv_y = kwargs.get("cv_y", None)
    point_y = kwargs.get("point_y", None)
    bandwidth_y = kwargs.get("bandwidth_y", None)

    bias = bias/kbT

    dist_x = (point_x - cv_x) / bandwidth_x
    if dim == 1:
        arg = bias - 0.5*dist_x*dist_x
    elif dim == 2:
        dist_y = (point_y - cv_y) / bandwidth_y
        arg = bias - 0.5*dist_x*dist_x - 0.5*dist_y*dist_y 
        arg1 = - 0.5*dist_x*dist_x - 0.5*dist_y*dist_y 

    arg = np.exp(arg) # G(s(R)-sk) * exp(beta*V)
    denominator = np.sum(arg) # < G(s(R)-sk) * exp(beta*V) >_V
    numerator = np.sum(np.exp(arg1)) # <  G(s(R)-sk) * exp(beta*V) * exp(-beta*V) >_V

    return numerator, denominator

def calc_one_c_of_t_point_wrapper(args):
        """ Wrapper function to unpack arguments for parallel execution. """
        i, j, fes_x, fes_y, cv_x, cv_y, bias, bandwidth_x, bandwidth_y, kbT, dim= args
        return i, j, calc_one_property_point(cv_x=cv_x,
                                        cv_y=cv_y,
                                        bias=bias,
                                        point_x=fes_x[i, j],
                                        point_y=fes_y[i, j],
                                        bandwidth_x=bandwidth_x,
                                        bandwidth_y=bandwidth_y,
                                        kbT=kbT,
                                        dim=dim,
                                        )

def calc_one_property_point_wrapper(args):
        """ Wrapper function to unpack arguments for parallel execution. """
        i, j, fes_x, fes_y, cv_x, cv_y, bias, bandwidth_x, bandwidth_y, kbT, dim, prop= args
        return i, j, calc_one_property_point(cv_x=cv_x,
                                        cv_y=cv_y,
                                        bias=bias,
                                        point_x=fes_x[i, j],
                                        point_y=fes_y[i, j],
                                        bandwidth_x=bandwidth_x,
                                        bandwidth_y=bandwidth_y,
                                        kbT=kbT,
                                        dim=dim,
                                        prop=prop)



def calc_c_of_t(cv_x: npt.NDArray,
                          bias: npt.NDArray,
                          bandwidth_x: float,
                          kbT: float,
                          nbins_x: int,
                          dim: int = 2,
                          **kwargs,
                          ):
    """
    Calculates the property surface based on the given input parameters.

    Parameters:
    """
    save_file = kwargs.get("save_file", True)
    name_file = kwargs.get("name_file", "prop_surface.dat")

    name_cv_x = kwargs.get("name_cv_x", "cv_x")

    min_cv_x = kwargs.get("min_cv_x", cv_x.min())
    max_cv_x = kwargs.get("max_cv_x", cv_x.max())

    property_name = kwargs.get("property_name", 'property_name')

    # the syntax is ugly.. any better way to do this?
    template_header_1d = \
"""FIELDS ${name_cv_x} ${property_name}
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}"""

    template_header_2d = \
"""FIELDS ${name_cv_x} ${name_cv_y} ${property_name} 
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}
SET min_${name_cv_y} ${min_cv_y}
SET max_${name_cv_y} ${max_cv_y}
SET num_grid_points_${name_cv_y} ${num_grid_points_y}"""

    logger.info("Calculating 2D property surface")

    name_cv_y = kwargs.get("name_cv_y", "cv_y")
    cv_y = kwargs.get("cv_y", None)
    bandwidth_y = kwargs.get("bandwidth_y", None)
    nbins_y = kwargs.get("nbins_y", None)
    min_cv_y = kwargs.get("min_cv_y", cv_y.min())
    max_cv_y = kwargs.get("max_cv_y", cv_y.max())

    fes_x, fes_y = make_grid(grid_bin_cv_x=nbins_x,
                            grid_bin_cv_y=nbins_y,
                            cv_x_min=min_cv_x,
                            cv_x_max=max_cv_x,
                            cv_y_min=min_cv_y,
                            cv_y_max=max_cv_y,
                            dim=dim,
                            )

    # Note the num_grid_points_x and num_grid_points_y are nbins_x+1 and nbins_y+1
    num_grid_points_x = len(fes_x)
    num_grid_points_y = len(fes_y)

    numerator_surface = np.zeros((num_grid_points_x, num_grid_points_y))
    denominator_surface = np.zeros((num_grid_points_x, num_grid_points_y))

    # toggle parallel execution
    parallel=False
    if parallel:

        tasks = [(i, j, fes_x, fes_y, cv_x, cv_y, bias, bandwidth_x, bandwidth_y, kbT, dim)
                 for i in range(num_grid_points_x)
                 for j in range(num_grid_points_y)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
                for idx, (i, j, num, denom) in enumerate(tqdm(executor.map(calc_one_c_of_t_point_wrapper, tasks), 
                                                              desc='working', total=len(tasks), ncols=80)):
                        numerator_surface[i, j], denominator_surface[i, j] = num, denom

    else:
        for i in range(num_grid_points_x):
            #logger.info(f'   working...  {(i/num_grid_points_x):.0%}')
            for j in range(num_grid_points_y):
                numerator_surface[i, j], denominator_surface[i, j] = calc_one_c_of_t_point(cv_x=cv_x,
                                                             cv_y=cv_y,
                                                             bias=bias,
                                                             point_x=fes_x[i, j],
                                                             point_y=fes_y[i, j],
                                                             bandwidth_x=bandwidth_x,
                                                             bandwidth_y=bandwidth_y,
                                                             kbT=kbT,
                                                             dim=dim,
                                                             )

        result= -1/kbT*np.log(np.sum(numerator_surface)/np.sum(denominator_surface))

    return result









def calc_property_surface(cv_x: npt.NDArray,
                          bias: npt.NDArray,
                          prop: npt.NDArray,
                          bandwidth_x: float,
                          kbT: float,
                          nbins_x: int,
                          dim: int = 2,
                          **kwargs,
                          ):
    """
    Calculates the property surface based on the given input parameters.

    Parameters:
    """
    save_file = kwargs.get("save_file", True)
    name_file = kwargs.get("name_file", "prop_surface.dat")

    name_cv_x = kwargs.get("name_cv_x", "cv_x")

    min_cv_x = kwargs.get("min_cv_x", cv_x.min())
    max_cv_x = kwargs.get("max_cv_x", cv_x.max())

    property_name= kwargs.get("property_name", 'property')

    # the syntax is ugly.. any better way to do this?
    template_header_1d = \
"""FIELDS ${name_cv_x}  ${property_name}
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}"""

    template_header_2d = \
"""FIELDS ${name_cv_x} ${name_cv_y}  ${property_name}
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}
SET min_${name_cv_y} ${min_cv_y}
SET max_${name_cv_y} ${max_cv_y}
SET num_grid_points_${name_cv_y} ${num_grid_points_y}"""

    if dim == 1:
        logger.info("Calculating 1D property surface")

        fes_x = make_grid(grid_bin_cv_x=nbins_x,
                          cv_x_min=min_cv_x,
                          cv_x_max=max_cv_x,
                          dim=dim
                          )
        num_grid_points_x = len(fes_x)
        prop_surface = np.zeros(num_grid_points_x)

        for i in range(num_grid_points_x):
            logger.info(f'   working...  {(i/num_grid_points_x):.0%}')
            prop_surface[i] = calc_one_property_point(cv_x=cv_x,
                                                      bias=bias,
                                                      point_x=fes_x[i],
                                                      bandwidth_x=bandwidth_x,
                                                      kbT=kbT,
                                                      dim=dim,
                                                      prop=prop,
                                                      )

        if save_file:
            header = Template(template_header_1d).substitute(name_cv_x=name_cv_x,
                                                             min_cv_x=f"{min_cv_x:10.3f}",
                                                             max_cv_x=f"{max_cv_x:10.3f}",
                                                             num_grid_points_x=num_grid_points_x,
                                                             property_name=property_name
                                                             )

            logger.info('saving file: name_file')
            np.savetxt(name_file,
                       np.array([fes_x.flatten(), prop_surface.flatten()]).T,
                       fmt="%.6f",
                       header=header,
                       )

        return prop_surface

    elif dim == 2:
        logger.info("Calculating 2D property surface")

        name_cv_y = kwargs.get("name_cv_y", "cv_y")
        cv_y = kwargs.get("cv_y", None)
        bandwidth_y = kwargs.get("bandwidth_y", None)
        nbins_y = kwargs.get("nbins_y", None)
        min_cv_y = kwargs.get("min_cv_y", cv_y.min())
        max_cv_y = kwargs.get("max_cv_y", cv_y.max())

        fes_x, fes_y = make_grid(grid_bin_cv_x=nbins_x,
                                grid_bin_cv_y=nbins_y,
                                cv_x_min=min_cv_x,
                                cv_x_max=max_cv_x,
                                cv_y_min=min_cv_y,
                                cv_y_max=max_cv_y,
                                dim=dim,
                                )

        # Note the num_grid_points_x and num_grid_points_y are nbins_x+1 and nbins_y+1
        num_grid_points_x = len(fes_x)
        num_grid_points_y = len(fes_y)

        prop_surface = np.zeros((num_grid_points_x, num_grid_points_y))
        
        parallel=True
        if parallel:
            tasks = [(i, j, fes_x, fes_y, cv_x, cv_y, bias, bandwidth_x, bandwidth_y, kbT, dim, prop)
                            for i in range(num_grid_points_x)
                            for j in range(num_grid_points_y)]

            with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
                    for idx, (i, j, value) in enumerate(tqdm(executor.map(calc_one_property_point_wrapper, tasks), desc='working', total=len(tasks))):
                            prop_surface[i, j] = value

        else:
            for i in range(num_grid_points_x):
                logger.info(f'   working...  {(i/num_grid_points_x):.0%}')
                for j in range(num_grid_points_y):
                    prop_surface[i, j] = calc_one_property_point(cv_x=cv_x,
                                                                 cv_y=cv_y,
                                                                 bias=bias,
                                                                 point_x=fes_x[i, j],
                                                                 point_y=fes_y[i, j],
                                                                 bandwidth_x=bandwidth_x,
                                                                 bandwidth_y=bandwidth_y,
                                                                 kbT=kbT,
                                                                 dim=dim,
                                                                 prop=prop,
                                                                 )

        if save_file:
            header = Template(template_header_2d).substitute(name_cv_x=name_cv_x,
                                                             name_cv_y=name_cv_y,
                                                             min_cv_x=f"{min_cv_x:10.3f}",
                                                             max_cv_x=f"{max_cv_x:10.3f}",
                                                             num_grid_points_x=num_grid_points_x,
                                                             min_cv_y=f"{min_cv_y:10.3f}",
                                                             max_cv_y=f"{max_cv_y:10.3f}",
                                                             num_grid_points_y=num_grid_points_y,
                                                             property_name=property_name
                                                             )
            logger.info(f'saving file: {name_file}')
            np.savetxt(name_file,
                       np.array([fes_x.flatten(), fes_y.flatten(), prop_surface.flatten()]).T,
                       fmt="%.6f",
                       header=header,
                       )

        return prop_surface

    else:
        raise ValueError("dim should be either 1 or 2")



def write_fes(X,Y,Z,filename,**options):

    template_header_2d = \
"""FIELDS ${name_cv_x} ${name_cv_y}  ${property_name}
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}
SET min_${name_cv_y} ${min_cv_y}
SET max_${name_cv_y} ${max_cv_y}
SET num_grid_points_${name_cv_y} ${num_grid_points_y}
SET log_weight_total ${log_weight_total}"""

    name_cv_x=options.get('name_cv_x','cv_x')
    min_cv_x=options.get('min_cv_x',-1)
    max_cv_x=options.get('max_cv_x',1)
    name_cv_y=options.get('name_cv_y','cv_y')
    min_cv_y=options.get('min_cv_y',-1)
    max_cv_y=options.get('max_cv_y',-1)

    num_grid_points_x=options.get('num_grid_points_x',-1)
    num_grid_points_y=options.get('num_grid_points_y',-1)

    property_name=options.get('property_name','')

    column_extra=options.get('column_extra',None)

    log_weight_total=options.get('log_weight_total',None)

    header = Template(template_header_2d).substitute(name_cv_x=name_cv_x,
                                                        name_cv_y=name_cv_y,
                                                        min_cv_x=f"{min_cv_x:10.3f}",
                                                        max_cv_x=f"{max_cv_x:10.3f}",
                                                        num_grid_points_x=num_grid_points_x,
                                                        min_cv_y=f"{min_cv_y:10.3f}",
                                                        max_cv_y=f"{max_cv_y:10.3f}",
                                                        num_grid_points_y=num_grid_points_y,
                                                        property_name=property_name,
                                                        log_weight_total=log_weight_total,
                                                        )

    if column_extra is not None:
        arr_to_save=np.array([X.flatten(), Y.flatten(), Z.flatten(), column_extra.flatten()]).T
    else:
        arr_to_save=np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    logger.info(f'saving file: {filename}')
    np.savetxt(filename,arr_to_save,fmt="%.6f",header=header, )


