# @author Akash Pallath
# Licensed under the MIT license, see LICENSE for details.
#
# Reference: Weinan E, "Simplified and improved string method for computing the minimum energy paths in barrier-crossing events",
# J. Chem. Phys. 126, 164103 (2007), https://doi.org/10.1063/1.2720838

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, griddata, interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from tqdm import tqdm

import numpy as np

def _as_point(p, name="point"):
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size != 2:
        raise ValueError(f"{name} must be length-2 (x,y), got shape {np.shape(p)}")
    return float(p[0]), float(p[1])

def _as_points_list(mid):
    if mid is None:
        return []
    # Accept: [] , [[x,y], ...] , or a single [x,y]
    arr = np.asarray(mid, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"mid must be a point [x,y] or list of points, got shape {arr.shape}")
        return [(_as_point(arr, "mid"))]
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"mid points must be shape (m,2), got {arr.shape}")
        return [(_as_point(arr[i], f"mid[{i}]")) for i in range(arr.shape[0])]
    raise ValueError(f"mid must be a point or list of points, got shape {arr.shape}")

def _piecewise_linear_string(begin, mid, end, npts):
    bx, by = _as_point(begin, "begin")
    ex, ey = _as_point(end, "end")
    mid_pts = _as_points_list(mid)

    # Build knots; sort by x; drop duplicate x (keeps first occurrence)
    xpts = [bx] + [p[0] for p in mid_pts] + [ex]
    ypts = [by] + [p[1] for p in mid_pts] + [ey]

    order = np.argsort(xpts)
    x_sorted = np.array([xpts[i] for i in order], dtype=float)
    y_sorted = np.array([ypts[i] for i in order], dtype=float)

    # Remove duplicate x values (np.interp requires strictly increasing xp)
    keep = [0]
    for i in range(1, len(x_sorted)):
        if x_sorted[i] != x_sorted[keep[-1]]:
            keep.append(i)
    xk = x_sorted[keep]
    yk = y_sorted[keep]

    # If all x are equal, fall back to linear in y to avoid division by zero logic
    if xk.size < 2:
        string_x = np.full(int(npts), bx, dtype=float)
        string_y = np.linspace(by, ey, int(npts), dtype=float)
        return np.column_stack([string_x, string_y])

    string_x = np.linspace(xk[0], xk[-1], int(npts), dtype=float)
    string_y = np.interp(string_x, xk, yk)  # piecewise linear

    return np.column_stack([string_x, string_y])




class String2D:
    """
    Class containing methods to compute the minimum energy path between two
    points on an energy landscape $V$.

    Args:
        x: Array of shape (nx,) specifying x-axis coordinates of grid.
        y: Array of shape (ny,) specifying y-axis coordinates of grid.
        V: Array of shape (ny, nx) or (nx, ny) specifying energy values at each point on the grid.
            Missing values should be set to np.inf.
        indexing: Indexing of V array ('xy' specifies (ny, nx), 'ij' specifies (nx, ny); default = 'xy').

    Attributes:
        x: Array of shape (nx,) specifying x-axis coordinates of grid.
        y: Array of shape (ny,) specifying y-axis coordinates of grid.
        V: Array of shape (ny, nx) or (nx, ny) specifying energy values at each point on the grid.
        X: Grid of shape (ny, nx) or (nx, ny) containing x-coordinates of each point on the grid.
        Y: Grid of shape (ny, nx) or (nx, ny) containing y-coordinates of each point on the grid.
        indexing: Indexing of V, X, and Y arrays ('xy' specifies (ny, nx), 'ij' specifies (nx, ny); default = 'xy').
        gradX: Gradient in x.
        gradY: Gradient in y.
        string_traj: Trajectory showing the evolution of the string (default=[]).
        mep: Converged minimum energy path (default=None, if not converged).
    """

    def __init__(self, x, y, V, indexing="xy"):
        self.x = x
        self.y = y
        self.V = V

        # Generate grids
        self.X, self.Y = np.meshgrid(x, y, indexing=indexing)
        self.grid = np.vstack([self.X.ravel(), self.Y.ravel()]).T

        # Compute gradients
        self.indexing = indexing
        if indexing == "xy":
            # V shape (ny, nx): axis0=y, axis1=x
            self.gradY, self.gradX = np.gradient(self.V, self.y, self.x)
        elif indexing == "ij":
            # V shape (nx, ny): axis0=x, axis1=y
            self.gradX, self.gradY = np.gradient(self.V, self.x, self.y)
        else:
            raise ValueError("Indexing method not recognized.")

        # String method variables
        self.string_traj = []
        self.mep = None
        self.convergence_history = []
        self.converged=False

        self.gradX_array = None
        self.gradY_array = None

    def step_euler_new(self, string, dt, flexible=True):
        """
        Euler step using fast bilinear interpolation of gradients.
        """
        gx, gy = self.interpolate_gradients_map(string)
        h = np.max(np.sqrt(gx**2 + gy**2))
    
        step = dt * np.vstack([gx, gy]).T / h
    
        if flexible:
            string = string - step
        else:
            string[1:-1] = string[1:-1] - step[1:-1]
    
        return string

    def to_grid_coords(self, points):
        """
        Maps physical coordinates to grid indices.
        """
        ix = (points[:, 0] - self.x[0]) / (self.x[1] - self.x[0])
        iy = (points[:, 1] - self.y[0]) / (self.y[1] - self.y[0])
        return np.vstack([ix, iy])
    
    def interpolate_gradients_map(self, points):
        coords = self.to_grid_coords(points)
        gx = map_coordinates(self.gradX_array, coords, order=2, mode='nearest')
        gy = map_coordinates(self.gradY_array, coords, order=2, mode='nearest')
        return gx, gy

    def prepare_interpolation(self):
        """
        Prepares gradient arrays for fast interpolation using map_coordinates.
        Should be called after gradients are computed.
        """
        if self.indexing == "xy":
            # Transpose to (len(x), len(y)) order for map_coordinates
            self.gradX_array = self.gradX.T
            self.gradY_array = self.gradY.T
        elif self.indexing == "ij":
            self.gradX_array = self.gradX
            self.gradY_array = self.gradY
        else:
            raise ValueError("Indexing method not recognized.")


    def compute_mep(
        self,
        begin,
        end,
        mid=[],
        spline_order=2,
        npts=100,
        integrator="forward_euler",
        dt=0.1,
        tol=None,
        maxsteps=100,
        traj_every=10,
        flexible=True,
    ):
        """
        Computes the minimum free energy path. The points `begin`
        and `end` and the midpoints passed through `mid` are used to generate
        an initial guess (a k-order spline which interpolates through all the points).
        If no midpoints are defined, then the initial guess is a line connecting `begin`
        and `end`. If `flexible` is set to False, the ends of the string are fixed to `begin`
        and `end`, otherwise the ends of the string are free to move.

        Args:
            begin: Array of shape (2,) specifying starting point of the string.
            end: Array of shape (2,) specifying end point of the string.
            mid: List of arrays of shape (2,) specifying points between `begin` and `end`
                to use for generating an initial guess of the minimum energy path (default=[]).
            spline_order: Order of spline interpolating begin, mid, and end (default=2).
            npts: Number of points between any two valuesalong the string (default=100).
            integrator: Integration scheme to use (default='forward_euler'). Options=['forward_euler'].
            dt: Integration timestep (default=0.1).
            tol: Convergence criterion; stop stepping if string has an RMSD < tol between
                consecutive steps (default = max{npts^-4, 10^-10}).
            maxsteps: Maximum number of steps to take (default=100).
            traj_every: Interval to store string trajectory (default=10).
            flexible: If False, the ends of the string are fixed (default=True).

        Returns:
            mep: Array of shape (npts, 2) specifying string images along the minimum energy path between `begin` and `end`.
        """
        # Calculate params
        if tol is None:
            tol = max([npts**-4, 1e-10])

        string = _piecewise_linear_string(begin, mid, end, npts)
        
        # Store initial guess
        self.string_traj = []
        self.string_traj.append(string)

        # fix array shapes (new implementation)
        self.prepare_interpolation()

        # Loop
        old_string = np.zeros_like(string)

        x = np.linspace(np.min(self.x), np.max(self.x), self.gradX.shape[0])
        y = np.linspace(np.min(self.y), np.max(self.y), self.gradX.shape[1])

        for tstep in tqdm(range(1, maxsteps + 1), total=maxsteps, ncols=80):
            # Integrator step
            if integrator == "forward_euler":
                old_string[:] = string
                string = self.step_euler_new(string, dt, flexible=flexible)
            else:
                raise ValueError("Invalid integrator")

            # Reparameterize string (equal arc length reparameterization)
            arclength = np.hstack([0, np.cumsum(np.linalg.norm(string[1:] - string[:-1], axis=1))])
            arclength /= arclength[-1]
            reparam_x = interp1d(arclength, string[:, 0])
            reparam_y = interp1d(arclength, string[:, 1])
            gamma = np.linspace(0, 1, npts)
            string = np.vstack([reparam_x(gamma), reparam_y(gamma)]).T
           
            self.convergence_history.append(np.sqrt(np.mean((string - old_string) ** 2))) 
            

            # Store
            if tstep % traj_every == 0:
                self.string_traj.append(string)
                # Print convergence
#                print(
#                    "Change in string: {:.10f}".format(
#                        np.sqrt(np.mean((string - old_string) ** 2))
#                    )
#                )

            # Test for convergence
            if np.sqrt(np.mean((string - old_string) ** 2)) < tol:
                self.converged=True                
                break

        if self.converged:
            print('convergence reached')
        else:
            print('max step reached, leaving loop')


        # Store minimum energy path
        self.mep = string

    def step_euler(self, string, dt, flexible=True):
        """
        Evolves string images in time in response to forces calculated from the energy landscape.

        Args:
            string: Array of shape (npts, 2) specifying string images at the previous timestep.
            dt: Timestep.
            flexible: If False, the ends of the string are fixed (default=True).

        Returns:
            newstring: Array of shape (npts, 2) specifying string images after a timestep.
        """
        # Compute gradients at string points

#        string_grad_x = griddata(self.grid, self.gradX.ravel(), string, method="linear")
#        string_grad_y = griddata(self.grid, self.gradY.ravel(), string, method="linear")
######################################3 START FROM HERE
        string_grad_x = self.gradX_interp(string)
        string_grad_y = self.gradY_interp(string)
        h = np.max(np.sqrt(string_grad_x**2 + string_grad_y**2))

        # Euler step
        if flexible:
            string = string - dt * np.vstack([string_grad_x, string_grad_y]).T / h
        else:
            string[1:-1] = (
                string[1:-1] - dt * np.vstack([string_grad_x, string_grad_y]).T[1:-1] / h
            )

        return string

    def get_mep_energy_profile(self):
        energy_mep = griddata(self.grid, self.V.ravel(), self.mep, method="linear")
        return self.mep, energy_mep

    def plot_V(self, clip_min=None, clip_max=None, levels=None, cmap="RdYlBu", dpi=300):
        """
        Generates a filled contour plot of the energy landscape $V$.

        Args:
            cmap: Colormap for plot.
            levels: Levels to plot contours at (see matplotlib contour/contourf docs for details).
            dpi: DPI.
        """
        fig, ax = plt.subplots(dpi=dpi)

        V = self.V
        if clip_min is not None:
            V = V.clip(min=clip_min)
        if clip_max is not None:
            V = V.clip(max=clip_max)

        cs = ax.contourf(self.X, self.Y, V, levels=levels, cmap=cmap)
        ax.contour(self.X, self.Y, V, levels=levels, colors="black", alpha=0.2)
        cbar = fig.colorbar(cs)
        return fig, ax, cbar

    def plot_mep(self, **plot_V_kwargs):
        """
        Plots the minimum energy path on the energy landscape $V$.

        Args:
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax, cbar = self.plot_V(**plot_V_kwargs)
        ax.scatter(self.mep[0, 0], self.mep[0, 1], color="C0")
        ax.scatter(self.mep[-1, 0], self.mep[-1, 1], color="C0")
        ax.plot(self.mep[:, 0], self.mep[:, 1])
        return fig, ax, cbar

    def plot_mep_energy_profile(self, dpi=300):
        """
        Plots the energy profile along the minimum energy path in $V$.
        """
        energy_mep = griddata(self.grid, self.V.ravel(), self.mep, method="linear")
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(np.linspace(0, 1, len(energy_mep)), energy_mep)
        return fig, ax

    def plot_string_evolution(self, string_cmap=cm.gray, **plot_V_kwargs):
        """
        Plots the evolution of the string on the energy landscape $V$.

        Args:
            string_cmap: Colormap to use for plotting the evolution of the string.
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax, cbar = self.plot_V(**plot_V_kwargs)
        colors = string_cmap(np.linspace(0, 1, len(self.string_traj)))
        for sidx, string in enumerate(self.string_traj):
            ax.plot(string[:, 0], string[:, 1], "--", color=colors[sidx])
        return fig, ax, cbar
