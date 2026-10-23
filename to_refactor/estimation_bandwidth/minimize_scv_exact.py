import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize


def _as_spd_matrix(B, d):
    """
    Accepts:
      - scalar b: returns (b)*I   (so pass h**2 if you want H = h^2 I)
      - (d,) vector: returns diag(B)
      - (d,d) matrix: returns B
    """
    B = np.asarray(B)
    if B.ndim == 0:
        return float(B) * np.eye(d)
    if B.ndim == 1:
        if B.shape[0] != d:
            raise ValueError(f"Vector bandwidth must have length d={d}, got {B.shape[0]}")
        return np.diag(B)
    if B.ndim == 2:
        if B.shape != (d, d):
            raise ValueError(f"Matrix bandwidth must be shape {(d,d)}, got {B.shape}")
        return B
    raise ValueError("Bandwidth must be scalar, (d,) vector, or (d,d) matrix.")


def _phi_cov(diffs, cov, chol=None, logdet=None):
    """
    Multivariate normal density N(0, cov) evaluated at rows of diffs (m,d).
    Returns shape (m,).
    Uses Cholesky for stability.
    """
    diffs = np.asarray(diffs)
    d = diffs.shape[1]

    if chol is None or logdet is None:
        chol = np.linalg.cholesky(cov)  # cov = L L^T
        logdet = 2.0 * np.log(np.diag(chol)).sum()

    # Solve L y = x  => y = L^{-1} x
    # quad = x^T cov^{-1} x = ||y||^2
    y = np.linalg.solve(chol, diffs.T).T  # (m,d)
    quad = np.einsum("ij,ij->i", y, y)

    log_norm = -0.5 * (d * np.log(2.0 * np.pi) + logdet)
    return np.exp(log_norm - 0.5 * quad)


def scv_objective_eq318(X, H, G, chunk_size=1024):
    r"""
    Computes SCV(H;G) from Eq. (3.18) for Gaussian kernels K=L=phi:

      SCV(H;G) = n^{-1} |H|^{-1/2} (4π)^{-d/2}
               + n^{-2} Σ_{i=1}^n Σ_{j=1}^n
                   ( φ_{2H+2G}(X_i - X_j)
                     - 2 φ_{H+2G}(X_i - X_j)
                     +   φ_{2G}(X_i - X_j) )

    Parameters
    ----------
    X : (n,d) array
    H : scalar, (d,), or (d,d) SPD (bandwidth/cov matrix)
    G : scalar, (d,), or (d,d) SPD (pilot/cov matrix)
    chunk_size : int, controls memory (computes pairs in blocks over i)

    Returns
    -------
    float
        SCV objective value.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n,d).")
    n, d = X.shape

    Hm = _as_spd_matrix(H, d)
    Gm = _as_spd_matrix(G, d)

    # First term: n^{-1} |H|^{-1/2} (4π)^{-d/2}
    cholH = np.linalg.cholesky(Hm)
    logdetH = 2.0 * np.log(np.diag(cholH)).sum()
    term1 = (1.0 / n) * np.exp(-0.5 * logdetH) * (4.0 * np.pi) ** (-0.5 * d)

    # Covariances in the Gaussian densities:
    C1 = 2.0 * (Hm + Gm)      # 2H + 2G
    C2 = Hm + 2.0 * Gm        # H + 2G
    C3 = 2.0 * Gm             # 2G

    # Pre-factorize for speed/stability
    chol1 = np.linalg.cholesky(C1)
    ld1 = 2.0 * np.log(np.diag(chol1)).sum()
    chol2 = np.linalg.cholesky(C2)
    ld2 = 2.0 * np.log(np.diag(chol2)).sum()
    chol3 = np.linalg.cholesky(C3)
    ld3 = 2.0 * np.log(np.diag(chol3)).sum()

    # Double sum, computed in blocks over i to avoid n*n*d memory
    s = 0.0
    for i0 in tqdm(range(0, n, chunk_size),total=(n + chunk_size - 1) // chunk_size, ncols=80):
        i1 = min(n, i0 + chunk_size)
        # diffs: (m,n,d)
        diffs = X[i0:i1, None, :] - X[None, :, :]
        diffs2 = diffs.reshape(-1, d)  # (m*n, d)

        v1 = _phi_cov(diffs2, C1, chol=chol1, logdet=ld1)
        v2 = _phi_cov(diffs2, C2, chol=chol2, logdet=ld2)
        v3 = _phi_cov(diffs2, C3, chol=chol3, logdet=ld3)

        s += (v1 - 2.0 * v2 + v3).sum()

    term2 = (1.0 / (n * n)) * s
    return float(term1 + term2)

def _diagH_from_u(u):
    """
    u: (d,) unconstrained
    Returns H = diag(h^2) with h = exp(u)
    """
    h = np.exp(u)
    return np.diag(h * h)

def scv_objective_diagH_u(u, X, G, chunk_size=256):
    """
    Returns SCV value for diagonal H parameterized by u.
    """
    H = _diagH_from_u(u)
    return scv_objective_eq318(X, H, G, chunk_size=chunk_size)


def optimize_diagH_scv(X, G, chunk_size=256, maxiter=200):
    X = np.asarray(X, float)
    n, d = X.shape

    u0 = np.array((0.1, 0.1))

    def f(u):
        return scv_objective_diagH_u(u, X, G, chunk_size=chunk_size)

    res = minimize(
        f,
        u0,
        method="L-BFGS-B",
        options={"maxiter": maxiter}
    )

    u_opt = res.x
    h_opt = np.exp(u_opt)
    H_opt = np.diag(h_opt * h_opt)
    return H_opt, h_opt, res






X=np.loadtxt('COLVAR',usecols=(1,2))[:2000]
print(X.shape)


n, d = X.shape
g = 0.05
G = (g**2)  # scalar -> interpreted as (g^2) I
G=np.array(([g**2,0.0],[0.0,g**2]))

H_opt, h_opt, res = optimize_diagH_scv(X, G, chunk_size=256)

print("h_opt =", h_opt)      # per-dimension bandwidths
print("H_opt diag =", np.diag(H_opt))
print("SCV(H_opt) =", res.fun)
print("converged =", res.success, res.message)



