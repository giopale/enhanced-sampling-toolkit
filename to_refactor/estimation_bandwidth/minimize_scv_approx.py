import numpy as np
from scipy.optimize import minimize
import sys
from pathlib import Path
import argparse


def _scv_term2_mc_pairs_diag(X, h2, g2, pairs_i, pairs_j, chunk_pairs=2_000_000):
    X = np.asarray(X, float)
    n, d = X.shape

    h2 = np.asarray(h2, float)          # (d,)
    if np.ndim(g2) == 0:
        g2 = np.full(d, float(g2))
    else:
        g2 = np.asarray(g2, float)

    v1 = 2.0 * (h2 + g2)                # C1 diag
    v2 = h2 + 2.0 * g2                  # C2 diag
    v3 = 2.0 * g2                       # C3 diag

    log_norm1 = -0.5 * (d * np.log(2.0 * np.pi) + np.log(v1).sum())
    log_norm2 = -0.5 * (d * np.log(2.0 * np.pi) + np.log(v2).sum())
    log_norm3 = -0.5 * (d * np.log(2.0 * np.pi) + np.log(v3).sum())

    M = pairs_i.shape[0]
    s = 0.0
    processed = 0

    for k0 in range(0, M, chunk_pairs):
        k1 = min(M, k0 + chunk_pairs)
        Xi = X[pairs_i[k0:k1]]
        Xj = X[pairs_j[k0:k1]]
        dif2 = (Xi - Xj) ** 2

        q1 = (dif2 / v1).sum(axis=1)
        q2 = (dif2 / v2).sum(axis=1)
        q3 = (dif2 / v3).sum(axis=1)

        phi1 = np.exp(log_norm1 - 0.5 * q1)
        phi2 = np.exp(log_norm2 - 0.5 * q2)
        phi3 = np.exp(log_norm3 - 0.5 * q3)

        s += (phi1 - 2.0 * phi2 + phi3).sum()
        processed += (k1 - k0)

    return s / processed


def scv_objective_eq318_diagH_mc_u(u, X, g, pairs_i, pairs_j, chunk_pairs=2_000_000):
    X = np.asarray(X, float)
    n, d = X.shape

    h = np.exp(np.asarray(u, float))
    h2 = h * h
    g2 = float(g) * float(g)

    # term1 for diagonal H: |H| = prod(h^2)
    logdetH = np.log(h2).sum()
    term1 = (1.0 / n) * np.exp(-0.5 * logdetH) * (4.0 * np.pi) ** (-0.5 * d)

    term2 = _scv_term2_mc_pairs_diag(X, h2=h2, g2=g2, pairs_i=pairs_i, pairs_j=pairs_j,
                                     chunk_pairs=chunk_pairs)
    return float(term1 + term2)


def optimize_diagH_scv_mc(
    X,
    g=0.05,
    M_pairs=2_000_000,
    seed=123,
    maxiter=80,
    h_init=1e-3,
    h_bounds=(1e-5, 5e-2),
    chunk_pairs=2_000_000,
):
    """
    d=2 assumed but works for any d.
    Uses fixed random pairs (common random numbers) => deterministic objective.
    Uses L-BFGS-B with bounds in u=log(h).
    """
    X = np.asarray(X, float)
    n, d = X.shape

    rng = np.random.default_rng(seed)
    pairs_i = rng.integers(0, n, size=M_pairs, dtype=np.int64)
    pairs_j = rng.integers(0, n, size=M_pairs, dtype=np.int64)

    # init near your Silverman scale
    u0 = np.full(d, np.log(h_init), dtype=float)

    # bounds on u = log(h)
    lo, hi = h_bounds
    bounds = [(np.log(lo), np.log(hi))] * d

    def f(u):
        return scv_objective_eq318_diagH_mc_u(u, X, g, pairs_i, pairs_j, chunk_pairs=chunk_pairs)

    res = minimize(
        f,
        u0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter}
    )

    h_opt = np.exp(res.x)
    H_opt = np.diag(h_opt * h_opt)
    return H_opt, h_opt, res



def parse_fields_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()

    if first.startswith("#! FIELDS"):
        fields = first.split()[2:]
    elif first.startswith("#"):
        fields = first.split()[1:]
    else:
        raise ValueError(f"{path}: first line must start with '#! FIELDS' or '#'")

    if not fields:
        raise ValueError(f"{path}: header has no fields")
    return fields


def load_numeric_block(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#", ndmin=2)
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def pick_time_col(fields, time_field) -> int:
    if time_field is not None:
        if time_field not in fields:
            raise ValueError(f"time field '{time_field}' not found in header. Available: {fields}")
        return fields.index(time_field)

    # Heuristics (common in PLUMED / CV outputs)
    candidates = ("time", "t", "step", "steps", "stride", "index", "idx")
    lower = [s.lower() for s in fields]
    for c in candidates:
        if c in lower:
            return lower.index(c)

    # Fallback: assume first column is time/index
    return 0


def dedup_by_time(data: np.ndarray, tcol: int, keep: str = "last") -> np.ndarray:
    """
    Deduplicate rows with same time index.
    keep = "first" or "last"
    """
    t = data[:, tcol]

    # Stable sort by time, then unique
    order = np.argsort(t, kind="mergesort")
    data_s = data[order]
    t_s = t[order]

    # Find unique times + which row to keep
    uniq_t, first_idx, counts = np.unique(t_s, return_index=True, return_counts=True)

    if keep == "first":
        keep_idx = first_idx
    elif keep == "last":
        keep_idx = first_idx + counts - 1
    else:
        raise ValueError("keep must be 'first' or 'last'")

    out = data_s[keep_idx]
    # out is already sorted by time (uniq_t ascending)
    return out


def align_on_common_time(
    tables: list[tuple[Path, list[str], np.ndarray, int]],
    mode: str = "intersection",
) -> tuple[np.ndarray, list[list[str]]]:
    """
    Align all tables on a common time vector.
    mode:
      - "intersection": keep only times present in ALL files (safest)
      - "union": keep all times (will insert NaNs where missing)  [optional, not implemented here]
    Returns:
      common_time (n,)
      aligned_data_per_file: list of arrays, each (n, ncols_file)
    """
    times = []
    for _, _, data, tcol in tables:
        times.append(data[:, tcol])

    if mode != "intersection":
        raise ValueError("Only mode='intersection' is implemented (recommended).")

    common = times[0]
    for t in times[1:]:
        common = np.intersect1d(common, t, assume_unique=True)

    # Select rows matching common times (data are sorted by time after dedup)
    aligned = []
    for path, fields, data, tcol in tables:
        t = data[:, tcol]
        idx = np.searchsorted(t, common)
        # Safety: confirm exact match
        if not np.all(t[idx] == common):
            raise ValueError(f"{path}: time alignment failed (unexpected missing times after intersect).")
        aligned.append(data[idx])

    return common, aligned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--x", required=True, help="Field name for X (e.g. d0)")
    ap.add_argument("--y", required=True, help="Field name for Y (e.g. cn)")
    ap.add_argument("--time-field", default=None,
                    help="Name of the time/index field in the header (default: auto-detect, else col0).")
    ap.add_argument("--keep", choices=("first", "last"), default="last",
                    help="When duplicate times exist in a file, keep the first or last occurrence.")
    args = ap.parse_args()

    files = [Path(f) for f in args.files]
    for p in files:
        if not p.exists():
            raise SystemExit(f"file not found: {p}")

    # Load + dedup each file
    tables = []
    file_bias=None
    for p in files:
        fields = parse_fields_header(p)
        data = load_numeric_block(p)
        if 'COLVAR' in p.name:
            file_bias=p

        # Reconcile header length vs numeric columns
        if data.shape[1] < len(fields):
            fields = fields[: data.shape[1]]
        elif data.shape[1] > len(fields):
            fields = fields + [f"col{j}" for j in range(len(fields), data.shape[1])]

        tcol = pick_time_col(fields, args.time_field)

        before = data.shape[0]
        data = dedup_by_time(data, tcol=tcol, keep=args.keep)
        after = data.shape[0]

        dup = before - after
        if dup > 0:
            print(f"[dedup] {p.name}: removed {dup} duplicate rows by time (kept {args.keep})")
        print(f"[load ] {p.name}: rows={after} cols={data.shape[1]} time_col={fields[tcol]}(idx={tcol})")

        tables.append((p, fields, data, tcol))

    # Align on common time
    common_time, aligned_data = align_on_common_time(tables, mode="intersection")
    print(f"[align] common rows (intersection) = {common_time.size}")

    # Build column names and hstack, keeping time only once (from first file)
    all_cols = []
    blocks = []

    # Add time column once (as "time")
    p0, fields0, data0, tcol0 = tables[0]
    blocks.append(aligned_data[0][:, [tcol0]])
    all_cols.append("time")

    # Add all non-time columns, prefixed by file stem to avoid collisions
    for (p, fields, _, tcol), data in zip(tables, aligned_data):
        stem = p.stem
        for j, name in enumerate(fields):
            if j == tcol:
                continue
            all_cols.append(f"{stem}:{name}")
        blocks.append(np.delete(data, tcol, axis=1))

    DATA = np.hstack(blocks)
    name_to_col = {name: i for i, name in enumerate(all_cols)}
    print("[done ] DATA shape:", DATA.shape)
    # Find bias column inside the already aligned/deduped data
    bias = None
    
    for (p, fields, _, tcol), data in zip(tables, aligned_data):
        if p == file_bias:
            # adjust the candidate names if needed
            candidates = ["opes.bias", "bias", "rbias"]
            for name in candidates:
                if name in fields:
                    bcol = fields.index(name)
                    bias = data[:, bcol]
                    break
            if bias is None:
                raise SystemExit(
                    f"Could not find a bias column in {p.name}. "
                    f"Available fields: {fields}"
                )
            break
    
    if bias is None:
        raise SystemExit("Could not locate the bias file in processed tables.")
    
    # --- Remove rows containing NaNs ---
    mask = ~np.isnan(DATA).any(axis=1)
    n_removed = DATA.shape[0] - np.sum(mask)
    if n_removed > 0:
        print(f"[clean] removed {n_removed} rows containing NaNs")
    DATA = DATA[mask]
    
    print("[clean] DATA shape after NaN removal:", DATA.shape)

    # Resolve requested x/y: allow bare field name if unique across files
    def resolve(query: str) -> str:
        if query in name_to_col:
            return query
        matches = [k for k in name_to_col if k.endswith(f":{query}")]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise SystemExit(f"field '{query}' not found. Example keys: {', '.join(list(name_to_col)[:10])} ...")
        raise SystemExit(f"field '{query}' is ambiguous: {matches}. Use e.g. '{matches[0]}'.")

    xname = resolve(args.x)
    yname = resolve(args.y)

    X = DATA[:, [name_to_col[xname], name_to_col[yname]]]
    print(f"[X    ] using {xname} and {yname} -> X shape {X.shape}")

    # Now you can use X[:,0], X[:,1]

    mu = X.mean(axis=0)
    sig = X.std(axis=0, ddof=1)
    
    Z = (X - mu) / sig
    
    
    # choose pilot from plug in
    n, d = Z.shape
    temp=350
    kbT = temp*0.0083144621

    # remove nans
    bias = bias[mask]
    weights = np.exp(bias/kbT)
    weights /= np.sum(weights)
    
    n_eff = 1/np.sum(weights**2)
    
    factor = n_eff ** (-1.0 / (d + 4))  # n^{-1/6} for d=2
    g0 = factor                      # since std(Z)=1
    g_list = [1.0 * g0, 1.5 * g0, 2.0 * g0]
    print(f"n_eff = {n_eff}, using pilot g0 = {g0}")
    
    # Choose pilot and init IN STANDARDIZED UNITS.
    # In Z-space, typical scales are O(1), so h ~ 1e-2 ... 1e-1 is a reasonable starting point.
    H_z, h_z, res = optimize_diagH_scv_mc(
        Z,
        g=g0,                    # pilot bandwidth in Z-units (tune 0.05-0.3)
        M_pairs=10_000_000,
        h_init=0.01,               # initial h in Z-units
        h_bounds=(1e-4, 0.5),      # bounds in Z-units
        seed=123,
        maxiter=60,
    )
    
    # Transform bandwidths back to original units:
    h_x = h_z * sig
    H_x = np.diag(h_x * h_x)
    
    print("h_z (standardized units) =", h_z)
    print("h_x (original units)     =", h_x)
    print("H_x diag                 =", np.diag(H_x))
    print("SCV(H_z)                 =", res.fun)
    print("converged                =", res.success, res.message)

if __name__ == "__main__":
    main()


