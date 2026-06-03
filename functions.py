import numpy as np
import pandas as pd
import pygimli as pg
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import csv
from sklearn.neighbors import NearestNeighbors


# =====================================================================
# 1. GLOBAL VARIABLES (populated by get_profile_info)
# =====================================================================

x_range       = None  # profile width (m)
z_range_dipol = None  # reference depth (~0.3 * x_range, dipole-dipole standard)
dx            = None  # mean electrode spacing
n_sensors     = None  # number of electrodes


# =====================================================================
# 2. GENERAL PARAMETERS (ROI / physical filtering)
# =====================================================================

PARAMS = {
    "zmin":         0.5,
    "zmax":         None,      # updated by get_profile_info()
    "roh_low":      9000.0,
    "roh_high":     12000.0,
    "eps_factor":   1.2,
    "min_samples":  8,
    "area_min":     0.5,
    "thickness_min":1.0,
    "return_top_k": 1,
}

# =====================================================================
# 3. WEIGHTING & FEATURE PARAMETERS
# =====================================================================

WEIGHTS = {
    "w_cov":  0.20,  # coverage quality weight
    "w_rho":  0.50,  # relative resistivity weight
    "w_prox": 0.30,  # spatial proximity to seed weight
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1."

FEATURE_PARAMS = {
    "coverage_L":           -4.0,  # log10(coverage) lower bound for normalization
    "coverage_U":            0.0,  # log10(coverage) upper bound
    "rho_p_lo":              5.0,  # lower percentile for log10(rho) normalization
    "rho_p_hi":             98.0,  # upper percentile
    "prox_L_frac":           0.10, # proximity length scale as fraction of x extent
    "rho_keep_min":          0.7,  # minimum normalized rho for pre-selection
    "core_keep_top_quantile":0.30, # fallback: keep top (1 - q) % by core score
}

DBSCAN_PARAMS = {
    "eps":         0.8,
    "min_samples": 8,
    "metric":      "euclidean",
}


# =====================================================================
# 4. PROFILE METADATA & TOPOGRAPHY
# =====================================================================

def get_profile_info(profile_name, metadata_path, survey_name):
    """Read electrode geometry from metadata CSV and update global profile variables."""

    global x_range, z_range_dipol, dx, n_sensors, PARAMS

    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        i_survey = header.index("survey_name")
        i_pos    = header.index("electrode_position")
        i_x      = header.index("x")

        in_block = False
        last_row = None

        for row in reader:
            if not in_block:
                if row[i_survey] == survey_name:
                    in_block = True
                    last_row = row
            else:
                if row[i_survey] == survey_name:
                    last_row = row
                else:
                    break

    n_sensors     = int(round(float(last_row[i_pos])))
    x_range       = float(last_row[i_x])
    dx            = x_range / (n_sensors - 1)
    z_range_dipol = 0.3 * x_range
    PARAMS["zmax"] = z_range_dipol

    return n_sensors, dx, x_range, z_range_dipol


def _make_topo_interpolator_from_data(
    metadatapath,
    survey_name,
    name_col="survey_name",
    x_col="x",
    y_col="y",
):
    """
    Build a 1D interpolator f_topo(x) from the topography file.
    Duplicate x values are averaged. Extrapolation is flat (nearest edge value).
    """
    df = pd.read_csv(metadatapath)
    df.columns = df.columns.str.strip().str.lower()

    name_col = name_col.lower()
    x_col    = x_col.lower()
    y_col    = y_col.lower()

    missing = [c for c in [name_col, x_col, y_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in {metadatapath}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    dff = df[df[name_col] == survey_name].copy()
    if dff.empty:
        avail = df[name_col].dropna().astype(str).value_counts().head(10).index.tolist()
        raise ValueError(
            f"No rows for survey_name='{survey_name}' in {metadatapath}. "
            f"Available examples: {avail}"
        )

    # Robust numeric conversion (handles comma-as-decimal)
    dff[x_col] = dff[x_col].astype(str).str.replace(",", ".", regex=False)
    dff[y_col] = dff[y_col].astype(str).str.replace(",", ".", regex=False)
    dff[x_col] = pd.to_numeric(dff[x_col], errors="coerce")
    dff[y_col] = pd.to_numeric(dff[y_col], errors="coerce")
    dff = dff[[x_col, y_col]].dropna()
    if dff.empty:
        raise ValueError(f"No valid (x, y) pairs after numeric conversion for '{survey_name}'.")

    dff = dff.groupby(x_col, as_index=False)[y_col].mean()
    dff = dff.sort_values(x_col)
    x_t = dff[x_col].to_numpy(dtype=float)
    y_t = dff[y_col].to_numpy(dtype=float)
    if x_t.size < 2:
        raise ValueError("At least 2 topo points are required for interpolation.")

    xmin, xmax = x_t.min(), x_t.max()

    def f_topo(xx):
        xx = np.asarray(xx, float)
        yy = np.interp(np.clip(xx, xmin, xmax), x_t, y_t)
        yy = np.where(xx < xmin, y_t[0],  yy)
        yy = np.where(xx > xmax, y_t[-1], yy)
        return yy

    return f_topo, x_t, y_t


# =====================================================================
# 5. MESH & GEOMETRY
# =====================================================================

def utils_mesh(data_inv, mesh_node, mesh_cell, survey_name, profile_name):
    """Reconstruct a pyGIMLi mesh from CSV node/cell/inversion files."""

    nodes = pd.read_csv(mesh_node)
    nodes.columns = nodes.columns.str.strip().str.lower()
    nodes = nodes[nodes["profile_name"] == profile_name][
        ["mesh_point_num", "x", "z"]
    ].copy()
    nodes["mesh_point_num"] = nodes["mesh_point_num"].astype(int)

    cells = pd.read_csv(mesh_cell)
    cells.columns = cells.columns.str.strip().str.lower()
    cells = cells[cells["profile_name"] == profile_name][
        ["cell_num", "cell_node_1", "cell_node_2", "cell_node_3"]
    ].copy()
    cells[["cell_num", "cell_node_1", "cell_node_2", "cell_node_3"]] = cells[
        ["cell_num", "cell_node_1", "cell_node_2", "cell_node_3"]
    ].astype(int)

    vals = pd.read_csv(data_inv)
    vals.columns = vals.columns.str.strip().str.lower()
    vals = vals[vals["survey_name"] == survey_name][
        ["mesh_cell_id", "resistivity", "coverage"]
    ].copy()
    vals["mesh_cell_id"] = vals["mesh_cell_id"].astype(int)

    mesh = pg.Mesh(2)
    id2node = {}

    for _, r in nodes.iterrows():
        nid      = int(r["mesh_point_num"])
        id2node[nid] = mesh.createNode(pg.Pos(float(r["x"]), float(r["z"])))

    file_cell_ids_in_order = []
    skipped = 0

    for _, r in cells.iterrows():
        try:
            n1 = id2node[int(r["cell_node_1"])]
            n2 = id2node[int(r["cell_node_2"])]
            n3 = id2node[int(r["cell_node_3"])]
        except KeyError:
            skipped += 1
            continue
        mesh.createTriangle(n1, n2, n3)
        file_cell_ids_in_order.append(int(r["cell_num"]))

    mesh.createNeighborInfos()

    rho_by_cell = {
        int(r["mesh_cell_id"]): (float(r["resistivity"]), float(r["coverage"]))
        for _, r in vals.iterrows()
    }

    if skipped > 0:
        print(f"[utils_mesh] Triangles skipped (missing node): {skipped}")

    return mesh, rho_by_cell, nodes, cells, file_cell_ids_in_order


def cell_centers_xyz(mesh):
    """Return (N, 2) array of cell centroid coordinates."""
    C  = mesh.cellCenters()
    xs = np.array([p.x() for p in C], float)
    ys = np.array([p.y() for p in C], float)
    return np.c_[xs, ys]


def rho_at_point(mesh, rho_by_cell, file_cell_ids_in_order, x_target, z_target):
    """Return resistivity and coverage of the cell closest to (x_target, z_target)."""
    best = None
    for ci, cell in enumerate(mesh.cells()):
        cx, cz = cell.center().x(), cell.center().y()
        d2  = (cx - x_target) ** 2 + (cz - z_target) ** 2
        fid = file_cell_ids_in_order[ci]
        if fid not in rho_by_cell:
            continue
        if (best is None) or (d2 < best[0]):
            best = (d2, ci)

    if best is None:
        return None, None, None, None

    ci  = best[1]
    fid = file_cell_ids_in_order[ci]
    rho, cov = rho_by_cell[fid]
    return rho, cov, ci, fid


def find_reference_point(model, mesh=None, centers=None):
    """Return position and index of the cell with maximum resistivity."""
    i   = int(np.nanargmax(model))
    out = {"cell_id": i, "rho_max": float(model[i])}

    if centers is None and mesh is not None:
        centers = cell_centers_xyz(mesh)
    if centers is not None:
        x, y = centers[i]
        out["x"] = float(x)
        out["y"] = float(y)

    return out


def roi_mask_topo(
    mesh,
    f_topo,
    z_range_dipol,
    offset_factor=0.07,
    x_window=None,
    x_range=None,
    x_borne=0.05,
):
    """
    Build a boolean mask (True = excluded) based on topography and lateral bounds.
    Cells above the topographic surface minus offset, or outside [xmin, xmax], are masked.
    """
    centers = cell_centers_xyz(mesh)
    xc, yc  = centers[:, 0], centers[:, 1]

    offset_m    = float(offset_factor) * float(z_range_dipol)
    y_lim       = f_topo(xc) - offset_m
    roi_mask_t  = yc > y_lim  # True = above surface limit → exclude

    if x_range is None:
        x_range = np.nanmax(xc) - np.nanmin(xc)
    x0   = np.nanmin(xc)
    xmin = x0 + x_borne * x_range
    xmax = x0 + (1.0 - x_borne) * x_range
    roi_mask_t |= (xc < xmin) | (xc > xmax)

    return roi_mask_t, offset_m, xc, yc, y_lim, xmin, xmax


# =====================================================================
# 6. POINT CLOUD PREPARATION FOR DBSCAN
# =====================================================================

def prepare_nuage(centers_xy: np.ndarray,
                  rho_linear: np.ndarray,
                  coverage_log10: np.ndarray,
                  weights: dict = WEIGHTS,
                  fparams: dict = FEATURE_PARAMS) -> dict:
    """
    Build the normalized feature space used by DBSCAN.
    Three signals are combined: coverage quality, relative resistivity, proximity to seed.
    Vertical axis is compressed by z_aniso_factor to favor horizontally-dominant bodies.
    """
    x   = centers_xy[:, 0]
    L_m = fparams["prox_L_frac"] * (np.nanmax(x) - np.nanmin(x) + 1e-12)

    # Seed at the highest-resistivity cell; fall back to centroid if no valid rho
    if np.isfinite(rho_linear).any():
        i_max    = int(np.nanargmax(rho_linear))
        seed_xy  = (centers_xy[i_max, 0], centers_xy[i_max, 1])
    else:
        seed_xy  = tuple(centers_xy.mean(axis=0))

    # Coverage: invert so that 1 = reliable, 0 = unreliable
    cov = np.array(coverage_log10, dtype=float)
    if np.nanmax(cov) > 0:
        print("[WARNING] Positive coverage (>0 in log10) — check units.")
    Lc, Uc   = fparams["coverage_L"], fparams["coverage_U"]
    cov_norm = np.clip((cov - Lc) / (Uc - Lc + 1e-12), 0.0, 1.0)
    cov_good = 1.0 - cov_norm

    # Resistivity: normalize log10(rho) to [0, 1] using robust percentiles
    rho  = np.where((rho_linear > 0) & np.isfinite(rho_linear), rho_linear, np.nan)
    lrho = np.log10(rho)
    lo   = np.nanpercentile(lrho, fparams["rho_p_lo"])
    hi   = np.nanpercentile(lrho, fparams["rho_p_hi"])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(lrho), np.nanmax(lrho)
    rho_rel = np.clip((lrho - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    # Proximity: exponential decay from seed
    d    = np.linalg.norm(centers_xy - np.array(seed_xy)[None, :], axis=1)
    prox = np.exp(-d / max(L_m, 1e-9))
    prox = np.clip(prox, 0.0, 1.0)

    w_cov, w_rho, w_prox = weights["w_cov"], weights["w_rho"], weights["w_prox"]
    core = w_cov * cov_good + w_rho * rho_rel + w_prox * prox

    # Pre-selection: keep cells above rho threshold; fall back to top core quantile
    rho_thr = float(fparams.get("rho_keep_min", 0.70))
    cand    = (rho_rel >= rho_thr) & np.isfinite(core)
    if np.count_nonzero(cand) < 20:
        q    = float(fparams["core_keep_top_quantile"])
        thr  = np.nanquantile(core, q)
        cand = (core >= thr)

    sel_idx = np.where(cand)[0]
    pts     = centers_xy[sel_idx, :]
    mean    = pts.mean(axis=0)
    std     = pts.std(axis=0)
    std[std == 0] = 1.0

    # Vertical anisotropy: stretch z distances to penalize deep fragmented clusters
    fac_z      = fparams.get("z_aniso_factor", 0.2)
    std_mod    = std.copy()
    std_mod[1] *= fac_z
    points_norm = (pts - mean) / std_mod

    print(f"[prepare] Selected points (rho >= thr or top core): {len(sel_idx)}")

    return {
        "sel_idx":    sel_idx,
        "points_norm":points_norm,
        "scaler_xy":  {"mean": mean, "std": std},
        "core":       core,
        "diag":       {"seed_xy": seed_xy, "L_m": L_m},
    }


def auto_eps_kdist(points_norm: np.ndarray, min_samples: int, q: float = 0.90) -> float:
    """Estimate DBSCAN eps from the k-distance graph at quantile q."""
    if points_norm.shape[0] < max(min_samples, 3):
        return 0.20  # default when too few points

    nbrs      = NearestNeighbors(n_neighbors=min_samples, metric="euclidean").fit(points_norm)
    dists, _  = nbrs.kneighbors(points_norm)
    kth       = np.sort(dists[:, -1])
    return float(np.quantile(kth, q))


def run_dbscan(points_norm: np.ndarray,
               eps:         float = DBSCAN_PARAMS["eps"],
               min_samples: int   = DBSCAN_PARAMS["min_samples"],
               metric:      str   = DBSCAN_PARAMS["metric"]) -> dict:
    """Run DBSCAN and return labels, cluster list, and noise count."""

    if points_norm.size == 0:
        print("[dbscan] No points to cluster.")
        return {"labels": np.array([], dtype=int), "clusters": [], "noise": 0}

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(points_norm)

    noise    = int(np.sum(labels == -1))
    labs     = [int(l) for l in np.unique(labels) if l >= 0]
    clusters = [(l, int(np.sum(labels == l))) for l in labs]
    print(
        f"[dbscan] eps={eps:.3f} min_samples={min_samples} | "
        f"clusters={len(labs)} {clusters} | noise={noise}"
    )

    return {"labels": labels, "clusters": clusters, "noise": noise}


# =====================================================================
# 7. LABEL POST-PROCESSING & STATISTICS
# =====================================================================

def labels_to_mask(n_cells:   int,
                   sel_idx:   np.ndarray,
                   labels:    np.ndarray,
                   core:      np.ndarray,
                   keep:      str = "all",  # "best" | "all" | "top2"
                   min_size:  int = 10) -> dict:
    """
    Convert DBSCAN labels to a boolean cell mask (False = kept).
    keep="all"  : merge all significant clusters (recommended for fragmented zones)
    keep="best" : retain only the highest-scoring cluster
    keep="top2" : retain the two highest-scoring clusters
    """
    mask = np.ones(n_cells, dtype=bool)
    if labels.size == 0 or sel_idx.size == 0:
        print("[mask] Nothing to keep (no points or labels).")
        return {"mask": mask, "kept_labels": []}

    labs = [int(l) for l in np.unique(labels) if l >= 0]
    if not labs:
        print("[mask] Only noise (-1).")
        return {"mask": mask, "kept_labels": []}

    scores = []
    for l in labs:
        idx_local = np.where(labels == l)[0]
        taille    = len(idx_local)
        print(f"   -> Cluster {l} : {taille} cells.", end=" ")
        if len(idx_local) < min_size:
            print(f"=> REJECTED (too small < {min_size})")
            continue
        idx_cells = sel_idx[idx_local]
        score     = float(np.nanmedian(core[idx_cells])) * np.log1p(len(idx_local))
        scores.append((l, score, len(idx_local)))

    if not scores:
        print("[mask] All clusters too small (< min_size).")
        return {"mask": mask, "kept_labels": []}

    scores.sort(key=lambda t: t[1], reverse=True)

    if keep == "all":
        kept = [l for (l, _, _) in scores]
    elif keep == "top2":
        kept = [l for (l, _, _) in scores[:2]]
    else:
        kept = [scores[0][0]]

    print(f"=> REJECTED (too small < {min_size})")

    keep_cells = np.zeros(n_cells, dtype=bool)
    for l in kept:
        idx_local = np.where(labels == l)[0]
        keep_cells[sel_idx[idx_local]] = True

    mask = ~keep_cells
    msg  = ", ".join([f"#{l}" for l in kept])
    print(f"[mask] Kept cluster(s): {msg} | cells kept={keep_cells.sum()} / {n_cells}")
    return {"mask": mask, "kept_labels": kept}


def summarize_rho(model,
                  keep_cells,
                  centers_xyz,
                  coverage_log10=None,
                  cell_areas=None):
    """Compute resistivity statistics and geometric indicators for the detected cluster."""

    model       = np.asarray(model)
    keep_cells  = np.asarray(keep_cells, bool)
    centers_xyz = np.asarray(centers_xyz)

    used = keep_cells & np.isfinite(model)
    n    = int(used.sum())

    if n == 0:
        return {
            "n_cells": 0, "mean_rho": np.nan, "median_rho": np.nan,
            "weighted_mean_rho": np.nan, "std_rho": np.nan,
            "p10_rho": np.nan, "p90_rho": np.nan,
            "span_x": 0.0, "span_z": 0.0, "span_xy": 0.0,
            "surface_m2": 0.0, "compacity": np.nan,
        }

    rho_vals = model[used]
    mean_rho   = float(np.mean(rho_vals))
    median_rho = float(np.median(rho_vals))
    std_rho    = float(np.std(rho_vals))
    p10_rho    = float(np.percentile(rho_vals, 10))
    p90_rho    = float(np.percentile(rho_vals, 90))

    # Coverage-weighted mean: weight = 1 / (1 + 10^cov), penalizes low-quality cells
    weighted_mean_rho = np.nan
    if coverage_log10 is not None:
        cov = np.asarray(coverage_log10)[used]
        w   = 1.0 / (1.0 + np.power(10.0, cov))
        if np.sum(w) > 0:
            weighted_mean_rho = float(np.sum(w * rho_vals) / np.sum(w))

    xs      = centers_xyz[used, 0]
    zs      = centers_xyz[used, 1]
    span_x  = float(xs.max() - xs.min())
    span_z  = float(zs.max() - zs.min())
    span_xy = float(np.hypot(span_x, span_z))

    surface_m2 = np.nan
    if cell_areas is not None:
        surface_m2 = float(np.asarray(cell_areas)[used].sum())

    compacity = float(span_x / span_z) if span_z > 0 else np.nan

    return {
        "n_cells": n, "mean_rho": mean_rho, "median_rho": median_rho,
        "weighted_mean_rho": weighted_mean_rho, "std_rho": std_rho,
        "p10_rho": p10_rho, "p90_rho": p90_rho,
        "span_x": span_x, "span_z": span_z, "span_xy": span_xy,
        "surface_m2": surface_m2, "compacity": compacity,
    }
