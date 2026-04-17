import numpy as np
import pandas as pd
import pygimli as pg
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import csv
from sklearn.neighbors import NearestNeighbors

# =====================================================================
# 1. VARIABLES GLOBALES (remplies par get_profile_info)
# =====================================================================

x_range = None        # largeur du profil (m)
z_range_dipol = None  # profondeur de référence (~0.3 * x_range)
dx = None             # espacement moyen entre électrodes
n_sensors = None      # nombre d’électrodes


# =====================================================================
# 2. PARAMÈTRES GÉNÉRAUX (ROI / filtrage physique)
# =====================================================================

PARAMS = {
    "zmin": 0.5,          # profondeur min. pour l’analyse (si tu l’utilises)
    "zmax": None,         # sera mis à jour par get_profile_info()
    "roh_low": 9000.0,    # plage de résistivité typique basse (Ω·m)
    "roh_high": 12000.0,  # plage haute (Ω·m)
    "eps_factor": 1.2,    # facteur pour eps si tu t’en sers
    "min_samples": 8,     # min_samples de base pour DBSCAN
    "area_min": 0.5,      # surface min. d’un patch (si utilisé)
    "thickness_min": 1.0, # épaisseur min.
    "return_top_k": 1,    # nombre de patches à garder
}

# =====================================================================
# 3. PARAMÈTRES DE PONDÉRATION & FEATURES
# =====================================================================

WEIGHTS = {
    "w_cov": 0.20,   # poids de la qualité de coverage
    "w_rho": 0.50,   # poids de la résistivité relative (élevée)
    "w_prox": 0.30,  # poids de la proximité spatiale au seed
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Les poids doivent sommer à 1."

FEATURE_PARAMS = {
    # Coverage: intervalle (log10) pour normaliser en [0,1]
    "coverage_L": -4.0,  # borne basse log10(coverage)
    "coverage_U":  0.0,  # borne haute

    # Rho: percentiles (sur log10(rho)) pour normaliser en [0,1]
    "rho_p_lo": 5.0,
    "rho_p_hi": 98.0,

    # Proximité: échelle L (m) = fraction de la largeur en X
    "prox_L_frac": 0.10,  # 10% de l’étendue en X

    # Seuil minimal sur rho_rel pour présélection (avant core)
    "rho_keep_min": 0.7,

    # Présélection par score 'core' (garde le top X %)
    "core_keep_top_quantile": 0.30  # 0.30 => garde top 70%
}

DBSCAN_PARAMS = {
    # On normalise (z-score) (x,y) avant DBSCAN → eps est sans unité
    "eps": 0.8,
    "min_samples": 8,
    "metric": "euclidean",
}


# =====================================================================
# 4. MÉTA-DONNÉES PROFIL & TOPO
# =====================================================================

def get_profile_info(profile_name, metadata_path, survey_name):

    global x_range, z_range_dipol, dx, n_sensors, PARAMS

    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        i_survey = header.index("survey_name")
        i_pos = header.index("electrode_position")
        i_x = header.index("x")

        in_block = False
        last_row = None

        for row in reader:
            if not in_block:
                if row[i_survey] == survey_name:
                    in_block = True
                    last_row = row
            else:
                if row[i_survey] == survey_name:
                    last_row = row  # on garde la dernière du bloc
                else:
                    break

    n_sensors = int(round(float(last_row[i_pos])))
    x_range   = float(last_row[i_x])

    dx = x_range / (n_sensors - 1)
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
    Construit un interpolateur 1D f_topo(x) à partir du fichier topo.
    - filtre sur survey_name
    - gère les virgules en décimal
    - regroupe les x dupliqués en moyennant y
    """
    df = pd.read_csv(metadatapath)
    df.columns = df.columns.str.strip().str.lower()

    name_col = name_col.lower()
    x_col = x_col.lower()
    y_col = y_col.lower()

    # 1) vérif colonnes requises
    missing = [c for c in [name_col, x_col, y_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Colonnes manquantes dans {metadatapath}: {missing}\n"
            f"Colonnes présentes: {list(df.columns)}"
        )

    # 2) filtre survey
    dff = df[df[name_col] == survey_name].copy()
    if dff.empty:
        avail = (
            df[name_col].dropna().astype(str).value_counts().head(10).index.tolist()
        )
        raise ValueError(
            f"Aucune ligne pour survey_name='{survey_name}' dans {metadatapath}. "
            f"Exemples disponibles: {avail}"
        )

    # 3) conversion numérique robuste
    dff[x_col] = dff[x_col].astype(str).str.replace(",", ".", regex=False)
    dff[y_col] = dff[y_col].astype(str).str.replace(",", ".", regex=False)
    dff[x_col] = pd.to_numeric(dff[x_col], errors="coerce")
    dff[y_col] = pd.to_numeric(dff[y_col], errors="coerce")
    dff = dff[[x_col, y_col]].dropna()
    if dff.empty:
        raise ValueError(
            f"Après conversion numérique, pas de (x,y) valides pour '{survey_name}'."
        )

    # 4) si x dupliqués → moyenne de y
    dff = dff.groupby(x_col, as_index=False)[y_col].mean()

    # 5) tri + arrays
    dff = dff.sort_values(x_col)
    x_t = dff[x_col].to_numpy(dtype=float)
    y_t = dff[y_col].to_numpy(dtype=float)
    if x_t.size < 2:
        raise ValueError("Il faut au moins 2 points topo pour interpoler.")

    xmin, xmax = x_t.min(), x_t.max()

    def f_topo(xx):
        xx = np.asarray(xx, float)
        yy = np.interp(np.clip(xx, xmin, xmax), x_t, y_t)
        # extrapolation plate (bord = valeur la plus proche)
        yy = np.where(xx < xmin, y_t[0], yy)
        yy = np.where(xx > xmax, y_t[-1], yy)
        return yy

    return f_topo, x_t, y_t


# =====================================================================
# 5. MESH & GÉOMÉTRIE
# =====================================================================

def utils_mesh(data_inv, mesh_node, mesh_cell, survey_name, profile_name):

    # --- NODES ---
    nodes = pd.read_csv(mesh_node)
    nodes.columns = nodes.columns.str.strip().str.lower()
    nodes = nodes[nodes["profile_name"] == profile_name][
        ["mesh_point_num", "x", "z"]
    ].copy()
    nodes["mesh_point_num"] = nodes["mesh_point_num"].astype(int)

    # --- CELLS ---
    cells = pd.read_csv(mesh_cell)
    cells.columns = cells.columns.str.strip().str.lower()
    cells = cells[cells["profile_name"] == profile_name][
        ["cell_num", "cell_node_1", "cell_node_2", "cell_node_3"]
    ].copy()
    cells[["cell_num", "cell_node_1", "cell_node_2", "cell_node_3"]] = cells[
        ["cell_num", "cell_node_1", "cell_node_2", "cell_node_3"]
    ].astype(int)

    # --- VALS (rho, coverage) ---
    vals = pd.read_csv(data_inv)
    vals.columns = vals.columns.str.strip().str.lower()
    vals = vals[vals["survey_name"] == survey_name][
        ["mesh_cell_id", "resistivity", "coverage"]
    ].copy()
    vals["mesh_cell_id"] = vals["mesh_cell_id"].astype(int)

    # --- CONSTRUCTION DU MESH PYGIMLI ---
    mesh = pg.Mesh(2)
    id2node = {}

    # 1) créer les nœuds (x,z)
    for _, r in nodes.iterrows():
        nid = int(r["mesh_point_num"])
        x = float(r["x"])
        z = float(r["z"])   # ici ton "z" est déjà une altitude/profondeur
        id2node[nid] = mesh.createNode(pg.Pos(x, z))

    file_cell_ids_in_order = []
    skipped = 0

    # 2) créer les cellules (triangles)
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

    # 3) dictionnaire des valeurs pour ce survey
    rho_by_cell = {}  # cell_id -> (rho, cov)
    for _, r in vals.iterrows():
        rho_by_cell[int(r["mesh_cell_id"])] = (
            float(r["resistivity"]),
            float(r["coverage"]),
        )

    if skipped > 0:
        print(f"[utils_mesh] triangles ignorés (nœud manquant): {skipped}")

    return mesh, rho_by_cell, nodes, cells, file_cell_ids_in_order


def cell_centers_xyz(mesh):

    C = mesh.cellCenters()  # R3Vector
    xs = np.array([p.x() for p in C], float)
    ys = np.array([p.y() for p in C], float)
    return np.c_[xs, ys]


def rho_at_point(mesh, rho_by_cell, file_cell_ids_in_order, x_target, z_target):

    best = None  # (d2, mesh_idx)
    for ci, cell in enumerate(mesh.cells()):
        cx, cz = cell.center().x(), cell.center().y()  # en 2D PG: y = ton z
        d2 = (cx - x_target) ** 2 + (cz - z_target) ** 2

        fid = file_cell_ids_in_order[ci]  # id de cellule côté fichiers
        if fid not in rho_by_cell:
            continue
        if (best is None) or (d2 < best[0]):
            best = (d2, ci)

    if best is None:
        return None, None, None, None

    ci = best[1]
    fid = file_cell_ids_in_order[ci]
    rho, cov = rho_by_cell[fid]
    return rho, cov, ci, fid


def find_reference_point(model, mesh=None, centers=None):

    i = int(np.nanargmax(model))
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

    centers = cell_centers_xyz(mesh)
    xc, yc = centers[:, 0], centers[:, 1]

    # limite verticale (topo descendue)
    offset_m = float(offset_factor) * float(z_range_dipol)
    y_lim = f_topo(xc) - offset_m
    roi_mask_t = yc > y_lim  # True => cacher

    # bornes horizontales
    if x_range is None:
        x_range = np.nanmax(xc) - np.nanmin(xc)
    x0 = np.nanmin(xc)
    xmin = x0 + x_borne * x_range
    xmax = x0 + (1.0 - x_borne) * x_range

    # masque latéral (hors [xmin, xmax])
    roi_mask_t |= (xc < xmin) | (xc > xmax)

    return roi_mask_t, offset_m, xc, yc, y_lim, xmin, xmax


# =====================================================================
# 6. PRÉPARATION DU NUAGE POUR DBSCAN
# =====================================================================

def prepare_nuage(centers_xy: np.ndarray,
                  rho_linear: np.ndarray,
                  coverage_log10: np.ndarray,
                  weights: dict = WEIGHTS,
                  fparams: dict = FEATURE_PARAMS) -> dict:

    x = centers_xy[:, 0]

    # --- prox: seed et échelle L
    L_m = fparams["prox_L_frac"] * (np.nanmax(x) - np.nanmin(x) + 1e-12)

    # seed = cellule au rho max (si dispo), sinon barycentre
    if np.isfinite(rho_linear).any():
        i_max = int(np.nanargmax(rho_linear))
        seed_xy = (centers_xy[i_max, 0], centers_xy[i_max, 1])
    else:
        seed_xy = tuple(centers_xy.mean(axis=0))


    cov = np.array(coverage_log10, dtype=float)
    if np.nanmax(cov) > 0:
        print("[WARN] Coverage positif (>0 en log10). Ce n'est pas attendu — vérifiez l’unité.")
    Lc, Uc = fparams["coverage_L"], fparams["coverage_U"]
    cov_norm = np.clip((cov - Lc) / (Uc - Lc + 1e-12), 0.0, 1.0)
    cov_good = 1.0 - cov_norm  # 1=bon, 0=mauvais


    rho = np.where((rho_linear > 0) & np.isfinite(rho_linear), rho_linear, np.nan)
    lrho = np.log10(rho)
    lo = np.nanpercentile(lrho, fparams["rho_p_lo"])
    hi = np.nanpercentile(lrho, fparams["rho_p_hi"])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(lrho), np.nanmax(lrho)
    rho_rel = np.clip((lrho - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    # --- prox = exp(-d/L)
    d = np.linalg.norm(centers_xy - np.array(seed_xy)[None, :], axis=1)
    prox = np.exp(-d / max(L_m, 1e-9))
    prox = np.clip(prox, 0.0, 1.0)

    # --- core pondéré
    w_cov, w_rho, w_prox = weights["w_cov"], weights["w_rho"], weights["w_prox"]
    core = w_cov * cov_good + w_rho * rho_rel + w_prox * prox

    # --- PRÉSÉLECTION par rho_rel
    rho_thr = float(fparams.get("rho_keep_min", 0.70))
    cand = (rho_rel >= rho_thr) & np.isfinite(core)

    # fallback si trop peu de points
    if np.count_nonzero(cand) < 20:
        q = float(fparams["core_keep_top_quantile"])
        thr = np.nanquantile(core, q)
        cand = (core >= thr)

    sel_idx = np.where(cand)[0]


    pts = centers_xy[sel_idx, :]
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    std[std == 0] = 1.0


    fac_z = fparams.get("z_aniso_factor", 0.2)  # ex: 0.5 ou 0.3
    std_mod = std.copy()
    std_mod[1] *= fac_z  # 1 = axe z

    points_norm = (pts - mean) / std_mod

    print(f"[prepare] retenus (ρ>=thr ou core top) : {len(sel_idx)} points")

    return {
        "sel_idx": sel_idx,
        "points_norm": points_norm,
        "scaler_xy": {"mean": mean, "std": std},
        "core": core,
        "diag": {"seed_xy": seed_xy, "L_m": L_m},
    }


def auto_eps_kdist(points_norm: np.ndarray, min_samples: int, q: float = 0.90) -> float:

    if points_norm.shape[0] < max(min_samples, 3):
        return 0.20  # valeur par défaut si pas assez de points

    nbrs = NearestNeighbors(n_neighbors=min_samples, metric="euclidean").fit(points_norm)
    dists, _ = nbrs.kneighbors(points_norm)
    kth = np.sort(dists[:, -1])
    return float(np.quantile(kth, q))


def run_dbscan(points_norm: np.ndarray,
               eps: float = DBSCAN_PARAMS["eps"],
               min_samples: int = DBSCAN_PARAMS["min_samples"],
               metric: str = DBSCAN_PARAMS["metric"]) -> dict:


    if points_norm.size == 0:
        print("[dbscan] Aucun point à clusteriser.")
        return {"labels": np.array([], dtype=int), "clusters": [], "noise": 0}

    algo = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = algo.fit_predict(points_norm)

    clusters = []
    noise = 0
    if labels.size:
        noise = int(np.sum(labels == -1))
        labs = [int(l) for l in np.unique(labels) if l >= 0]
        clusters = [(l, int(np.sum(labels == l))) for l in labs]
        print(
            f"[dbscan] eps={eps:.3f} min_samples={min_samples} | "
            f"clusters={len(labs)} {clusters} | noise={noise}"
        )

    return {"labels": labels, "clusters": clusters, "noise": noise}


# =====================================================================
# 7. POST-TRAITEMENT DES LABELS & STATISTIQUES
# =====================================================================

def labels_to_mask(n_cells: int,
                   sel_idx: np.ndarray,
                   labels: np.ndarray,
                   core: np.ndarray,
                   keep: str = "all",      # "best" | "all" | "top2"
                   min_size: int = 10) -> dict:

    mask = np.ones(n_cells, dtype=bool)  # tout masqué au départ
    if labels.size == 0 or sel_idx.size == 0:
        print("[mask] Rien à garder (pas de points/labels).")
        return {"mask": mask, "kept_labels": []}

    labs = [int(l) for l in np.unique(labels) if l >= 0]
    if not labs:
        print("[mask] Seulement du bruit (-1).")
        return {"mask": mask, "kept_labels": []}

    scores = []
    for l in labs:
        idx_local = np.where(labels == l)[0]
        taille = len(idx_local)# indices dans sel_idx

        print(f"   -> Cluster {l} : {taille} cellules.", end=" ")
        if len(idx_local) < min_size:
            print(f"=> REJETÉ (Trop petit < {min_size})")
            continue
        idx_cells = sel_idx[idx_local]        # indices de cellules
        score = float(np.nanmedian(core[idx_cells])) * np.log1p(len(idx_local))
        scores.append((l, score, len(idx_local)))

    if not scores:
        print("[mask] Clusters trop petits (< min_size).")
        return {"mask": mask, "kept_labels": []}

    scores.sort(key=lambda t: t[1], reverse=True)

    if keep == "all":
        kept = [l for (l, _, _) in scores]
    elif keep == "top2":
        kept = [l for (l, _, _) in scores[:2]]
    else:
        kept = [scores[0][0]]  # best

    print(f"=> REJETÉ (Trop petit < {min_size})")

    keep_cells = np.zeros(n_cells, dtype=bool)
    for l in kept:
        idx_local = np.where(labels == l)[0]
        keep_cells[sel_idx[idx_local]] = True

    mask = ~keep_cells
    msg = ", ".join([f"#{l}" for l in kept])
    print(f"[mask] Gardé(s) cluster(s): {msg} | cellules gardées={keep_cells.sum()} / {n_cells}")
    return {"mask": mask, "kept_labels": kept}


def summarize_rho(model,
                  keep_cells,
                  centers_xyz,
                  coverage_log10=None,
                  cell_areas=None):

    model = np.asarray(model)
    keep_cells = np.asarray(keep_cells, bool)
    centers_xyz = np.asarray(centers_xyz)

    used = keep_cells & np.isfinite(model)
    n = int(used.sum())

    if n == 0:
        return {
            "n_cells": 0,
            "mean_rho": np.nan,
            "median_rho": np.nan,
            "weighted_mean_rho": np.nan,
            "std_rho": np.nan,
            "p10_rho": np.nan,
            "p90_rho": np.nan,
            "span_x": 0.0,
            "span_z": 0.0,
            "span_xy": 0.0,
            "surface_m2": 0.0,
            "compacity": np.nan,
        }

    # ===============================================================
    # 1) RÉSISTIVITÉ
    # ===============================================================
    rho_vals = model[used]

    mean_rho   = float(np.mean(rho_vals))
    median_rho = float(np.median(rho_vals))
    std_rho    = float(np.std(rho_vals))
    p10_rho    = float(np.percentile(rho_vals, 10))
    p90_rho    = float(np.percentile(rho_vals, 90))

    # 1b) moyenne pondérée par coverage
    weighted_mean_rho = np.nan
    if coverage_log10 is not None:
        cov = np.asarray(coverage_log10)[used]

        w = 1.0 / (1.0 + np.power(10.0, cov))
        if np.sum(w) > 0:
            weighted_mean_rho = float(np.sum(w * rho_vals) / np.sum(w))

    # ===============================================================
    # 2) GÉOMÉTRIE (span_x, span_z, span_xy)
    # ===============================================================
    xs = centers_xyz[used, 0]
    zs = centers_xyz[used, 1]

    span_x  = float(xs.max() - xs.min())
    span_z  = float(zs.max() - zs.min())
    span_xy = float(np.hypot(span_x, span_z))

    # ===============================================================
    # 3) SURFACE RÉELLE
    # ===============================================================
    surface_m2 = np.nan
    if cell_areas is not None:
        cell_areas = np.asarray(cell_areas)
        surface_m2 = float(cell_areas[used].sum())

    # ===============================================================
    # 4) COMPACITÉ (horizontalité/verticalité)
    # ===============================================================
    compacity = np.nan
    if span_z > 0:
        compacity = float(span_x / span_z)

    return {
        "n_cells": n,
        "mean_rho": mean_rho,
        "median_rho": median_rho,
        "weighted_mean_rho": weighted_mean_rho,
        "std_rho": std_rho,
        "p10_rho": p10_rho,
        "p90_rho": p90_rho,
        "span_x": span_x,
        "span_z": span_z,
        "span_xy": span_xy,
        "surface_m2": surface_m2,
        "compacity": compacity,
    }