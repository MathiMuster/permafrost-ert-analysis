import numpy as np
import pandas as pd
import os
from datetime import datetime
import pygimli as pg
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import sys

from functions import (
    get_profile_info,
    find_reference_point,
    utils_mesh,
    rho_at_point,
    cell_centers_xyz,
    _make_topo_interpolator_from_data,
    roi_mask_topo,
    prepare_nuage,
    run_dbscan,
    labels_to_mask,
    summarize_rho,
    auto_eps_kdist,
)

# ==============================
# GLOBAL PARAMETERS
# ==============================

PARAMS = {
    "paths": {
        "metadata":  "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/topo.txt",
        "mesh_cell": "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/mesh_cell.txt",
        "mesh_node": "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/mesh_node.txt",
        "data_inv":  "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/data_inv.txt",
    },
    "roi": {
        "offset_factor": 0.07,  # fraction of z_range_dipol subtracted from topography
        "x_borne": 0.03,        # lateral edge exclusion as fraction of profile length
    },
    "dbscan": {
        "min_samples": 8,
        "q": 0.90,              # quantile for auto_eps_kdist
        "metric": "euclidean",
        "keep": "all",          # "best" | "top2" | "all"
        "min_size": 8,
    },
    # Required by rho_at_point — removing it causes a crash
    "test_point": {
        "x_target": 150.0,
        "z_target": -40.0,
    },
}


def main():

    if len(sys.argv) > 2:
        profile_name = sys.argv[1]
        survey_name  = sys.argv[2]
        keep         = sys.argv[3] if len(sys.argv) > 3 else "all"
    else:
        print("[ERROR] Usage: python main.py <profile_name> <survey_name> [keep]")
        sys.exit(1)

    print(f"=== Running: {profile_name} | {survey_name} ===")

    paths    = PARAMS["paths"]
    roi_cfg  = PARAMS["roi"]
    db_cfg   = PARAMS["dbscan"]
    test_pt  = PARAMS["test_point"]

    # 1) Profile geometry
    try:
        n_sensors, dx, x_range, z_range_dipol = get_profile_info(
            profile_name=profile_name,
            metadata_path=paths["metadata"],
            survey_name=survey_name,
        )
    except ValueError as e:
        print("[ERROR]", e)
        return

    # 2) Mesh + inverted data
    mesh, rho_by_cell, nodes_df, cells_df, file_cell_ids_in_order = utils_mesh(
        data_inv=paths["data_inv"],
        mesh_node=paths["mesh_node"],
        mesh_cell=paths["mesh_cell"],
        survey_name=survey_name,
        profile_name=profile_name,
    )

    # 3) Topography interpolator
    f_topo, x_t, y_t = _make_topo_interpolator_from_data(
        metadatapath=paths["metadata"],
        survey_name=survey_name,
    )
    print(f"[OK] Topo interpolator built with {len(x_t)} points.")

    # 4) ROI mask (topography + lateral bounds)
    roi_mask_t, offset_m, xc, yc, y_lim, xmin, xmax = roi_mask_topo(
        mesh, f_topo, z_range_dipol,
        offset_factor=roi_cfg["offset_factor"],
        x_range=x_range,
        x_borne=roi_cfg["x_borne"],
    )
    print(f"Left bound  : {xmin:.2f} m")
    print(f"Right bound : {xmax:.2f} m")
    print(f"ROI offset  : {offset_m:.2f} m ({100 * roi_cfg['offset_factor']:.0f}% of z_range_dipol)")

    # 5) Resistivity and coverage arrays (one value per mesh cell)
    N              = mesh.cellCount()
    model          = np.full(N, np.nan, dtype=float)
    coverage_log10 = np.full(N, np.nan, dtype=float)
    COVERAGE_IS_LOG10 = True  # coverage stored as log10 in the database

    for ci, fid in enumerate(file_cell_ids_in_order):
        pair = rho_by_cell.get(fid)
        if pair is not None:
            rho, cov = pair
            model[ci]          = float(rho)
            coverage_log10[ci] = (
                float(cov) if COVERAGE_IS_LOG10
                else np.log10(max(float(cov), 1e-12))
            )

    # 6) Sanity check at a fixed test point
    rho, cov, ci, fid = rho_at_point(
        mesh, rho_by_cell, file_cell_ids_in_order,
        x_target=test_pt["x_target"],
        z_target=test_pt["z_target"],
    )
    print(f"rho={rho}  coverage={cov}  mesh_idx={ci}  file_id={fid}")

    # 7) Cell centers + reference point
    XYZ  = cell_centers_xyz(mesh)
    info = find_reference_point(model, centers=XYZ)

    # 8) Geometric ROI (topography + lateral bounds)
    XYZ  = cell_centers_xyz(mesh)
    x_c, y_c = XYZ[:, 0], XYZ[:, 1]

    y_lim_cells  = f_topo(x_c) - offset_m
    under_topo   = (y_c <= y_lim_cells)
    inside_x     = (x_c >= xmin) & (x_c <= xmax)
    mask_roi_geom = under_topo & inside_x

    print(f"[DEBUG] Total mesh cells       : {len(XYZ)}")
    print(f"[DEBUG] Cells in geometric ROI : {mask_roi_geom.sum()}")

    # 9) Prepare point cloud for DBSCAN (NaN outside geometric ROI)
    rho_for_cloud = model.copy()
    cov_for_cloud = coverage_log10.copy()
    rho_for_cloud[~mask_roi_geom] = np.nan
    cov_for_cloud[~mask_roi_geom] = np.nan

    cloud = prepare_nuage(
        centers_xy=XYZ,
        rho_linear=rho_for_cloud,
        coverage_log10=cov_for_cloud,
    )
    print(f"[DEBUG] Points selected for DBSCAN: {len(cloud['sel_idx'])}")

    # 10) DBSCAN
    res   = run_dbscan(cloud["points_norm"])
    final = labels_to_mask(
        n_cells=mesh.cellCount(),
        sel_idx=cloud["sel_idx"],
        labels=res["labels"],
        core=cloud["core"],
        keep="all",
        min_size=10,
    )
    roi_mask_dbscan = final["mask"]
    kept_labels     = final["kept_labels"]
    print(f"[summary] Kept clusters: {kept_labels} | DBSCAN cells kept: {(~roi_mask_dbscan).sum()}")

    # 11) Final ROI = geometric ROI ∩ DBSCAN
    keep_db       = ~roi_mask_dbscan
    keep_final    = mask_roi_geom & keep_db
    mask_drop_final = ~keep_final  # True = masked for pg.show

    # 12) Plot tomogram with ROI and clusters
    try:
        ax, cb = pg.show(
            mesh,
            data=model,
            mask=mask_drop_final,
            logScale=True,
            showMesh=True,
            cMap="Spectral_r",
        )
    except ValueError as e:
        print(f"[WARNING] pg.show failed for {profile_name} | {survey_name}: {e}")
        print("[WARNING] Skipping plot — summary and CSV are unaffected.")
        ax = None
        cb = None

    x_line = np.linspace(x_c.min(), x_c.max(), 400)
    ax.plot(x_line, f_topo(x_line) - offset_m,
            'k--', lw=1.2, label=f'ROI topo -{offset_m:.2f} m')
    ax.plot(x_t, y_t, 'm.-', lw=1.2, label='topographie')
    ax.axvline(xmin, color='green', linestyle='--', lw=1.2, label=f'Borne gauche ({xmin:.1f} m)')
    ax.axvline(xmax, color='red',   linestyle='--', lw=1.2, label=f'Borne droite ({xmax:.1f} m)')

    sel_idx = cloud["sel_idx"]
    labels  = res["labels"]
    XY_sel  = XYZ[sel_idx]
    for lab in sorted(set(labels) - {-1}):
        m = (labels == lab)
        ax.scatter(XY_sel[m, 0], XY_sel[m, 1], s=25, label=f"cluster {lab}", zorder=4)

    ax.set_title("ROI finale = Topo ∩ bornes X ∩ DBSCAN")
    ax.legend()
    fig_name = f"fig_{survey_name}.png"
    ax.figure.savefig(fig_name, dpi=300, bbox_inches="tight")
    print(f"[OK] Figure saved: {fig_name}")

    # 13) Summary statistics over final ROI
    cell_areas = np.array([c.size() for c in mesh.cells()])
    summary    = summarize_rho(
        model=model,
        keep_cells=keep_final,
        centers_xyz=XYZ,
        coverage_log10=coverage_log10,
        cell_areas=cell_areas,
    )

    print("\n=== ROI Summary (Topo ∩ X ∩ DBSCAN) ===")
    print(f"  n_cells         : {summary['n_cells']}  (expected: {keep_final.sum()})")
    print(f"  span XY         : {summary['span_xy']:.2f} m")
    print(f"  surface         : {summary['surface_m2']:.2f} m2")
    print(f"  weighted mean   : {summary['weighted_mean_rho']:.1f}")
    print(f"  % of mesh       : {100 * summary['n_cells'] / N:.1f}%")

    # 14) Append results to CSV
    result = {
        "survey_name":       survey_name,
        "profile_name":      profile_name,
        "year":              int(survey_name.split("_")[3][:4]),
        "date":              survey_name.split("_")[3],
        "n_cells":           summary["n_cells"],
        "mesh_total_cells":  N,
        "percent_used":      100 * summary["n_cells"] / N,
        "span_x":            summary["span_x"],
        "span_z":            summary["span_z"],
        "span_xy":           summary["span_xy"],
        "surface_m2":        summary["surface_m2"],
        "weighted_mean_rho": summary["weighted_mean_rho"],
        "median_rho":        summary["median_rho"],
        "std_rho":           summary["std_rho"],
    }

    out_csv = "summary_results.csv"
    if os.path.exists(out_csv):
        pd.concat(
            [pd.read_csv(out_csv), pd.DataFrame([result])],
            ignore_index=True,
        ).to_csv(out_csv, index=False)
    else:
        pd.DataFrame([result]).to_csv(out_csv, index=False)

    print(f"[OK] Results appended to {out_csv}")
    print("--- Done ---")


if __name__ == "__main__":
    main()
