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
# PARAMÈTRES GLOBAUX
# ==============================

PARAMS = {
    # chemins vers les fichiers
    "paths": {
        "metadata":  "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/topo.txt",
        "mesh_cell": "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/mesh_cell.txt",
        "mesh_node": "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/mesh_node.txt",
        "data_inv":  "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/data_inv.txt",
    },

    # réglages ROI
    "roi": {
        "offset_factor": 0.07,  # 10% de z_range_dipol sous la topo
        "x_borne": 0.03,        # enlève 5% de la longueur en x au début/à la fin
    },

    # réglages DBSCAN
    "dbscan": {
        "min_samples": 8,
        "q": 0.90,              # quantile pour auto_eps_kdist
        "metric": "euclidean",
        "keep": "all",         # "best" | "top2" | "all"
        "min_size": 8,         # taille min. de cluster
    },

    # point test (rho_at_point)
    #inutile parcontre sa bug sans ...
    "test_point": {
        "x_target": 150.0,
        "z_target": -40.0,
    },
}





def main():
    # ==============================
    # ARGUMENTS LIGNE DE COMMANDE
    # ==============================
    if len(sys.argv) > 2:
        profile_name = sys.argv[1]
        survey_name = sys.argv[2]
        keep = sys.argv[3] if len(sys.argv) > 3 else "all"
    else:
        print("❌ Usage: python main.py <profile_name> <survey_name> [keep]")
        sys.exit(1)

    print(f"=== Lancement pour {profile_name} | {survey_name} ===")

    paths = PARAMS["paths"]
    roi_cfg = PARAMS["roi"]
    db_cfg = PARAMS["dbscan"]
    test_pt = PARAMS["test_point"]

    # ==============================
    # 1) Paramètres de profil
    # ==============================
    try:
        n_sensors, dx, x_range, z_range_dipol = get_profile_info(
            profile_name=profile_name,
            metadata_path=paths["metadata"],
            survey_name=survey_name,
        )
    except ValueError as e:
        print("❌", e)
        return


    # ==============================
    # 2) Mesh + données inversées
    # ==============================
    mesh, rho_by_cell, nodes_df, cells_df, file_cell_ids_in_order = utils_mesh(
        data_inv=paths["data_inv"],
        mesh_node=paths["mesh_node"],
        mesh_cell=paths["mesh_cell"],
        survey_name=survey_name,
        profile_name=profile_name,
    )

    # ==============================
    # 3) Topographie & interpolateur
    # ==============================
    f_topo, x_t, y_t = _make_topo_interpolator_from_data(
        metadatapath=paths["metadata"],
        survey_name=survey_name,
    )
    print("✅ Interpolateur construit avec", len(x_t), "points de topographie.")

    # ==============================
    # 4) ROI topo + bornes X
    # ==============================
    roi_mask_t, offset_m, xc, yc, y_lim, xmin, xmax = roi_mask_topo(
        mesh, f_topo, z_range_dipol,
        offset_factor=roi_cfg["offset_factor"],
        x_range=x_range,
        x_borne=roi_cfg["x_borne"],
    )

    print(f"Borne gauche : {xmin:.2f} m")
    print(f"Borne droite : {xmax:.2f} m")
    print(
        f"ROI: topo descendue de {offset_m:.2f} m "
        f"(= {100 * roi_cfg['offset_factor']:.0f}% de z_range_dipol)"
    )

    # ==============================
    # 5) Construction du modèle ρ et coverage_log10
    # ==============================
    N = mesh.cellCount()
    model = np.full(N, np.nan, dtype=float)
    coverage_log10 = np.full(N, np.nan, dtype=float)
    COVERAGE_IS_LOG10 = True

    # file_cell_ids_in_order: mapping index PG -> id de cellule des fichiers
    for ci, fid in enumerate(file_cell_ids_in_order):
        pair = rho_by_cell.get(fid)  # (rho, cov) ou None
        if pair is not None:
            rho, cov = pair
            model[ci] = float(rho)
            coverage_log10[ci] = (
                float(cov)
                if COVERAGE_IS_LOG10
                else np.log10(max(float(cov), 1e-12))
            )

    # ==============================
    # 6) Point test (debug / sanity check)
    # ==============================
    rho, cov, ci, fid = rho_at_point(
        mesh,
        rho_by_cell,
        file_cell_ids_in_order,
        x_target=test_pt["x_target"],
        z_target=test_pt["z_target"],
    )
    print("ρ =", rho, "coverage =", cov, "mesh_idx =", ci, "file_id =", fid)

    # ==============================
    # 7) Centres de cellules & point de référence
    # ==============================
    XYZ = cell_centers_xyz(mesh)
    info = find_reference_point(model, centers=XYZ)


    # ==============================
    # 8) Préparation du nuage pour DBSCAN
    # ==============================
    cloud = prepare_nuage(
        centers_xy=XYZ,
        rho_linear=model,
        coverage_log10=coverage_log10,
    )

    pts = cloud["points_norm"]

    # auto eps via k-distance
    min_samples = db_cfg["min_samples"]
    eps = auto_eps_kdist(pts, min_samples=min_samples, q=db_cfg["q"])

    # ==============================
    # 9) DBSCAN
    # ==============================
    res = run_dbscan(
        points_norm=pts,
        eps=eps,
        min_samples=min_samples,
        metric=db_cfg["metric"],
    )

    # ==============================
    # 10) Labels -> masque DBSCAN
    # ==============================
    db_best = labels_to_mask(
        n_cells=N,
        sel_idx=cloud["sel_idx"],
        labels=res["labels"],
        core=cloud["core"],
        keep=db_cfg["keep"],
        min_size=db_cfg["min_size"]
    )
    roi_mask_dbscan = db_best["mask"]  # True = exclure
    kept_labels = db_best["kept_labels"]
    print(f"[summary] Clusters gardés: {kept_labels} | "
          f"cellules gardées={(~roi_mask_dbscan).sum()} / {N}")

    # ==============================
    # 11) Résumé tabulaire simple
    # ==============================
    row = {
        "profile_name": profile_name,
        "survey_name": survey_name,
        "n_sensors": n_sensors,
        "dx": dx,
        "x_range": x_range,
        "zmax": z_range_dipol,
        "xmin_roi": xmin,
        "xmax_roi": xmax,
        "n_cells_roi_db": int((~roi_mask_dbscan).sum()),
    }
    df = pd.DataFrame([row])
    print("\nRésumé :")
    print(df.to_string(index=False))

    # ==============================
    # 12) Préparation courbe limite pour l'affichage
    # ==============================
    x_line = np.linspace(x_t.min(), x_t.max(), 400)
    y_line = f_topo(x_line) - offset_m



    # ==============================
    # 8) Centres + ROI géométrique (topo + bornes X)
    # ==============================
    XYZ = cell_centers_xyz(mesh)
    x_c, y_c = XYZ[:, 0], XYZ[:, 1]

    y_lim_cells = f_topo(x_c) - offset_m
    under_topo = (y_c <= y_lim_cells)          # sous la topo descendue
    inside_x   = (x_c >= xmin) & (x_c <= xmax) # entre les bornes verticales

    mask_roi_geom = under_topo & inside_x      # ROI géométrique pure

    print(f"[DEBUG] Nombre total de cellules mesh : {len(XYZ)}")
    print(f"[DEBUG] Cellules dans ROI géométrique : {mask_roi_geom.sum()}")

    # ==============================
    # 9) Préparer nuage pour DBSCAN (NaN hors ROI)
    # ==============================
    rho_for_cloud = model.copy()
    cov_for_cloud = coverage_log10.copy()

    rho_for_cloud[~mask_roi_geom] = np.nan
    cov_for_cloud[~mask_roi_geom] = np.nan

    cloud = prepare_nuage(
        centers_xy=XYZ,
        rho_linear=rho_for_cloud,
        coverage_log10=cov_for_cloud,
    )

    print(f"[DEBUG] Points sélectionnés pour DBSCAN (sel_idx) : {len(cloud['sel_idx'])}")

    # ==============================
    # 10) DBSCAN sur CE nuage
    # ==============================
    res = run_dbscan(cloud["points_norm"])  # ta fonction Mo.run_dbscan gère eps/min_samples

    final = labels_to_mask(
        n_cells=mesh.cellCount(),
        sel_idx=cloud["sel_idx"],
        labels=res["labels"],
        core=cloud["core"],
        keep="all",     # "best" | "top2" | "all"
        min_size=10,
    )
    roi_mask_dbscan = final["mask"]      # True = exclure
    kept_labels     = final["kept_labels"]

    print(f"[summary] Clusters gardés : {kept_labels} | "
          f"cellules gardées DBSCAN = {(~roi_mask_dbscan).sum()}")

    # ==============================
    # 11) ROI finale = Topo ∩ X ∩ DBSCAN
    # ==============================
    keep_db    = ~roi_mask_dbscan            # True = gardée par DBSCAN
    keep_final = mask_roi_geom & keep_db     # topo & bornes X & DBSCAN
    mask_drop_final = ~keep_final            # True = MASQUER pour pg.show

    # ==============================
    # 12) Affichage tomogramme + ROI + clusters
    # ==============================
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
        print(f"[WARN] pg.show a échoué pour {profile_name} | {survey_name} : {e}")
        print("[WARN] On saute juste l'affichage, mais le résumé et le CSV sont OK.")
        ax = None
        cb = None

    # courbe limite topo-ROI
    x_line = np.linspace(x_c.min(), x_c.max(), 400)
    ax.plot(x_line, f_topo(x_line) - offset_m,
            'k--', lw=1.2, label=f'ROI topo -{offset_m:.2f} m')
    # topo originale
    ax.plot(x_t, y_t, 'm.-', lw=1.2, label='topographie')

    # bornes X
    ax.axvline(xmin, color='green', linestyle='--', lw=1.2,
               label=f'Borne gauche ({xmin:.1f} m)')
    ax.axvline(xmax, color='red', linestyle='--', lw=1.2,
               label=f'Borne droite ({xmax:.1f} m)')

    # points DBSCAN (sur les points sélectionnés seulement)
    sel_idx = cloud["sel_idx"]
    labels  = res["labels"]
    XY_sel  = XYZ[sel_idx]



    for lab in sorted(set(labels) - {-1}):
        m = (labels == lab)
        ax.scatter(XY_sel[m, 0], XY_sel[m, 1],
                   s=25, label=f"cluster {lab}", zorder=4)

    ax.set_title("ROI finale = Topo ∩ bornes X ∩ DBSCAN")
    ax.legend()
    fig = ax.figure
    fig_name = f"fig_{survey_name}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")
    print(f"✅ Figure sauvegardée : {fig_name}")

    # ==============================
    # 13) Résumé statistique sur le ROI
    # ==============================


    cell_areas = np.array([c.size() for c in mesh.cells()])


    summary = summarize_rho(
        model=model,
        keep_cells=keep_final,
        centers_xyz=XYZ,
        coverage_log10=coverage_log10,
        cell_areas=cell_areas
    )

    N = mesh.cellCount()
    print("\n=== Résumé ROI (Topo ∩ X ∩ DBSCAN) ===")
    print(f"• n_cells   : {summary['n_cells']}  (attendu = {keep_final.sum()})")
    print(f"• span XY   : {summary['span_xy']:.2f} m")
    print(f"• Surface   : {summary['surface_m2']:.2f} m²")  # Vérif visuelle
    print(f"• Moyenne P.: {summary['weighted_mean_rho']:.1f}")  # Vérif visuelle
    print(f"• % mesh    : {100 * summary['n_cells'] / N:.1f}%")
    # ==============================
    # 14) Sauvegarde CSV
    # ==============================
    result = {
        "survey_name": survey_name,
        "profile_name": profile_name,
        "year": int(survey_name.split("_")[3][:4]),
        "date": survey_name.split("_")[3],
        # géométrie / comptage
        "n_cells": summary["n_cells"],
        "mesh_total_cells": N,
        "percent_used": 100 * summary["n_cells"] / N,
        "span_x": summary["span_x"],
        "span_z": summary["span_z"],
        "span_xy": summary["span_xy"],
        "surface_m2": summary["surface_m2"],

        # statistiques résistivité
        "weighted_mean_rho": summary["weighted_mean_rho"],
        "median_rho": summary["median_rho"],
        "std_rho": summary["std_rho"],


    }

    out_csv = "summary_results.csv"
    if os.path.exists(out_csv):
        pd.concat(
            [pd.read_csv(out_csv), pd.DataFrame([result])],
            ignore_index=True
        ).to_csv(out_csv, index=False)
    else:
        pd.DataFrame([result]).to_csv(out_csv, index=False)

    print(f"✅ Résultats ajoutés à {out_csv}")
    print("\n--- Fin du programme ---")





if __name__ == "__main__":
    main()

