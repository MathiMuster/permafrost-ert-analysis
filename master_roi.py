# =============================================================================
#  master_roi.py — Temporal monitoring via fixed reference points
#  Author: Mathilda Muster Labeau
# =============================================================================

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors
import pygimli as pg

from functions import (
    get_profile_info,
    utils_mesh,
    cell_centers_xyz,
    _make_topo_interpolator_from_data,
    roi_mask_topo,
    prepare_nuage,
    run_dbscan,
    labels_to_mask,
    FEATURE_PARAMS,
)


# =============================================================================
#  PARAMETERS
# =============================================================================

PATHS = {
    "metadata":  "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/topo.txt",
    "mesh_cell": "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/mesh_cell.txt",
    "mesh_node": "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/mesh_node.txt",
    "data_inv":  "./resibase-v3/ert-dashboard-main/data_inversion/inverted_data_public/tables/data_inv.txt",
}

# Standard ROI — used for display of all surveys
ROI_CFG = {
    "offset_factor": 0.09,
    "x_borne":       0.03,
}

# Relaxed ROI for the reference survey — captures deeper cells for master points
ROI_CFG_REF = {
    "offset_factor": 0.04,
    "x_borne":       0.03,
}

# Standard DBSCAN
DB_CFG = {
    "eps":         0.8,
    "min_samples": 8,
    "metric":      "euclidean",
    "keep":        "best",
    "min_size":    8,
}

# Relaxed DBSCAN for reference survey — keeps deeper and more fragmented clusters
DB_CFG_REF = {
    "eps":         0.8,
    "min_samples": 6,
    "metric":      "euclidean",
    "keep":        "all",
    "min_size":    6,
}

# Relaxed feature params for reference survey
FPARAMS_REF = FEATURE_PARAMS.copy()
FPARAMS_REF["rho_keep_min"]            = 0.50
FPARAMS_REF["z_aniso_factor"]          = 0.45
FPARAMS_REF["core_keep_top_quantile"]  = 0.30

MAX_DIST_M = 5.0  # maximum matching distance (m) between master point and mesh cell

SAVE_DIR = "master_points"
OUT_CSV  = "summary_master.csv"

# Surveys in chronological order — first entry is the reference survey
ATT_SURVEYS = [
    "CH_ATT_MV1_2007-08-25_01",
    "CH_ATT_MV1_2008-07-16_01",
    "CH_ATT_MV1_2008-11-10_01",
    "CH_ATT_MV1_2010-07-14_01",
    "CH_ATT_MV1_2014-09-30_01",
    "CH_ATT_MV1_2016-09-23_01",
    "CH_ATT_MV1_2017-09-25_01",
    "CH_ATT_MV1_2018-10-15_01",
    "CH_ATT_MV1_2019-09-17_01",
    "CH_ATT_MV1_2020-09-22_01",
    "CH_ATT_MV1_2021-09-24_01",
    "CH_ATT_MV1_2022-10-07_01",
]


# =============================================================================
#  INTERNAL HELPERS
# =============================================================================

def _load_survey(profile_name, survey_name):
    """Load mesh, resistivity model, coverage and topography for one survey."""

    n_sensors, dx, x_range, z_range_dipol = get_profile_info(
        profile_name=profile_name,
        metadata_path=PATHS["metadata"],
        survey_name=survey_name,
    )
    mesh, rho_by_cell, _, _, file_cell_ids = utils_mesh(
        data_inv=PATHS["data_inv"],
        mesh_node=PATHS["mesh_node"],
        mesh_cell=PATHS["mesh_cell"],
        survey_name=survey_name,
        profile_name=profile_name,
    )
    f_topo, x_t, y_t = _make_topo_interpolator_from_data(
        metadatapath=PATHS["metadata"],
        survey_name=survey_name,
    )

    N              = mesh.cellCount()
    model          = np.full(N, np.nan)
    coverage_log10 = np.full(N, np.nan)
    for ci, fid in enumerate(file_cell_ids):
        pair = rho_by_cell.get(fid)
        if pair is not None:
            model[ci]          = float(pair[0])
            coverage_log10[ci] = float(pair[1])

    return {
        "mesh": mesh, "model": model, "coverage_log10": coverage_log10,
        "f_topo": f_topo, "x_t": x_t, "y_t": y_t,
        "x_range": x_range, "z_range_dipol": z_range_dipol,
    }


def _apply_roi(data, roi_cfg):
    """Apply topographic mask and return geometric ROI mask with bounds."""

    _, offset_m, xc, yc, y_lim, xmin, xmax = roi_mask_topo(
        data["mesh"], data["f_topo"], data["z_range_dipol"],
        offset_factor=roi_cfg["offset_factor"],
        x_range=data["x_range"],
        x_borne=roi_cfg["x_borne"],
    )
    XYZ      = cell_centers_xyz(data["mesh"])
    x_c, y_c = XYZ[:, 0], XYZ[:, 1]
    y_lim_c  = data["f_topo"](x_c) - offset_m
    mask_roi_geom = (y_c <= y_lim_c) & (x_c >= xmin) & (x_c <= xmax)

    return mask_roi_geom, offset_m, xmin, xmax, XYZ


# =============================================================================
#  1. COMPUTE MASTER POINTS (reference survey)
# =============================================================================

def compute_master_points(profile_name, ref_survey_name):
    """
    Run the full pipeline on the reference survey using relaxed ROI/DBSCAN parameters.
    Save centroid coordinates of retained cells as master reference points.
    """

    print(f"\n{'='*60}")
    print(f"  COMPUTING MASTER POINTS")
    print(f"  Profile   : {profile_name}")
    print(f"  Reference : {ref_survey_name}")
    print(f"{'='*60}")

    data = _load_survey(profile_name, ref_survey_name)
    mask_roi_geom, offset_m, xmin, xmax, XYZ = _apply_roi(data, ROI_CFG_REF)

    rho_c = data["model"].copy()
    cov_c = data["coverage_log10"].copy()
    rho_c[~mask_roi_geom] = np.nan
    cov_c[~mask_roi_geom] = np.nan

    cloud = prepare_nuage(
        centers_xy=XYZ,
        rho_linear=rho_c,
        coverage_log10=cov_c,
        fparams=FPARAMS_REF,
    )
    res = run_dbscan(
        points_norm=cloud["points_norm"],
        eps=DB_CFG_REF["eps"],
        min_samples=DB_CFG_REF["min_samples"],
        metric=DB_CFG_REF["metric"],
    )
    final = labels_to_mask(
        n_cells=data["mesh"].cellCount(),
        sel_idx=cloud["sel_idx"],
        labels=res["labels"],
        core=cloud["core"],
        keep=DB_CFG_REF["keep"],
        min_size=DB_CFG_REF["min_size"],
    )

    keep_final = mask_roi_geom & ~final["mask"]
    n_kept     = int(keep_final.sum())

    if n_kept == 0:
        raise ValueError(
            f"[master] No cells retained for {ref_survey_name}.\n"
            f"Adjust ROI_CFG_REF, DB_CFG_REF or FPARAMS_REF."
        )

    master_pts = XYZ[keep_final]
    print(f"  -> {n_kept} master points extracted.")

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{profile_name}_master_pts.csv")
    pd.DataFrame(master_pts, columns=["x", "y"]).assign(
        profile_name=profile_name, ref_survey=ref_survey_name
    ).to_csv(save_path, index=False)
    print(f"  -> Master points saved: {save_path}")

    _plot_validation(data, mask_roi_geom, master_pts, cloud, res,
                     offset_m, xmin, xmax, profile_name, ref_survey_name)

    return master_pts


# =============================================================================
#  2. APPLY MASTER POINTS TO A SURVEY
# =============================================================================

def apply_master_points(master_pts, data):
    """
    For each master point, find the nearest cell in the current survey mesh
    and extract resistivity and coverage values.
    """

    XYZ           = cell_centers_xyz(data["mesh"])
    nn            = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(XYZ)
    dists, idxs   = nn.kneighbors(master_pts)
    dists         = dists[:, 0]
    idxs          = idxs[:, 0]

    N       = len(master_pts)
    rho_out = np.full(N, np.nan)
    cov_out = np.full(N, np.nan)
    valid   = dists <= MAX_DIST_M

    for i, (ci, ok) in enumerate(zip(idxs, valid)):
        if ok:
            if np.isfinite(data["model"][ci]):
                rho_out[i] = data["model"][ci]
            if np.isfinite(data["coverage_log10"][ci]):
                cov_out[i] = data["coverage_log10"][ci]

    print(f"  [apply] {int(valid.sum())}/{N} points matched (dist <= {MAX_DIST_M} m)")
    return {"rho": rho_out, "cov": cov_out, "dist": dists, "valid": valid}


# =============================================================================
#  3. FULL PIPELINE
# =============================================================================

def run_master_pipeline(profile_name, surveys_ordered):
    """
    Run the Master ROI pipeline over all surveys for a profile.
    The first survey in the list is used as the reference.
    """

    if not surveys_ordered:
        print(f"[master] No surveys provided for {profile_name}.")
        return

    ref_survey = surveys_ordered[0]
    print(f"\n{'='*60}")
    print(f"  MASTER ROI PIPELINE — {profile_name}")
    print(f"  Reference survey : {ref_survey}")
    print(f"  Total surveys    : {len(surveys_ordered)}")
    print(f"{'='*60}")

    pts_path = os.path.join(SAVE_DIR, f"{profile_name}_master_pts.csv")
    if os.path.exists(pts_path):
        print(f"\n  Master points already computed — loading from {pts_path}")
        master_pts = pd.read_csv(pts_path)[["x", "y"]].to_numpy()
        print(f"  -> {len(master_pts)} master points loaded.")
    else:
        master_pts = compute_master_points(profile_name, ref_survey)

    results = []
    for survey_name in surveys_ordered:
        print(f"\n  --- Survey: {survey_name} ---")
        try:
            data    = _load_survey(profile_name, survey_name)
            matched = apply_master_points(master_pts, data)

            rho_v   = matched["rho"][matched["valid"]]
            cov_v   = matched["cov"][matched["valid"]]
            n_valid = int(matched["valid"].sum())

            if n_valid == 0:
                mean_rho = median_rho = weighted_mean_rho = std_rho = np.nan
            else:
                rho_f             = rho_v[np.isfinite(rho_v)]
                mean_rho          = float(np.nanmean(rho_f))   if len(rho_f) else np.nan
                median_rho        = float(np.nanmedian(rho_f)) if len(rho_f) else np.nan
                std_rho           = float(np.nanstd(rho_f))    if len(rho_f) else np.nan
                w                 = 1.0 / (1.0 + np.power(10.0, cov_v[np.isfinite(rho_v)]))
                weighted_mean_rho = float(np.sum(w * rho_f) / np.sum(w)) if np.sum(w) > 0 else np.nan

            date_str = survey_name.split("_")[3]
            results.append({
                "profile_name":      profile_name,
                "survey_name":       survey_name,
                "date":              date_str,
                "year":              int(date_str[:4]),
                "is_reference":      survey_name == ref_survey,
                "n_master_pts":      len(master_pts),
                "n_valid_pts":       n_valid,
                "pct_valid":         100.0 * n_valid / len(master_pts),
                "mean_rho":          mean_rho,
                "median_rho":        median_rho,
                "weighted_mean_rho": weighted_mean_rho,
                "std_rho":           std_rho,
            })
            print(f"  [OK] n_valid={n_valid} | "
                  f"mean={mean_rho:.0f} ohm.m | weighted={weighted_mean_rho:.0f} ohm.m")

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "profile_name": profile_name, "survey_name": survey_name,
                "date": "error", "year": -1,
                "is_reference": survey_name == ref_survey,
                "n_master_pts": len(master_pts), "n_valid_pts": 0,
                "pct_valid": 0.0, "mean_rho": np.nan, "median_rho": np.nan,
                "weighted_mean_rho": np.nan, "std_rho": np.nan,
            })

    df_out = pd.DataFrame(results).sort_values("date")
    if os.path.exists(OUT_CSV):
        df_ex  = pd.read_csv(OUT_CSV)
        df_ex  = df_ex[df_ex["profile_name"] != profile_name]
        df_out = pd.concat([df_ex, df_out], ignore_index=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n  [OK] Results saved to {OUT_CSV}")

    return df_out


# =============================================================================
#  4. VALIDATION FIGURE
# =============================================================================

def _plot_validation(data, mask_roi_geom, master_pts, cloud, res,
                     offset_m, xmin, xmax, profile_name, ref_survey_name):

    try:
        ax, _ = pg.show(
            data["mesh"], data=data["model"],
            mask=~mask_roi_geom, logScale=True,
            showMesh=True, cMap="Spectral_r",
        )
        ylo, yhi = ax.get_ylim()
        if ylo > yhi:
            ax.set_ylim(yhi, ylo)

        x_line = np.linspace(data["x_t"].min(), data["x_t"].max(), 400)
        ax.plot(x_line, data["f_topo"](x_line) - offset_m,
                "k--", lw=1.2, label=f"ROI topo -{offset_m:.2f} m")
        ax.plot(data["x_t"], data["y_t"], "m.-", lw=1.2, label="topographie")
        ax.axvline(xmin, color="green", linestyle="--", lw=1.2,
                   label=f"Borne gauche ({xmin:.1f} m)")
        ax.axvline(xmax, color="red", linestyle="--", lw=1.2,
                   label=f"Borne droite ({xmax:.1f} m)")

        XYZ_sel = cell_centers_xyz(data["mesh"])[cloud["sel_idx"]]
        labels  = res["labels"]
        for lab in sorted(set(labels) - {-1}):
            ax.scatter(XYZ_sel[labels == lab, 0], XYZ_sel[labels == lab, 1],
                       s=20, label=f"cluster {lab}", zorder=4)

        ax.scatter(master_pts[:, 0], master_pts[:, 1],
                   s=15, c="black", alpha=0.7, zorder=5,
                   label=f"Points maitres (n={len(master_pts)})")

        ax.set_title(f"Points maitres — {profile_name}")
        ax.legend(fontsize=8)
        fig_path = os.path.join(SAVE_DIR, f"{profile_name}_master_validation.png")
        ax.figure.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"  -> Validation figure saved: {fig_path}")

    except Exception as e:
        print(f"  [WARNING] Validation figure failed: {e}")


# =============================================================================
#  5. TREND PLOT
# =============================================================================

def plot_master_trend(profile_name, out_csv=OUT_CSV, save_dir=SAVE_DIR):
    """
    Temporal trend chart from summary_master.csv.
    Panel 1: weighted mean + median + interquartile range.
    Panel 2: percentage of valid matched points (quality indicator).
    """
    import matplotlib.dates as mdates

    df  = pd.read_csv(out_csv)
    sub = df[df["profile_name"] == profile_name].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub[sub["n_valid_pts"] > 0].sort_values("date").dropna(subset=["date"])

    if sub.empty:
        print(f"[plot_trend] No valid data for {profile_name}.")
        return

    has_quantiles = "p25_rho" in sub.columns and "p75_rho" in sub.columns

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(f"Evolution temporelle — {profile_name}\n",
                 fontsize=12, fontweight="bold")

    ax1.set_title("Resistivite aux points maitres", fontsize=10, loc="left", color="#555")

    valid   = sub["weighted_mean_rho"].notna() & (sub["weighted_mean_rho"] > 0)
    dates_v = sub["date"][valid]
    wmean   = sub["weighted_mean_rho"][valid]
    median  = sub["median_rho"][valid]

    if has_quantiles:
        p25 = sub["p25_rho"][valid]
        p75 = sub["p75_rho"][valid]
    else:
        # Approximation from std when quantile columns are absent
        std = sub["std_rho"][valid]
        p25 = np.maximum(1e-1, wmean - 0.35 * std)
        p75 = wmean + 0.35 * std

    ax1.fill_between(dates_v, p25, p75, color="#D85A30", alpha=0.18,
                     label="intervalle central (P40-P60)")
    ax1.plot(dates_v, wmean,  "o-",  color="#D85A30", lw=2,
             label=r"$\rho$ moyenne ponderee")
    ax1.plot(dates_v, median, "v--", color="#888780", lw=1.5,
             label=r"$\rho$ mediane")

    ax1.set_yscale("log")
    ax1.set_ylabel(r"Resistivite ($\Omega\cdot$m) (log)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, which="both", alpha=0.35)

    etés_chauds = {"2003": "Canicule\n2003", "2015": "Ete chaud\n2015", "2019": "Canicule\n2019"}
    ymin_ax, _ = ax1.get_ylim()
    for annee, label in etés_chauds.items():
        ts = pd.Timestamp(f"{annee}-07-01")
        if sub["date"].min() <= ts <= sub["date"].max():
            ax1.axvline(ts, color="#aaaaaa", linestyle=":", lw=1.0, alpha=0.8)
            ax1.text(ts + pd.Timedelta(days=60), ymin_ax * 1.15,
                     label, fontsize=7.5, color="#888888", va="bottom")

    ax2.set_title("Qualite du matching", fontsize=9, loc="left", color="#888")
    ax2.plot(sub["date"], sub["pct_valid"], "p-",
             color="#534AB7", lw=1.5, markersize=5,
             label="% points maitres matches")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("% valides", fontsize=9)
    ax2.set_xlabel("Date d'acquisition", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"{profile_name}_master_trend.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Trend figure saved: {fig_path}")
    return fig_path


# =============================================================================
#  6. MULTI-TOMOGRAM FIGURE
# =============================================================================

def plot_master_timeseries(profile_name, surveys_ordered, master_pts,
                           ncols=4, figsize_per_ax=(4, 3)):
    """
    Grid of tomograms with master points overlaid.
    Reference survey is highlighted with an orange border.
    """

    n     = len(surveys_ordered)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows + 0.6),
        squeeze=False,
    )
    fig.suptitle(
        f"Suivi temporel — {profile_name}\n"
        f"(reference = {surveys_ordered[0].split('_')[3]})",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for ax_idx, survey_name in enumerate(surveys_ordered):
        ax       = axes[ax_idx // ncols][ax_idx % ncols]
        date_lbl = survey_name.split("_")[3]
        is_ref   = (ax_idx == 0)
        print(f"  [plot] {survey_name} ...", end=" ", flush=True)

        try:
            data = _load_survey(profile_name, survey_name)
            mask_roi_geom, offset_m, xmin, xmax, XYZ = _apply_roi(data, ROI_CFG)

            pg.show(
                data["mesh"], data=data["model"],
                mask=~mask_roi_geom, logScale=True,
                showMesh=False, cMap="Spectral_r",
                ax=ax, colorBar=False,
            )

            x_line = np.linspace(data["x_t"].min(), data["x_t"].max(), 300)
            ax.plot(x_line, data["f_topo"](x_line) - offset_m, "k--", lw=0.8)
            ax.plot(data["x_t"], data["y_t"], "m.-", lw=0.8)
            ax.axvline(xmin, color="green", linestyle="--", lw=0.8)
            ax.axvline(xmax, color="red",   linestyle="--", lw=0.8)

            # Point size proportional to log(rho) for visual emphasis
            nn = NearestNeighbors(n_neighbors=1).fit(XYZ)
            dists, idxs = nn.kneighbors(master_pts)
            valid = dists[:, 0] <= MAX_DIST_M

            rho_at = np.array([
                data["model"][idxs[i, 0]]
                if valid[i] and np.isfinite(data["model"][idxs[i, 0]])
                else np.nan
                for i in range(len(master_pts))
            ])
            fin = np.isfinite(rho_at) & valid
            if fin.any():
                lr     = np.log10(np.where(fin, rho_at, 1.0))
                lo, hi = np.nanpercentile(lr[fin], [10, 90])
                sizes  = 5 + 20 * np.clip((lr - lo) / max(hi - lo, 0.1), 0, 1)
            else:
                sizes = np.full(len(master_pts), 8.0)

            ax.scatter(master_pts[valid, 0], master_pts[valid, 1],
                       s=sizes[valid], c="black", alpha=0.55,
                       linewidths=0, zorder=5)

            ax.set_title(date_lbl, fontsize=9,
                         color="darkorange" if is_ref else "black",
                         fontweight="bold" if is_ref else "normal")
            ax.set_xlabel("x in m", fontsize=7)
            ax.set_ylabel("y in m", fontsize=7)
            ax.tick_params(labelsize=7)

            if is_ref:
                for spine in ax.spines.values():
                    spine.set_edgecolor("darkorange")
                    spine.set_linewidth(2)

            print("OK")

        except Exception as e:
            ax.set_visible(False)
            print(f"ERROR: {e}")

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.legend(
        handles=[
            mpatches.Patch(edgecolor="darkorange", facecolor="none",
                           linewidth=2, label="survey de reference"),
            mpatches.Patch(color="black", alpha=0.55,
                           label=f"points maitres (n={len(master_pts)})"),
        ],
        loc="lower center", ncol=2, fontsize=8,
        frameon=False, bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, f"{profile_name}_timeseries_tomo.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Multi-tomogram figure saved: {fig_path}")
    return fig_path


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    df_results = run_master_pipeline(
        profile_name="CH_ATT_MV1",
        surveys_ordered=ATT_SURVEYS,
    )

    if df_results is not None:
        print("\n=== Results preview ===")
        print(df_results[["date", "n_valid_pts", "pct_valid",
                           "weighted_mean_rho", "median_rho"]].to_string(index=False))

        master_pts = pd.read_csv(
            os.path.join(SAVE_DIR, "CH_ATT_MV1_master_pts.csv")
        )[["x", "y"]].to_numpy()

        plot_master_timeseries(
            profile_name="CH_ATT_MV1",
            surveys_ordered=ATT_SURVEYS,
            master_pts=master_pts,
            ncols=4,
        )
        plot_master_trend(profile_name="CH_ATT_MV1")
