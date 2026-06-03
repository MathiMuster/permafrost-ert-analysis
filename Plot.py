import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys


def plot_single_profile(df: pd.DataFrame, profile_name: str):
    """Plot resistivity, geometry and detection quality over time for one ERT profile."""

    sub = df[df["profile_name"] == profile_name].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors='coerce')
    sub = sub[sub["n_cells"] > 0].sort_values("date").dropna(subset=["date"])

    if sub.empty:
        print(f"[WARNING] No valid data for {profile_name}, skipping.")
        return

    dates = sub["date"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Suivi temporel du site : {profile_name}", fontsize=16, fontweight='bold')

    # --- Panel 1: Resistivity (log scale) ---
    ax1.set_title("Evolution des indicateurs de Resistivite", fontsize=10, loc='left', color='#555')
    rho_mean   = sub["weighted_mean_rho"]
    rho_median = sub["median_rho"]
    rho_std    = sub["std_rho"]

    valid_rho_mask = rho_mean.notna() & (rho_mean > 0)

    if valid_rho_mask.any():
        dates_valid = dates[valid_rho_mask]

        ax1.plot(dates_valid, rho_mean[valid_rho_mask], 'o-', color='#d62728', linewidth=2,
                 label=r"$\rho$ Moyenne Ponderee")
        ax1.plot(dates_valid, rho_median[valid_rho_mask], 'v--', color='#ff7f0e', linewidth=1.5,
                 label=r"$\rho$ Mediane")

        if rho_std.notna().any():
            std_vals    = rho_std[valid_rho_mask].fillna(0)
            mean_vals   = rho_mean[valid_rho_mask]
            # Lower bound clamped to 0.1 to stay within log domain
            lower_bound = np.maximum(1e-1, mean_vals - std_vals)
            upper_bound = mean_vals + std_vals
            ax1.fill_between(dates_valid, lower_bound, upper_bound,
                             color='#d62728', alpha=0.2, label=r"$\pm 1\sigma$")

        ax1.set_yscale('log')
        ax1.set_ylabel(r"Resistivite ($\Omega\cdot m$) (Log)", fontsize=12)
        ax1.legend(loc="upper right", fontsize=10)
    else:
        ax1.text(0.5, 0.5, "Missing or non-positive resistivity data.",
                 transform=ax1.transAxes, ha="center", va="center", color='red')
        ax1.set_ylabel(r"Resistivite ($\Omega\cdot m$)", fontsize=12)

    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # --- Panel 2: Cluster geometry ---
    ax2.set_title("Evolution de la Geometrie du Noyau de Glace", fontsize=10, loc='left', color='#555')

    line_xy, = ax2.plot(dates, sub["span_xy"], 's-', color='#1f77b4', linewidth=2,   label="Span XY (Diagonale)")
    line_x,  = ax2.plot(dates, sub["span_x"],  'd:', color='#17becf', linewidth=1.5, label="Span X (Longueur)")
    line_z,  = ax2.plot(dates, sub["span_z"],  '^-', color='#2ca02c', linewidth=1.5, label="Span Z (Profondeur)")

    ax2.set_ylabel("Extension (m)", fontsize=12)
    ax2.grid(True, alpha=0.5)
    lines_list = [line_xy, line_x, line_z]

    # Surface on secondary axis to avoid scale conflict with span values
    if "surface_m2" in sub.columns and sub["surface_m2"].notna().any():
        ax2_bis = ax2.twinx()
        line_surface, = ax2_bis.plot(dates, sub["surface_m2"], 'k-.', alpha=0.7, label=r"Surface ($m^2$)")
        ax2_bis.set_ylabel(r"Surface ($m^2$)", color='k')
        ax2_bis.tick_params(axis='y', labelcolor='k')
        ax2_bis.grid(False)
        lines_list.append(line_surface)

    labels = [l.get_label() for l in lines_list]
    ax2.legend(lines_list, labels, loc="upper right", fontsize=10)

    # --- Panel 3: Detection quality ---
    ax3.set_title("Qualite de detection : Proportion de cellules utilisees", fontsize=10, loc='left', color='#555')
    ax3.plot(dates, sub["percent_used"], 'p-', color='#8c564b', linewidth=2, label="% du maillage total utilise")
    ax3.set_ylabel("% du maillage total", fontsize=12)
    ax3.set_ylim(bottom=0, top=max(100, sub["percent_used"].max() * 1.1) if sub["percent_used"].max() > 0 else 100)
    ax3.grid(True, alpha=0.5)
    ax3.legend(loc='best')

    ax3.set_xlabel("Date d'acquisition", fontsize=12)
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    os.makedirs("plots_evolution", exist_ok=True)
    filename = f"plots_evolution/{profile_name}_trend_full_detail.png"
    plt.savefig(filename, dpi=300)
    print(f"[OK] {filename}")
    plt.close(fig)


def plot_all_profiles(csv_path: str):
    """Load results CSV and generate a detailed trend plot for each unique profile."""

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        return

    required_cols = ["profile_name", "date", "weighted_mean_rho", "span_xy", "percent_used"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Missing column: '{col}'")
            return

    unique_profiles = df["profile_name"].unique()
    print(f"Plotting {len(unique_profiles)} profile(s): {list(unique_profiles)}")

    for profile in unique_profiles:
        plot_single_profile(df, profile)

    print("Done.")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "summary_results.csv"
    plot_all_profiles(csv_file)
