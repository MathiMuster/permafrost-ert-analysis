import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys


# =====================================================================
# 1. FONCTION DE TRACÉ DÉTAILLÉ (Par profil)
# =====================================================================

def plot_single_profile(df: pd.DataFrame, profile_name: str):

    # 1. Préparation des données pour le profil
    sub = df[df["profile_name"] == profile_name].copy()


    sub["date"] = pd.to_datetime(sub["date"], errors='coerce')
    sub = sub[sub["n_cells"] > 0].sort_values("date").dropna(subset=["date"])

    if sub.empty:
        print(f"⚠️ Profil ignoré : aucune donnée valide ou datée pour {profile_name}.")
        return

    dates = sub["date"]


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Suivi temporel du site : {profile_name}", fontsize=16, fontweight='bold')

    # ---------------------------------------------------------
    # PANNEAU 1 : Résistivité (Log Y)
    # ---------------------------------------------------------
    ax1.set_title("Évolution des indicateurs de Résistivité", fontsize=10, loc='left', color='#555')
    rho_mean = sub["weighted_mean_rho"]
    rho_median = sub["median_rho"]
    rho_std = sub["std_rho"]

    # Masque pour ne garder que les valeurs positives finies (obligatoire pour l'axe log)
    valid_rho_mask = rho_mean.notna() & (rho_mean > 0)

    if valid_rho_mask.any():
        dates_valid = dates[valid_rho_mask]

        # 1. Courbe Moyenne pondérée (ROUGE, principale)
        ax1.plot(dates_valid, rho_mean[valid_rho_mask], 'o-', color='#d62728', linewidth=2,
                 label=r"$\rho$ Moyenne Pondérée")

        # 2. Courbe Médiane (ORANGE, pour la robustesse)
        ax1.plot(dates_valid, rho_median[valid_rho_mask], 'v--', color='#ff7f0e', linewidth=1.5,
                 label=r"$\rho$ Médiane")

        # 3. Zone d'incertitude (écart-type)
        if rho_std.notna().any():
            std_vals = rho_std[valid_rho_mask].fillna(
                0)  # Remplacer les éventuels NaN d'écart-type par 0 pour l'affichage
            mean_vals = rho_mean[valid_rho_mask]

            # Limiter les bornes à des valeurs positives ou très proches de zéro pour l'affichage log
            lower_bound = np.maximum(1e-1, mean_vals - std_vals)
            upper_bound = mean_vals + std_vals

            ax1.fill_between(dates_valid, lower_bound, upper_bound,
                             color='#d62728', alpha=0.2, label=r"$\pm 1\sigma$")

        ax1.set_yscale('log')
        ax1.set_ylabel(r"Résistivité ($\Omega\cdot m$) (Log)", fontsize=12)
        ax1.legend(loc="upper right", fontsize=10)
    else:

        ax1.text(0.5, 0.5, "❌ Données de résistivité manquantes ou non-positives.",
                 transform=ax1.transAxes, ha="center", va="center", color='red')
        ax1.set_ylabel(r"Résistivité ($\Omega\cdot m$)", fontsize=12)

    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # ---------------------------------------------------------
    # PANNEAU 2 : Extension Spatiale (Linéaire Y)
    # ---------------------------------------------------------
    ax2.set_title("Évolution de la Géométrie du Noyau de Glace", fontsize=10, loc='left', color='#555')

    # Tracé des Spans (Axe Y Gauche)
    line_xy, = ax2.plot(dates, sub["span_xy"], 's-', color='#1f77b4', linewidth=2, label="Span XY (Diagonale)")
    line_x, = ax2.plot(dates, sub["span_x"], 'd:', color='#17becf', linewidth=1.5, label="Span X (Longueur)")
    line_z, = ax2.plot(dates, sub["span_z"], '^-', color='#2ca02c', linewidth=1.5, label="Span Z (Profondeur)")

    ax2.set_ylabel("Extension (m)", fontsize=12)
    ax2.grid(True, alpha=0.5)

    lines_list = [line_xy, line_x, line_z]

    # Surface (Axe Y Droit - Twinx)
    if "surface_m2" in sub.columns and sub["surface_m2"].notna().any():
        ax2_bis = ax2.twinx()
        line_surface, = ax2_bis.plot(dates, sub["surface_m2"], 'k-.', alpha=0.7, label=r"Surface ($m^2$)")
        ax2_bis.set_ylabel(r"Surface ($m^2$)", color='k')
        ax2_bis.tick_params(axis='y', labelcolor='k')
        ax2_bis.grid(False)
        lines_list.append(line_surface)

    # Légende (fusion des deux axes)
    labels = [l.get_label() for l in lines_list]
    ax2.legend(lines_list, labels, loc="upper right", fontsize=10)

    # ---------------------------------------------------------
    # PANNEAU 3 : Qualité de détection
    # ---------------------------------------------------------
    ax3.set_title("Qualité de détection : Proportion de cellules utilisées", fontsize=10, loc='left', color='#555')
    ax3.plot(dates, sub["percent_used"], 'p-', color='#8c564b', linewidth=2, label="% du maillage total utilisé")
    ax3.set_ylabel("% du maillage total", fontsize=12)
    ax3.set_ylim(bottom=0, top=max(100, sub["percent_used"].max() * 1.1) if sub["percent_used"].max() > 0 else 100)
    ax3.grid(True, alpha=0.5)
    ax3.legend(loc='best')

    # Format de l'axe des temps (X)
    ax3.set_xlabel("Date d'acquisition", fontsize=12)
    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax3.xaxis.set_major_locator(locator)
    ax3.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Sauvegarde
    os.makedirs("plots_evolution", exist_ok=True)
    filename = f"plots_evolution/{profile_name}_trend_full_detail.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Graphique détaillé généré : {filename}")
    plt.close(fig)


# =====================================================================
# 2. FONCTION PRINCIPALE
# =====================================================================

def plot_all_profiles(csv_path: str):
    """
    Charge le CSV de résultats et génère un graphique de tendance
    détaillé pour chaque profil unique.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Erreur : Fichier non trouvé à {csv_path}")
        print("Assure-toi de l'avoir généré en lançant main.py d'abord.")
        return

    print(f"Chargement des données depuis {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du CSV : {e}")
        return

    # Vérification des colonnes nécessaires
    required_cols = ["profile_name", "date", "weighted_mean_rho", "span_xy", "percent_used"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Erreur de colonne : La colonne '{col}' est manquante.")
            print("As-tu bien mis à jour et relancé main.py pour générer le nouveau CSV ?")
            return

    unique_profiles = df["profile_name"].unique()
    print(f"Tâches de tracé lancées pour {len(unique_profiles)} profils: {list(unique_profiles)}")

    for profile in unique_profiles:
        plot_single_profile(df, profile)

    print("\nOpération de tracé terminée.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "summary_results.csv"  # Nom de fichier par défaut

    plot_all_profiles(csv_file)