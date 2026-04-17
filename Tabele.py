import matplotlib.pyplot as plt
import pandas as pd

data = [
    # GÉOMÉTRIE
    ["Géométrie (ROI)", "Profondeur (z_range)", "0.3 × x_range (m)", "Standard dipôle-dipôle"],
    ["", "Marge surface (offset)", "7 % → 0.07 × z_range (m)", "Exclusion de la couche active bruitée"],
    ["", "Troncature latérale", "3 % → 0.03 × x_range (m)", "Suppression des effets de bord"],
    # NORMALISATION
    ["Normalisation", "Facteur pondération Z\n(z_weight_factor)", "0.3 (sans unité)", "Réduit le poids de la profondeur\ndans la métrique de distance"],
    ["", "Bornes coverage", "[-4.0, 0.0] (log₁₀)", "Normalisation de la qualité de mesure"],
    ["", "Échelle proximité", "0.10 × x_range (m)", "Rayon d'influence spatial"],
    # PONDÉRATION
    ["Pondération", "Poids résistivité (w_rho)", "0.50 (sans unité)", "Priorité à la valeur physique"],
    ["", "Poids proximité (w_prox)", "0.30 (sans unité)", "Cohérence spatiale"],
    ["", "Poids coverage (w_cov)", "0.20 (sans unité)", "Pénalité de fiabilité de mesure"],
    # CLUSTERING
    ["Clustering (DBSCAN)", "Rayon (ε)", "0.8 (espace normalisé)", "Voisinage dans l'espace de clustering"],
    ["", "Densité minimale (min_pts)", "8 (cellules)", "Taille minimale d'un noyau de cluster"],
    ["", "Seuil pré-sélection", "0.70 (ρ relatif)", "Filtre initial sur résistivité normalisée"],
    ["", "Stratégie conservation", "'all'", "Fusion de tous les fragments significatifs"],
]

columns = ["Catégorie", "Paramètre", "Valeur", "Justification / Rôle"]

fig, ax = plt.subplots(figsize=(16, 9))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=data,
    colLabels=columns,
    cellLoc='left',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.8)

# Largeurs des colonnes
col_widths = [0.18, 0.22, 0.22, 0.38]
for (row, col), cell in table.get_celld().items():
    cell.set_width(col_widths[col])

header_color = '#40466e'
row_colors = ['#f1f1f2', 'w']

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='w')
        cell.set_facecolor(header_color)
        cell.set_edgecolor('w')
    else:
        cell.set_facecolor(row_colors[row % 2])
        cell.set_edgecolor('w')
        if col == 0 and cell.get_text().get_text() != "":
            cell.set_text_props(weight='bold')
        # Alignement centré pour la colonne valeur
        if col == 2:
            cell.set_text_props(ha='center')

plt.title(
    "Synthèse des paramètres fixes de l'algorithme",
    fontsize=14, weight='bold', y=0.98
)

plt.savefig("tableau_parametres.png", dpi=300, bbox_inches='tight')
print("✅ tableau_parametres.png généré !")
plt.show()