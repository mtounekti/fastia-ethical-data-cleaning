# BRIEF 2 – ANALYSE EXPLORATOIRE (EDA)
# Projet FastIA – Dataset Mixte
# Ce script réalise l'analyse exploratoire complète du dataset brut :
#   1. Aperçu général du dataset
#   2. Analyse des valeurs manquantes
#   3. Distribution des variables numériques
#   4. Analyse des variables catégorielles
#   5. Détection des outliers
#   6. Analyse des variables temporelles

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# Paramètres globaux
FICHIER_SOURCE = "fichier-de-donnees-mixtes-6920344a2a6cd267411281.csv"
COULEUR_PRINCIPALE = "#5B8DB8"
COULEUR_ALERTE     = "#E07B54"

os.makedirs("graphiques", exist_ok=True)

# SECTION 1: CHARGEMENT ET APERÇU GÉNÉRAL
print("=" * 65)
print("  ÉTAPE 1 – CHARGEMENT DU DATASET")
print("=" * 65)

df = pd.read_csv(FICHIER_SOURCE)

print(f"\n✔  Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n── Colonnes ──")
print(list(df.columns))
print(f"\n── Types de données ──")
print(df.dtypes.to_string())
print(f"\n── Aperçu des 5 premières lignes ──")
print(df.head().to_string())
print(f"\n── Statistiques descriptives (colonnes numériques) ──")
print(df.describe().round(2).to_string())

# SECTION 2: ANALYSE DES VALEURS MANQUANTES
print("\n" + "=" * 65)
print("  ÉTAPE 2 – VALEURS MANQUANTES")
print("=" * 65)

nb_manquants  = df.isnull().sum()
pct_manquants = (nb_manquants / len(df) * 100).round(2)

rapport_manquants = pd.DataFrame({
    "Valeurs manquantes": nb_manquants,
    "Pourcentage (%)":    pct_manquants
}).sort_values("Pourcentage (%)", ascending=False)

print("\n── Rapport des valeurs manquantes ──")
print(rapport_manquants[rapport_manquants["Valeurs manquantes"] > 0].to_string())

# graphique valeurs manquantes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Analyse des valeurs manquantes – Dataset brut",
             fontsize=14, fontweight="bold")

# matrice style missingno
ax1 = axes[0]
data_nan   = df.isnull().astype(int)
echantillon = data_nan.sample(300, random_state=42).reset_index(drop=True)
im = ax1.imshow(echantillon.T, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
ax1.set_yticks(range(len(df.columns)))
ax1.set_yticklabels(df.columns, fontsize=8)
ax1.set_xlabel("Échantillon de 300 lignes", fontsize=9)
ax1.set_title("Matrice des valeurs manquantes\n(vert=présent, rouge=absent)", fontsize=10)
plt.colorbar(im, ax=ax1, fraction=0.03)

# baarres du pourcentage manquant
ax2 = axes[1]
colors = [COULEUR_ALERTE if p > 40 else COULEUR_PRINCIPALE for p in pct_manquants.values]
barres = ax2.barh(pct_manquants.index, pct_manquants.values, color=colors, edgecolor="white")
ax2.axvline(40, color="red", linestyle="--", linewidth=1.5, label="Seuil 40% (suppression)")
ax2.set_xlabel("% de valeurs manquantes", fontsize=9)
ax2.set_title("Taux de valeurs manquantes\npar colonne", fontsize=10)
ax2.legend(fontsize=8)
for barre, val in zip(barres, pct_manquants.values):
    if val > 0:
        ax2.text(val + 0.3, barre.get_y() + barre.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("graphiques/01_valeurs_manquantes.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔  Graphique sauvegardé : graphiques/01_valeurs_manquantes.png")

# SECTION 3 DISTRIBUTION DES VARIABLES NUMÉRIQUES
print("\n" + "=" * 65)
print("  ÉTAPE 3 – DISTRIBUTIONS NUMÉRIQUES")
print("=" * 65)

colonnes_num = ["age", "taille", "poids", "revenu_estime_mois",
                "risque_personnel", "loyer_mensuel", "montant_pret"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
fig.suptitle("Distribution des variables numériques – Dataset brut",
             fontsize=14, fontweight="bold")

for i, col in enumerate(colonnes_num):
    ax = axes[i]
    donnees = df[col].dropna()
    sns.histplot(donnees, kde=True, ax=ax, color=COULEUR_PRINCIPALE, edgecolor="white")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.axvline(donnees.mean(),   color="navy",  linestyle="--", linewidth=1.2,
               label=f"Moy. {donnees.mean():.1f}")
    ax.axvline(donnees.median(), color="green", linestyle=":",  linewidth=1.2,
               label=f"Méd. {donnees.median():.1f}")
    ax.legend(fontsize=7)

# masquer le dernier axe vide
axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig("graphiques/02_distributions_numeriques.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/02_distributions_numeriques.png")

# section 4: analyse des variables catégorielles
print("\n" + "=" * 65)
print("  ÉTAPE 4 – VARIABLES CATÉGORIELLES")
print("=" * 65)

colonnes_cat = ["sexe", "sport_licence", "niveau_etude", "region",
                "smoker", "nationalité_francaise", "situation_familiale"]

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle("Distribution des variables catégorielles – Dataset brut",
             fontsize=14, fontweight="bold")

for i, col in enumerate(colonnes_cat):
    ax = axes[i]
    counts = df[col].value_counts(dropna=False)
    # renommer NaN pour l'affichage
    counts.index = [str(x) if pd.notna(x) else "NaN" for x in counts.index]

    barres = ax.bar(counts.index, counts.values, color=COULEUR_PRINCIPALE, edgecolor="white")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)

    # afficher le % sur chaque barre
    total = counts.sum()
    for barre, val in zip(barres, counts.values):
        ax.text(barre.get_x() + barre.get_width()/2, barre.get_height() + 50,
                f"{val/total*100:.1f}%", ha="center", fontsize=8)

    print(f"\n  {col} :")
    for val, cnt in counts.items():
        print(f"    {val} : {cnt} ({cnt/total*100:.1f}%)")

axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig("graphiques/03_variables_categorielles.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔  Graphique sauvegardé : graphiques/03_variables_categorielles.png")

# section 5: détection des outliers

print("\n" + "=" * 65)
print("  ÉTAPE 5 – DÉTECTION DES OUTLIERS")
print("=" * 65)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
fig.suptitle("Boxplots – Détection des outliers (dataset brut)",
             fontsize=14, fontweight="bold")

for i, col in enumerate(colonnes_num):
    ax = axes[i]
    donnees = df[col].dropna()

    Q1  = donnees.quantile(0.25)
    Q3  = donnees.quantile(0.75)
    IQR = Q3 - Q1
    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR
    nb_outliers = ((donnees < borne_basse) | (donnees > borne_haute)).sum()
    pct_outliers = nb_outliers / len(donnees) * 100

    sns.boxplot(y=donnees, ax=ax, color=COULEUR_ALERTE,
                flierprops=dict(marker="o", markerfacecolor="red",
                                markersize=3, alpha=0.5))
    ax.set_title(f"{col}\n({nb_outliers} outliers, {pct_outliers:.1f}%)", fontsize=9)
    ax.set_ylabel("")

    print(f"  {col} : {nb_outliers} outliers ({pct_outliers:.1f}%) "
          f"| bornes [{borne_basse:.2f} ; {borne_haute:.2f}]")

axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig("graphiques/04_boxplots_outliers.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔  Graphique sauvegardé : graphiques/04_boxplots_outliers.png")

# SECTION 6: analyse de la variable temporelle

print("\n" + "=" * 65)
print("  ÉTAPE 6 – VARIABLE TEMPORELLE")
print("=" * 65)

# conversion de date_creation_compte en ancienneté (jours)
df["date_creation_compte"] = pd.to_datetime(df["date_creation_compte"])
date_reference = df["date_creation_compte"].max()
df["anciennete_jours"] = (date_reference - df["date_creation_compte"]).dt.days

print(f"\n  date_creation_compte → ancienneté en jours")
print(f"  Min : {df['anciennete_jours'].min()} jours")
print(f"  Max : {df['anciennete_jours'].max()} jours")
print(f"  Moyenne : {df['anciennete_jours'].mean():.0f} jours")

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["anciennete_jours"], kde=True, ax=ax,
             color=COULEUR_PRINCIPALE, edgecolor="white")
ax.set_title("Distribution de l'ancienneté du compte (en jours)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Ancienneté (jours)")
plt.tight_layout()
plt.savefig("graphiques/05_anciennete_compte.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/05_anciennete_compte.png")

# SECTION 7: matrice de corrélation

print("\n" + "=" * 65)
print("  ÉTAPE 7 – CORRÉLATIONS")
print("=" * 65)

colonnes_corr = colonnes_num + ["anciennete_jours"]
corr = df[colonnes_corr].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Matrice de corrélation – Variables numériques",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("graphiques/06_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/06_correlation.png")

print("\n" + "=" * 65)
print("  EDA TERMINÉE ✅")
print("=" * 65)
print(f"\n  6 graphiques générés dans le dossier graphiques/")
print(f"  Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes analysées")