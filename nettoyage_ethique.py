# NETTOYAGE ÉTHIQUE – DATASET V2
# Projet FastIA – Dataset Mixte
# Ce script applique les décisions éthiques documentées dans la datasheet :
#   1. Suppression des données directement identifiantes (RGPD)
#   2. Suppression des variables discriminantes
#   3. Transformation de l'âge en tranches
#   4. Export du dataset v2 éthique

# Ce script part du dataset v1 propre (nettoyage_technique.py)
# et applique en plus les décisions éthiques.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

FICHIER_V1    = "datasets/dataset_v1_propre.csv"
FICHIER_V2    = "datasets/dataset_v2_ethique.csv"
COULEUR_AVANT = "#E07B54"
COULEUR_APRES = "#5B8DB8"

os.makedirs("datasets",   exist_ok=True)
os.makedirs("graphiques", exist_ok=True)

# section 1: loading dataset v1
print("=" * 65)
print("  ÉTAPE 1 – CHARGEMENT DU DATASET V1")
print("=" * 65)

# On recharge le dataset brut pour récupérer l'âge brut avant scaling
df_brut = pd.read_csv("fichier-de-donnees-mixtes-6920344a2a6cd267411281.csv")
df = pd.read_csv(FICHIER_V1)

print(f"\n✔  Dataset v1 chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   Colonnes : {list(df.columns)}")

# section2: – SUPPRESSION DONNÉES IDENTIFIANTES (RGPD)

print("\n" + "=" * 65)
print("  ÉTAPE 2 – SUPPRESSION DONNÉES IDENTIFIANTES (RGPD)")
print("=" * 65)

# nom et prénom sont des données directement identifiantes
# Article 4 du RGPD → suppression obligatoire
colonnes_identifiantes = ["nom", "prenom"]
df = df.drop(columns=colonnes_identifiantes)

print(f"\n  ❌ Supprimées (données personnelles identifiantes) :")
for col in colonnes_identifiantes:
    print(f"     - {col}")
print(f"\n✔  Dimensions restantes : {df.shape}")

# section 3: SUPPRESSION VARIABLES DISCRIMINANTES

print("\n" + "=" * 65)
print("  ÉTAPE 3 – SUPPRESSION VARIABLES DISCRIMINANTES")
print("=" * 65)

# ces variables peuvent entraîner des décisions automatisées discriminatoires
# article 22 du RGPD → interdit de les utiliser pour des décisions automatisées
colonnes_discriminantes = ["sexe", "nationalité_francaise", "smoker", "taille", "poids"]

# dans le v1 ces colonnes ont été encodées en numérique
# on les supprime directement
colonnes_a_supprimer = [col for col in colonnes_discriminantes if col in df.columns]

print(f"\n  ❌ Supprimées (risque de discrimination) :")
for col in colonnes_a_supprimer:
    print(f"     - {col}")

df = df.drop(columns=colonnes_a_supprimer)
print(f"\n✔  Dimensions restantes : {df.shape}")

# SECTION 4 – TRANSFORMATION DE L'ÂGE EN TRANCHES
print("\n" + "=" * 65)
print("  ÉTAPE 4 – TRANSFORMATION ÂGE → TRANCHES")
print("=" * 65)

# L'âge brut est trop granulaire et peut introduire de la discrimination
# On le transforme en tranches pour réduire ce risque
# On repart de l'âge brut (avant scaling du v1)
age_brut = df_brut["age"].values

tranches = pd.cut(
    age_brut,
    bins=[17, 30, 45, 60, 75],
    labels=["18-30", "31-45", "46-60", "61-75"]
)

# encodage ordinal des tranches
mapping_tranches = {"18-30": 0, "31-45": 1, "46-60": 2, "61-75": 3}
df["age_tranche"] = [mapping_tranches[str(t)] for t in tranches]

# suppression de l'âge scalé (remplacé par les tranches)
df = df.drop(columns=["age"])

print(f"\n  age (scalé) → age_tranche (ordinal 0-3)")
print(f"  Mapping : {mapping_tranches}")
print(f"\n  Distribution des tranches :")
tranches_counts = pd.Series(df["age_tranche"].value_counts().sort_index())
for val, label in enumerate(["18-30", "31-45", "46-60", "61-75"]):
    count = tranches_counts.get(val, 0)
    print(f"    [{val}] {label} : {count} personnes ({count/len(df)*100:.1f}%)")

print(f"\n✔  Dimensions restantes : {df.shape}")

# section5: one last verification

print("\n" + "=" * 65)
print("  ÉTAPE 5 – VÉRIFICATION FINALE")
print("=" * 65)

print(f"\n  Dimensions finales : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"  Valeurs manquantes : {df.isnull().sum().sum()}")
print(f"\n  Colonnes du dataset v2 :")
for col in df.columns:
    print(f"    - {col}")

# section6 – exporting

df.to_csv(FICHIER_V2, index=False)
print(f"\n✔  Dataset v2 éthique exporté : {FICHIER_V2}")

# SECTION7 – GRAPHIQUE COMPARATIF V1 vs V2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Comparaison V1 (technique) vs V2 (éthique)",
             fontsize=13, fontweight="bold")

df_v1 = pd.read_csv(FICHIER_V1)

# nombre de colonnes
ax1 = axes[0]
noms   = ["Dataset brut", "Dataset v1\n(technique)", "Dataset v2\n(éthique)"]
valeurs = [19, df_v1.shape[1], df.shape[1]]
couleurs = ["#AAAAAA", COULEUR_AVANT, COULEUR_APRES]
barres = ax1.bar(noms, valeurs, color=couleurs, edgecolor="white", width=0.5)
ax1.set_title("Nombre de colonnes", fontsize=11)
ax1.set_ylabel("Colonnes")
for barre, val in zip(barres, valeurs):
    ax1.text(barre.get_x() + barre.get_width()/2, barre.get_height() + 0.3,
             str(val), ha="center", fontsize=11, fontweight="bold")

# distribution age_tranche dans v2
ax2 = axes[1]
labels_tranches = ["18-30", "31-45", "46-60", "61-75"]
counts_tranches = df["age_tranche"].value_counts().sort_index()
ax2.bar(labels_tranches, counts_tranches.values,
        color=COULEUR_APRES, edgecolor="white")
ax2.set_title("Distribution des tranches d'âge (v2)", fontsize=11)
ax2.set_xlabel("Tranche d'âge")
ax2.set_ylabel("Nombre de personnes")
for i, val in enumerate(counts_tranches.values):
    ax2.text(i, val + 30, f"{val/len(df)*100:.1f}%",
             ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("graphiques/08_comparaison_v1_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/08_comparaison_v1_v2.png")

# résumé final

print("\n" + "=" * 65)
print("  résume – décisiosn éthiques appliquées")
print("=" * 65)
print(f"""
  Dataset v1 (technique) : {df_v1.shape[0]} lignes × {df_v1.shape[1]} colonnes
  Dataset v2 (éthique)   : {df.shape[0]} lignes × {df.shape[1]} colonnes

  Décisions éthiques appliquées :
  ┌─────────────────────────────────────────────────────────────┐
  │ ❌ suppression RGPD (identifiantes) : nom, prenom           │
  │ ❌ suppression discrimination :                             │
  │    sexe, nationalité_francaise, smoker, taille, poids       │
  │ 🔄 age → age_tranche [18-30, 31-45, 46-60, 61-75]          │
  │ ✅ region conservée (pertinence métier, à surveiller)       │
  │ ✅ situation_familiale conservée (pertinence métier)        │
  └─────────────────────────────────────────────────────────────┘

  référence réglementaire :
  - RGPD Article 4  : données personnelles identifiantes
  - RGPD Article 22 : décisions automatisées discriminatoires
""")

print("  Nettoyage éthique v2 terminé ✅")