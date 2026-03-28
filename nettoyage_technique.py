# NETTOYAGE TECHNIQUE – DATASET V1
# Ce script réalise le nettoyage technique complet du dataset brut :
#   1. Suppression des colonnes quasi-vides (> 40% NaN)
#   2. Imputation des valeurs manquantes
#   3. Traitement des outliers (winsorisation)
#   4. Encodage des variables catégorielles et ordinales
#   5. Transformation de la variable temporelle
#   6. Normalisation / Standardisation
#   7. Export du dataset v1 propre
# NOTE : Ce dataset v1 conserve encore les colonnes sensibles (nom, prénom,
# sexe, etc.). Le nettoyage éthique sera appliqué dans le script suivant
# (nettoyage_ethique.py) pour produire le dataset v2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
import warnings
import os

warnings.filterwarnings("ignore")

#  global params
FICHIER_SOURCE  = "fichier-de-donnees-mixtes-6920344a2a6cd267411281.csv"
FICHIER_V1  = "datasets/dataset_v1_propre.csv"
SEUIL_COL_VIDE  = 0.40   # suppression colonnes > 40% NaN
SEUIL_IQR = 1.5    # multiplicateur IQR pour winsorisation

COULEUR_AVANT = "#E07B54"
COULEUR_APRES = "#5B8DB8"

os.makedirs("datasets",   exist_ok=True)
os.makedirs("graphiques", exist_ok=True)

# section 1: loading
print("=" * 65)
print("  ÉTAPE 1 – CHARGEMENT DU DATASET BRUT")
print("=" * 65)

df_brut = pd.read_csv(FICHIER_SOURCE)
df = df_brut.copy()

print(f"\n✔  Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# section2: SUPPRESSION DES COLONNES QUASI-VIDES

print("\n" + "=" * 65)
print("  ÉTAPE 2 – SUPPRESSION COLONNES QUASI-VIDES (> 40% NaN)")
print("=" * 65)

# historique_credits (52.9%) et score_credit (53.1%) → trop vides
# toute imputation introduirait un biais trop important
colonnes_a_supprimer = [col for col in df.columns
                        if df[col].isnull().mean() > SEUIL_COL_VIDE]

for col in colonnes_a_supprimer:
    print(f"\n  - {col} ({df[col].isnull().mean()*100:.1f}% NaN) → supprimée")

df = df.drop(columns=colonnes_a_supprimer)
print(f"\n✔  Dimensions restantes : {df.shape}")

# SECTION 3 – IMPUTATION DES VALEURS MANQUANTES
print("\n" + "=" * 65)
print("  ÉTAPE 3 – IMPUTATION DES VALEURS MANQUANTES")
print("=" * 65)

print(f"\n  NaN restants avant imputation :")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# 3.1 situation_familiale (23.5% NaN) → mode (valeur la plus fréquente)
# Donnée catégorielle → on impute avec le mode
mode_situation = df["situation_familiale"].mode()[0]
df["situation_familiale"] = df["situation_familiale"].fillna(mode_situation)
print(f"\n  [3.1] situation_familiale → imputée avec le mode : '{mode_situation}'")

# 3.2 loyer_mensuel (29.1% NaN) → KNN Imputer (k=5)
# Variable numérique corrélée aux autres → KNN plus précis que la médiane
colonnes_numeriques = df.select_dtypes(include=np.number).columns.tolist()
imputer = KNNImputer(n_neighbors=5)
df[colonnes_numeriques] = imputer.fit_transform(df[colonnes_numeriques])
print(f"  [3.2] loyer_mensuel → imputé par KNN (k=5)")

print(f"\n  NaN restants après imputation : {df.isnull().sum().sum()}")
print("  ✔  Aucune valeur manquante restante.")


# SECTION 4: TRAITEMENT DES OUTLIERS (WINSORISATION)

print("\n" + "=" * 65)
print("  ÉTAPE 4 – TRAITEMENT DES OUTLIERS (WINSORISATION IQR × 1.5)")
print("=" * 65)

# On ne winseorise pas taille/poids car ils seront supprimés dans v2
# On winsorise les colonnes financières et revenu
colonnes_a_clipper = ["revenu_estime_mois", "loyer_mensuel", "montant_pret"]

for col in colonnes_a_clipper:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    borne_basse = Q1 - SEUIL_IQR * IQR
    borne_haute = Q3 + SEUIL_IQR * IQR
    nb_avant = ((df[col] < borne_basse) | (df[col] > borne_haute)).sum()
    df[col] = df[col].clip(lower=borne_basse, upper=borne_haute)
    print(f"\n  {col} : {nb_avant} valeurs winsorisées [{borne_basse:.2f} ; {borne_haute:.2f}]")

# section5: TRANSFORMATION DE LA VARIABLE TEMPORELLE

print("\n" + "=" * 65)
print("  ÉTAPE 5 – VARIABLE TEMPORELLE")
print("=" * 65)

# date_creation_compte → ancienneté en jours
# La date brute n'est pas exploitable par un modèle ML
df["date_creation_compte"] = pd.to_datetime(df["date_creation_compte"])
date_reference = df["date_creation_compte"].max()
df["anciennete_jours"] = (date_reference - df["date_creation_compte"]).dt.days
df = df.drop(columns=["date_creation_compte"])

print(f"\n  date_creation_compte → anciennete_jours")
print(f"  Min : {df['anciennete_jours'].min()} | Max : {df['anciennete_jours'].max()} | "
      f"Moy : {df['anciennete_jours'].mean():.0f} jours")
print("  ✔  Transformation effectuée")

# section 6: ENCODAGE DES VARIABLES CATÉGORIELLES

print("\n" + "=" * 65)
print("  ÉTAPE 6 – ENCODAGE DES VARIABLES CATÉGORIELLES")
print("=" * 65)

# 6.1 Variables binaires → label encoding (0/1)
# sport_licence, smoker, nationalité_francaise, sexe → oui/non ou H/F
binaires = {
    "sport_licence":        {"oui": 1, "non": 0},
    "smoker":               {"oui": 1, "non": 0},
    "nationalité_francaise": {"oui": 1, "non": 0},
    "sexe":                 {"H": 1, "F": 0},
}

for col, mapping in binaires.items():
    df[col] = df[col].map(mapping)
    print(f"\n  [6.1] {col} → Label Encoding {mapping}")

# 6.2 niveau_etude → ordinal encoding
# Variable ordinale : il y a un ordre naturel entre les niveaux d'étude
ordre_etude = ["aucun", "bac", "bac+2", "master", "doctorat"]
df["niveau_etude"] = df["niveau_etude"].map(
    {val: i for i, val in enumerate(ordre_etude)}
)
print(f"\n  [6.2] niveau_etude → Ordinal Encoding {dict(enumerate(ordre_etude))}")

# 6.3 situation_familiale → One-Hot Encoding
# Variable nominale sans ordre → on crée des colonnes binaires
df = pd.get_dummies(df, columns=["situation_familiale"], prefix="sit_fam", dtype=int)
print(f"\n  [6.3] situation_familiale → One-Hot Encoding")
print(f"        Nouvelles colonnes : {[c for c in df.columns if c.startswith('sit_fam')]}")

# 6.4 region → One-Hot Encoding
df = pd.get_dummies(df, columns=["region"], prefix="region", dtype=int)
print(f"\n  [6.4] region → One-Hot Encoding")
print(f"        Nouvelles colonnes : {[c for c in df.columns if c.startswith('region_')]}")

# 6.5 nom / prénom → conservés dans v1 seront supprimés dans v2
print(f"\n  [6.5] nom, prenom → conservés dans v1 (supprimés dans v2 éthique)")

print(f"\n✔  Dimensions après encodage : {df.shape}")

# section 7: NORMALISATION / STANDARDISATION
print("\n" + "=" * 65)
print("  ÉTAPE 7 – NORMALISATION / STANDARDISATION")
print("=" * 65)

# Règle: distribution normale → StandardScaler
#  distribution non normale → MinMaxScaler
# (basé sur l'observation des distributions dans l'EDA)

colonnes_standard = ["age", "revenu_estime_mois", "risque_personnel",
                     "taille", "poids", "anciennete_jours", "niveau_etude"]
colonnes_minmax   = ["loyer_mensuel", "montant_pret"]

# on ne scale pas les colonnes binaires ni les one-hot :)

scaler_std    = StandardScaler()
scaler_minmax = MinMaxScaler()

df[colonnes_standard] = scaler_std.fit_transform(df[colonnes_standard])
df[colonnes_minmax]   = scaler_minmax.fit_transform(df[colonnes_minmax])

print(f"\n  StandardScaler : {colonnes_standard}")
print(f"  MinMaxScaler   : {colonnes_minmax}")
print(f"\n✔  Scaling appliqué")

# section 8: VÉRIFICATION ET EXPORT
print("\n" + "=" * 65)
print("  ÉTAPE 8 – VÉRIFICATION ET EXPORT")
print("=" * 65)

print(f"\n  Dimensions finales : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"  Valeurs manquantes : {df.isnull().sum().sum()}")
print(f"  Colonnes : {list(df.columns)}")

df.to_csv(FICHIER_V1, index=False)
print(f"\n✔  Dataset v1 exporté : {FICHIER_V1}")

# section 9: GRAPHIQUE COMPARATIF
colonnes_communes_num = ["age", "revenu_estime_mois", "risque_personnel",
                         "loyer_mensuel", "montant_pret"]

fig, axes = plt.subplots(2, len(colonnes_communes_num), figsize=(18, 7))
fig.suptitle("Distributions avant / après nettoyage technique (v1)",
             fontsize=13, fontweight="bold")

for i, col in enumerate(colonnes_communes_num):
    # le avant
    sns.histplot(df_brut[col].dropna(), kde=True, ax=axes[0][i],
                 color=COULEUR_AVANT, edgecolor="white")
    axes[0][i].set_title(f"{col}\nAVANT", fontsize=9)

    # l'après
    sns.histplot(df[col], kde=True, ax=axes[1][i],
                 color=COULEUR_APRES, edgecolor="white")
    axes[1][i].set_title(f"{col}\nAPRÈS", fontsize=9)

plt.tight_layout()
plt.savefig("graphiques/07_comparaison_v1.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/07_comparaison_v1.png")

print("\n" + "=" * 65)
print("  NETTOYAGE TECHNIQUE V1 TERMINÉ ✅")
print("=" * 65)
print(f"""
  Dataset initial : {df_brut.shape[0]} lignes × {df_brut.shape[1]} colonnes
  Dataset v1      : {df.shape[0]} lignes × {df.shape[1]} colonnes

  Opérations réalisées :
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Suppression colonnes > 40% NaN :                         │
  │    historique_credits (52.9%), score_credit (53.1%)         │
  │ 2. Imputation : situation_familiale (mode),                 │
  │    loyer_mensuel (KNN k=5)                                  │
  │ 3. Winsorisation outliers (IQR × 1.5) :                     │
  │    revenu_estime_mois, loyer_mensuel, montant_pret          │
  │ 4. date_creation_compte → anciennete_jours                  │
  │ 5. Label Encoding : sport_licence, smoker,                  │
  │    nationalité_francaise, sexe                              │
  │ 6. Ordinal Encoding : niveau_etude                          │
  │ 7. One-Hot Encoding : situation_familiale, region           │
  │ 8. StandardScaler + MinMaxScaler selon distribution         │
  └─────────────────────────────────────────────────────────────┘
""")