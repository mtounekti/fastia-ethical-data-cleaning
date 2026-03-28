# Brief 2 – Analyse Éthique et Nettoyage Complet d'un Dataset

## Description du projet

FastIA dispose d'un nouveau dataset brut contenant des données mixtes (numériques + catégorielles). avant toute utilisation par un modèle d'IA, ces données doivent être **analysées sous l'angle éthique**, nettoyées techniquement puis transformées conformément aux décisions éthiques et réglementaires (RGPD)

Le pipeline produit **deux datasets** :
- `dataset_v1_propre.csv` → nettoyage technique uniquement
- `dataset_v2_ethique.csv` → nettoyage technique + décisions éthiques appliquées

---

## Structure du dépôt

```
fastia-ethical-data-cleaning/
├── eda.py                        # analyse exploratoire (EDA)
├── nettoyage_technique.py        # pipeline nettoyage technique → v1
├── nettoyage_ethique.py          # pipeline nettoyage éthique → v2
├── datasheet.md                  # analyse éthique et décisions par feature
├── README.md                     # documentation
├── requirements.txt              # dépendances Python
├── datasets/
│   ├── dataset_v1_propre.csv     # dataset nettoyé techniquement
│   └── dataset_v2_ethique.csv    # dataset nettoyé éthiquement
└── graphiques/
    ├── 01_valeurs_manquantes.png
    ├── 02_distributions_numeriques.png
    ├── 03_variables_categorielles.png
    ├── 04_boxplots_outliers.png
    ├── 05_anciennete_compte.png
    ├── 06_correlation.png
    ├── 07_comparaison_v1.png
    └── 08_comparaison_v1_v2.png
```

---

## Description du dataset initial

| Propriété | Valeur |
|---|---|
| Lignes | 10 000 |
| Colonnes | 19 |
| Types | Numériques + catégorielles + textuelles + temporelle |
| Contexte métier | Évaluation du risque crédit / octroi de prêts |

### anomalies détectées

- `historique_credits` : 52.9% de NaN → supprimée
- `score_credit` : 53.1% de NaN → supprimée
- `loyer_mensuel` : 29.1% de NaN → imputée par KNN
- `situation_familiale` : 23.5% de NaN → imputée par mode
- `loyer_mensuel` min = -395€ → valeur aberrante corrigée
- `montant_pret` : 103 outliers → winsorisés

---

## analyse éthique

La datasheet complète est disponible dans `datasheet.md`. Voici le résumé des décisions :

| Décision | Colonnes concernées | Justification |
|---|---|---|
| ❌ Suppression obligatoire | `nom`, `prenom` | Données identifiantes – RGPD Art. 4 |
| ❌ Suppression discrimination | `sexe`, `nationalité_francaise`, `smoker`, `taille`, `poids` | Décisions automatisées – RGPD Art. 22 |
| ❌ Suppression (trop vide) | `historique_credits`, `score_credit` | > 40% NaN |
| 🔄 Transformation | `age` → tranches [18-30, 31-45, 46-60, 61-75] | Réduction granularité discriminante |
| 🔄 Transformation | `date_creation_compte` → `anciennete_jours` | Non exploitable brut par un modèle |
| ✅ Conservée avec vigilance | `region` | Pertinence métier – proxy à surveiller |
| ✅ Conservée | `situation_familiale` | Pertinence métier |

---

## méthodologie

### pipeline nettoyage technique (v1)

1. **Suppression colonnes quasi-vides** (> 40% NaN) : `historique_credits`, `score_credit`
2. **Imputation** : `situation_familiale` → mode, `loyer_mensuel` → KNN (k=5)
3. **Winsorisation** outliers (IQR × 1.5) : `revenu_estime_mois`, `loyer_mensuel`, `montant_pret`
4. **Variable temporelle** : `date_creation_compte` → `anciennete_jours`
5. **Label Encoding** (0/1) : `sport_licence`, `smoker`, `nationalité_francaise`, `sexe`
6. **Ordinal Encoding** : `niveau_etude` (aucun=0 → doctorat=4)
7. **One-Hot Encoding** : `situation_familiale`, `region`
8. **Scaling** : StandardScaler (distributions normales) + MinMaxScaler (distributions asymétriques)

### pipeline nettoyage éthique (v2)

À partir du dataset v1, application des décisions éthiques :

1. **Suppression RGPD** : `nom`, `prenom`
2. **Suppression discrimination** : `sexe`, `nationalité_francaise`, `smoker`, `taille`, `poids`
3. **Transformation âge** → tranches ordinales (0 à 3)

---

## comparatif des datasets

| | Dataset brut | Dataset v1 (technique) | Dataset v2 (éthique) |
|---|---|---|---|
| Lignes | 10 000 | 10 000 | 10 000 |
| Colonnes | 19 | 27 | 20 |
| Valeurs manquantes | 10 464 | **0** | **0** |
| Colonnes sensibles | ✅ présentes | ✅ présentes | ❌ supprimées |
| Colonnes encodées | ❌ | ✅ | ✅ |
| Scaling | ❌ | ✅ | ✅ |

> Le v1 contient plus de colonnes que le brut car l'encodage One-Hot crée de nouvelles colonnes (région × 8, situation familiale × 4).

---

## colonnes du dataset v2 éthique (final)

| Colonne | Description |
|---|---|
| `sport_licence` | Licence sportive (0/1) |
| `niveau_etude` | Niveau d'étude ordinal (0=aucun → 4=doctorat) |
| `revenu_estime_mois` | Revenu mensuel standardisé |
| `risque_personnel` | Score de risque standardisé |
| `loyer_mensuel` | Loyer normalisé [0-1] |
| `montant_pret` | Montant prêt normalisé [0-1] |
| `anciennete_jours` | Ancienneté du compte standardisée |
| `sit_fam_*` | Situation familiale (One-Hot × 4) |
| `region_*` | Région (One-Hot × 8) |
| `age_tranche` | Tranche d'âge ordinale (0=18-30 → 3=61-75) |

---

## reproductibilité

```bash
# installation des dépendances
pip install -r requirements.txt

# 1. analyse exploratoire
python3 eda.py

# 2. nettoyage technique → dataset v1
python3 nettoyage_technique.py

# 3. nettoyage éthique → dataset v2
python3 nettoyage_ethique.py
```

---

## 🔗 Références réglementaires

- [RGPD – Article 4 : données personnelles](https://www.cnil.fr/fr/reglement-europeen-protection-donnees)
- [RGPD – Article 22 : décisions automatisées](https://www.cnil.fr/fr/reglement-europeen-protection-donnees/chapitre3#Article22)
- [CNIL – Guide IA et RGPD](https://www.cnil.fr/fr/ia-et-rgpd)