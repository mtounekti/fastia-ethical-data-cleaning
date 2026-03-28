# Datasheet – Dataset Mixte FastIA
> Projet : Brief 2 – Analyse Éthique et Nettoyage Complet
> Auteur : FastIA Data Team
> Date : 2025

---

## 1. Origine des données

| Propriété | Valeur |
|---|---|
| Source | Données synthétiques fournies par FastIA |
| Contexte métier | Évaluation du risque crédit et octroi de prêts |
| Nombre de lignes | 10 000 |
| Nombre de colonnes | 19 |
| Types de données | Numériques + catégorielles + textuelles |

---

## 2. Description et classification des features

### 2.1 Données directement identifiantes ❌ RGPD

| Colonne | Type | Sensibilité | Décision |
|---|---|---|---|
| `nom` | Texte | 🔴 Directement identifiante | **Suppression obligatoire** |
| `prenom` | Texte | 🔴 Directement identifiante | **Suppression obligatoire** |

> **Justification** : Le nom et le prénom sont des données à caractère personnel au sens de l'article 4 du RGPD. Leur présence dans un dataset d'entraînement est illégale sans consentement explicite et anonymisation préalable.

---

### 2.2 Données sensibles discriminantes ⚠️

| Colonne | Type | Sensibilité | Décision |
|---|---|---|---|
| `sexe` | Catégorielle (H/F) | 🔴 Discriminante | **Suppression** |
| `nationalité_francaise` | Catégorielle (oui/non) | 🔴 Discriminante | **Suppression** |
| `smoker` | Catégorielle (oui/non) | 🟠 Potentiellement discriminante | **Suppression** |
| `taille` | Numérique | 🟠 Donnée corporelle personnelle | **Suppression** |
| `poids` | Numérique | 🟠 Donnée corporelle personnelle | **Suppression** |

> **Justification** :
> - `sexe` et `nationalité_francaise` : l'article 22 du RGPD interdit les décisions automatisées basées sur des caractéristiques discriminantes. Un modèle entraîné sur ces variables risque de reproduire et amplifier des biais systémiques.
> - `smoker` : variable de santé pouvant entraîner une discrimination indirecte.
> - `taille` et `poids` : données corporelles sans pertinence métier pour l'évaluation d'un crédit.

---

### 2.3 Données à traitement prudent 🟡

| Colonne | Type | Sensibilité | Décision |
|---|---|---|---|
| `age` | Numérique | 🟡 Potentiellement discriminant | **Transformation en tranches d'âge** |
| `region` | Catégorielle | 🟡 Proxy potentiel (origine sociale/ethnique) | **Conservée avec vigilance** |
| `situation_familiale` | Catégorielle | 🟡 Potentiellement discriminante | **Conservée – pertinence métier** |

> **Justification** :
> - `age` : variable utile métier mais discriminante si utilisée brute. On la transforme en tranches pour réduire la granularité discriminante : [18-30], [31-45], [46-60], [61-75].
> - `region` : peut servir de proxy pour l'origine sociale. Conservée car pertinente pour l'évaluation du marché immobilier local (loyer, prix). À surveiller lors de l'évaluation du modèle.
> - `situation_familiale` : pertinente pour l'évaluation de la stabilité financière. Conservée mais à surveiller pour éviter une discrimination envers les personnes célibataires ou divorcées.

---

### 2.4 Données catégorielles non sensibles ✅

| Colonne | Type | Nature | Encodage |
|---|---|---|---|
| `sport_licence` | Catégorielle nominale (oui/non) | Non sensible | **Label Encoding** (0/1) |
| `niveau_etude` | Catégorielle **ordinale** | Non sensible | **Ordinal Encoding** (ordre croissant) |

> **Ordre ordinal `niveau_etude`** : aucun(0) → bac(1) → bac+2(2) → master(3) → doctorat(4)

---

### 2.5 Données numériques ✅

| Colonne | Type | % NaN | Distribution | Traitement |
|---|---|---|---|---|
| `age` | Numérique entier | 0% | Approximativement normale | Tranches d'âge |
| `revenu_estime_mois` | Numérique entier | 0% | Approximativement normale | StandardScaler |
| `historique_credits` | Numérique float | **52.9%** | — | **Suppression** (trop vide) |
| `risque_personnel` | Numérique float | 0% | Approximativement normale | StandardScaler |
| `score_credit` | Numérique float | **53.1%** | — | **Suppression** (trop vide) |
| `loyer_mensuel` | Numérique float | 29.1% | Bimodale | KNN Imputation + MinMaxScaler |
| `montant_pret` | Numérique float | 0% | Asymétrique droite | MinMaxScaler |

---

### 2.6 Données temporelles et textuelles

| Colonne | Type | Décision |
|---|---|---|
| `date_creation_compte` | Date (texte) | **Transformation** → ancienneté en jours |

> **Justification** : La date brute n'est pas exploitable par un modèle. On la transforme en nombre de jours depuis la création du compte, ce qui est une variable métier pertinente (fidélité client).

---

## 3. Résumé des décisions éthiques

| Décision | Colonnes concernées |
|---|---|
| ❌ Suppression obligatoire (RGPD) | `nom`, `prenom` |
| ❌ Suppression (discrimination) | `sexe`, `nationalité_francaise`, `smoker`, `taille`, `poids` |
| ❌ Suppression (trop vide) | `historique_credits`, `score_credit` |
| 🔄 Transformation | `age` → tranches, `date_creation_compte` → ancienneté |
| ✅ Conservée avec encodage | `niveau_etude`, `sport_licence`, `region`, `situation_familiale` |
| ✅ Conservée avec scaling | `revenu_estime_mois`, `risque_personnel`, `loyer_mensuel`, `montant_pret` |

---

## 4. Datasets produits

| Dataset | Description |
|---|---|
| `dataset_v1_propre.csv` | Nettoyage technique : NaN, outliers, encodage, scaling. Colonnes sensibles encore présentes. |
| `dataset_v2_ethique.csv` | Nettoyage éthique : suppression des colonnes discriminantes, transformation de `age` et `date_creation_compte`. |

---

## 5. Références réglementaires

- [RGPD – Article 4 : définition des données personnelles](https://www.cnil.fr/fr/reglement-europeen-protection-donnees)
- [RGPD – Article 22 : décisions automatisées](https://www.cnil.fr/fr/reglement-europeen-protection-donnees/chapitre3#Article22)
- [Loi Informatique et Libertés](https://www.cnil.fr/fr/la-loi-informatique-et-libertes)