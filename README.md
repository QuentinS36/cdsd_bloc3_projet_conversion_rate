# Projet Conversion rate challenge 

Certification CDSD RNCP35288 - Bloc 3 : Analyse prédictive de données
structurées par l'intelligence artificielle


## Objectif

Prédire si un visiteur du site *datascienceweekly.org* s'inscrit à la
newsletter, et identifier des leviers pour améliorer le taux de conversion.


## Données

Jeu de données 'data/conversion_data_train.csv' pour réaliser le projet 
et 'data/conversion_data_test.csv' réservé exclusivement à la soumission 
dans le cadre du challenge.


## Etapes du projet

1. **Données et nettoyage** : 284 580 visiteurs, 5 variables (`country`,
   `age`, `new_user`, `source`, `total_pages_visited`), aucune valeur
   manquante. 2 âges considérés aberrants (111 et 123 ans) supprimés. 94 % de lignes
   sont des doublons stricts, conservés car ils représentent des visiteurs
   distincts au même profil.
2. **Analyse exploratoire** : cible très déséquilibrée (3,2 % de conversions),
   conversion croissante avec le nombre de pages, décroissante avec l'âge,
   très inégale selon le pays (Chine notamment) et le statut de nouvel utilisateur.
3. **Préparation** : séparation stratifiée 80 / 20 (test conservé à l'écart, 
   à ne pas confondre avec le .csv test fourni pour le classement du challenge),
   `ColumnTransformer` (imputation, standardisation, one-hot) intégré au
   pipeline pour éviter toute fuite d'information.
4. **Évaluation** : validation croisée stratifiée à 5 plis sur le train
   uniquement ; F1 comme métrique principale, PR-AUC et ROC-AUC en complément.
5. **Modèles comparés** : régression logistique (avec et sans
   `class_weight="balanced"`), Random Forest, XGBoost.
6. **Amélioration** : optimisation du seuil de décision sur les prédictions
   out-of-fold, `GridSearchCV` léger (critère PR-AUC).
7. **Évaluation finale** : une seule fois sur le jeu de test.
8. **Interprétation** : odds ratios de la régression logistique.
9. **Suivi MLflow** et **export de la soumission**, après ré-entraînement sur
   l'ensemble des données étiquetées.


## Résultats en bref

| Indicateur | Valeur |
|---|---|
| Modèle retenu | Régression logistique (`C=1`), seuil de décision 0,43 |
| **F1-score, jeu de test** | **0,764** (précision 0,83, rappel 0,71) |
| F1-score, validation croisée (5 plis) | 0,772 |
| ROC-AUC / PR-AUC (test) | 0,986 / 0,839 |

Points clés :
- Le **plafond de performance est d'environ 0,77 de F1** : Random Forest,
  XGBoost et réglage des hyperparamètres apportent peu par rapport à
  une régression logistique.
- Le **réglage du seuil de décision** (0,43 au lieu de 0,5) est le seul levier
  simple qui améliore le F1, et son gain est modeste (moins d'un point).
  `class_weight="balanced"` fait chuter le F1 à 0,51 au seuil 0,5 : à éviter
  quand on vise le F1 au seuil par défaut.
- Toutes choses égales par ailleurs, les chances de conversion d'un visiteur
  **chinois** (24 % du trafic ; 0,13 % de conversion contre 3,8 % aux
  États-Unis) sont environ 20 fois plus faibles que celles d'un visiteur
  américain, celles des **nouveaux utilisateurs** environ 5 à 6 fois plus
  faibles que celles des utilisateurs connus, et la **source de trafic** n'a
  qu'un effet faible (Direct un peu en dessous de Seo).
- ⚠️ **`total_pages_visited` n'est pas un levier prouvé** : le taux de
  conversion y atteint 100 % à partir de 21 pages, ce qui suggère que la
  variable peut être en partie une conséquence de l'inscription. À clarifier
  avec l'équipe qui a produit les données.


## Recommandations

1. **Comprendre la Chine** : un quart du trafic pour 0,13 % de conversion,
   alors que les pages vues par visiteur sont comparables (4,5 contre 4,9).
   Pistes : langue, accessibilité du formulaire, trafic non qualifié. Ordre
   de grandeur si la Chine convertissait comme les États-Unis : environ
   +2 500 inscriptions (+28 %), hypothèse haute.
2. **Travailler la conversion des nouveaux visiteurs** (1,4 % contre 7,2 %
   pour les visiteurs connus) et favoriser leur retour.
3. **Développer l'acquisition en Allemagne et au Royaume-Uni**, meilleurs taux
   de conversion mais faible part du trafic.
4. **Cibler les publics plus jeunes**, qui convertissent davantage.
5. **Ne pas arbitrer les canaux sur la seule conversion** : les écarts entre
   `Ads` (3,5 %), `Seo` (3,3 %) et `Direct` (2,8 %) sont modestes ; le choix
   doit se faire surtout sur le coût d'acquisition.
6. **Valider la mesure de `total_pages_visited`** et tester tout levier
   éventuel par une expérience contrôlée (test A/B).


## Limites

- Seulement cinq variables : au-delà de ~0,77 de F1, il faudrait des données
  de comportement plus riches (temps passé, page d'entrée, appareil).
- Les odds ratios décrivent des associations, pas des relations causales.
- Les très nombreux profils identiques limitent la capacité de n'importe quel
  modèle à les départager.
- La stabilité du score n'a pas été vérifiée par des répétitions de la
  validation croisée (piste d'amélioration).


## Structure du dépôt

```
cdsd_bloc3_projet_conversion_rate/
├── assets
│   └── images
│       ├── 01_target_variable.png
│       ├── 02_distributions.png
│       ├── 03_rate_pages_age.png
│       ├── 04_rate_categories.png
│       ├── 05_baselines.png
│       ├── 06_threshold.png
│       ├── 07_confusion_matrix.png
│       └── 08_odds_ratios.png
├── data
│   ├── conversion_data_test.csv
│   └── conversion_data_train.csv
├── mlruns
│   └── mlflow.db
├── notebook
│   └── projet_conversion_rate_challenge.ipynb
├── submissions
│   └── conversion_data_test_predictions_qs.csv
├── README.md
└── requirements.txt
```

## Installation et exécution

```powershell
conda create -n cdsd_b3_conversion_rate python=3.11.15 pip -y
conda activate cdsd_b3_conversion_rate
python -m pip install -r requirements.txt
```

1. Placer `conversion_data_train.csv` et `conversion_data_test.csv` dans
   `data/`.
2. Lancer Jupyter et ouvrir `notebooks/conversion_rate_challenge.ipynb`.
3. Exécuter toutes les cellules (**Restart & Run All**, environ 2 à 3 minutes).

Le notebook retrouve automatiquement la racine du projet, qu'il soit lancé
depuis `notebooks/` ou depuis la racine.

Pour consulter le suivi des expériences :

```powershell
python -m mlflow ui --backend-store-uri "sqlite:///mlruns/mlflow.db"
```

## Soumission

Le notebook écrit `submissions/conversion_data_test_predictions_<nom>.csv`
avec une seule colonne `converted` (0/1) et sans index, dans l'ordre des
lignes de `conversion_data_test.csv`, comme attendu dans le cadre du challenge. 
Le nom est défini par la variable `SUBMISSION_NAME` (cellule de configuration) :
**à remplacer par le nom de l'utilisateur**, au format `prénom_nom`.


