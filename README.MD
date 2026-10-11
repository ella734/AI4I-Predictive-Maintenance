# AI4I Predictive Maintenance

## Présentation du projet

Dans le cadre de notre projet de Data Mining, j'ai travaillé sur la détection des défaillances des machines industrielles à l'aide de plusieurs algorithmes de Machine Learning.

L'objectif est d'étudier les données de fonctionnement des machines, de comparer différentes méthodes de classification et de développer une application web permettant de visualiser les résultats et d'effectuer des prédictions.

Pour cela, j'ai utilisé le dataset **AI4I 2020 Predictive Maintenance**, disponible sur UCI Machine Learning Repository.

## 1. Dataset utilisé

Le dataset contient 10 000 observations et plusieurs informations sur le fonctionnement des machines.

Pour entraîner les modèles, j'ai utilisé les six variables suivantes :

- Type de produit
- Température de l'air
- Température du processus
- Vitesse de rotation
- Couple
- Usure de l'outil

La variable cible est `Machine failure`, qui indique si une défaillance est présente (`1`) ou non (`0`).

Lors de l'exploration des données, j'ai remarqué que les classes étaient déséquilibrées : seulement 339 observations correspondent à une panne, contre 9 661 observations normales.

## 2. Exploration des données

J'ai commencé par analyser le dataset pour mieux comprendre sa structure.

Cette étape comprend :
- La vérification des dimensions et des types des variables
- La recherche de valeurs manquantes et de doublons
- L'étude de la répartition des pannes
- L'analyse des variables numériques à l'aide de graphiques
- L'étude des relations entre les caractéristiques des machines et les défaillances

Cette exploration m'a permis de mieux comprendre les données avant de passer à la modélisation.

## 3. Prétraitement

Avant l'entraînement, j'ai préparé les données avec un pipeline Scikit-learn.

J'ai utilisé :
- `SimpleImputer` pour gérer les éventuelles valeurs manquantes
- `OneHotEncoder` pour transformer la variable catégorielle `Type`
- `StandardScaler` pour standardiser les variables numériques lorsque le modèle le nécessite

J'ai ensuite séparé les données en deux ensembles :
- 80 % pour l'entraînement
- 20 % pour le test

J'ai utilisé une séparation stratifiée afin de conserver la proportion des pannes dans les deux ensembles.

## 4. Modèles de Machine Learning

J'ai entraîné et comparé cinq algorithmes :

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Random Forest
5. XGBoost

Pour améliorer leurs performances, j'ai utilisé `RandomizedSearchCV` avec une validation croisée stratifiée à cinq plis.

J'ai choisi l'**Average Precision** comme métrique d'optimisation, car les observations correspondant aux pannes sont minoritaires.

## 5. Résultats obtenus

Après l'optimisation, j'ai évalué les modèles sur le jeu de test.

| Modèle | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|
| Logistic Regression | 0.1600 | 0.8983 | 0.4134 |
| Decision Tree | 0.7692 | 0.9057 | 0.7654 |
| KNN | 0.2532 | 0.8921 | 0.5632 |
| Random Forest | 0.6542 | 0.9721 | 0.7858 |
| XGBoost | 0.6707 | 0.9749 | 0.8281 |

L'arbre de décision a obtenu le meilleur F1-score sur le jeu de test, tandis que XGBoost a obtenu les meilleurs scores ROC-AUC et Average Precision.

J'ai également étudié l'influence du seuil de classification sur les performances de XGBoost. J'ai constaté qu'un seuil plus faible permet généralement de détecter davantage de pannes, mais augmente aussi le nombre de fausses alertes.

## 6. Évaluation des modèles

Pour comparer les performances, j'ai utilisé plusieurs métriques :

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

J'ai aussi généré les matrices de confusion, les courbes ROC et les courbes précision-rappel.

Comme le dataset est déséquilibré, je ne me suis pas limitée à l'accuracy pour évaluer les modèles.

## 7. Application web

Une interface web est en cours de développement avec Streamlit.

Elle permettra de :
- Explorer les données à l'aide de graphiques interactifs
- Consulter les étapes de prétraitement
- Comparer les performances des modèles
- Visualiser les résultats de l'évaluation
- Saisir les caractéristiques d'une machine pour obtenir une prédiction

## 8. API

J'ai également préparé une API avec FastAPI pour utiliser le modèle XGBoost depuis l'application.
Elle comprend trois endpoints :

- `GET /health` : vérifier le fonctionnement de l'API
- `POST /predict` : effectuer une prédiction
- `GET /metrics` : consulter les résultats des modèles

L'intégration et les tests de l'API restent à finaliser.

## 9. Exécution du projet

Après installation des dépendances, l'application Streamlit pourra être lancée avec :

`python -m streamlit run app.py`

L'API pourra être lancée avec :

`python -m uvicorn api:app --reload`

## 10. Conclusion

Ce projet m'a permis de mettre en pratique les différentes étapes d'un projet de Data Mining, depuis l'exploration des données jusqu'à l'entraînement et à l'évaluation de plusieurs modèles.
J'ai notamment appris à mieux gérer un problème de classification avec des classes déséquilibrées et à comprendre pourquoi il est important d'utiliser plusieurs métriques pour comparer les modèles.
La prochaine étape consiste à finaliser l'interface Streamlit et à intégrer l'API pour rendre les prédictions accessibles depuis une application web.
**Remarque :** le dataset AI4I est synthétique. Les modèles apprennent à reconnaître les observations associées à une défaillance ; ils ne permettent pas de prévoir avec certitude la date d'une future panne.