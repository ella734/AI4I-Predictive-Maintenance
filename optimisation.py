
import os
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

dataset = fetch_ucirepo(id=601)
df = dataset.data.original.copy()

features = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]

X = df[features].copy()
y = df["Machine failure"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dimensions Train :", X_train.shape)
print("Dimensions Test :", X_test.shape)

numeric_features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]

categorical_features = ["Type"]

# 5. Pretraitement avec standardisation
numeric_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_scaled = ColumnTransformer([
    ("num", numeric_scaled, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 6. Pretraitement sans standardisation
numeric_unscaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor_unscaled = ColumnTransformer([
    ("num", numeric_unscaled, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 7. Calculer le poids des pannes
nb_normales = (y_train == 0).sum()
nb_pannes = (y_train == 1).sum()

poids_pannes = nb_normales / nb_pannes

print("Poids des pannes :", poids_pannes)

# 8. Validation croisee stratifiee
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 9. Definir les cinq modeles
configurations = {

    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "preprocessor": preprocessor_scaled,
        "params": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"]
        }
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(
            random_state=42
        ),
        "preprocessor": preprocessor_unscaled,
        "params": {
            "classifier__max_depth": [3, 5, 10, 15, None],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    },

    "KNN": {
        "model": KNeighborsClassifier(),
        "preprocessor": preprocessor_scaled,
        "params": {
            "classifier__n_neighbors": [3, 5, 7, 11, 15],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ),
        "preprocessor": preprocessor_unscaled,
        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [5, 10, 15, None],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__class_weight": [None, "balanced"]
        }
    },

    "XGBoost": {
        "model": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        ),
        "preprocessor": preprocessor_unscaled,
        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 4, 6],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__scale_pos_weight": [1, poids_pannes],
            "classifier__subsample": [0.8, 1.0]
        }
    }
}

# 10. Creer le dossier de sauvegarde
os.makedirs("models", exist_ok=True)

# 11. Optimiser chaque modele
resultats = []

for nom, config in configurations.items():

    print("\n================================")
    print("Optimisation :", nom)
    print("================================")

    # Construire le pipeline complet
    pipeline = Pipeline([
        ("preprocessing", config["preprocessor"]),
        ("classifier", config["model"])
    ])

    # Recherche des meilleurs hyperparametres
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=config["params"],
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Entrainement avec validation croisee
    search.fit(X_train, y_train)

    print("\nMeilleurs hyperparametres :")
    print(search.best_params_)

    print("\nMeilleur score CV :")
    print(search.best_score_)

    # Recuperer le meilleur modele
    best_model = search.best_estimator_

    # Predictions sur Test
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculer les metriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    print("\nAccuracy :", accuracy)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1-score :", f1)
    print("ROC-AUC :", roc_auc)
    print("Average Precision :", ap)

    # Sauvegarder le modele
    nom_fichier = nom.lower().replace(" ", "_")
    chemin = f"models/{nom_fichier}.joblib"

    joblib.dump(best_model, chemin)

    print("\nModele sauvegarde :", chemin)

    # Stocker les resultats
    resultats.append({
        "Modele": nom,
        "CV Average Precision": search.best_score_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "Average Precision": ap
    })

# 12. Creer le tableau comparatif
resultats_df = pd.DataFrame(resultats)

resultats_df = resultats_df.sort_values(
    by="CV Average Precision",
    ascending=False
)


print("TABLEAU COMPARATIF FINAL")


print(resultats_df.to_string(index=False))

# 13. Enregistrer le tableau
resultats_df.to_csv(
    "resultats_optimisation.csv",
    index=False
)

print("\nResultats enregistres dans resultats_optimisation.csv")
print("Modeles enregistres dans le dossier models/")
