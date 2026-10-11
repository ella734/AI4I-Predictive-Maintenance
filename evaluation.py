
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Charger les donnees
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
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Liste des modeles sauvegardes
modeles = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Random Forest": "models/random_forest.joblib",
    "XGBoost": "models/xgboost.joblib"
}

# 4. Creer le dossier des graphiques
os.makedirs("figures", exist_ok=True)

# 5. Initialiser les graphiques
fig_roc, ax_roc = plt.subplots(figsize=(9, 6))
fig_pr, ax_pr = plt.subplots(figsize=(9, 6))

# 6. Evaluer les cinq modeles
for nom, chemin in modeles.items():

    print("\nEvaluation :", nom)

    model = joblib.load(chemin)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normale", "Panne"]
    )

    disp.plot(
        ax=ax_cm,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax_cm.set_title("Matrice de confusion - " + nom)
    fig_cm.tight_layout()

    fichier = nom.lower().replace(" ", "_")

    fig_cm.savefig(
        f"figures/confusion_{fichier}.png",
        dpi=150
    )

    plt.close(fig_cm)

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    ax_roc.plot(
        fpr,
        tpr,
        label=f"{nom} (AUC={auc:.3f})"
    )

    # Courbe precision-rappel
    precision, recall, _ = precision_recall_curve(
        y_test,
        y_proba
    )

    ap = average_precision_score(y_test, y_proba)

    ax_pr.plot(
        recall,
        precision,
        label=f"{nom} (AP={ap:.3f})"
    )

    print("ROC-AUC :", round(auc, 4))
    print("Average Precision :", round(ap, 4))

# 7. Finaliser la courbe ROC
ax_roc.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Classification aleatoire"
)

ax_roc.set_xlabel("Taux de faux positifs")
ax_roc.set_ylabel("Taux de vrais positifs")
ax_roc.set_title("Comparaison des courbes ROC")
ax_roc.legend()
ax_roc.grid(alpha=0.3)

fig_roc.tight_layout()
fig_roc.savefig("figures/roc_comparison.png", dpi=150)
plt.close(fig_roc)

# 8. Finaliser la courbe precision-rappel
baseline = y_test.mean()

ax_pr.axhline(
    baseline,
    linestyle="--",
    color="gray",
    label=f"Baseline ({baseline:.3f})"
)

ax_pr.set_xlabel("Rappel")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Comparaison des courbes Precision-Rappel")
ax_pr.legend()
ax_pr.grid(alpha=0.3)

fig_pr.tight_layout()
fig_pr.savefig("figures/pr_comparison.png", dpi=150)
plt.close(fig_pr)

# 9. Analyse des seuils pour XGBoost
model_xgb = joblib.load("models/xgboost.joblib")
proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

seuils = np.arange(0.1, 1.0, 0.1)

resultats_seuils = []

for seuil in seuils:

    predictions = (proba_xgb >= seuil).astype(int)

    resultats_seuils.append({
        "Seuil": round(float(seuil), 2),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

df_seuils = pd.DataFrame(resultats_seuils)

print("\nANALYSE DES SEUILS - XGBOOST")
print(df_seuils.to_string(index=False))

df_seuils.to_csv(
    "resultats_seuils.csv",
    index=False
)

# 10. Graphique des seuils
fig_seuils, ax_seuils = plt.subplots(figsize=(9, 6))

ax_seuils.plot(
    df_seuils["Seuil"],
    df_seuils["Precision"],
    marker="o",
    label="Precision"
)

ax_seuils.plot(
    df_seuils["Seuil"],
    df_seuils["Recall"],
    marker="o",
    label="Recall"
)

ax_seuils.plot(
    df_seuils["Seuil"],
    df_seuils["F1-score"],
    marker="o",
    label="F1-score"
)

ax_seuils.set_xlabel("Seuil de classification")
ax_seuils.set_ylabel("Score")
ax_seuils.set_title("Influence du seuil - XGBoost")
ax_seuils.legend()
ax_seuils.grid(alpha=0.3)

fig_seuils.tight_layout()
fig_seuils.savefig(
    "figures/threshold_xgboost.png",
    dpi=150
)

plt.close(fig_seuils)

print("\nEvaluation terminee.")
print("Graphiques enregistres dans figures/")
print("Resultats des seuils enregistres dans resultats_seuils.csv")
