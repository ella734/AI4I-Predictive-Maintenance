"""Explications SHAP et LIME pour le pipeline XGBoost du projet AI4I."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

FEATURES = [
    "Type", "Air temperature", "Process temperature",
    "Rotational speed", "Torque", "Tool wear"
]


def _transform(pipeline, X):
    """Applique le ColumnTransformer déjà ajusté pendant l'entraînement."""
    preprocessing = pipeline.named_steps["preprocessing"]
    X_processed = preprocessing.transform(X.loc[:, FEATURES])
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    names = preprocessing.get_feature_names_out().tolist()
    return pd.DataFrame(np.asarray(X_processed, dtype=float), columns=names)


def _classifier(pipeline):
    return pipeline.named_steps["classifier"]


def shap_global(pipeline, X):
    """Diagramme des importances SHAP moyennes sur X."""
    transformed = _transform(pipeline, X)
    explainer = shap.TreeExplainer(_classifier(pipeline))
    explanation = explainer(transformed)
    plt.close("all")
    shap.plots.bar(explanation, max_display=len(transformed.columns), show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig, explanation


def shap_local(pipeline, observation):
    """Waterfall SHAP d'une observation, en log-odds pour XGBoost binaire."""
    transformed = _transform(pipeline, observation.iloc[[0]])
    explainer = shap.TreeExplainer(_classifier(pipeline))
    explanation = explainer(transformed)
    plt.close("all")
    shap.plots.waterfall(explanation[0], max_display=len(transformed.columns), show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def lime_local(pipeline, X_reference, observation):
    """LIME sur les colonnes encodées, avec fond issu du jeu d'entraînement."""
    background = _transform(pipeline, X_reference)
    instance = _transform(pipeline, observation.iloc[[0]])
    explainer = LimeTabularExplainer(
        training_data=background.to_numpy(),
        feature_names=background.columns.tolist(),
        class_names=["Pas de panne", "Panne"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )
    explanation = explainer.explain_instance(
        data_row=instance.iloc[0].to_numpy(),
        predict_fn=_classifier(pipeline).predict_proba,
        num_features=background.shape[1],
        labels=(1,),
        num_samples=1000
    )
    fig = explanation.as_pyplot_figure(label=1)
    fig.tight_layout()
    return fig
