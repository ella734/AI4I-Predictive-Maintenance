
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests

from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

st.set_page_config(
    page_title="AI4I - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8001"

FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib"
}


@st.cache_data
def load_data():
    dataset = fetch_ucirepo(id=601)
    return pd.concat(
        [dataset.data.features, dataset.data.targets],
        axis=1
    )


@st.cache_resource
def load_model(name):
    path = BASE_DIR / "models" / MODEL_FILES[name]
    return joblib.load(path)


@st.cache_data
def load_results():
    path = BASE_DIR / "resultats_optimisation.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def predict_with_api(machine_data):
    response = requests.post(
        f"{API_URL}/predict",
        json=machine_data,
        timeout=15
    )
    response.raise_for_status()
    return response.json()


df = load_data()
X = df[FEATURES].copy()
y = df["Machine failure"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.sidebar.title("⚙️ AI4I")
st.sidebar.write("Predictive Maintenance")

page = st.sidebar.radio(
    "Navigation",
    [
        "Accueil",
        "Exploration des données",
        "Prétraitement",
        "Comparaison des modèles",
        "Évaluation avancée",
        "Prédiction"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Projet Data Mining - AI4I 2020")


if page == "Accueil":
    st.title("⚙️ AI4I - Predictive Maintenance")

    st.write(
        "Application de Data Mining pour analyser les défaillances "
        "des machines industrielles à l'aide du Machine Learning."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Observations", len(df))
    col2.metric("Variables du dataset", df.shape[1])
    col3.metric("Pannes", int(y.sum()))
    col4.metric("Taux de panne", f"{y.mean() * 100:.2f}%")

    st.subheader("Présentation du projet")

    st.write(
        "L'objectif est de comparer cinq algorithmes de classification "
        "pour identifier les observations associées à une panne."
    )

    st.subheader("Aperçu des données")
    st.dataframe(df.head(15), width="stretch")

    st.info(
        "Le dataset AI4I est synthétique. Les prédictions correspondent "
        "à une classification des défaillances et non à une estimation "
        "de la date d'une future panne."
    )


elif page == "Exploration des données":
    st.title("Exploration interactive des données")

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("Répartition des pannes")

    counts = y.value_counts().sort_index()

    repartition = pd.DataFrame({
        "État": ["Sans panne", "Avec panne"],
        "Nombre": [
            counts.get(0, 0),
            counts.get(1, 0)
        ]
    })

    fig = px.bar(
        repartition,
        x="État",
        y="Nombre",
        color="État",
        title="Répartition des observations"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Distribution des variables")

    numerical_features = FEATURES[1:]

    variable = st.selectbox(
        "Choisir une variable",
        numerical_features
    )

    df_plot = df.copy()

    df_plot["État"] = df_plot["Machine failure"].map({
        0: "Sans panne",
        1: "Avec panne"
    })

    fig = px.histogram(
        df_plot,
        x=variable,
        color="État",
        nbins=40,
        barmode="overlay",
        title=f"Distribution de {variable}"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Relation entre deux variables")

    col1, col2 = st.columns(2)

    with col1:
        x_variable = st.selectbox(
            "Variable X",
            numerical_features,
            index=2
        )

    with col2:
        y_variable = st.selectbox(
            "Variable Y",
            numerical_features,
            index=3
        )

    fig = px.scatter(
        df_plot,
        x=x_variable,
        y=y_variable,
        color="État",
        opacity=0.5,
        title="Relation entre les caractéristiques"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Matrice de corrélation")

    corr = df[
        numerical_features + ["Machine failure"]
    ].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Corrélation entre les variables numériques"
    )

    st.plotly_chart(fig, width="stretch")


elif page == "Prétraitement":
    st.title("Prétraitement des données")

    st.write(
        "Les modèles utilisent six variables explicatives. "
        "Les identifiants et les indicateurs de type de panne "
        "sont exclus afin d'éviter les fuites de données."
    )

    st.subheader("Variables utilisées")

    st.dataframe(
        X.head(10),
        width="stretch"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Entraînement", len(X_train))
    col2.metric("Test", len(X_test))
    col3.metric(
        "Valeurs manquantes",
        int(X.isna().sum().sum())
    )

    st.subheader("Étapes du prétraitement")

    st.markdown(
        """
        1. Sélection des variables explicatives.
        2. Imputation des valeurs manquantes si nécessaire.
        3. Encodage One-Hot de la variable Type.
        4. Standardisation pour les modèles qui en ont besoin.
        5. Séparation stratifiée des données en entraînement et test.
        """
    )

    st.subheader("Répartition de la cible")

    repartition = pd.DataFrame({
        "Ensemble": ["Entraînement", "Test"],
        "Taux de panne (%)": [
            y_train.mean() * 100,
            y_test.mean() * 100
        ]
    })

    fig = px.bar(
        repartition,
        x="Ensemble",
        y="Taux de panne (%)",
        title="Proportion des pannes après séparation"
    )

    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Les transformations sont ajustées dans les pipelines "
        "d'entraînement pour éviter les fuites de données."
    )


elif page == "Comparaison des modèles":
    st.title("Comparaison des modèles optimisés")

    results = load_results()

    if results is None:
        st.error(
            "Le fichier resultats_optimisation.csv est introuvable."
        )

    else:
        st.subheader("Résultats de l'optimisation")

        st.dataframe(
            results,
            width="stretch"
        )

        numeric_columns = results.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            metric = st.selectbox(
                "Choisir une métrique",
                numeric_columns
            )

            model_column = next(
                (
                    c for c in results.columns
                    if c.lower() in [
                        "model",
                        "modèle",
                        "modele"
                    ]
                ),
                results.columns[0]
            )

            fig = px.bar(
                results,
                x=model_column,
                y=metric,
                color=model_column,
                title=f"Comparaison selon {metric}"
            )

            st.plotly_chart(fig, width="stretch")

        st.info(
            "Pour comparer des modèles sur des classes déséquilibrées, "
            "la précision, le rappel, le F1-score et l'Average Precision "
            "sont particulièrement importants."
        )


elif page == "Évaluation avancée":
    st.title("Évaluation avancée")

    model_name = st.selectbox(
        "Choisir un modèle",
        list(MODEL_FILES.keys())
    )

    try:
        model = load_model(model_name)

        probabilities = model.predict_proba(X_test)[:, 1]

        threshold = st.slider(
            "Seuil de classification",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Precision",
            f"{precision_score(y_test, predictions, zero_division=0):.3f}"
        )

        col2.metric(
            "Recall",
            f"{recall_score(y_test, predictions, zero_division=0):.3f}"
        )

        col3.metric(
            "F1-score",
            f"{f1_score(y_test, predictions, zero_division=0):.3f}"
        )

        col4.metric(
            "ROC-AUC",
            f"{roc_auc_score(y_test, probabilities):.3f}"
        )

        st.subheader("Matrice de confusion")

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Prédit normal", "Prédit panne"],
            y=["Réel normal", "Réel panne"],
            title="Matrice de confusion"
        )

        st.plotly_chart(fig, width="stretch")

        col_roc, col_pr = st.columns(2)

        with col_roc:
            fpr, tpr, _ = roc_curve(
                y_test,
                probabilities
            )

            fig_roc = go.Figure()

            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=model_name
                )
            )

            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Aléatoire",
                    line=dict(dash="dash")
                )
            )

            fig_roc.update_layout(
                title="Courbe ROC",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )

            st.plotly_chart(
                fig_roc,
                width="stretch"
            )

        with col_pr:
            precision_values, recall_values, _ = (
                precision_recall_curve(
                    y_test,
                    probabilities
                )
            )

            fig_pr = go.Figure()

            fig_pr.add_trace(
                go.Scatter(
                    x=recall_values,
                    y=precision_values,
                    mode="lines",
                    name=model_name
                )
            )

            fig_pr.update_layout(
                title="Courbe précision-rappel",
                xaxis_title="Recall",
                yaxis_title="Precision"
            )

            st.plotly_chart(
                fig_pr,
                width="stretch"
            )

        st.metric(
            "Average Precision",
            f"{average_precision_score(y_test, probabilities):.3f}"
        )

        st.warning(
            "Le curseur sert à illustrer l'effet du seuil sur le jeu "
            "de test. Le seuil final doit être choisi sur un ensemble "
            "de validation indépendant."
        )

    except FileNotFoundError:
        st.error(
            "Modèle introuvable dans le dossier models/. "
            "Vérifie le nom du fichier .joblib."
        )


elif page == "Prédiction":
    st.title("Prédiction de défaillance")

    st.write(
        "Saisis les caractéristiques d'une machine. "
        "Les données seront envoyées à l'API FastAPI, "
        "qui utilisera le modèle XGBoost pour effectuer "
        "la prédiction."
    )

    st.subheader("Connexion à l'API")

    if st.button("Vérifier la connexion"):
        api_status = check_api()

        if api_status is not None:
            st.success(
                f"API connectée : {api_status.get('status', 'OK')}"
            )
            st.write(
                "Modèle utilisé :",
                api_status.get("model", "Non précisé")
            )
        else:
            st.error(
                "Impossible de joindre FastAPI. "
                "Vérifie que le serveur fonctionne sur le port 8001."
            )

    st.divider()

    st.subheader("Caractéristiques de la machine")

    with st.form("prediction_form"):

        machine_type = st.selectbox(
            "Type de produit",
            ["L", "M", "H"]
        )

        air_temperature = st.number_input(
            "Température de l'air (K)",
            min_value=0.0,
            value=298.1
        )

        process_temperature = st.number_input(
            "Température du processus (K)",
            min_value=0.0,
            value=308.6
        )

        rotational_speed = st.number_input(
            "Vitesse de rotation (rpm)",
            min_value=1.0,
            value=1551.0
        )

        torque = st.number_input(
            "Couple (Nm)",
            min_value=0.0,
            value=42.8
        )

        tool_wear = st.number_input(
            "Usure de l'outil (min)",
            min_value=0.0,
            value=0.0
        )

        submitted = st.form_submit_button(
            "Prédire la défaillance"
        )

    if submitted:

        machine_data = {
            "type": machine_type,
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear
        }

        try:
            result = predict_with_api(machine_data)

            probability = float(
                result["probability"]
            )

            prediction = int(
                result["prediction"]
            )

            threshold = float(
                result["threshold"]
            )

            st.subheader("Résultat de la prédiction")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Probabilité estimée de panne",
                    f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Seuil de décision",
                    f"{threshold:.2f}"
                )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if prediction == 1:
                st.error(
                    "Le modèle classe cette observation "
                    "comme une défaillance."
                )
            else:
                st.success(
                    "Le modèle classe cette observation "
                    "comme un fonctionnement normal."
                )

            st.write(
                "Statut retourné par l'API :",
                result.get("status", "Non disponible")
            )

            st.caption(
                "Cette prédiction est indicative. "
                "Le modèle est entraîné sur des données "
                "synthétiques et ne remplace pas un diagnostic "
                "de maintenance industrielle."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Connexion impossible avec FastAPI. "
                "Démarre le serveur sur le port 8001."
            )

        except requests.exceptions.Timeout:
            st.error(
                "L'API met trop de temps à répondre."
            )

        except requests.exceptions.HTTPError as error:
            st.error(
                f"Erreur HTTP retournée par l'API : {error}"
            )

        except (ValueError, KeyError, TypeError) as error:
            st.error(
                f"Réponse inattendue de l'API : {error}"
            )
