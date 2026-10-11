
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

dataset = fetch_ucirepo(id=601)
df = dataset.data.original.copy()

print("Valeurs manquantes :")
print(df.isnull().sum())

print("\nNombre de doublons :")
print(df.duplicated().sum())

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

print("\nDimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDimensions X_train :", X_train.shape)
print("Dimensions X_test :", X_test.shape)

print("\nRepartition Train :")
print(y_train.value_counts(normalize=True))

print("\nRepartition Test :")
print(y_test.value_counts(normalize=True))

# 7. Definir les variables numeriques
numeric_features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]

categorical_features = ["Type"]


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train_prepared = preprocessor.fit_transform(X_train)

X_test_prepared = preprocessor.transform(X_test)

print("\nDimensions apres preprocessing :")
print("X_train_prepared :", X_train_prepared.shape)
print("X_test_prepared :", X_test_prepared.shape)

print("\nNoms des variables transformees :")
print(preprocessor.get_feature_names_out())

print("\nPremieres observations transformees :")
print(X_train_prepared[:5])
