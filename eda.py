
import pandas as pd
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=601)
df = dataset.data.original.copy()

print("Dimensions du dataset :")
print(df.shape)

print("\nStatistiques descriptives :")
print(df.describe())

print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

print("\nNombre de lignes dupliquees :")
print(df.duplicated().sum())

print("\nTypes des variables :")
print(df.dtypes)
