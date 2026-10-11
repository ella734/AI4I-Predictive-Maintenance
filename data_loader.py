
import pandas as pd
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=601)

df = dataset.data.original.copy()

print("Dimensions du dataset :", df.shape)
print("\nLes 5 premieres lignes :")
print(df.head())

print("\nNoms des colonnes :")
print(df.columns.tolist())

print("\nInformations sur les donnees :")
df.info()

print("\nDistribution de la variable cible :")
print(df["Machine failure"].value_counts())
