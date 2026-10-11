
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=601)
df = dataset.data.original.copy()

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
#  Repartition de la variable cible
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Machine failure")
plt.title("Repartition des pannes")
plt.xlabel("Machine failure (0 = normale, 1 = panne)")
plt.ylabel("Nombre de machines")
plt.tight_layout()
plt.show()

#  Histogrammes des variables numeriques
df[features].hist(figsize=(12, 8), bins=30)
plt.suptitle("Distribution des variables numeriques")
plt.tight_layout()
plt.show()

# Boxplots des variables numeriques
fig, axes = plt.subplots(1, 5, figsize=(17, 4))

for i, col in enumerate(features):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(col, fontsize=9)

plt.tight_layout()
plt.show()

# 4. Matrice de correlation
columns = features + ["Machine failure"]
correlation = df[columns].corr()

plt.figure(figsize=(9, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de correlation")
plt.tight_layout()
plt.show()
