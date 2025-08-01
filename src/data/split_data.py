import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement des données
data = pd.read_csv('data/raw/raw.csv')

# Suppression de la colonne date qui n'est pas utile pour le modèle
data = data.drop(columns=['date'])

# Séparation features/cible (silica_concentrate est la dernière colonne)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sauvegarde
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)