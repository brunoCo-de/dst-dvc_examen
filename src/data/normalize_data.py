import pandas as pd
from sklearn.preprocessing import StandardScaler

# Chargement des données splitées
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Conversion en DataFrame et sauvegarde
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
    'data/processed/X_train_scaled.csv', index=False
)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
    'data/processed/X_test_scaled.csv', index=False
)