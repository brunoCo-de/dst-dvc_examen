import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import os

def train_model():
    # Load data and best parameters
    X_train = pd.read_csv('data/processed/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    with open('models/params/best_params.pkl', 'rb') as f:
        best_params = pickle.load(f)
    
    # Initialize and train model with best parameters
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    # Save trained model
    os.makedirs('models', exist_ok=True)
    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    train_model()