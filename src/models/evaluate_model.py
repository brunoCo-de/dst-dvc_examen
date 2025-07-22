import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import os

def evaluate_model():
    # Load data and model
    X_test = pd.read_csv('data/processed/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    with open('models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    # Save predictions
    pd.DataFrame(y_pred, columns=['predictions']).to_csv(
        'data/predictions.csv', index=False
    )
    
    # Create metrics directory if not exists
    os.makedirs('metrics', exist_ok=True)
    
    # Save metrics
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    evaluate_model()