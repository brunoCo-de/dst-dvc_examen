import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import os

def grid_search():
    # Load scaled data
    X_train = pd.read_csv('data/processed/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    # Initialize model
    rf = RandomForestRegressor(random_state=42)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Grid search with 5-fold CV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    
    # Create models directory if not exists
    os.makedirs('models/params', exist_ok=True)
    
    # Save best parameters
    with open('models/params/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

if __name__ == '__main__':
    grid_search()