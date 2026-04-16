"""
model_training.py
-----------------
Trains Linear Regression, Random Forest, and XGBoost regressors.
XGBoost uses tuned hyperparameters for best performance.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.append(os.path.dirname(__file__))
from data_preparation import get_prepared_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"  Saved: {path}")


def evaluate(y_test, preds, name):
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  R²:   {r2:.4f}")
    return {'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2}


def train_linear_regression(X_train, X_test, y_train, y_test):
    print("\n--- Linear Regression ---")
    model = LinearRegression()
    model.fit(X_train, y_train)
    save_model(model, 'lr_model.pkl')
    return model, evaluate(y_test, model.predict(X_test), 'Linear Regression')


def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n--- Random Forest (Tuned) ---")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    save_model(model, 'rf_model.pkl')
    return model, evaluate(y_test, model.predict(X_test), 'Random Forest')


def train_xgboost(X_train, X_test, y_train, y_test):
    print("\n--- XGBoost (Tuned) ---")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    save_model(model, 'xgb_model.pkl')
    return model, evaluate(y_test, model.predict(X_test), 'XGBoost')


def train_all():
    print("=== Training Melbourne Housing Price Models ===")
    X_train, X_test, y_train, y_test = get_prepared_data()
    lr_model, lr_r   = train_linear_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_r   = train_random_forest(X_train, X_test, y_train, y_test)
    xgb_model, xgb_r = train_xgboost(X_train, X_test, y_train, y_test)

    print("\n=== Model Comparison ===")
    for r in [lr_r, rf_r, xgb_r]:
        print(f"{r['model']}: R²={r['r2']:.4f}, MAE=${r['mae']:,.0f}, RMSE=${r['rmse']:,.0f}")
    print("\n=== All models trained and saved ===")
    return lr_model, rf_model, xgb_model


if __name__ == '__main__':
    train_all()
