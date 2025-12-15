import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
import optuna
from optuna.integration.mlflow import MLflowCallback
from pathlib import Path

def load_processed_data(filepath):
    """Load processed training data."""
    data = pd.read_csv(filepath)
    return data

def prepare_features_target(data):
    """
    Separate features and target variable.
    
    Args:
        data: Processed DataFrame
    
    Returns:
        X (features), y (target)
    """
    exclude_cols = ['unit', 'cycle', 'RUL']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols]
    y = data['RUL']
    
    return X, y

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        Validation RMSE (to minimize)
    """
    # Suggest hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42
    }
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Predict and calculate RMSE
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    
    return rmse

def train_with_best_params(X_train, y_train, X_val, y_val, best_params):
    """
    Train final model with best parameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        best_params: Best hyperparameters from Optuna
    
    Returns:
        Trained model
    """
    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    return model

def evaluate_model(model, X, y, dataset_name=""):
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True target values
        dataset_name: Name for logging (train/val/test)
    
    Returns:
        Dictionary of metrics, predictions
    """
    predictions = model.predict(X)
    
    metrics = {
        f'{dataset_name}_rmse': np.sqrt(mean_squared_error(y, predictions)),
        f'{dataset_name}_mae': mean_absolute_error(y, predictions),
        f'{dataset_name}_r2': r2_score(y, predictions)
    }
    
    return metrics, predictions

def main(use_tuning=True, n_trials=50):
    """
    Main training pipeline with optional hyperparameter tuning.
    
    Args:
        use_tuning: Whether to run Optuna tuning
        n_trials: Number of Optuna trials
    """
    
    # Set MLflow experiment
    mlflow.set_experiment("turbofan_rul_prediction")
    
    print("Loading data...")
    data = load_processed_data('data/processed/train_FD001_processed.csv')
    
    # Prepare features and target
    X, y = prepare_features_target(data)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    
    if use_tuning:
        print(f"\nStarting hyperparameter tuning with {n_trials} trials...")
        print("This may take several minutes...")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            study_name='turbofan_rul_tuning',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        # MLflow callback for Optuna
        mlflowc = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name='val_rmse'
        )
        
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            callbacks=[mlflowc],
            show_progress_bar=True
        )
        
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING RESULTS")
        print("="*60)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best RMSE: {study.best_value:.2f}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("="*60)
        
        # Get best parameters
        best_params = study.best_params.copy()
        best_params['objective'] = 'reg:squarederror'
        best_params['random_state'] = 42
        
    else:
        # Default parameters
        print("\nUsing default parameters (no tuning)")
        best_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    # Train final model with best parameters
    with mlflow.start_run(run_name="best_model"):
        
        print("\nTraining final model with best parameters...")
        model = train_with_best_params(X_train, y_train, X_val, y_val, best_params)
        
        # Log parameters
        mlflow.log_params(best_params)
        mlflow.log_param("dataset", "FD001")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("tuning_enabled", use_tuning)
        if use_tuning:
            mlflow.log_param("n_trials", n_trials)
        
        # Evaluate on training set
        print("\nEvaluating on training set...")
        train_metrics, _ = evaluate_model(model, X_train, y_train, "train")
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        val_metrics, val_predictions = evaluate_model(model, X_val, y_val, "val")
        
        # Combine all metrics
        all_metrics = {**train_metrics, **val_metrics}
        
        # Log metrics
        mlflow.log_metrics(all_metrics)
        
        # Print results
        print("\n" + "="*60)
        print("FINAL MODEL RESULTS")
        print("="*60)
        print(f"Train RMSE: {train_metrics['train_rmse']:.2f}")
        print(f"Train MAE:  {train_metrics['train_mae']:.2f}")
        print(f"Train R²:   {train_metrics['train_r2']:.4f}")
        print(f"\nVal RMSE:   {val_metrics['val_rmse']:.2f}")
        print(f"Val MAE:    {val_metrics['val_mae']:.2f}")
        print(f"Val R²:     {val_metrics['val_r2']:.4f}")
        print("="*60)
        
        # Log model
        mlflow.xgboost.log_model(
            model, 
            "model",
            registered_model_name="turbofan_rul_xgboost"
        )
        
        # Save feature names
        mlflow.log_dict({"features": list(X.columns)}, "features.json")
        
        print(f"\n✅ Model logged to MLflow")
        print(f"🔗 View results at: http://localhost:5000")

if __name__ == "__main__":
    # Run with hyperparameter tuning (50 trials)
    main(use_tuning=True, n_trials=50)
    
    # To run without tuning (faster):
    # main(use_tuning=False)