import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_model_from_mlflow(model_name="turbofan_rul_xgboost", stage="None"):
    """
    Load model from MLflow Model Registry.
    
    Args:
        model_name: Registered model name
        stage: Model stage (None, Staging, Production)
    
    Returns:
        Loaded model
    """
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.xgboost.load_model(model_uri)
    return model

def load_test_data(filepath):
    """Load processed test data."""
    data = pd.read_csv(filepath)
    return data

def prepare_features_target(data):
    """Separate features and target."""
    exclude_cols = ['unit', 'cycle', 'RUL']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols]
    y = data['RUL']
    
    return X, y

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics

def plot_predictions(y_true, y_pred, output_dir='reports/figures'):
    """Create prediction visualization plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Actual vs Predicted scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title('Actual vs Predicted RUL')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/actual_vs_predicted.png')
    plt.close()
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted RUL')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals.png')
    plt.close()
    
    # 3. Error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png')
    plt.close()

def save_results(metrics, predictions_df, output_dir='reports'):
    """Save evaluation results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(f'{output_dir}/test_metrics.txt', 'w') as f:
        f.write("TEST SET EVALUATION METRICS\n")
        f.write("="*60 + "\n")
        for key, value in metrics.items():
            f.write(f"{key.upper()}: {value:.4f}\n")
    
    # Save predictions
    predictions_df.to_csv(f'{output_dir}/test_predictions.csv', index=False)

def main():
    """Main evaluation pipeline."""
    
    print("="*60)
    print("MODEL EVALUATION ON TEST DATA")
    print("="*60)
    
    # Load latest model from MLflow
    print("\nLoading model from MLflow...")
    try:
        # Load model directly from latest run instead of registry
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("turbofan_rul_tuning")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'best_model'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            print("No 'best_model' run found. Run training first.")
            return

        run_id = runs[0].info.run_id
        model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure you've trained and registered a model first")
        return
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data('data/processed/test_FD001_processed.csv')
    print(f"✓ Test data loaded: {test_data.shape}")
    
    # Prepare features and target
    X_test, y_test = prepare_features_target(test_data)
    print(f"Test features shape: {X_test.shape}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_test, y_pred)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"RMSE:  {metrics['rmse']:.2f}")
    print(f"MAE:   {metrics['mae']:.2f}")
    print(f"R²:    {metrics['r2']:.4f}")
    print(f"MAPE:  {metrics['mape']:.2f}%")
    print("="*60)
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'actual_RUL': y_test,
        'predicted_RUL': y_pred,
        'error': y_test - y_pred,
        'absolute_error': np.abs(y_test - y_pred)
    })
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_predictions(y_test, y_pred)
    print("✓ Plots saved to reports/figures/")
    
    # Save results
    print("\nSaving results...")
    save_results(metrics, predictions_df)
    print("✓ Results saved to reports/")
    
    # Log to MLflow
    with mlflow.start_run(run_name="test_evaluation"):
        mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
        # mlflow.log_artifacts('reports/figures')
        print("\n✓ Results logged to MLflow")
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()