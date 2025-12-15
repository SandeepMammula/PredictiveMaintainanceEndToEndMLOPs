import mlflow
from mlflow.tracking import MlflowClient

def get_best_run(experiment_name="turbofan_rul_prediction", metric="val_rmse"):
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize (lower is better)
    
    Returns:
        Best run object
    """
    client = MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None
    
    # Search for best run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1
    )
    
    if not runs:
        print("No runs found in experiment")
        return None
    
    return runs[0]

def register_best_model(experiment_name="turbofan_rul_prediction", 
                       model_name="turbofan_rul_xgboost",
                       metric="val_rmse"):
    """
    Register the best model to MLflow Model Registry.
    
    Args:
        experiment_name: Experiment name
        model_name: Name for registered model
        metric: Metric to determine best model
    """
    client = MlflowClient()
    
    print("="*60)
    print("MODEL REGISTRATION")
    print("="*60)
    
    # Get best run
    print(f"\nFinding best run based on {metric}...")
    best_run = get_best_run(experiment_name, metric)
    
    if best_run is None:
        return
    
    run_id = best_run.info.run_id
    val_rmse = best_run.data.metrics.get('val_rmse', 'N/A')
    
    print(f"✓ Best run found: {run_id}")
    print(f"  Validation RMSE: {val_rmse}")
    
    # Register model
    print(f"\nRegistering model as '{model_name}'...")
    model_uri = f"runs:/{run_id}/model"
    
    try:
        mv = mlflow.register_model(model_uri, model_name)
        print(f"✓ Model registered successfully")
        print(f"  Name: {mv.name}")
        print(f"  Version: {mv.version}")
        
        # Transition to Staging
        print(f"\nTransitioning model version {mv.version} to 'Staging'...")
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging"
        )
        print(f"✓ Model transitioned to Staging stage")
        
        # Add description
        client.update_model_version(
            name=model_name,
            version=mv.version,
            description=f"XGBoost model for RUL prediction. Val RMSE: {val_rmse}"
        )
        
        print("\n" + "="*60)
        print("✅ MODEL REGISTRATION COMPLETE")
        print("="*60)
        print(f"\nModel: {model_name}")
        print(f"Version: {mv.version}")
        print(f"Stage: Staging")
        print(f"Run ID: {run_id}")
        
    except Exception as e:
        print(f"✗ Error registering model: {e}")

def promote_to_production(model_name="turbofan_rul_xgboost", version=None):
    """
    Promote a model version to Production stage.
    
    Args:
        model_name: Registered model name
        version: Model version to promote (latest if None)
    """
    client = MlflowClient()
    
    # Get model versions
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    if not model_versions:
        print(f"No versions found for model '{model_name}'")
        return
    
    # Use specified version or latest
    if version is None:
        # Get latest version
        version = max([int(mv.version) for mv in model_versions])
    
    print(f"\nPromoting model '{model_name}' version {version} to Production...")
    
    # Archive any current production models
    for mv in model_versions:
        if mv.current_stage == "Production":
            print(f"Archiving current production version {mv.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived"
            )
    
    # Promote to production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    
    print(f"✓ Model version {version} promoted to Production")

def list_registered_models():
    """List all registered models and their versions."""
    client = MlflowClient()
    
    print("\n" + "="*60)
    print("REGISTERED MODELS")
    print("="*60)
    
    for rm in client.search_registered_models():
        print(f"\nModel: {rm.name}")
        print(f"Description: {rm.description or 'N/A'}")
        print("Versions:")
        
        for mv in client.search_model_versions(f"name='{rm.name}'"):
            print(f"  - Version {mv.version}: {mv.current_stage}")
            print(f"    Run ID: {mv.run_id}")
            if mv.description:
                print(f"    Description: {mv.description}")

if __name__ == "__main__":
    # Register best model
    register_best_model(
        experiment_name="turbofan_rul_prediction",
        model_name="turbofan_rul_xgboost",
        metric="val_rmse"
    )
    
    # List all registered models
    list_registered_models()
    
    # Uncomment to promote to production after testing
    # promote_to_production("turbofan_rul_xgboost", version=1)