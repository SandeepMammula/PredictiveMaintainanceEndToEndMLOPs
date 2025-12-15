import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# Transition version 3 to Staging
client.transition_model_version_stage(
    name="turbofan_rul_xgboost",
    version=3,
    stage="Staging"
)

print("✅ Version 3 transitioned to Staging")

# Verify
versions = client.search_model_versions("name='turbofan_rul_xgboost'")
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")