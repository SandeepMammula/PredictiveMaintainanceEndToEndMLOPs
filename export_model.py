import mlflow
import shutil
from pathlib import Path

mlflow.set_tracking_uri("file:./mlruns")

# Your run_id
run_id = "cd606c65805f4b3488c28c4ad86de07e"

# Download model artifact to local path
model_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model")

# Copy to a clean location
target = Path("model_artifact")
if target.exists():
    shutil.rmtree(target)
shutil.copytree(model_path, target)

print(f"✓ Model exported to: {target}")