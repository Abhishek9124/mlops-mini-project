"""
Model Registration Script (Corrected)
-------------------------------------
- robust authentication using dagshub.init()
- handles model version creation and staging promotion
- explicit error handling for MLflow
- Handles missing 'model_name' in experiment info
"""

import json
import logging
import os
import time
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# ==============================
# Configuration
# ==============================
REPO_OWNER = "Abhishek9124"
REPO_NAME = "mlops-mini-project"
EXPERIMENT_INFO_PATH = "reports/experiment_info.json"
DEFAULT_MODEL_NAME = "mlops_model"  # Fallback name if missing in JSON

# ==============================
# Logging Configuration
# ==============================
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ==============================
# Helper Functions
# ==============================
def load_model_info(file_path: str) -> dict:
    """Load model metadata saved during training."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Experiment info file not found at: {file_path}")
        
    try:
        with open(file_path, "r") as f:
            model_info = json.load(f)
        logger.info(f"Loaded model info from {file_path}")
        
        # Debug: Print keys to help user identify what is missing
        logger.debug(f"Keys found in experiment info: {list(model_info.keys())}")
        
        return model_info
    except Exception as e:
        logger.error(f"Failed to load model info: {e}")
        raise

def wait_for_version_ready(client, name, version, max_retries=10):
    """Wait for the model version to be ready before transitioning."""
    for _ in range(max_retries):
        model_version_details = client.get_model_version(name=name, version=version)
        status = model_version_details.status
        if status == "READY":
            return True
        logger.info(f"Model version status: {status}. Waiting...")
        time.sleep(1)
    return False

def register_model(model_name: str, model_info: dict):
    """
    Register model in MLflow Model Registry and promote to STAGING.
    """
    client = MlflowClient()
    
    # Construct the run URI
    # Default to 'model' if model_path is missing
    model_path = model_info.get('model_path', 'model')
    clean_path = model_path.lstrip('/')
    
    if 'run_id' not in model_info:
        raise KeyError("Critical error: 'run_id' missing from experiment info.")
        
    model_uri = f"runs:/{model_info['run_id']}/{clean_path}"

    # 1. Create Registered Model
    try:
        client.create_registered_model(model_name)
        logger.info(f"Created new registered model: {model_name}")
    except MlflowException as e:
        # Check if error is "Resource already exists"
        if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
            logger.info(f"Registered model '{model_name}' already exists.")
        else:
            logger.error(f"Error creating registered model: {e}")
            raise

    # 2. Create Model Version
    try:
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=model_info["run_id"],
        )
        logger.info(f"Created version {model_version.version} for model {model_name}")
    except MlflowException as e:
        logger.error(f"Failed to create model version: {e}")
        raise

    # 3. Wait for Ready state (Critical for DagsHub/Remote backends)
    if not wait_for_version_ready(client, model_name, model_version.version):
        logger.warning("Model version not ready. Skipping stage transition.")
        return

    # 4. Promote to STAGING
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True,
        )
        logger.info(f"Model {model_name} version {model_version.version} promoted to STAGING")
    except MlflowException as e:
        logger.error(f"Failed to transition model stage: {e}")
        raise

# ==============================
# Main
# ==============================
def main():
    # 1. Initialize DagsHub (Handles Auth & Tracking URI automatically)
    try:
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    except Exception as e:
        logger.error("Failed to initialize DagsHub. Check your internet or Repo details.")
        raise e

    # 2. Load Info
    model_info = load_model_info(EXPERIMENT_INFO_PATH)
    
    # Determine Model Name (Fallback to default if missing)
    model_name = model_info.get("model_name", DEFAULT_MODEL_NAME)
    if "model_name" not in model_info:
        logger.warning(f"'model_name' not found in {EXPERIMENT_INFO_PATH}. Using default: {model_name}")

    # 3. Log Registration Event (Optional separate run)
    with mlflow.start_run(run_name="model_registration") as run:
        mlflow.set_tag("event", "model_registration")
        mlflow.set_tag("target_model", model_name)
        mlflow.set_tag("source_run_id", model_info.get("run_id", "unknown"))
        
        logger.info(f"Registration event logged in run: {run.info.run_id}")

    # 4. Perform Registration
    register_model(
        model_name=model_name,
        model_info=model_info,
    )

if __name__ == "__main__":
    main()