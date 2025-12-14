# register model

import json
import mlflow
import logging
import os
import dagshub

# ==============================
# DagsHub Authentication
# ==============================
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

dagshub.auth.add_app_token(dagshub_token)

# ==============================
# MLflow Tracking Configuration
# ==============================
mlflow.set_tracking_uri(
    "https://dagshub.com/Abhishek9124/mlops-mini-project.mlflow"
)

# ==============================
# Logging Configuration
# ==============================
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_registration_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ==============================
# Helper Functions
# ==============================
def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error(
            "Unexpected error occurred while loading the model info: %s", e
        )
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry (DagsHub-safe)."""
    try:
        client = mlflow.tracking.MlflowClient()

        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Step 1: Create registered model if it does not exist
        try:
            client.create_registered_model(model_name)
            logger.debug("Registered model %s created", model_name)
        except Exception:
            logger.debug("Registered model %s already exists", model_name)

        # Step 2: Create model version
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=model_info["run_id"],
        )

        # Step 3: Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )

        logger.debug(
            "Model %s version %s successfully moved to Staging",
            model_name,
            model_version.version,
        )

    except Exception as e:
        logger.error("Error during model registration: %s", e)
        raise



# ==============================
# Main
# ==============================
def main():
    model_info = json.load(open("reports/experiment_info.json"))

    with mlflow.start_run(run_id=model_info["run_id"]):
        mlflow.set_tag("model_status", "staging_candidate")
        mlflow.set_tag("model_path", model_info["model_path"])
        mlflow.set_tag("lifecycle", "validated")

    logger.debug("Model marked as staging candidate via MLflow tags")

if __name__ == "__main__":
    main()

