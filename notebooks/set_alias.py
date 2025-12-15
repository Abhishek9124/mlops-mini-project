import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(
    "https://dagshub.com/Abhishek9124/mlops-mini-project.mlflow"
)

client = MlflowClient()

client.set_registered_model_alias(
    name="mlops_model",
    alias="prod",
    version="2"
)

print("Alias 'prod' set successfully")
