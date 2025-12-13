import dagshub
import mlflow

mlflow.set_tracking_uri('https://dagshub.com/Abhishek9124/mlops-mini-project.mlflow')
dagshub.init(repo_owner='Abhishek9124', repo_name='mlops-mini-project', mlflow=True)

with mlflow.start_run():
  mlflow.log_metric('accuracy', 42)
  mlflow.log_param('Param name', 'Value')