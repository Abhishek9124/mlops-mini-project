import dagshub
dagshub.init(repo_owner='Abhishek9124', repo_name='mlops-mini-project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_metric('accuracy', 42)
  mlflow.log_param('Param name', 'Value')