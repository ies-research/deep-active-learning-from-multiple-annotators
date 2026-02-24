import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:////home/mherde/PycharmProjects/scikit-activeml/z_non_github_stuff/dalc/scripts/mlflow/mlruns.db")
client = MlflowClient()

# List experiments
exps = client.search_experiments()  # or mlflow.search_experiments()
for e in exps:
    print(e.experiment_id, e.name, e.artifact_location)

# Show recent runs for the first experiment
exp_id = exps[0].experiment_id
runs = client.search_runs([exp_id], order_by=["attributes.start_time DESC"], max_results=20)
for r in runs:
    print(r.info.run_id, r.info.status, r.data.metrics, r.data.params)

