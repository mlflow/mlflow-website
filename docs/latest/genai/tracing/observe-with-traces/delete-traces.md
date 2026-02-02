# Delete Traces

You can delete traces based on specific criteria using the [`mlflow.client.MlflowClient.delete_traces()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.delete_traces) method. This method allows you to delete traces by **timestamp** or **trace IDs**.

Deletion is irreversible

Deleting a trace cannot be undone. Ensure that the parameters provided to the `delete_traces` API meet the intended range for deletion.

## Deleting traces from MLflow UI[​](#deleting-traces-from-mlflow-ui "Direct link to Deleting traces from MLflow UI")

![Delete Traces from MLflow UI](/mlflow-website/docs/latest/images/llms/tracing/delete-traces.png)

## Delete traces older than a specific timestamp:[​](#delete-traces-older-than-a-specific-timestamp "Direct link to Delete traces older than a specific timestamp:")

python

```python
from datetime import datetime, timedelta

# Calculate timestamp for 7 days ago
seven_days_ago = datetime.now() - timedelta(days=7)
timestamp_ms = int(seven_days_ago.timestamp() * 1000)

deleted_count = client.delete_traces(
    experiment_id="1",
    max_timestamp_millis=timestamp_ms,
)

print(f"Deleted {deleted_count} traces")

```

Delete specific traces by their trace IDs:

python

```python
from mlflow import MlflowClient

client = MlflowClient()

# Delete specific traces
trace_ids = ["trace_id_1", "trace_id_2", "trace_id_3"]

deleted_count = client.delete_traces(experiment_id="1", trace_ids=trace_ids)

print(f"Deleted {deleted_count} traces")

```
