# MLflow 3 Migration Guide

This guide covers breaking changes and API updates when migrating from MLflow 2.x to MLflow 3.x.

## Installation[​](#installation "Direct link to Installation")

Install MLflow 3 by running:

bash

```bash
pip install "mlflow>=3.1"

```

Resources: [Website](https://mlflow.org/) | [Documentation](https://mlflow.org/docs/latest/index.html) | [Release Notes](https://mlflow.org/releases/3)

## Key Changes from MLflow 2.x[​](#key-changes-from-mlflow-2x "Direct link to Key Changes from MLflow 2.x")

### Model Logging API Changes[​](#model-logging-api-changes "Direct link to Model Logging API Changes")

**MLflow 2.x:**

python

```python
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=python_model,
    )

```

**MLflow 3:**

python

```python
# No longer requires starting a Run before logging models
mlflow.pyfunc.log_model(
    name="model",  # Use 'name' instead of 'artifact_path'
    python_model=python_model,
)

```

note

Models are now first-class entities in MLflow 3. You can call `log_model` directly without the `mlflow.start_run()` context manager. Use the `name` parameter to enable searching for LoggedModels.

### Model Artifacts Storage Location[​](#model-artifacts-storage-location "Direct link to Model Artifacts Storage Location")

**MLflow 2.x:**

shell

```shell
experiments/
  └── <experiment_id>/
    └── <run_id>/
      └── artifacts/
        └── ... # model artifacts stored here

```

**MLflow 3:**

shell

```shell
experiments/
  └── <experiment_id>/
    └── models/
      └── <model_id>/
        └── artifacts/
          └── ... # model artifacts stored here

```

warning

This change impacts the behavior of [`mlflow.client.MlflowClient.list_artifacts()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.list_artifacts). Model artifacts are no longer stored as run artifacts.

### UI Changes[​](#ui-changes "Direct link to UI Changes")

**Artifacts Tab**

In MLflow 3.x, the **Artifacts** tab in the run page no longer displays model artifacts. Model artifacts are now accessed through the **Logged Models** page, which provides a dedicated view for model-specific information and artifacts.

## Breaking Changes[​](#breaking-changes "Direct link to Breaking Changes")

### Removed Features[​](#removed-features "Direct link to Removed Features")

* **MLflow Recipes**: Completely removed ([#15250](https://github.com/mlflow/mlflow/pull/15250)). Migrate to standard MLflow tracking and model registry functionality or consider MLflow Projects.

* **Model Flavors**: The following flavors are no longer supported:

  * `fastai` ([#15255](https://github.com/mlflow/mlflow/pull/15255)) - Use `mlflow.pyfunc` with custom wrapper
  * `mleap` ([#15259](https://github.com/mlflow/mlflow/pull/15259)) - Use `mlflow.onnx` or `mlflow.pyfunc`
  * `diviner` - Use `mlflow.pyfunc` with custom wrapper
  * `gluon` - Use `mlflow.pytorch` or `mlflow.onnx`

* **AI Gateway**: The 'routes' and 'route\_type' config keys removed ([#15331](https://github.com/mlflow/mlflow/pull/15331)). Use the new configuration format.

* **Deployment Server**: The deployment server and `start-server` CLI command removed ([#15327](https://github.com/mlflow/mlflow/pull/15327)). Use `mlflow models serve` or containerized deployments.

### Tracking API Changes[​](#tracking-api-changes "Direct link to Tracking API Changes")

#### `run_uuid` Attribute Removed[​](#run_uuid-attribute-removed "Direct link to run_uuid-attribute-removed")

Replace `run_uuid` with `run_id`:

python

```python
# MLflow 2.x
run_info.run_uuid

# MLflow 3
run_info.run_id

```

#### Git Tags Removed[​](#git-tags-removed "Direct link to Git Tags Removed")

The following run tags have been removed ([#15366](https://github.com/mlflow/mlflow/pull/15366)):

* `mlflow.gitBranchName`
* `mlflow.gitRepoURL`

#### TensorFlow Autologging[​](#tensorflow-autologging "Direct link to TensorFlow Autologging")

The `every_n_iter` parameter removed from TensorFlow autologging ([#15412](https://github.com/mlflow/mlflow/pull/15412)). Implement custom logging callbacks for fine-tuned logging frequency.

### Model API Changes[​](#model-api-changes "Direct link to Model API Changes")

#### Removed Parameters[​](#removed-parameters "Direct link to Removed Parameters")

The following parameters have been removed from model logging/saving APIs:

* `example_no_conversion` ([#15322](https://github.com/mlflow/mlflow/pull/15322))
* `code_path` ([#15368](https://github.com/mlflow/mlflow/pull/15368)) - Use default code directory structure
* `requirements_file` from PyTorch flavor ([#15369](https://github.com/mlflow/mlflow/pull/15369)) - Use `pip_requirements` or `extra_pip_requirements`
* `inference_config` from Transformers flavor ([#15415](https://github.com/mlflow/mlflow/pull/15415)) - Set configuration before logging

#### ModelInfo Changes[​](#modelinfo-changes "Direct link to ModelInfo Changes")

The `signature_dict` property removed from `ModelInfo` ([#15367](https://github.com/mlflow/mlflow/pull/15367)). Use the `signature` property instead.

### Evaluation API Changes[​](#evaluation-api-changes "Direct link to Evaluation API Changes")

#### Baseline Model Comparison[​](#baseline-model-comparison "Direct link to Baseline Model Comparison")

The `baseline_model` parameter removed ([#15362](https://github.com/mlflow/mlflow/pull/15362)). Use `mlflow.validate_evaluation_results` API to compare models:

python

```python
# For classical ML models, use mlflow.models.evaluate
result_1 = mlflow.models.evaluate(
    model_1, data, targets="label", model_type="classifier"
)
result_2 = mlflow.models.evaluate(
    model_2, data, targets="label", model_type="classifier"
)

# Compare results
mlflow.validate_evaluation_results(result_1, result_2)

```

note

For GenAI evaluation, use `mlflow.genai.evaluate` with the new evaluation framework. See the **[GenAI Evaluation Migration Guide](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/)** for details on migrating from the legacy LLM evaluation approach.

#### MetricThreshold Changes[​](#metricthreshold-changes "Direct link to MetricThreshold Changes")

Use `greater_is_better` instead of `higher_is_better`:

python

```python
# MLflow 2.x
threshold = MetricThreshold(higher_is_better=True)

# MLflow 3
threshold = MetricThreshold(greater_is_better=True)

```

#### Custom Metrics[​](#custom-metrics "Direct link to Custom Metrics")

The `custom_metrics` parameter removed ([#15361](https://github.com/mlflow/mlflow/pull/15361)). Use the newer custom metrics approach in the evaluation API.

#### Explainer Logging[​](#explainer-logging "Direct link to Explainer Logging")

`mlflow.models.evaluate` no longer logs an explainer as a model by default. To enable:

python

```python
mlflow.models.evaluate(
    ...,
    evaluator_config={
        "log_model_explainability": True,
        "log_explainer": True,
    },
)

```

### Environment Variables[​](#environment-variables "Direct link to Environment Variables")

`MLFLOW_GCS_DEFAULT_TIMEOUT` removed ([#15365](https://github.com/mlflow/mlflow/pull/15365)). Configure timeouts using standard GCS client library approaches.

## Migration FAQs[​](#migration-faqs "Direct link to Migration FAQs")

### Can MLflow 3.x load resources created with MLflow 2.x?[​](#can-mlflow-3x-load-resources-created-with-mlflow-2x "Direct link to Can MLflow 3.x load resources created with MLflow 2.x?")

Yes, MLflow 3.x can load resources (runs, models, traces, etc.) created with MLflow 2.x. However, the reverse is not true.

warning

When testing MLflow 3.x, use **a separate environment** to avoid conflicts with MLflow 2.x.

### `load_model` throws `ResourceNotFound` error. What's wrong?[​](#load_model-throws-resourcenotfound-error-whats-wrong "Direct link to load_model-throws-resourcenotfound-error-whats-wrong")

In MLflow 3.x, model artifacts are stored in a different location. Use the model URI returned by `log_model`:

python

```python
# ❌ Don't use mlflow.get_artifact_uri("model")
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(my_model, name="model")
    mlflow.sklearn.load_model(mlflow.get_artifact_uri("model"))  # Fails!

# ✅ Use the model URI from log_model
with mlflow.start_run() as run:
    info = mlflow.sklearn.log_model(my_model, name="model")

    # Recommended: use model_uri from result
    mlflow.sklearn.load_model(info.model_uri)

    # Alternative: use model_id
    mlflow.sklearn.load_model(f"models:/{info.model_id}")

    # Deprecated: use run_id (will be removed in future)
    mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")

```

### How do I modify model requirements?[​](#how-do-i-modify-model-requirements "Direct link to How do I modify model requirements?")

Use [`mlflow.models.update_model_requirements()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.models.html#mlflow.models.update_model_requirements):

python

```python
import mlflow


class DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: list[str]) -> list[str]:
        return model_input


model_info = mlflow.pyfunc.log_model(name="model", python_model=DummyModel())
mlflow.models.update_model_requirements(
    model_uri=model_info.model_uri,
    operation="add",
    requirement_list=["scikit-learn"],
)

```

### How do I stay on MLflow 2.x?[​](#how-do-i-stay-on-mlflow-2x "Direct link to How do I stay on MLflow 2.x?")

Pin MLflow to the latest 2.x version:

bash

```bash
pip install 'mlflow<3'

```

## Compatibility[​](#compatibility "Direct link to Compatibility")

We strongly recommend upgrading **both client and server** to MLflow 3.x for the best experience. A mismatch between client and server versions may lead to unexpected behavior.

## Getting Help[​](#getting-help "Direct link to Getting Help")

For detailed guidance on migrating specific code, please consult:

* [MLflow Documentation](https://mlflow.org/docs/latest)
* [MLflow Community Forum](https://github.com/mlflow/mlflow/discussions)
* [MLflow Slack](https://mlflow.org/slack)
* [Release Notes](https://mlflow.org/releases/3)
