# MLflow Plugins

MLflow's plugin architecture enables seamless integration with third-party tools and custom infrastructure. As a framework-agnostic platform, MLflow provides developer APIs for extending functionality across storage, authentication, execution backends, and model evaluation.

## Quick Start[​](#quick-start "Direct link to Quick Start")

### Installing and Using a Plugin[​](#installing-and-using-a-plugin "Direct link to Installing and Using a Plugin")

Try the built-in test plugin to see how plugins work:

bash

```bash
# Clone MLflow and install example plugin
git clone https://github.com/mlflow/mlflow
cd mlflow
pip install -e . tests/resources/mlflow-test-plugin

```

bash

```bash
# Use the plugin with custom tracking URI scheme
MLFLOW_TRACKING_URI=file-plugin:$(PWD)/mlruns python examples/quickstart/mlflow_tracking.py

# Launch MLflow UI to view results
mlflow server --backend-store-uri ./mlruns

```

Open <http://localhost:5000> to see your tracked experiment:

![Quickstart UI](/mlflow-website/docs/latest/assets/images/quickstart_ui_screenshot-3ea56ba2b1c9970e1ff466b8dc57b37d.png)

Plugin Benefits

Plugins let you integrate MLflow with your existing infrastructure without modifying core MLflow code, ensuring smooth upgrades and maintenance.

## Plugin Types & Use Cases[​](#plugin-types--use-cases "Direct link to Plugin Types & Use Cases")

MLflow supports eight types of plugins, each addressing different integration needs:

### **Storage & Persistence**[​](#storage--persistence "Direct link to storage--persistence")

| Plugin Type              | Purpose                        | Example Use Cases                                  |
| ------------------------ | ------------------------------ | -------------------------------------------------- |
| **Tracking Store**       | Custom experiment data storage | Enterprise databases, cloud data warehouses        |
| **Artifact Repository**  | Custom artifact storage        | In-house blob storage, specialized file systems    |
| **Model Registry Store** | Custom model registry backend  | Enterprise model catalogs, version control systems |

### **Authentication & Headers**[​](#authentication--headers "Direct link to authentication--headers")

| Plugin Type                 | Purpose                | Example Use Cases                              |
| --------------------------- | ---------------------- | ---------------------------------------------- |
| **Request Auth Provider**   | Custom authentication  | OAuth, API keys, certificate-based auth        |
| **Request Header Provider** | Custom HTTP headers    | Environment identification, compliance headers |
| **Run Context Provider**    | Automatic run metadata | Git info, environment details, custom tags     |

### **Execution & Evaluation**[​](#execution--evaluation "Direct link to execution--evaluation")

| Plugin Type         | Purpose                       | Example Use Cases                                  |
| ------------------- | ----------------------------- | -------------------------------------------------- |
| **Project Backend** | Custom execution environments | Internal clusters, job schedulers, cloud platforms |
| **Model Evaluator** | Custom evaluation metrics     | Domain-specific validation, custom test suites     |
| **Deployment**      | Custom serving platforms      | Internal serving infrastructure, edge deployment   |

## Developing Custom Plugins[​](#developing-custom-plugins "Direct link to Developing Custom Plugins")

### Plugin Structure[​](#plugin-structure "Direct link to Plugin Structure")

Create a plugin as a standalone Python package:

python

```python
# setup.py
from setuptools import setup

setup(
    name="my-mlflow-plugin",
    version="0.1.0",
    install_requires=["mlflow>=2.0.0"],
    entry_points={
        # Define plugin entry points
        "mlflow.tracking_store": "my-scheme=my_plugin.store:MyTrackingStore",
        "mlflow.artifact_repository": "my-scheme=my_plugin.artifacts:MyArtifactRepo",
        "mlflow.run_context_provider": "unused=my_plugin.context:MyContextProvider",
        "mlflow.request_auth_provider": "unused=my_plugin.auth:MyAuthProvider",
        "mlflow.model_evaluator": "my-evaluator=my_plugin.evaluator:MyEvaluator",
        "mlflow.project_backend": "my-backend=my_plugin.backend:MyBackend",
        "mlflow.deployments": "my-target=my_plugin.deployment",
        "mlflow.app": "my-app=my_plugin.app:create_app",
    },
)

```

### Storage Plugins[​](#storage-plugins "Direct link to Storage Plugins")

* Tracking Store
* Artifact Repository
* Model Registry Store

python

```python
# my_plugin/store.py
from mlflow.store.tracking.abstract_store import AbstractStore


class MyTrackingStore(AbstractStore):
    """Custom tracking store for scheme 'my-scheme://'"""

    def __init__(self, store_uri):
        super().__init__()
        self.store_uri = store_uri
        # Initialize your custom storage backend

    def create_experiment(self, name, artifact_location=None, tags=None):
        # Implement experiment creation logic
        pass

    def log_metric(self, run_id, metric):
        # Implement metric logging logic
        pass

    def log_param(self, run_id, param):
        # Implement parameter logging logic
        pass

    # Implement other required AbstractStore methods...

```

python

```python
# my_plugin/artifacts.py
from mlflow.store.artifact.artifact_repo import ArtifactRepository


class MyArtifactRepo(ArtifactRepository):
    """Custom artifact repository for scheme 'my-scheme://'"""

    def __init__(self, artifact_uri):
        super().__init__(artifact_uri)
        # Initialize your artifact storage backend

    def log_artifact(self, local_file, artifact_path=None):
        # Upload file to your storage system
        pass

    def log_artifacts(self, local_dir, artifact_path=None):
        # Upload directory to your storage system
        pass

    def list_artifacts(self, path=None):
        # List artifacts in your storage system
        pass

    def download_artifacts(self, artifact_path, dst_path=None):
        # Download artifacts from your storage system
        pass

```

python

```python
# my_plugin/registry.py
from mlflow.store.model_registry.abstract_store import AbstractStore


class MyModelRegistryStore(AbstractStore):
    """Custom model registry store for scheme 'my-scheme://'"""

    def __init__(self, store_uri):
        super().__init__()
        self.store_uri = store_uri
        # Initialize your model registry backend

    def create_registered_model(self, name, tags=None, description=None):
        # Implement model registration logic
        pass

    def create_model_version(
        self, name, source, run_id=None, tags=None, run_link=None, description=None
    ):
        # Implement model version creation logic
        pass

    def get_registered_model(self, name):
        # Implement model retrieval logic
        pass

    # Implement other required AbstractStore methods...

```

### Authentication Plugins[​](#authentication-plugins "Direct link to Authentication Plugins")

* Request Auth Provider
* Run Context Provider
* Request Header Provider

python

```python
# my_plugin/auth.py
from mlflow.tracking.request_auth.abstract_request_auth_provider import (
    RequestAuthProvider,
)


class MyAuthProvider(RequestAuthProvider):
    """Custom authentication provider"""

    def get_name(self):
        return "my_auth_provider"

    def get_auth(self):
        # Return authentication object for HTTP requests
        # Can be anything that requests.auth accepts
        return MyCustomAuth()


class MyCustomAuth:
    """Custom authentication class"""

    def __call__(self, request):
        # Add authentication headers to request
        token = self._get_token()
        request.headers["Authorization"] = f"Bearer {token}"
        return request

    def _get_token(self):
        # Implement token retrieval logic
        # E.g., read from file, environment, or API call
        pass

```

**Usage:**

bash

```bash
export MLFLOW_TRACKING_AUTH=my_auth_provider
python your_mlflow_script.py

```

python

```python
# my_plugin/context.py
from mlflow.tracking.context.abstract_context import RunContextProvider


class MyContextProvider(RunContextProvider):
    """Automatically add custom tags to runs"""

    def in_context(self):
        # Return True if this context applies
        return True

    def tags(self):
        # Return dictionary of tags to add to runs
        return {
            "environment": self._get_environment(),
            "team": self._get_team(),
            "cost_center": self._get_cost_center(),
        }

    def _get_environment(self):
        # Detect environment (dev/staging/prod)
        pass

    def _get_team(self):
        # Get team from environment or config
        pass

    def _get_cost_center(self):
        # Get cost center for billing
        pass

```

python

```python
# my_plugin/headers.py
from mlflow.tracking.request_header.abstract_request_header_provider import (
    RequestHeaderProvider,
)


class MyHeaderProvider(RequestHeaderProvider):
    """Add custom headers to MLflow requests"""

    def in_context(self):
        return True

    def request_headers(self):
        return {
            "X-Client-Version": self._get_client_version(),
            "X-Environment": self._get_environment(),
            "X-User-Agent": "MyOrganization-MLflow-Client",
        }

    def _get_client_version(self):
        # Return your client version
        return "1.0.0"

    def _get_environment(self):
        # Detect environment context
        return os.getenv("DEPLOYMENT_ENV", "development")

```

### Execution Plugins[​](#execution-plugins "Direct link to Execution Plugins")

#### Project Backend Plugin[​](#project-backend-plugin "Direct link to Project Backend Plugin")

python

```python
# my_plugin/backend.py
from mlflow.projects.backend import AbstractBackend
from mlflow.projects.submitted_run import SubmittedRun


class MyBackend(AbstractBackend):
    """Custom execution backend for MLflow Projects"""

    def run(
        self,
        project_uri,
        entry_point,
        parameters,
        version,
        backend_config,
        tracking_uri,
        experiment_id,
    ):
        """Execute project on custom infrastructure"""

        # Parse backend configuration
        cluster_config = backend_config.get("cluster_config", {})

        # Submit job to your execution system
        job_id = self._submit_job(
            project_uri=project_uri,
            entry_point=entry_point,
            parameters=parameters,
            cluster_config=cluster_config,
        )

        # Return SubmittedRun for monitoring
        return MySubmittedRun(job_id, tracking_uri)

    def _submit_job(self, project_uri, entry_point, parameters, cluster_config):
        # Implement job submission to your infrastructure
        # Return job ID for monitoring
        pass


class MySubmittedRun(SubmittedRun):
    """Handle for submitted run"""

    def __init__(self, job_id, tracking_uri):
        self.job_id = job_id
        self.tracking_uri = tracking_uri
        super().__init__()

    def wait(self):
        # Wait for job completion and return success status
        return self._poll_job_status()

    def cancel(self):
        # Cancel the running job
        self._cancel_job()

    def get_status(self):
        # Get current job status
        return self._get_job_status()

```

### Model Evaluation Plugin[​](#model-evaluation-plugin "Direct link to Model Evaluation Plugin")

python

```python
# my_plugin/evaluator.py
from mlflow.models.evaluation import ModelEvaluator
from mlflow.models import EvaluationResult


class MyEvaluator(ModelEvaluator):
    """Custom model evaluator"""

    def can_evaluate(self, *, model_type, evaluator_config, **kwargs):
        """Check if this evaluator can handle the model type"""
        supported_types = ["classifier", "regressor"]
        return model_type in supported_types

    def evaluate(
        self, *, model, model_type, dataset, run_id, evaluator_config, **kwargs
    ):
        """Perform custom evaluation"""

        # Get predictions
        predictions = model.predict(dataset.features_data)

        # Compute custom metrics
        metrics = self._compute_custom_metrics(
            predictions, dataset.labels_data, evaluator_config
        )

        # Generate custom artifacts
        artifacts = self._generate_artifacts(predictions, dataset, evaluator_config)

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def _compute_custom_metrics(self, predictions, labels, config):
        # Implement domain-specific metrics
        return {
            "custom_score": self._calculate_custom_score(predictions, labels),
            "business_metric": self._calculate_business_metric(predictions, labels),
        }

    def _generate_artifacts(self, predictions, dataset, config):
        # Generate custom plots, reports, etc.
        return {}

```

### Popular Community Plugins[​](#popular-community-plugins "Direct link to Popular Community Plugins")

* Storage Solutions
* Model Deployment
* Model Evaluation
* Execution Backends
* Enterprise Solutions

#### **SQL Server Plugin**[​](#sql-server-plugin "Direct link to sql-server-plugin")

Store artifacts directly in SQL Server databases:

bash

```bash
pip install mlflow[sqlserver]

```

python

```python
import mlflow

# Use SQL Server as artifact store
db_uri = "mssql+pyodbc://user:pass@host:port/db?driver=ODBC+Driver+17+for+SQL+Server"
mlflow.create_experiment("sql_experiment", artifact_location=db_uri)

with mlflow.start_run():
    mlflow.onnx.log_model(model, name="model")  # Stored as BLOB in SQL Server

```

#### **Alibaba Cloud OSS Plugin**[​](#alibaba-cloud-oss-plugin "Direct link to alibaba-cloud-oss-plugin")

Integrate with Aliyun Object Storage Service:

bash

```bash
pip install mlflow[aliyun-oss]

```

python

```python
import os
import mlflow

# Configure OSS credentials
os.environ["MLFLOW_OSS_ENDPOINT_URL"] = "https://oss-region.aliyuncs.com"
os.environ["MLFLOW_OSS_KEY_ID"] = "your_access_key"
os.environ["MLFLOW_OSS_KEY_SECRET"] = "your_secret_key"

# Use OSS as artifact store
mlflow.create_experiment("oss_experiment", artifact_location="oss://bucket/path")

```

#### **XetHub Plugin**[​](#xethub-plugin "Direct link to xethub-plugin")

Use XetHub for versioned artifact storage:

bash

```bash
pip install mlflow[xethub]

```

python

```python
import mlflow

# Authenticate with XetHub (via CLI or environment variables)
mlflow.create_experiment(
    "xet_experiment", artifact_location="xet://username/repo/branch"
)

```

#### **Elasticsearch Plugin**[​](#elasticsearch-plugin "Direct link to elasticsearch-plugin")

Use Elasticsearch for experiment tracking:

bash

```bash
pip install mlflow-elasticsearchstore

```

| Plugin                | Target Platform | Installation                    |
| --------------------- | --------------- | ------------------------------- |
| **mlflow-redisai**    | RedisAI         | `pip install mlflow-redisai`    |
| **mlflow-torchserve** | TorchServe      | `pip install mlflow-torchserve` |
| **mlflow-ray-serve**  | Ray Serve       | `pip install mlflow-ray-serve`  |
| **mlflow-azureml**    | Azure ML        | Built-in with Azure ML          |
| **oci-mlflow**        | Oracle Cloud    | `pip install oci-mlflow`        |

**Example deployment usage:**

python

```python
import mlflow.deployments

# Deploy to custom target
client = mlflow.deployments.get_deploy_client("redisai")
client.create_deployment(
    name="my_model", model_uri="models:/MyModel/1", config={"device": "GPU"}
)

```

#### **Giskard Plugin**[​](#giskard-plugin "Direct link to giskard-plugin")

Comprehensive model validation and bias detection:

bash

```bash
pip install mlflow-giskard

```

python

```python
import mlflow

# Evaluate with Giskard
result = mlflow.models.evaluate(
    model,
    eval_data,
    targets="label",
    model_type="classifier",
    evaluators=["giskard"],
    evaluator_config={
        "giskard": {
            "test_suite": "full_suite",
            "bias_tests": True,
            "performance_tests": True,
        }
    },
)

```

**Detects vulnerabilities:**

* Performance bias
* Ethical bias
* Data leakage
* Overconfidence/Underconfidence
* Spurious correlations

#### **Trubrics Plugin**[​](#trubrics-plugin "Direct link to trubrics-plugin")

Advanced model validation framework:

bash

```bash
pip install trubrics-sdk

```

| Plugin          | Platform     | Use Case                     |
| --------------- | ------------ | ---------------------------- |
| **mlflow-yarn** | Hadoop/YARN  | Big data processing clusters |
| **oci-mlflow**  | Oracle Cloud | Oracle Cloud Infrastructure  |

**Example usage:**

bash

```bash
# Run MLflow project on YARN
mlflow run . --backend yarn --backend-config yarn-config.json

```

#### **JFrog Artifactory Plugin**[​](#jfrog-artifactory-plugin "Direct link to jfrog-artifactory-plugin")

Enterprise artifact governance:

bash

```bash
pip install mlflow[jfrog]

```

**Key Features:**

* Artifacts stored in JFrog Artifactory
* Full lifecycle management
* Enterprise security and compliance
* Integration with JFrog platform tools

**Setup:**

bash

```bash
export ARTIFACTORY_AUTH_TOKEN="your_token"

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --artifacts-destination "artifactory://artifactory.company.com/artifactory/ml-models"

```

**Usage:**

python

```python
import mlflow
from transformers import pipeline

mlflow.set_tracking_uri("http://your-mlflow-server:5000")
classifier = pipeline("sentiment-analysis", model="bert-base-uncased")

with mlflow.start_run():
    mlflow.transformers.log_model(
        transformers_model=classifier, name="model"
    )  # Automatically stored in JFrog Artifactory

```

**Configuration:**

bash

```bash
# Optional: Use HTTP instead of HTTPS
export ARTIFACTORY_NO_SSL=true

# Optional: Enable debug logging
export ARTIFACTORY_DEBUG=true

# Optional: Skip artifact deletion during garbage collection
export ARTIFACTORY_ARTIFACTS_DELETE_SKIP=true

```

## Testing Your Plugin[​](#testing-your-plugin "Direct link to Testing Your Plugin")

* Unit Testing
* Integration Testing
* Performance Testing

python

```python
# tests/test_my_plugin.py
import pytest
import mlflow
from my_plugin.store import MyTrackingStore


class TestMyTrackingStore:
    def setup_method(self):
        self.store = MyTrackingStore("my-scheme://test")

    def test_create_experiment(self):
        experiment_id = self.store.create_experiment("test_exp")
        assert experiment_id is not None

    def test_log_metric(self):
        experiment_id = self.store.create_experiment("test_exp")
        run = self.store.create_run(experiment_id, "user", "test_run")

        metric = mlflow.entities.Metric("accuracy", 0.95, 12345, 0)
        self.store.log_metric(run.info.run_id, metric)

        # Verify metric was logged correctly
        stored_run = self.store.get_run(run.info.run_id)
        assert "accuracy" in stored_run.data.metrics
        assert stored_run.data.metrics["accuracy"] == 0.95

    def test_log_artifact(self):
        # Test artifact logging functionality
        pass

```

python

```python
# tests/test_integration.py
import tempfile
import mlflow
from my_plugin import setup_test_environment


def test_end_to_end_workflow():
    with setup_test_environment():
        # Set tracking URI to use your plugin
        mlflow.set_tracking_uri("my-scheme://test")

        # Test full MLflow workflow
        with mlflow.start_run():
            mlflow.log_param("alpha", 0.5)
            mlflow.log_metric("rmse", 0.8)

            # Create and log a simple model
            with tempfile.NamedTemporaryFile() as f:
                f.write(b"model data")
                f.flush()
                mlflow.log_artifact(f.name, "model")

        # Verify everything was stored correctly
        runs = mlflow.search_runs()
        assert len(runs) == 1
        assert runs.iloc[0]["params.alpha"] == "0.5"
        assert runs.iloc[0]["metrics.rmse"] == 0.8

```

python

```python
# tests/test_performance.py
import time
import mlflow
import pytest
import threading
from my_plugin.store import MyTrackingStore


class TestPerformance:
    def test_bulk_logging_performance(self):
        store = MyTrackingStore("my-scheme://perf-test")
        experiment_id = store.create_experiment("perf_test")
        run = store.create_run(experiment_id, "user", "perf_run")

        # Test bulk metric logging
        start_time = time.time()
        for i in range(1000):
            metric = mlflow.entities.Metric(f"metric_{i}", i * 0.1, 12345, i)
            store.log_metric(run.info.run_id, metric)

        elapsed = time.time() - start_time
        assert elapsed < 10.0  # Should complete within 10 seconds

        # Verify all metrics were logged
        stored_run = store.get_run(run.info.run_id)
        assert len(stored_run.data.metrics) == 1000

    def test_concurrent_access(self):
        store = MyTrackingStore("my-scheme://concurrent-test")
        results = []

        def create_experiment(name):
            exp_id = store.create_experiment(f"concurrent_{name}")
            results.append(exp_id)

        threads = [
            threading.Thread(target=create_experiment, args=[i]) for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 10  # All unique experiment IDs

```

## Distribution & Publishing[​](#distribution--publishing "Direct link to Distribution & Publishing")

### Package Structure[​](#package-structure "Direct link to Package Structure")

text

```text
my-mlflow-plugin/
├── setup.py                    # Package configuration
├── README.md                   # Plugin documentation
├── my_plugin/
│   ├── __init__.py
│   ├── store.py               # Tracking store implementation
│   ├── artifacts.py           # Artifact repository implementation
│   ├── auth.py                # Authentication provider
│   └── evaluator.py           # Model evaluator
├── tests/
│   ├── test_store.py
│   ├── test_artifacts.py
│   └── test_integration.py
└── examples/
    └── example_usage.py

```

### Publishing to PyPI[​](#publishing-to-pypi "Direct link to Publishing to PyPI")

bash

```bash
# Build distribution packages
python setup.py sdist bdist_wheel

# Upload to PyPI
pip install twine
twine upload dist/*

```

### Documentation Template[​](#documentation-template "Direct link to Documentation Template")

markdown

````markdown
# My MLflow Plugin

## Installation
```bash
pip install my-mlflow-plugin

````

## Configuration[​](#configuration "Direct link to Configuration")

bash

```bash
export MY_PLUGIN_CONFIG="value"

```

## Usage[​](#usage "Direct link to Usage")

python

```python
import mlflow

mlflow.set_tracking_uri("my-scheme://config")

```

## Features[​](#features "Direct link to Features")

* Feature 1
* Feature 2

## Examples[​](#examples "Direct link to Examples")

See examples/ directory for complete usage examples.

## Best Practices[​](#best-practices "Direct link to Best Practices")

### **Plugin Development**[​](#plugin-development "Direct link to plugin-development")

* **Follow MLflow interfaces** - Implement all required abstract methods
* **Handle errors gracefully** - Provide clear error messages for configuration issues
* **Support authentication** - Integrate with existing credential systems
* **Add comprehensive logging** - Help users debug configuration problems
* **Version compatibility** - Test against multiple MLflow versions

### **Performance Optimization**[​](#performance-optimization "Direct link to performance-optimization")

* **Batch operations** - Implement efficient bulk logging when possible
* **Connection pooling** - Reuse connections to external systems
* **Async operations** - Use async I/O for storage operations when beneficial
* **Caching** - Cache frequently accessed metadata

### **Security Considerations**[​](#security-considerations "Direct link to security-considerations")

* **Credential management** - Never log or expose sensitive credentials
* **Input validation** - Validate all user inputs and URIs
* **Access controls** - Respect existing authentication and authorization
* **Secure communication** - Use TLS/SSL for network communications

### **Testing Strategy**[​](#testing-strategy "Direct link to testing-strategy")

* **Unit tests** - Test individual plugin components
* **Integration tests** - Test full workflows with MLflow
* **Performance tests** - Verify acceptable performance characteristics
* **Compatibility tests** - Test with different MLflow versions

***

**Ready to extend MLflow?** Start with the [example test plugin](https://github.com/mlflow/mlflow/tree/master/tests/resources/mlflow-test-plugin) to see all plugin types in action, then build your custom integration!
