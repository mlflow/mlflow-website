# MLflow Model Serving

Transform your trained models into production-ready inference servers with MLflow's comprehensive serving capabilities. Deploy locally, in the cloud, or through managed endpoints with standardized REST APIs.

#### REST API Endpoints

Automatic generation of standardized REST endpoints for model inference with consistent request/response formats.

#### Multi-Framework Support

Serve models from any ML framework through MLflow's flavor system with unified deployment patterns.

#### Custom Applications

Build sophisticated serving applications with custom logic, preprocessing, and business rules.

#### Scalable Deployment

Deploy to various targets from local development servers to cloud platforms and Kubernetes clusters.

## Quick Start[​](#quick-start "Direct link to Quick Start")

Get your model serving in minutes with these simple steps:

* 1\. Serve Model
* 2\. Make Predictions

Choose your serving approach:

bash

```
# Serve a logged model
mlflow models serve -m "models:/<model-id>" -p 5000

# Serve a registered model
mlflow models serve -m "models:/<model-name>/<model-version>" -p 5000

# Serve a model from local path
mlflow models serve -m ./path/to/model -p 5000
```

Your model will be available at `http://localhost:5000`

Send prediction requests via HTTP:

bash

```
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[1, 2, 3, 4]]}'
```

**Using Python:**

python

```
import requests
import json

data = {
    "dataframe_split": {
        "columns": ["feature1", "feature2", "feature3", "feature4"],
        "data": [[1, 2, 3, 4]],
    }
}

response = requests.post(
    "http://localhost:5000/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data),
)

print(response.json())
```

## How Model Serving Works[​](#how-model-serving-works "Direct link to How Model Serving Works")

MLflow transforms your trained models into production-ready HTTP servers through a carefully orchestrated process that handles everything from model loading to request processing.

### Server Startup and Model Loading[​](#server-startup-and-model-loading "Direct link to Server Startup and Model Loading")

When you run `mlflow models serve`, MLflow begins by analyzing your model's metadata to determine how to load it. Each model contains an `MLmodel` file that specifies which "flavor" it uses - whether it's scikit-learn, PyTorch, TensorFlow, or a custom PyFunc model.

MLflow downloads the model artifacts to a local directory and creates a FastAPI server with standardized endpoints. The server loads your model using the appropriate flavor-specific loading logic. For example, a scikit-learn model is loaded using pickle, while a PyTorch model loads its state dictionary and model class.

The server exposes four key endpoints:

* **`POST /invocations`** - The main prediction endpoint
* **`GET /ping`** and **`GET /health`** - Health checks for monitoring
* **`GET /version`** - Returns server and model information

### Request Processing Pipeline[​](#request-processing-pipeline "Direct link to Request Processing Pipeline")

When a prediction request arrives at `/invocations`, MLflow processes it through several validation and transformation steps:

**Input Format Detection**: MLflow automatically detects which input format you're using. It supports multiple formats to accommodate different use cases:

* `dataframe_split`: Pandas DataFrame with separate columns and data arrays
* `dataframe_records`: List of dictionaries representing rows
* `instances`: TensorFlow Serving format for individual predictions
* `inputs`: Named tensor format for more complex inputs

**Schema Validation**: If your model includes a signature (input/output schema), MLflow validates the incoming data against it. This catches type mismatches and missing columns before they reach your model.

**Parameter Extraction**: MLflow separates prediction data from optional parameters. Parameters like `temperature` for language models or `threshold` for classifiers are extracted and passed separately to models that support them.

### Model Prediction and Response[​](#model-prediction-and-response "Direct link to Model Prediction and Response")

Once the input is validated and formatted, MLflow calls your model's `predict()` method. The framework automatically detects whether your model accepts parameters and calls it appropriately:

python

```
# For models that accept parameters
raw_predictions = model.predict(data, params=params)

# For traditional models
raw_predictions = model.predict(data)
```

MLflow then serializes the predictions back to JSON, handling various data types including NumPy arrays, pandas DataFrames, and Python lists. The response format depends on your input format - traditional requests get wrapped in a `predictions` object, while LLM-style requests return unwrapped results.

### The Flavor System[​](#the-flavor-system "Direct link to The Flavor System")

MLflow's flavor system is what makes serving work consistently across different ML frameworks. Each flavor implements framework-specific loading and prediction logic while exposing a unified interface.

When you log a model using `mlflow.sklearn.log_model()` or `mlflow.pytorch.log_model()`, MLflow creates both a flavor-specific representation and a PyFunc wrapper. The PyFunc wrapper provides the standardized `predict()` interface that the serving layer expects, while the flavor handles the framework-specific details like tensor operations or data preprocessing.

This architecture means you can serve scikit-learn, PyTorch, TensorFlow, and custom models using identical serving commands and APIs.

### Error Handling and Debugging[​](#error-handling-and-debugging "Direct link to Error Handling and Debugging")

MLflow's serving infrastructure includes comprehensive error handling to help you debug issues:

* **Schema Errors**: Detailed messages about data type mismatches or missing columns
* **Format Errors**: Clear guidance when input format is incorrect or ambiguous
* **Model Errors**: Full stack traces from your model's prediction code
* **Server Errors**: Timeout and resource-related error handling

The server logs all requests and errors, making it easier to diagnose production issues.

### Input Format Examples[​](#input-format-examples "Direct link to Input Format Examples")

Here are the main input formats MLflow accepts:

json

```
// dataframe_split format
{
  "dataframe_split": {
    "columns": ["feature1", "feature2", "feature3"],
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  }
}

// dataframe_records format
{
  "dataframe_records": [
    {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
    {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0}
  ]
}

// instances format (for simple models)
{
  "instances": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
}
```

All formats return a consistent response structure with your predictions and any additional metadata your model provides.

## Key Implementation Concepts[​](#key-implementation-concepts "Direct link to Key Implementation Concepts")

* Model Preparation
* Server Configuration
* Advanced Patterns

**Prepare your models for successful serving:**

* **Model Signatures**: Define input/output schemas for automatic request validation
* **Environment Management**: Capture dependencies to ensure reproducible deployments
* **Model Registry**: Use aliases for seamless production updates
* **Metadata**: Include relevant context for debugging and monitoring

python

```
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Log model with comprehensive serving metadata
signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(
    sk_model=model,
    name="my_model",
    signature=signature,
    registered_model_name="production_model",
    input_example=X_train[:5],  # Visible example for the MLflow UI
)

# Use aliases for production deployment
client = MlflowClient()
client.set_registered_model_alias(
    name="production_model", alias="production", version="1"
)
```

**Configure your serving infrastructure for optimal performance:**

* **Request Handling**: Set appropriate timeouts and batch sizes
* **Resource Allocation**: Configure workers based on model complexity and load
* **Input Formats**: Choose the right format for your data type
* **Error Handling**: Implement proper logging and monitoring

bash

```
# Configure server for production workloads
export MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT=60

# For uvicorn-based deployments (Docker):
export MLFLOW_MODELS_WORKERS=4

# For deployments that use gunicorn:
export GUNICORN_CMD_ARGS="--timeout 60 --workers 4"

# Serve with optimal settings
mlflow models serve \
  --model-uri models:/my_model@production \
  --port 5000 \
  --env-manager local  # For production, use conda or virtualenv
```

**Implement advanced serving patterns with custom PyFunc models:**

* **Preprocessing Logic**: Handle data transformation within the model
* **Multi-Model Ensembles**: Combine predictions from multiple models
* **Business Logic**: Integrate validation and post-processing rules
* **Performance Optimization**: Batch processing and caching strategies

python

```
import joblib
import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any


class EnsembleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Load multiple models for ensemble prediction"""
        self.model_a = joblib.load(context.artifacts["model_a"])
        self.model_b = joblib.load(context.artifacts["model_b"])
        self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        # Load ensemble weights from config
        self.weights = context.model_config.get("weights", [0.5, 0.5])

    def predict(self, model_input: pd.DataFrame) -> np.ndarray:
        """Combine predictions from multiple models"""
        # Preprocess input
        processed = self.preprocessor.transform(model_input)

        # Get predictions from both models
        pred_a = self.model_a.predict(processed)
        pred_b = self.model_b.predict(processed)

        # Weighted average ensemble
        ensemble_pred = self.weights[0] * pred_a + self.weights[1] * pred_b

        return ensemble_pred


# Log ensemble model with artifacts
artifacts = {
    "model_a": "path/to/model_a.pkl",
    "model_b": "path/to/model_b.pkl",
    "preprocessor": "path/to/preprocessor.pkl",
}

mlflow.pyfunc.log_model(
    name="ensemble_model",
    python_model=EnsembleModel(),
    artifacts=artifacts,
    model_config={"weights": [0.6, 0.4]},
    pip_requirements=["scikit-learn", "pandas", "numpy"],
)
```

## Complete Example: Train to Production[​](#complete-example-train-to-production "Direct link to Complete Example: Train to Production")

Follow this step-by-step guide to go from model training to a deployed REST API:

* 1\. Train & Log
* 2\. Promote Model
* 3\. Start Server
* 4\. Make Predictions

Train a simple model with automatic logging:

python

```
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Load sample data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Enable sklearn autologging with model registration
mlflow.sklearn.autolog(registered_model_name="iris_classifier")

# Train model - MLflow automatically logs everything
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Autologging automatically captures:
    # - Model artifacts
    # - Training parameters (n_estimators, random_state, etc.)
    # - Training metrics (score on training data)
    # - Model signature (inferred from training data)
    # - Input example

    # Optional: Log additional custom metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", accuracy)

    print(f"Run ID: {run.info.run_id}")
    print("Model automatically logged and registered!")
```

Set up model alias for production:

python

```
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest registered version (autologging creates version 1)
model_version = client.get_registered_model("iris_classifier").latest_versions[0]

# Set production alias (replaces deprecated stages)
client.set_registered_model_alias(
    name="iris_classifier", alias="production", version=model_version.version
)

print(f"Model version {model_version.version} tagged as 'production'")

# Model URI for serving (using alias)
model_uri = "models:/iris_classifier@production"
print(f"Production model URI: {model_uri}")
```

Serve the registered model:

bash

```
# Serve using model alias (MLflow 3.x way)
mlflow models serve \
  --model-uri "models:/iris_classifier@production" \
  --port 5000 \
  --env-manager local

# Server will start at http://localhost:5000
# Available endpoints:
# - POST /invocations (predictions)
# - GET /ping (health check)
# - GET /version (model info)
```

**Alternative serving approaches:**

bash

```
# Serve by specific version number
mlflow models serve \
  --model-uri "models:/iris_classifier/1" \
  --port 5000

# Serve from run URI
mlflow models serve \
  --model-uri "runs:/<run-id>/model" \
  --port 5000
```

Send requests to your served model:

python

```
import requests
import json
import pandas as pd

# Prepare test data (same format as training)
test_data = {
    "dataframe_split": {
        "columns": [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # setosa
            [6.2, 2.9, 4.3, 1.3],  # versicolor
            [7.3, 2.9, 6.3, 1.8],  # virginica
        ],
    }
}

# Make prediction request
response = requests.post(
    "http://localhost:5000/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(test_data),
)

# Parse response
if response.status_code == 200:
    predictions = response.json()
    print("Predictions:", predictions)
    # Output: {"predictions": [0, 1, 2]}
else:
    print(f"Error: {response.status_code}, {response.text}")

# Health check
health = requests.get("http://localhost:5000/ping")
print("Health status:", health.status_code)  # Should be 200

# Model info
info = requests.get("http://localhost:5000/version")
print("Model version info:", info.json())
```

## Next Steps[​](#next-steps "Direct link to Next Steps")

Ready to build more advanced serving applications? Explore these specialized topics:

Get Started

The examples in each section are designed to be practical and ready-to-use. Start with the Quick Start above, then explore the use cases that match your deployment needs.

### [Custom Applications](/mlflow-website/docs/latest/genai/serving/custom-apps.md)

[Build sophisticated serving logic with custom preprocessing, routing, and business rules](/mlflow-website/docs/latest/genai/serving/custom-apps.md)

[Build custom apps →](/mlflow-website/docs/latest/genai/serving/custom-apps.md)

### [Responses Agents](/mlflow-website/docs/latest/genai/serving/responses-agent.md)

[Handle complex response patterns and multi-step inference workflows](/mlflow-website/docs/latest/genai/serving/responses-agent.md)

[Learn about agents →](/mlflow-website/docs/latest/genai/serving/responses-agent.md)
