# MLflow Keras 3.0 Integration

## Introduction[​](#introduction "Direct link to Introduction")

**Keras 3.0** is a high-level neural networks API that runs on TensorFlow, JAX, and PyTorch backends. It provides a user-friendly interface for building and training deep learning models with the flexibility to switch backends without changing your code.

MLflow's Keras integration provides experiment tracking, model versioning, and deployment capabilities for deep learning workflows.

## Why MLflow + Keras?[​](#why-mlflow--keras "Direct link to Why MLflow + Keras?")

#### Autologging

Enable comprehensive experiment tracking with one line: mlflow\.tensorflow\.autolog() automatically logs metrics, parameters, and models.

#### Experiment Tracking

Track training metrics, hyperparameters, model architectures, and artifacts across all Keras experiments.

#### Model Registry

Version, stage, and deploy Keras models with MLflow's model registry and serving infrastructure.

#### Multi-Backend Support

Track experiments consistently across TensorFlow, JAX, and PyTorch backends.

## Autologging[​](#autologging "Direct link to Autologging")

Enable comprehensive autologging with a single line:

python

```python
import mlflow
import numpy as np
from tensorflow import keras

# Enable autologging
mlflow.tensorflow.autolog()

# Prepare sample data
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(0, 2, 1000)

# Define model
model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training with automatic logging
with mlflow.start_run():
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

```

Autologging captures training metrics, model parameters, optimizer configuration, and model artifacts automatically.

Configure autologging behavior:

python

```python
mlflow.tensorflow.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True,
    log_every_n_steps=1,
)

```

## Manual Logging with Keras Callback[​](#manual-logging-with-keras-callback "Direct link to Manual Logging with Keras Callback")

For more control, use [`mlflow.tensorflow.MlflowCallback()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.tensorflow.html#mlflow.tensorflow.MlflowCallback):

python

```python
import mlflow
import numpy as np
from tensorflow import keras

# Prepare sample data
X_train = np.random.rand(100, 20)
y_train = np.random.randint(0, 2, 100)

# Define and compile model
model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(20,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Create an MLflow run and add the callback
with mlflow.start_run() as run:
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[mlflow.tensorflow.MlflowCallback(run)],
    )

```

## Model Logging[​](#model-logging "Direct link to Model Logging")

Save Keras models with [`mlflow.tensorflow.log_model()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.tensorflow.html#mlflow.tensorflow.log_model):

python

```python
import mlflow
from tensorflow import keras

# Define model
model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(20,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Train model (code omitted for brevity)

# Log the model to MLflow
model_info = mlflow.tensorflow.log_model(model, name="model")

# Later, load the model for inference
loaded_model = mlflow.tensorflow.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)

```

## Model Registry Integration[​](#model-registry-integration "Direct link to Model Registry Integration")

Register Keras models for version control and deployment:

python

```python
import mlflow
from tensorflow import keras
from mlflow import MlflowClient

with mlflow.start_run():
    # Create a simple model for demonstration
    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Log model to registry
    model_info = mlflow.tensorflow.log_model(
        model, name="keras_model", registered_model_name="ImageClassifier"
    )

    # Tag for tracking
    mlflow.set_tags({"model_type": "cnn", "dataset": "mnist", "framework": "keras"})

# Set alias for production deployment
client = MlflowClient()
client.set_registered_model_alias(
    name="ImageClassifier",
    alias="champion",
    version=model_info.registered_model_version,
)

```

## Learn More[​](#learn-more "Direct link to Learn More")

### [Model Registry](/mlflow-website/docs/latest/ml/model-registry.md)

[Version and manage Keras models](/mlflow-website/docs/latest/ml/model-registry.md)

[Learn more →](/mlflow-website/docs/latest/ml/model-registry.md)

### [MLflow Tracking](/mlflow-website/docs/latest/ml/tracking.md)

[Track experiments, parameters, and metrics](/mlflow-website/docs/latest/ml/tracking.md)

[Learn more →](/mlflow-website/docs/latest/ml/tracking.md)
