# Get Started with MLflow + Tensorflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/tensorflow/quickstart/quickstart-tensorflow.ipynb)

In this guide, we will show how to train your model with Tensorflow and log your training using MLflow.

We will use the [Databricks Free Trial](https://mlflow.org/docs/latest/getting-started/databricks-trial.html), which has built-in support for MLflow. The Databricks Free Trial provides an opportunity to use Databricks platform for free, if you haven't, please register an account via [link](https://signup.databricks.com/?destination_url=/ml/experiments-signup?source=OSS_DOCS\&dbx_source=TRY_MLFLOW\&signup_experience_step=EXPRESS\&provider=MLFLOW\&utm_source=OSS_DOCS).

You can run code in this guide from cloud-based notebooks like Databricks notebook or Google Colab, or run it on your local machine.

## Install dependencies[​](#install-dependencies "Direct link to Install dependencies")

Let's install the `mlflow` package.

text

```
%pip install -q mlflow
```

Then let's import the packages.

python

```
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
```

## Load the dataset[​](#load-the-dataset "Direct link to Load the dataset")

We will do a simple image classification on handwritten digits with [mnist dataset](https://en.wikipedia.org/wiki/MNIST_database).

Let's load the dataset using `tensorflow_datasets` (`tfds`), which returns datasets in the format of `tf.data.Dataset`.

python

```
# Load the mnist dataset.
train_ds, test_ds = tfds.load(
  "mnist",
  split=["train", "test"],
  shuffle_files=True,
)
```

```
Downloading and preparing dataset 11.06 MiB (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /root/tensorflow_datasets/mnist/3.0.1...
```

```
Dl Completed...:   0%|          | 0/5 [00:00<?, ? file/s]
```

```
Dataset mnist downloaded and prepared to /root/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.
```

Let's preprocess our data with the following steps:

* Scale each pixel's value to `[0, 1)`.
* Batch the dataset.
* Use `prefetch` to speed up the training.

python

```
def preprocess_fn(data):
  image = tf.cast(data["image"], tf.float32) / 255
  label = data["label"]
  return (image, label)


train_ds = train_ds.map(preprocess_fn).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_fn).batch(128).prefetch(tf.data.AUTOTUNE)
```

## Define the Model[​](#define-the-model "Direct link to Define the Model")

Let's define a convolutional neural network as our classifier. We can use `keras.Sequential` to stack up the layers.

python

```
input_shape = (28, 28, 1)
num_classes = 10

model = keras.Sequential(
  [
      keras.Input(shape=input_shape),
      keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(num_classes, activation="softmax"),
  ]
)
```

Set training-related configs, optimizers, loss function, metrics.

python

```
model.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(),
  optimizer=keras.optimizers.Adam(0.001),
  metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

## Set up tracking/visualization tool[​](#set-up-trackingvisualization-tool "Direct link to Set up tracking/visualization tool")

In this tutorial, we will use Databricks Free Trial for MLflow tracking server. For other options such as using your local MLflow server, please read the [Tracking Server Overview](https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html).

If you have not, please set up your account and access token of the Databricks Free Trial by following [this guide](https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html). It should take no longer than 5 mins to register. The Databricks Free Trial is a way for users to try out Databricks features for free. For this guide, we need the ML experiment dashboard for us to track our training progress.

After successfully registering an account on the Databricks Free Trial, let's connnect MLflow to the Databricks Workspace. You will need to enter following information:

* **Databricks Host**: https\://\<your workspace host>.cloud.databricks.com/
* **Token**: You Personal Access Token

python

```
import mlflow

mlflow.login()
```

Now this colab is connected to the hosted tracking server. Let's configure MLflow metadata. Two things to set up:

* `mlflow.set_tracking_uri`: always use "databricks".
* `mlflow.set_experiment`: pick up a name you like, start with `/`.

## Logging with MLflow[​](#logging-with-mlflow "Direct link to Logging with MLflow")

There are two ways you can log to MLflow from your Tensorflow pipeline:

* MLflow auto logging.
* Use a callback.

Auto logging is simple to configure, but gives you less control. Using a callback is more flexible. Let's see how each way is done.

### MLflow Auto Logging[​](#mlflow-auto-logging "Direct link to MLflow Auto Logging")

All you need to do is to call `mlflow.tensorflow.autolog()` before kicking off the training, then the backend will automatically log the metrics into the server you configured earlier. In our case, Databricks Workspace.

python

```
# Choose any name that you like.
mlflow.set_experiment("/Users/<your email>/mlflow-tf-keras-mnist")

mlflow.tensorflow.autolog()

model.fit(x=train_ds, epochs=3)
```

```
2023/11/15 01:53:35 INFO mlflow.utils.autologging_utils: Created MLflow autologging run with ID '7c1db53e417b43f0a1d9e095c9943acb', which will track hyperparameters, performance metrics, model artifacts, and lineage information for the current tensorflow workflow
```

```
Epoch 1/3
469/469 [==============================] - 13s 7ms/step - loss: 0.3610 - sparse_categorical_accuracy: 0.8890
Epoch 2/3
469/469 [==============================] - 3s 6ms/step - loss: 0.1035 - sparse_categorical_accuracy: 0.9681
Epoch 3/3
469/469 [==============================] - 4s 8ms/step - loss: 0.0798 - sparse_categorical_accuracy: 0.9760
```

```
2023/11/15 01:54:05 WARNING mlflow.tensorflow: Failed to infer model signature: could not sample data to infer model signature: tuple index out of range
2023/11/15 01:54:05 WARNING mlflow.models.model: Model logged without a signature. Signatures will be required for upcoming model registry features as they validate model inputs and denote the expected schema of model outputs. Please visit https://www.mlflow.org/docs/2.8.1/models.html#set-signature-on-logged-model for instructions on setting a model signature on your logged model.
2023/11/15 01:54:05 WARNING mlflow.tensorflow: You are saving a TensorFlow Core model or Keras model without a signature. Inference with mlflow.pyfunc.spark_udf() will not work unless the model's pyfunc representation accepts pandas DataFrames as inference inputs.
2023/11/15 01:54:13 WARNING mlflow.utils.autologging_utils: MLflow autologging encountered a warning: "/usr/local/lib/python3.10/dist-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils."
```

```
Uploading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]
```

```
2023/11/15 01:54:13 INFO mlflow.store.artifact.cloud_artifact_repo: The progress bar can be disabled by setting the environment variable MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR to false
```

```
Uploading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]
```

```
<keras.src.callbacks.History at 0x7d48e6556b60>
```

While your training is ongoing, you can find this training in your dashboard. Log in to your Databricks Workspace, and click on the `Experiments tab`. See the screenshot below: ![landing page](https://i.imgur.com/bBiIPqp.png)

After clicking the `Experiments` button, it will bring you to the experiments page, where you can find your runs. Clicking on the most recent experiment and run, you can find your metrics there, similar to: ![experiment page](https://i.imgur.com/Idddpqe.png)

You can click on metrics to see the chart.

Let's evaluate the training result.

python

```
score = model.evaluate(test_ds)

print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]: .2f}")
```

```
79/79 [==============================] - 1s 12ms/step - loss: 0.0484 - sparse_categorical_accuracy: 0.9838
Test loss: 0.05
Test accuracy:  0.98
```

### Log with MLflow Callback[​](#log-with-mlflow-callback "Direct link to Log with MLflow Callback")

Auto logging is powerful and convenient, but if you are looking for a more native way as Tensorflow pipelines, you can use `mlflow.tensorflow.MllflowCallback` inside `model.fit()`, it will log:

* Your model configuration, layers, hyperparameters and so on.
* The training stats, including losses and metrics configured with `model.compile()`.

python

```
from mlflow.tensorflow import MlflowCallback

# Turn off autologging.
mlflow.tensorflow.autolog(disable=True)

with mlflow.start_run() as run:
  model.fit(
      x=train_ds,
      epochs=2,
      callbacks=[MlflowCallback(run)],
  )
```

```
Epoch 1/2
469/469 [==============================] - 5s 10ms/step - loss: 0.0473 - sparse_categorical_accuracy: 0.9851
Epoch 2/2
469/469 [==============================] - 4s 8ms/step - loss: 0.0432 - sparse_categorical_accuracy: 0.9866
```

Going to the Databricks Workspace experiment view, you will see a similar dashboard as before.

### Customize the MLflow Callback[​](#customize-the-mlflow-callback "Direct link to Customize the MLflow Callback")

If you want to add extra logging logic, you can customize the MLflow callback. You can either subclass from `keras.callbacks.Callback` and write everything from scratch or subclass from `mlflow.tensorflow.MllflowCallback` to add you custom logging logic.

Let's look at an example that we want to replace the loss with its log value to log to MLflow.

python

```
import math


# Create our own callback by subclassing `MlflowCallback`.
class MlflowCustomCallback(MlflowCallback):
  def on_epoch_end(self, epoch, logs=None):
      if not self.log_every_epoch:
          return
      loss = logs["loss"]
      logs["log_loss"] = math.log(loss)
      del logs["loss"]
      mlflow.log_metrics(logs, epoch)
```

Train the model with the new callback.

python

```
with mlflow.start_run() as run:
  run_id = run.info.run_id
  model.fit(
      x=train_ds,
      epochs=2,
      callbacks=[MlflowCustomCallback(run)],
  )
```

```
Epoch 1/2
469/469 [==============================] - 5s 10ms/step - loss: 0.0537 - sparse_categorical_accuracy: 0.9834 - log_loss: -2.9237
Epoch 2/2
469/469 [==============================] - 4s 9ms/step - loss: 0.0497 - sparse_categorical_accuracy: 0.9846 - log_loss: -3.0022
```

Going to your Databricks Workspace page, you should find the `log_loss` is replacing the `loss` metric, similar to what is shown in the screenshot below.

![log loss screenshot](https://i.imgur.com/tgP4Cji.png)

## Wrap up[​](#wrap-up "Direct link to Wrap up")

Now you have learned the basic integration between MLflow and Tensorflow. There are a few things not covered by this quickstart, e.g., saving TF model to MLflow and loading it back. For a detailed guide, please refer to our main guide for integration between MLflow and Tensorflow.
