# Quickstart with MLflow PyTorch Flavor

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/pytorch/quickstart/quickstart-pytorch.ipynb)

In this quickstart guide, we will walk you through how to log your PyTorch experiments to MLflow. After reading this quickstart, you will learn the basics of logging PyTorch experiments to MLflow, and how to view the experiment results in the MLflow UI.

This quickstart guide is compatible with cloud-based notebook such as Google Colab and Databricks notebook, you can also run it locally.

## Install Required Packages[​](#install-required-packages "Direct link to Install Required Packages")

python

```
%pip install -q mlflow torchmetrics torchinfo
```

python

```
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor

import mlflow
```

## Task Overview[​](#task-overview "Direct link to Task Overview")

In this guide, we will demonstrate the functionality of MLflow with PyTorch through a simple MNIST image classification task. We will build a convolutional neural network as the image classifier, and log the following information to mlflow:

* **Training Metrics**: training loss and accuracy.
* **Evalluation Metrics**: evaluation loss and accuracy.
* **Training Configs**: learning rate, batch size, etc.
* **Model Information**: model structure.
* **Saved Model**: model instance after training.

Now let's dive into the details!

## Prepare the Data[​](#prepare-the-data "Direct link to Prepare the Data")

Let's load our training data `FashionMNIST` from `torchvision`, which has already been preprocessed into scale the \[0, 1). We then wrap the dataset into an instance of `torch.utils.data.Dataloader`.

python

```
training_data = datasets.FashionMNIST(
  root="data",
  train=True,
  download=True,
  transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=ToTensor(),
)
```

Let's look into our data.

python

```
print(f"Image size: {training_data[0][0].shape}")
print(f"Size of training dataset: {len(training_data)}")
print(f"Size of test dataset: {len(test_data)}")
```

```
Image size: torch.Size([1, 28, 28])
Size of training dataset: 60000
Size of test dataset: 10000
```

We wrap the dataset a `Dataloader` instance for batching purposes. `Dataloader` is a useful tool for data preprocessing. For more details, you can refer to the [developer guide from PyTorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders).

python

```
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
```

## Define our Model[​](#define-our-model "Direct link to Define our Model")

Now, let's define our model. We will build a simple convolutional neural network as the classifier. To define a PyTorch model, you will need to subclass from `torch.nn.Module` and override `__init__` to define model components, as well as the `forward()` method to implement the forward-pass logic.

We will build a simple convolution neural network (CNN) consisting of 2 convolutional layers as the image classifier. CNN is a common architecture used in image classification task, for more details about CNN please read [this doc](https://en.wikipedia.org/wiki/Convolutional_neural_network). Our model output will be the logits of each class (10 classes in total). Applying softmax on logits yields the probability distribution across classes.

python

```
class ImageClassifier(nn.Module):
  def __init__(self):
      super().__init__()
      self.model = nn.Sequential(
          nn.Conv2d(1, 8, kernel_size=3),
          nn.ReLU(),
          nn.Conv2d(8, 16, kernel_size=3),
          nn.ReLU(),
          nn.Flatten(),
          nn.LazyLinear(10),  # 10 classes in total.
      )

  def forward(self, x):
      return self.model(x)
```

## Connect to MLflow Tracking Server[​](#connect-to-mlflow-tracking-server "Direct link to Connect to MLflow Tracking Server")

Before implementing the training loop, we need to configure the MLflow tracking server because we will log data into MLflow during training.

In this guide, we will use [Databricks Free Trial](https://mlflow.org/docs/latest/getting-started/databricks-trial.html) for MLflow tracking server. For other options such as using your local MLflow server, please read the [Tracking Server Overview](https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html).

If you have not, please set up your account and access token of the Databricks Free Trial by following [this guide](https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html). It should take no longer than 5 mins to register. The Databricks Free Trial is a way for users to try out Databricks features for free. For this guide, we need the ML experiment dashboard for us to track our training progress.

After successfully registering an account on the Databricks Free Trial, let's connnect MLflow to the Databricks Workspace. You will need to enter following information:

* **Databricks Host**: https\://\<your workspace host>.cloud.databricks.com/
* **Token**: You Personal Access Token

python

```
mlflow.login()
```

Now you have successfully connected to MLflow tracking server on your Databricks Workspace, and let's give our experiment a name.

python

```
mlflow.set_experiment("/Users/<your email>/mlflow-pytorch-quickstart")
```

```
<Experiment: artifact_location='dbfs:/databricks/mlflow-tracking/1078557169589361', creation_time=1703121702068, experiment_id='1078557169589361', last_update_time=1703194525608, lifecycle_stage='active', name='/mlflow-pytorch-quickstart', tags={'mlflow.experiment.sourceName': '/mlflow-pytorch-quickstart',
'mlflow.experimentType': 'MLFLOW_EXPERIMENT',
'mlflow.ownerEmail': 'qianchen94era@gmail.com',
'mlflow.ownerId': '3209978630771139'}>
```

## Implement the Training Loop[​](#implement-the-training-loop "Direct link to Implement the Training Loop")

Now let's define the training loop, which basically iterating through the dataset and applying a forward and backward pass on each data batch.

Get the device info, as PyTorch requires manual device management.

python

```
# Get cpu or gpu for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Define the training function.

python

```
def train(dataloader, model, loss_fn, metrics_fn, optimizer, epoch):
  """Train the model on a single pass of the dataloader.

  Args:
      dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
      model: an instance of `torch.nn.Module`, the model to be trained.
      loss_fn: a callable, the loss function.
      metrics_fn: a callable, the metrics function.
      optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
      epoch: an integer, the current epoch number.
  """
  model.train()
  for batch, (X, y) in enumerate(dataloader):
      X = X.to(device)
      y = y.to(device)

      pred = model(X)
      loss = loss_fn(pred, y)
      accuracy = metrics_fn(pred, y)

      # Backpropagation.
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if batch % 100 == 0:
          loss_value = loss.item()
          current = batch
          step = batch // 100 * (epoch + 1)
          mlflow.log_metric("loss", f"{loss_value:2f}", step=step)
          mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
          print(f"loss: {loss_value:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")
```

Define the evaluation function, which will be run at the end of each epoch.

python

```
def evaluate(dataloader, model, loss_fn, metrics_fn, epoch):
  """Evaluate the model on a single pass of the dataloader.

  Args:
      dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
      model: an instance of `torch.nn.Module`, the model to be trained.
      loss_fn: a callable, the loss function.
      metrics_fn: a callable, the metrics function.
      epoch: an integer, the current epoch number.
  """
  num_batches = len(dataloader)
  model.eval()
  eval_loss = 0
  eval_accuracy = 0
  with torch.no_grad():
      for X, y in dataloader:
          X = X.to(device)
          y = y.to(device)
          pred = model(X)
          eval_loss += loss_fn(pred, y).item()
          eval_accuracy += metrics_fn(pred, y)

  eval_loss /= num_batches
  eval_accuracy /= num_batches
  mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
  mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch)

  print(f"Eval metrics: 
Accuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} 
")
```

## Start Training[​](#start-training "Direct link to Start Training")

It's time to start the training! First let's define training hyperparameters, create our model, declare our loss function and instantiate our optimizer.

python

```
epochs = 3
loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
model = ImageClassifier().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

```
/usr/local/lib/python3.10/dist-packages/torch/nn/modules/lazy.py:180: UserWarning: Lazy modules are a new feature under heavy development so changes to the API or functionality can happen at any moment.
warnings.warn('Lazy modules are a new feature under heavy development '
```

Putting everything together, let's kick off the training and log information to MLflow. At the beginning of training, we log training and model information to MLflow, and during training, we log training and evaluation metrics. After everything is done, we log the trained model.

python

```
with mlflow.start_run() as run:
  params = {
      "epochs": epochs,
      "learning_rate": 1e-3,
      "batch_size": 64,
      "loss_function": loss_fn.__class__.__name__,
      "metric_function": metric_fn.__class__.__name__,
      "optimizer": "SGD",
  }
  # Log training parameters.
  mlflow.log_params(params)

  # Log model summary.
  with open("model_summary.txt", "w") as f:
      f.write(str(summary(model)))
  mlflow.log_artifact("model_summary.txt")

  for t in range(epochs):
      print(f"Epoch {t + 1}
-------------------------------")
      train(train_dataloader, model, loss_fn, metric_fn, optimizer, epoch=t)
      evaluate(test_dataloader, model, loss_fn, metric_fn, epoch=0)

  # Save the trained model to MLflow.
  model_info = mlflow.pytorch.log_model(model, name="model")
```

```
Epoch 1
-------------------------------
loss: 2.294313 accuracy: 0.046875 [0 / 938]
loss: 2.151955 accuracy: 0.515625 [100 / 938]
loss: 1.825312 accuracy: 0.640625 [200 / 938]
loss: 1.513407 accuracy: 0.593750 [300 / 938]
loss: 1.059044 accuracy: 0.718750 [400 / 938]
loss: 0.931140 accuracy: 0.687500 [500 / 938]
loss: 0.889886 accuracy: 0.703125 [600 / 938]
loss: 0.742625 accuracy: 0.765625 [700 / 938]
loss: 0.786106 accuracy: 0.734375 [800 / 938]
loss: 0.788444 accuracy: 0.781250 [900 / 938]
Eval metrics: 
Accuracy: 0.75, Avg loss: 0.719401 

Epoch 2
-------------------------------
loss: 0.649325 accuracy: 0.796875 [0 / 938]
loss: 0.756684 accuracy: 0.718750 [100 / 938]
loss: 0.488664 accuracy: 0.828125 [200 / 938]
loss: 0.780433 accuracy: 0.718750 [300 / 938]
loss: 0.691777 accuracy: 0.656250 [400 / 938]
loss: 0.670005 accuracy: 0.750000 [500 / 938]
loss: 0.712286 accuracy: 0.687500 [600 / 938]
loss: 0.644150 accuracy: 0.765625 [700 / 938]
loss: 0.683426 accuracy: 0.750000 [800 / 938]
loss: 0.659378 accuracy: 0.781250 [900 / 938]
Eval metrics: 
Accuracy: 0.77, Avg loss: 0.636072 

Epoch 3
-------------------------------
loss: 0.528523 accuracy: 0.781250 [0 / 938]
loss: 0.634942 accuracy: 0.750000 [100 / 938]
loss: 0.420757 accuracy: 0.843750 [200 / 938]
loss: 0.701463 accuracy: 0.703125 [300 / 938]
loss: 0.649267 accuracy: 0.656250 [400 / 938]
loss: 0.624556 accuracy: 0.812500 [500 / 938]
loss: 0.648762 accuracy: 0.718750 [600 / 938]
loss: 0.630074 accuracy: 0.781250 [700 / 938]
loss: 0.682306 accuracy: 0.718750 [800 / 938]
loss: 0.587403 accuracy: 0.750000 [900 / 938]
```

```
2023/12/21 21:39:55 WARNING mlflow.models.model: Model logged without a signature. Signatures will be required for upcoming model registry features as they validate model inputs and denote the expected schema of model outputs. Please visit https://www.mlflow.org/docs/2.9.2/models.html#set-signature-on-logged-model for instructions on setting a model signature on your logged model.
2023/12/21 21:39:56 WARNING mlflow.utils.requirements_utils: Found torch version (2.1.0+cu121) contains a local version label (+cu121). MLflow logged a pip requirement for this package as 'torch==2.1.0' without the local version label to make it installable from PyPI. To specify pip requirements containing local version labels, please use `conda_env` or `pip_requirements`.
```

```
Eval metrics: 
Accuracy: 0.77, Avg loss: 0.616615
```

```
2023/12/21 21:40:02 WARNING mlflow.utils.requirements_utils: Found torch version (2.1.0+cu121) contains a local version label (+cu121). MLflow logged a pip requirement for this package as 'torch==2.1.0' without the local version label to make it installable from PyPI. To specify pip requirements containing local version labels, please use `conda_env` or `pip_requirements`.
/usr/local/lib/python3.10/dist-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
warnings.warn("Setuptools is replacing distutils.")
```

```
Uploading artifacts:   0%|          | 0/6 [00:00<?, ?it/s]
```

While your training is ongoing, you can find this training in your dashboard. Log in to your Databricks Workspace, and click on the `Experiments tab`. See the screenshot below: ![landing page](https://i.imgur.com/bBiIPqp.png)

After clicking the `Experiments` tab, it will bring you to the experiment page, where you can find your runs. Clicking on the most recent experiment and run, you can find your metrics there, similar to: ![experiment page](https://i.imgur.com/jNa04eT.png)

Under artifact section you can see our model is successfully logged: ![saved model](https://i.imgur.com/qTi4nW7.png)

For the last step, let's load back the model and run inference on it.

python

```
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
```

```
Downloading artifacts:   0%|          | 0/6 [00:00<?, ?it/s]
```

There is a caveat that the input to the loaded model has to be a `numpy` array or `pandas` Dataframe, so we need to cast the tensor to `numpy` format explicitly.

python

```
outputs = loaded_model.predict(training_data[0][0][None, :].numpy())
```
