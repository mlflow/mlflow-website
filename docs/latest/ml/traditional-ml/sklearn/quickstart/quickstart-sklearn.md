# Get Started with MLflow + Scikit-learn

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/sklearn/quickstart/quickstart-sklearn.ipynb)

In this guide, we will show you how to train a model with scikit-learn and log your training using MLflow.

We will be using the [Databricks Free Trial](https://mlflow.org/docs/latest/ml/getting-started/databricks-trial.html), which has built-in support for MLflow. The Databricks Free Trial provides an opportunity to use Databricks platform for free. If you haven't already, please register for an account via [this link](https://signup.databricks.com/?destination_url=/ml/experiments-signup?source=OSS_DOCS\&dbx_source=TRY_MLFLOW\&signup_experience_step=EXPRESS\&provider=MLFLOW\&utm_source=OSS_DOCS).

You can run code in this guide from cloud-based notebooks like Databricks notebook or Google Colab, or run it on your local machine.

## Install dependencies[​](#install-dependencies "Direct link to Install dependencies")

Let's install the `mlflow` package.

python

```
%pip install mlflow
```

Then let's import the packages

python

```
from sklearn.datasets import load_iris
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
```

## Load and prepare the dataset[​](#load-and-prepare-the-dataset "Direct link to Load and prepare the dataset")

We will train a simple multi-class classification model for Iris flowers using the [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set).

Let's load the dataset using `load_iris()` into a pandas Dataframe and take a look at the data.

python

```
iris_df = load_iris(as_frame=True).frame
iris_df
```

|     | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | target |
| --- | ----------------- | ---------------- | ----------------- | ---------------- | ------ |
| 0   | 5.1               | 3.5              | 1.4               | 0.2              | 0      |
| 1   | 4.9               | 3.0              | 1.4               | 0.2              | 0      |
| 2   | 4.7               | 3.2              | 1.3               | 0.2              | 0      |
| 3   | 4.6               | 3.1              | 1.5               | 0.2              | 0      |
| 4   | 5.0               | 3.6              | 1.4               | 0.2              | 0      |
| ... | ...               | ...              | ...               | ...              | ...    |
| 145 | 6.7               | 3.0              | 5.2               | 2.3              | 2      |
| 146 | 6.3               | 2.5              | 5.0               | 1.9              | 2      |
| 147 | 6.5               | 3.0              | 5.2               | 2.0              | 2      |
| 148 | 6.2               | 3.4              | 5.4               | 2.3              | 2      |
| 149 | 5.9               | 3.0              | 5.1               | 1.8              | 2      |

150 rows × 5 columns

Now we'll split our dataset into training and testing sets

python

```
# Split into 80% training and 20% testing
train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42)
train_df.shape, test_df.shape
```

```
((120, 5), (30, 5))
```

python

```
# Separate the target column for the training set
train_dataset = mlflow.data.from_pandas(train_df, name="train")
train_x = train_dataset.df.drop(["target"], axis=1)
train_y = train_dataset.df[["target"]]

train_x.shape, train_y.shape
```

```
((120, 4), (120, 1))
```

python

```
# Separate the target column for the testing set
test_dataset = mlflow.data.from_pandas(test_df, name="test")
test_x = test_dataset.df.drop(["target"], axis=1)
test_y = test_dataset.df[["target"]]

test_x.shape, test_y.shape
```

```
((30, 4), (30, 1))
```

## Define the Model[​](#define-the-model "Direct link to Define the Model")

For this example, we'll use an ElasticNet model with some pre-defined hyperparameters. Let's also define a helper function to compute some metrics to evaluate our model's performance.

python

```
lr = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
```

python

```
def compute_metrics(actual, predicted):
  rmse = mean_squared_error(actual, predicted)
  mae = mean_absolute_error(actual, predicted)
  r2 = r2_score(actual, predicted)

  return rmse, mae, r2
```

## Connect to MLflow Tracking Server[​](#connect-to-mlflow-tracking-server "Direct link to Connect to MLflow Tracking Server")

Before training, we need to configure the MLflow tracking server because we will log data into MLflow. In this tutorial, we will use Databricks Free Trial for MLflow tracking server. For other options such as using your local MLflow server, please read the [Tracking Server Overview](https://mlflow.org/docs/latest/ml/getting-started/tracking-server-overview/).

If you have not, please set up your account and access token of the Databricks Free Trial by following [this guide](https://mlflow.org/docs/latest/ml/getting-started/tracking-server-overview/). It should take no longer than 5 mins to register. For this guide, we need the ML experiment dashboard for us to track our training progress.

After successfully registering an account on the Databricks Free Trial, let's connnect MLflow to the Databricks Workspace. You will need to enter following information:

* **Databricks Host**: https\://\<your workspace host>.cloud.databricks.com
* **Token**: You Personal Access Token

python

```
mlflow.login()
```

Now this notebook is connected to the hosted tracking server. Let's configure some MLflow metadata. Two things to set up:

* `mlflow.set_tracking_uri`: always use "databricks".
* `mlflow.set_experiment`: pick up a name you like, start with `/Users/<your email>/`.

python

```
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/<your email>/mlflow-sklearn-quickstart")
```

## Logging with MLflow[​](#logging-with-mlflow "Direct link to Logging with MLflow")

MLflow has powerful tracking APIs that let's us log runs and models along with their associated metadata such as parameters and metrics. Let's first start a training run to train our model.

python

```
# Start a training run
with mlflow.start_run() as training_run:
  # Log the parameters for our model
  mlflow.log_param("alpha", 0.5)
  mlflow.log_param("l1_ratio", 0.5)

  # Train and log our model, which inherits the parameters
  lr.fit(train_x, train_y)
  model_info = mlflow.sklearn.log_model(sk_model=lr, name="elasticnet", input_example=train_x)

  # Evaluate the model on the training dataset and log metrics
  # These metrics will be linked to both the model and run
  predictions = lr.predict(train_x)
  (rmse, mae, r2) = compute_metrics(train_y, predictions)
  mlflow.log_metrics(
      metrics={
          "rmse": rmse,
          "r2": r2,
          "mae": mae,
      },
      dataset=train_dataset,
  )
```

Let's now evaluate our model on the test dataset

python

```
# Start an evaluation run
with mlflow.start_run() as evaluation_run:
  # Load our previous model
  logged_model = mlflow.sklearn.load_model(f"models:/{model_info.model_id}")

  # Evaluate the model on the training dataset and log metrics
  predictions = logged_model.predict(test_x)
  (rmse, mae, r2) = compute_metrics(test_y, predictions)
  mlflow.log_metrics(
      metrics={
          "rmse": rmse,
          "r2": r2,
          "mae": mae,
      },
      dataset=test_dataset,
      model_id=model_info.model_id,
  )
```

## View results[​](#view-results "Direct link to View results")

Let's look at our training and testing results. Log in to your Databricks Workspace, and click on the `Experiments` tab from the left menu. The initial page displays a list of runs, where we can see our training and evaluation runs.

![runs page](https://imgur.com/xtIRa9P.png)

Now let's head to the models tab, where we can see the model that we logged

![models page](https://imgur.com/hapuVHr.png)

Clicking on the model name will take you to the model details page, with information on its parameters, metrics across both runs, and other metadata.

![model details page](https://imgur.com/dAXNEWV.png)

We can also inspect our model using the API

python

```
logged_model = mlflow.get_logged_model(model_info.model_id)

logged_model, logged_model.metrics, logged_model.params
```
