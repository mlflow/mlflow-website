# Get Started with MLflow + XGBoost

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/xgboost/quickstart/quickstart-xgboost.ipynb)

In this guide, we will show you how to train a model with XGBoost and log your training using MLflow.

We will be using the [Databricks Free Trial](https://mlflow.org/docs/latest/ml/getting-started/databricks-trial.html), which has built-in support for MLflow. The Databricks Free Trial provides an opportunity to use Databricks platform for free. If you haven't already, please register for an account via [this link](https://signup.databricks.com/?destination_url=/ml/experiments-signup?source=TRY_MLFLOW\&dbx_source=TRY_MLFLOW\&signup_experience_step=EXPRESS\&provider=MLFLOW).

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
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature
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
X_train = train_dataset.df.drop(["target"], axis=1)
y_train = train_dataset.df[["target"]]

dtrain = xgb.DMatrix(X_train, label=y_train)
```

python

```
# Separate the target column for the testing set
test_dataset = mlflow.data.from_pandas(test_df, name="test")
X_test = test_dataset.df.drop(["target"], axis=1)
y_test = test_dataset.df[["target"]]

dtest = xgb.DMatrix(X_test, label=y_test)
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
mlflow.set_experiment("/Users/<your email>/mlflow-xgboost-quickstart")
```

## Logging with MLflow[​](#logging-with-mlflow "Direct link to Logging with MLflow")

MLflow has powerful tracking APIs that let's us log runs and models along with their associated metadata such as parameters and metrics. Let's train and evaluate our model.

python

```
# Start a training run
with mlflow.start_run() as run:
  # Define and log the parameters for our model
  params = {
      "objective": "multi:softprob",
      "num_class": len(set(train_df["target"])),
      "max_depth": 8,
      "learning_rate": 0.05,
      "subsample": 0.9,
      "colsample_bytree": 0.9,
      "min_child_weight": 1,
      "gamma": 0,
      "reg_alpha": 0,
      "reg_lambda": 1,
      "random_state": 42,
  }
  training_config = {
      "num_boost_round": 200,
      "early_stopping_rounds": 20,
  }
  mlflow.log_params(params)
  mlflow.log_params(training_config)

  # Custom evaluation tracking
  eval_results = {}
  # Train model with custom callback
  model = xgb.train(
      params=params,
      dtrain=dtrain,
      num_boost_round=training_config["num_boost_round"],
      evals=[(dtrain, "train"), (dtest, "test")],
      early_stopping_rounds=training_config["early_stopping_rounds"],
      evals_result=eval_results,
      verbose_eval=False,
  )

  # Log training history to the run
  for epoch, (train_metrics, test_metrics) in enumerate(
      zip(eval_results["train"]["mlogloss"], eval_results["test"]["mlogloss"])
  ):
      mlflow.log_metrics(
          {"train_logloss": train_metrics, "test_logloss": test_metrics}, step=epoch
      )

  # Final evaluation
  y_pred_proba = model.predict(dtest)
  y_pred = np.argmax(y_pred_proba, axis=1)
  final_metrics = {
      "accuracy": accuracy_score(y_test, y_pred),
      "roc_auc": roc_auc_score(y_test, y_pred_proba, multi_class="ovr"),
  }
  mlflow.log_metrics(final_metrics, step=model.best_iteration)

  # Log the model at the best iteration, linked with all params and metrics
  model_info = mlflow.xgboost.log_model(
      xgb_model=model,
      name="xgboost_model",
      signature=infer_signature(X_train, y_pred_proba),
      input_example=X_train[:5],
      step=model.best_iteration,
  )
```

## View results[​](#view-results "Direct link to View results")

Let's look at our training and testing results. Log in to your Databricks Workspace, and click on the `Experiments` tab from the left menu. The initial page displays a list of runs, where we can see our run.

![runs page](https://imgur.com/2D2X5kK.png)

Now let's head to the models tab, where we can see the model that we logged

![models page](https://imgur.com/7si9VP8.png)

Clicking on the model name will take you to the model details page, with information on its parameters, metrics, and other metadata.

![model details page](https://imgur.com/mJhwhSr.png)

We can also inspect our model using the API

python

```
logged_model = mlflow.get_logged_model(model_info.model_id)

logged_model, logged_model.metrics, logged_model.params
```
