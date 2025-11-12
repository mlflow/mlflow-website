# Deploy an MLflow `PyFunc` model with Model Serving

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial.ipynb)

In this notebook, learn how to deploy a custom MLflow PyFunc model to a serving endpoint. MLflow pyfunc offers greater flexibility and customization to your deployment. You can run any custom model, add preprocessing or post-processing logic, or execute any arbitrary Python code. While using the MLflow built-in flavor is recommended for optimal performance, you can use MLflow PyFunc models where more customization is required.

## Install and import libraries[​](#install-and-import-libraries "Direct link to Install and import libraries")

python

```python
%pip install --upgrade mlflow scikit-learn -q

```

python

```python
import json
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor

import mlflow
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")

```

python

```python
DOW_MODEL_NAME_PREFIX = "DOW_model_"
MME_MODEL_NAME = "MME_DOW_model"

```

## 1 - Create Some Sample Models[​](#1---create-some-sample-models "Direct link to 1 - Create Some Sample Models")

#### 1.1 - Create Dummy Data[​](#11---create-dummy-data "Direct link to 1.1 - Create Dummy Data")

python

```python
def create_weekly_dataset(n_dates, n_observations_per_date):
  rng = pd.date_range(start="today", periods=n_dates, freq="D")
  df = pd.DataFrame(
      np.random.randn(n_dates * n_observations_per_date, 4),
      columns=["x1", "x2", "x3", "y"],
      index=np.tile(rng, n_observations_per_date),
  )
  df["dow"] = df.index.dayofweek
  return df


df = create_weekly_dataset(n_dates=30, n_observations_per_date=500)
print(df.shape)
df.head()

```

|                            | x1        | x2        | x3        | y         | dow |
| -------------------------- | --------- | --------- | --------- | --------- | --- |
| 2024-01-26 18:30:42.810981 | -1.137854 | 0.165915  | 0.711107  | 0.046467  | 4   |
| 2024-01-27 18:30:42.810981 | 0.475331  | -0.749121 | 0.318395  | 0.520535  | 5   |
| 2024-01-28 18:30:42.810981 | 2.525948  | 1.019708  | 0.038251  | -0.270675 | 6   |
| 2024-01-29 18:30:42.810981 | 1.113931  | 0.376434  | -1.464181 | -0.069208 | 0   |
| 2024-01-30 18:30:42.810981 | -0.304569 | 1.389245  | -1.152598 | -1.137589 | 1   |

#### 1.2 - Train Models for Each Day of the Week[​](#12---train-models-for-each-day-of-the-week "Direct link to 1.2 - Train Models for Each Day of the Week")

python

```python
for dow in df["dow"].unique():
  # Create dataset corresponding to a single day of the week
  X = df.loc[df["dow"] == dow]
  X.pop("dow")  # Remove DOW as a predictor column
  y = X.pop("y")

  # Fit our DOW model
  model = RandomForestRegressor().fit(X, y)

  # Infer signature of the model
  signature = infer_signature(X, model.predict(X))

  with mlflow.start_run():
      model_path = f"model_{dow}"

      # Log and register our DOW model with signature
      mlflow.sklearn.log_model(
          model,
          name=model_path,
          signature=signature,
          registered_model_name=f"{DOW_MODEL_NAME_PREFIX}{dow}",
      )
      mlflow.set_tag("dow", dow)

```

#### 1.3 - Test inference on our DOW models[​](#13---test-inference-on-our-dow-models "Direct link to 1.3 - Test inference on our DOW models")

python

```python
# Load Tuesday's model
tuesday_dow = 1
model_name = f"{DOW_MODEL_NAME_PREFIX}{tuesday_dow}"
model_uri = f"models:/{model_name}/latest"
model = mlflow.sklearn.load_model(model_uri)

# Perform inference using our training data for Tuesday
predictor_columns = [column for column in df.columns if column not in {"y", "dow"}]
head_of_training_data = df.loc[df["dow"] == tuesday_dow, predictor_columns].head()
tuesday_fitted_values = model.predict(head_of_training_data)
print(tuesday_fitted_values)

```

## 2 - Create an MME Custom PyFunc Model[​](#2---create-an-mme-custom-pyfunc-model "Direct link to 2 - Create an MME Custom PyFunc Model")

#### 2.1 - Create a Child Implementation of `mlflow.pyfunc.PythonModel`[​](#21---create-a-child-implementation-of-mlflowpyfuncpythonmodel "Direct link to 21---create-a-child-implementation-of-mlflowpyfuncpythonmodel")

python

```python
class DOWModel(mlflow.pyfunc.PythonModel):
  def __init__(self, model_uris):
      self.model_uris = model_uris
      self.models = {}

  @staticmethod
  def _model_uri_to_dow(model_uri: str) -> int:
      return int(model_uri.split("/")[-2].split("_")[-1])

  def load_context(self, context):
      self.models = {
          self._model_uri_to_dow(model_uri): mlflow.sklearn.load_model(model_uri)
          for model_uri in self.model_uris
      }

  def predict(self, context, model_input, params):
      # Parse the dow parameter
      dow = params.get("dow")
      if dow is None:
          raise ValueError("DOW param is not passed.")

      # Get the model associated with the dow parameter
      model = self.models.get(dow)
      if model is None:
          raise ValueError(f"Model {dow} version was not found: {self.models.keys()}.")

      # Perform inference
      return model.predict(model_input)

```

#### 2.2 - Test our Implementation[​](#22---test-our-implementation "Direct link to 2.2 - Test our Implementation")

python

```python
head_of_training_data

```

|                            | x1        | x2        | x3        |
| -------------------------- | --------- | --------- | --------- |
| 2024-01-30 18:30:42.810981 | -0.304569 | 1.389245  | -1.152598 |
| 2024-02-06 18:30:42.810981 | 0.521323  | 0.814452  | 0.115571  |
| 2024-02-13 18:30:42.810981 | 0.229761  | -1.936210 | 0.139201  |
| 2024-02-20 18:30:42.810981 | -0.865488 | 1.024857  | -0.857649 |
| 2024-01-30 18:30:42.810981 | -1.454631 | 0.462055  | 0.703858  |

python

```python
# Instantiate our DOW MME
model_uris = [f"models:/{DOW_MODEL_NAME_PREFIX}{i}/latest" for i in df["dow"].unique()]
dow_model = DOWModel(model_uris)
dow_model.load_context(None)
print("Model URIs:")
print(model_uris)

# Perform inference using our training data for Tuesday
params = {"dow": 1}
mme_tuesday_fitted_values = dow_model.predict(None, head_of_training_data, params=params)
assert all(tuesday_fitted_values == mme_tuesday_fitted_values)

print("
Tuesday fitted values:")
print(mme_tuesday_fitted_values)

```

#### 2.3 - Register our Custom PyFunc Model[​](#23---register-our-custom-pyfunc-model "Direct link to 2.3 - Register our Custom PyFunc Model")

python

```python
with mlflow.start_run():
  # Instantiate the custom pyfunc model
  model = DOWModel(model_uris)
  model.load_context(None)
  model_path = "MME_model_path"

  signature = infer_signature(
      model_input=head_of_training_data,
      model_output=tuesday_fitted_values,
      params=params,
  )
  print(signature)

  # Log the model to the experiment
  mlflow.pyfunc.log_model(
      name=model_path,
      python_model=model,
      signature=signature,
      pip_requirements=["scikit-learn=1.3.2"],
      registered_model_name=MME_MODEL_NAME,  # also register the model for easy access
  )

  # Set some relevant information about our model
  # (Assuming model has a property 'models' that can be counted)
  mlflow.log_param("num_models", len(model.models))

```

## 3 - Serve our Model[​](#3---serve-our-model "Direct link to 3 - Serve our Model")

To test our endpoint, let's serve our model on our local machine.

1. Open a new shell window in the root containing `mlruns` directory e.g. the same directory you ran this notebook.
2. Ensure mlflow is installed: `pip install --upgrade mlflow scikit-learn`
3. Run the bash command printed below.

python

```python
PORT = 1234
print(
  f"""Run the below command in a new window. You must be in the same repo as your mlruns directory and have mlflow installed...
  mlflow models serve -m "models:/{MME_MODEL_NAME}/latest" --env-manager local -p {PORT}"""
)

```

## 4 - Query our Served Model[​](#4---query-our-served-model "Direct link to 4 - Query our Served Model")

python

```python
def score_model(pdf, params):
  headers = {"Content-Type": "application/json"}
  url = f"http://127.0.0.1:{PORT}/invocations"
  ds_dict = {"dataframe_split": pdf, "params": params}
  data_json = json.dumps(ds_dict, allow_nan=True)

  response = requests.request(method="POST", headers=headers, url=url, data=data_json)
  response.raise_for_status()

  return response.json()


print("Inference on dow model 1 (Tuesday):")
inference_df = head_of_training_data.reset_index(drop=True).to_dict(orient="split")
print(score_model(inference_df, params={"dow": 1}))

```
