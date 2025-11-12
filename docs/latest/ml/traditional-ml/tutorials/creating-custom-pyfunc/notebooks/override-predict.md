# Customizing a Model's predict method

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict.ipynb)

In this tutorial, we will explore the process of customizing the predict method of a model in the context of MLflow's PyFunc flavor. This is particularly useful when you want to have more flexibility in how your model behaves after you've deployed it using MLflow.

To illustrate this, we'll use the famous Iris dataset and build a basic Logistic Regression model with scikit-learn.

python

```python
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

```

#### Configure the tracking server uri[​](#configure-the-tracking-server-uri "Direct link to Configure the tracking server uri")

This step is important to ensure that all of the calls to MLflow that we're going to be doing within this notebook will actually be logged to our tracking server that is running locally.

If you are following along with this notebook in a different environment and wish to execute the remainder of this notebook to a remote tracking server, change the following cell.

Databricks: `mlflow.set_tracking_uri("databricks")`

Your hosted MLflow: `mlflow.set_tracking_uri("http://my.company.mlflow.tracking.server:<port>)`

Your local tracking server As in the introductory tutorial, we can start a local tracking server via command line as follows:

bash

```bash
mlflow server --host 127.0.0.1 --port 8080

```

And the MLflow UI server can be started locally via:

bash

```bash
mlflow ui --host 127.0.0.1 --port 8090

```

python

```python
mlflow.set_tracking_uri("http://localhost:8080")

```

Let's begin by loading the Iris dataset and splitting it into training and testing sets. We'll then train a simple Logistic Regression model on the training data.

python

```python
iris = load_iris()
x = iris.data[:, 2:]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9001)

model = LogisticRegression(random_state=0, max_iter=5_000, solver="newton-cg").fit(x_train, y_train)

```

This is a common scenario in machine learning. We have a trained model, and we want to use it to make predictions. With scikit-learn, the model provides a few methods to do this:

* `predict` - to predict class labels
* `predict_proba` - to get class membership probabilities
* `predict_log_proba` - to get logarithmic probabilities for each class

We can predict the class labels, as shown below.

python

```python
model.predict(x_test)[:5]

```

We can also get the class membership probability.

python

```python
model.predict_proba(x_test)[:5]

```

As well as generate logarithmic probabilites for each class.

python

```python
model.predict_log_proba(x_test)[:5]

```

While using the model directly within the same Python session is straightforward, what happens when we want to save this model and load it elsewhere, especially when using MLflow's PyFunc flavor? Let's explore this scenario.

python

```python
mlflow.set_experiment("Overriding Predict Tutorial")

sklearn_path = "/tmp/sklearn_model"

with mlflow.start_run() as run:
  mlflow.sklearn.save_model(
      sk_model=model,
      path=sklearn_path,
      input_example=x_train[:2],
  )

```

Once the model is loaded as a pyfunc, the default behavior only supports the predict method. This is evident when you try to call other methods like predict\_proba, leading to an AttributeError. This can be limiting, especially when you want to preserve the full capability of the original model.

python

```python
loaded_logreg_model = mlflow.pyfunc.load_model(sklearn_path)

```

python

```python
loaded_logreg_model.predict(x_test)

```

This works precisely as we expect. The output is the same as the model direct usage prior to saving.

Let's try to use the predict\_proba method.

We're not actually going to run this, as it will raise an Exception. Here is the behavior if we try to execute this:

python

```python
loaded_logreg_model.predict_proba(x_text)

```

Which will result in this error:

shell

```shell
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/var/folders/cd/n8n0rm2x53l_s0xv_j_xklb00000gp/T/ipykernel_15410/1677830262.py in <cell line: 1>()
----> 1 loaded_logreg_model.predict_proba(x_text)

AttributeError: 'PyFuncModel' object has no attribute 'predict_proba'

```

### What can we do to support the original behavior of the model when deployed?[​](#what-can-we-do-to-support-the-original-behavior-of-the-model-when-deployed "Direct link to What can we do to support the original behavior of the model when deployed?")

We can create a custom pyfunc that overrides the behavior of the `predict` method.

For the example below, we're going to be showing two features of pyfunc that can be leveraged to handle custom model logging capabilities:

* override of the predict method
* custom loading of an artifact

A key thing to note is the use of joblib for serialization. While pickle has been historically used for serializing scikit-learn models, joblib is now recommended as it provides better performance and support, especially for large numpy arrays.

We'll be using `joblib` and it's `dump` and `load` APIs to handle loading of our model object into our custom pyfunc implementation. This process of using the load\_context method to handle loading files when instantiating the pyfunc object is particularly useful for models that have very large or numerous artifact dependencies (such as LLMs) and can help to dramatically lessen the total memory footprint of a pyfunc that is being loaded in a distributed system (such as Apache Spark or Ray).

python

```python
from joblib import dump

from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

```

To see how we can leverage the `load_context` functionality within a custom Python Model, we'll first serialize our model locally using `joblib`. The usage of `joblib` here is purely to demonstrate a non-standard method (one that is not natively supported in MLflow) to illustrate the flexibility of the Python Model implementation. Provided that we import this library within the `load_context` and have it available in the environment where we will be loading this model, the model artifact will be deserialized properly.

python

```python
model_directory = "/tmp/sklearn_model.joblib"
dump(model, model_directory)

```

#### Defining our custom `PythonModel`[​](#defining-our-custom-pythonmodel "Direct link to defining-our-custom-pythonmodel")

The `ModelWrapper` class below is an example of a custom `pyfunc` that extends MLflow's `PythonModel`. It provides flexibility in the prediction method by using the `params` argument of the `predict method`. This way, we can specify if we want the regular `predict`, `predict_proba`, or `predict_log_proba` behavior when we call the `predict` method on the loaded `pyfunc` instance.

python

```python
class ModelWrapper(PythonModel):
  def __init__(self):
      self.model = None

  def load_context(self, context):
      from joblib import load

      self.model = load(context.artifacts["model_path"])

  def predict(self, context, model_input, params=None):
      params = params or {"predict_method": "predict"}
      predict_method = params.get("predict_method")

      if predict_method == "predict":
          return self.model.predict(model_input)
      elif predict_method == "predict_proba":
          return self.model.predict_proba(model_input)
      elif predict_method == "predict_log_proba":
          return self.model.predict_log_proba(model_input)
      else:
          raise ValueError(f"The prediction method '{predict_method}' is not supported.")

```

After defining the custom `pyfunc`, the next steps involve saving the model with MLflow and then loading it back. The loaded model will retain the flexibility we built into the custom `pyfunc`, allowing us to choose the prediction method dynamically.

**NOTE**: The `artifacts` reference below is incredibly important. In order for the `load_context` to have access to the path that we are specifying as the location of our saved model, this must be provided as a dictionary that maps the appropriate access key to the relevant value. Failing to provide this dictionary as part of the `mlflow.save_model()` or `mlflow.log_model()` will render this custom `pyfunc` model unable to be properly loaded.

python

```python
# Define the required artifacts associated with the saved custom pyfunc
artifacts = {"model_path": model_directory}

# Define the signature associated with the model
signature = infer_signature(x_train, params={"predict_method": "predict_proba"})

```

We can see how the defined params are used within the signature definition. As is shown below, the params receive a slight alteration when logged. We have a param key that is defined (`predict_method`), and expected type (`string`), and a default value. What this ends up meaning for this `params` definition is:

* We can only provide a `params` override for the key `predict_method`. Anything apart from this will be ignored and a warning will be shown indicating that the unknown parameter will not be passed to the underlying model.

* The value associated with `predict_method` must be a string. Any other type will not be permitted and will raise an Exception for an unexpected type.

* If no value for the `predict_method` is provided when calling `predict`, the default value of `predict_proba` will be used by the model.

python

```python
signature

```

We can now save our custom model. We're providing a path to save it to, as well as the `artifacts` definition that contains the location of the manually serialized instance that we stored via `joblib`. Also included is the `signature`, which is a **key component** to making this example work; without the paramater defined within the signature, we wouldn't be able to override the method of prediction that the `predict` method will use.

**Note** that we're overriding the `pip_requirements` here to ensure that we specify the requirements for our two dependent libraries: `joblib` and `sklearn`. This helps to ensure that whatever environment that we deploy this model to will pre-load both of these dependencies prior to loading this saved model.

python

```python
pyfunc_path = "/tmp/dynamic_regressor"

with mlflow.start_run() as run:
  mlflow.pyfunc.save_model(
      path=pyfunc_path,
      python_model=ModelWrapper(),
      input_example=x_train,
      signature=signature,
      artifacts=artifacts,
      pip_requirements=["joblib", "sklearn"],
  )

```

We can now load our model back by using the `mlflow.pyfunc.load_model` API.

python

```python
loaded_dynamic = mlflow.pyfunc.load_model(pyfunc_path)

```

Let's see what the pyfunc model will produce with no overrides to the `params` argument.

python

```python
loaded_dynamic.predict(x_test)

```

As expected, it returned the default value of `params` `predict_method`, that of `predict_proba`. We can now attempt to override that functionality to return the class predictions.

python

```python
loaded_dynamic.predict(x_test, params={"predict_method": "predict"})

```

We can also override it to return the `predict_log_proba` logarithmic probailities of class membership.

python

```python
loaded_dynamic.predict(x_test, params={"predict_method": "predict_log_proba"})

```

We've successfully created a pyfunc model that retains the full capabilities of the original scikit-learn model, while simultaneously using a custom loader methodology that eschews the standard pickle methodology.

This tutorial highlights the power and flexibility of MLflow's PyFunc flavor, demonstrating how you can tailor it to fit your specific needs. As you continue building and deploying models, consider how custom pyfuncs can be used to enhance your model's capabilities and adapt to various scenarios.
