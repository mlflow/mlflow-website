# Creating a Custom Model: "Add N" Model

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction.ipynb) Our first example is simple yet illustrative. We'll create a model that adds a specified numeric value, n, to all columns of a Pandas DataFrame input. This will demonstrate the process of defining a custom model, saving it, loading it back, and performing predictions.

#### Step 1: Define the Model Class[​](#step-1-define-the-model-class "Direct link to Step 1: Define the Model Class")

We begin by defining a Python class for our model. This class should inherit from mlflow\.pyfunc.PythonModel and implement the necessary methods.

python

```
import mlflow.pyfunc


class AddN(mlflow.pyfunc.PythonModel):
  """
  A custom model that adds a specified value `n` to all columns of the input DataFrame.

  Attributes:
  -----------
  n : int
      The value to add to input columns.
  """

  def __init__(self, n):
      """
      Constructor method. Initializes the model with the specified value `n`.

      Parameters:
      -----------
      n : int
          The value to add to input columns.
      """
      self.n = n

  def predict(self, context, model_input, params=None):
      """
      Prediction method for the custom model.

      Parameters:
      -----------
      context : Any
          Ignored in this example. It's a placeholder for additional data or utility methods.

      model_input : pd.DataFrame
          The input DataFrame to which `n` should be added.

      params : dict, optional
          Additional prediction parameters. Ignored in this example.

      Returns:
      --------
      pd.DataFrame
          The input DataFrame with `n` added to all columns.
      """
      return model_input.apply(lambda column: column + self.n)
```

#### Step 2: Save the Model[​](#step-2-save-the-model "Direct link to Step 2: Save the Model")

Now that our model class is defined, we can instantiate it and save it using MLflow.

python

```
# Define the path to save the model
model_path = "/tmp/add_n_model"

# Create an instance of the model with `n=5`
add5_model = AddN(n=5)

# Save the model using MLflow
mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)
```

```
/Users/benjamin.wilson/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils.
warnings.warn("Setuptools is replacing distutils.")
```

#### Step 3: Load the Model[​](#step-3-load-the-model "Direct link to Step 3: Load the Model")

With our model saved, we can load it back using MLflow and then use it for predictions.

python

```
# Load the saved model
loaded_model = mlflow.pyfunc.load_model(model_path)
```

#### Step 4: Evaluate the Model[​](#step-4-evaluate-the-model "Direct link to Step 4: Evaluate the Model")

Let's now use our loaded model to perform predictions on a sample input and verify its correctness.

python

```
import pandas as pd

# Define a sample input DataFrame
model_input = pd.DataFrame([range(10)])

# Use the loaded model to make predictions
model_output = loaded_model.predict(model_input)
```

python

```
model_output
```

|   | 0 | 1 | 2 | 3 | 4 | 5  | 6  | 7  | 8  | 9  |
| - | - | - | - | - | - | -- | -- | -- | -- | -- |
| 0 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |

#### Conclusion[​](#conclusion "Direct link to Conclusion")

This simple example demonstrates the power and flexibility of MLflow's custom pyfunc. By encapsulating arbitrary Python code and its dependencies, custom pyfunc models ensure a consistent and unified interface for a wide range of use cases. Whether you're working with a niche machine learning framework, need custom preprocessing steps, or want to integrate unique prediction logic, pyfunc is the tool for the job.
