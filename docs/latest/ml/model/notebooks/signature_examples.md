# MLflow Signature Playground Notebook

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/model/notebooks/signature_examples.ipynb) Welcome to the MLflow Signature Playground! This interactive Jupyter notebook is designed to guide you through the foundational concepts of [Model Signatures](https://mlflow.org/docs/latest/ml/model/signatures.html) within the MLflow ecosystem. As you progress through the notebook, you'll gain practical experience with defining, enforcing, and utilizing model signatures—a critical aspect of model management that enhances reproducibility, reliability, and ease of use.

### Why Model Signatures Matter[​](#why-model-signatures-matter "Direct link to Why Model Signatures Matter")

In the realm of machine learning, defining the inputs and outputs of models with precision is key to ensuring smooth operations. Model signatures serve as the schema definition for the data your model expects and produces, acting as a blueprint for both model developers and users. This not only clarifies expectations but also facilitates automatic validation checks, streamlining the process from model training to deployment.

### Signature Enforcement in Action[​](#signature-enforcement-in-action "Direct link to Signature Enforcement in Action")

By exploring the code cells in this notebook, you'll witness firsthand how model signatures can enforce data integrity, prevent common errors, and provide descriptive feedback when discrepancies occur. This is invaluable for maintaining the quality and consistency of model inputs, especially when models are served in production environments.

### Practical Examples for a Deeper Understanding[​](#practical-examples-for-a-deeper-understanding "Direct link to Practical Examples for a Deeper Understanding")

The notebook includes a range of examples showcasing different data types and structures, from simple scalars to complex nested dictionaries. These examples demonstrate how signatures are inferred, logged, and updated, providing you with a comprehensive understanding of the signature lifecycle. As you interact with the provided PythonModel instances and invoke their predict methods, you'll learn how to handle various input scenarios—accounting for both required and optional data fields—and how to update existing models to include detailed signatures. Whether you're a data scientist looking to refine your model management practices or a developer integrating MLflow into your workflow, this notebook is your sandbox for mastering model signatures. Let's dive in and explore the robust capabilities of MLflow signatures!

> NOTE: Several of the features shown in this notebook are only available in version 2.10.0 and higher of MLflow. In particular, the support for the `Array` and `Object` types are not available prior to version 2.10.0.

python

```
import numpy as np
import pandas as pd

import mlflow
from mlflow.models.signature import infer_signature, set_signature


def report_signature_info(input_data, output_data=None, params=None):
  inferred_signature = infer_signature(input_data, output_data, params)

  report = f"""
The input data: 
	{input_data}.
The data is of type: {type(input_data)}.
The inferred signature is:

{inferred_signature}
"""
  print(report)
```

### Scalar Support in MLflow Signatures[​](#scalar-support-in-mlflow-signatures "Direct link to Scalar Support in MLflow Signatures")

In this segment of the tutorial, we explore the critical role of scalar data types in the context of MLflow's model signatures. Scalar types, such as strings, integers, floats, doubles, booleans, and datetimes, are fundamental to defining the schema for a model's input and output. Accurate representation of these types is essential for ensuring that models process data correctly, which directly impacts the reliability and accuracy of predictions.

By examining examples of various scalar types, this section demonstrates how MLflow infers and records the structure and nature of data. We'll see how MLflow signatures cater to different scalar types, ensuring that the data fed into the model matches the expected format. This understanding is crucial for any machine learning practitioner, as it helps in preparing and validating data inputs, leading to smoother model operations and more reliable results.

Through practical examples, including lists of strings, floats, and other types, we illustrate how MLflow's `infer_signature` function can accurately deduce the data format. This capability is a cornerstone in MLflow's ability to handle diverse data inputs and forms the basis for more complex data structures in machine learning models. By the end of this section, you'll have a clear grasp of how scalar data is represented within MLflow signatures and why this is important for your ML projects.

python

```
# List of strings

report_signature_info(["a", "list", "of", "strings"])
```

```

The input data: 
['a', 'list', 'of', 'strings'].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
[string (required)]
outputs: 
None
params: 
None
```

python

```
# List of floats

report_signature_info([np.float32(0.117), np.float32(1.99)])
```

```

The input data: 
[0.117, 1.99].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
[float (required)]
outputs: 
None
params: 
None
```

python

```
# Adding a column header to a list of doubles
my_data = pd.DataFrame({"input_data": [np.float64(0.117), np.float64(1.99)]})
report_signature_info(my_data)
```

```

The input data: 
   input_data
0       0.117
1       1.990.
The data is of type: <class 'pandas.core.frame.DataFrame'>.
The inferred signature is:

inputs: 
['input_data': double (required)]
outputs: 
None
params: 
None
```

python

```
# List of Dictionaries
report_signature_info([{"a": "a1", "b": "b1"}, {"a": "a2", "b": "b2"}])
```

```

The input data: 
[{'a': 'a1', 'b': 'b1'}, {'a': 'a2', 'b': 'b2'}].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
['a': string (required), 'b': string (required)]
outputs: 
None
params: 
None
```

python

```
# List of Arrays of strings
report_signature_info([["a", "b", "c"], ["d", "e", "f"]])
```

```

The input data: 
[['a', 'b', 'c'], ['d', 'e', 'f']].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
[Array(string) (required)]
outputs: 
None
params: 
None
```

python

```
# List of Arrays of Dictionaries
report_signature_info(
  [[{"a": "a", "b": "b"}, {"a": "a", "b": "b"}], [{"a": "a", "b": "b"}, {"a": "a", "b": "b"}]]
)
```

```

The input data: 
[[{'a': 'a', 'b': 'b'}, {'a': 'a', 'b': 'b'}], [{'a': 'a', 'b': 'b'}, {'a': 'a', 'b': 'b'}]].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
[Array({a: string (required), b: string (required)}) (required)]
outputs: 
None
params: 
None
```

### Understanding Type Conversion: Int to Long[​](#understanding-type-conversion-int-to-long "Direct link to Understanding Type Conversion: Int to Long")

In this section of the tutorial, we observe an interesting aspect of type conversion in MLflow's schema inference. When reporting the signature information for a list of integers, you might notice that the inferred data type is `long` instead of `int`. This conversion from int to long is not an error or bug but a valid and intentional type conversion within MLflow's schema inference mechanism.

#### Why Integers are Inferred as Long[​](#why-integers-are-inferred-as-long "Direct link to Why Integers are Inferred as Long")

* **Broader Compatibility:** The conversion to `long` ensures compatibility across various platforms and systems. Since the size of an integer (int) can vary depending on the system architecture, using `long` (which has a more consistent size specification) avoids potential discrepancies and data overflow issues.
* **Data Integrity:** By inferring integers as long, MLflow ensures that larger integer values, which might exceed the typical capacity of an int, are accurately represented and handled without data loss or overflow.
* **Consistency in Machine Learning Models:** In many machine learning frameworks, especially those involving larger datasets or computations, long integers are often the standard data type for numerical operations. This standardization in the inferred schema aligns with common practices in the machine learning community.

python

```
# List of integers
report_signature_info([1, 2, 3])
```

```

The input data: 
[1, 2, 3].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
[long (required)]
outputs: 
None
params: 
None
```

```
/Users/benjamin.wilson/repos/mlflow-fork/mlflow/mlflow/types/utils.py:378: UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values>`_ for more details.
warnings.warn(
```

python

```
# List of Booleans
report_signature_info([True, False, False, False, True])
```

```

The input data: 
[True, False, False, False, True].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
[boolean (required)]
outputs: 
None
params: 
None
```

python

```
# List of Datetimes
report_signature_info([np.datetime64("2023-12-24 11:59:59"), np.datetime64("2023-12-25 00:00:00")])
```

```

The input data: 
[numpy.datetime64('2023-12-24T11:59:59'), numpy.datetime64('2023-12-25T00:00:00')].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
[datetime (required)]
outputs: 
None
params: 
None
```

python

```
# Complex list of Dictionaries
report_signature_info([{"a": "b", "b": [1, 2, 3], "c": {"d": [4, 5, 6]}}])
```

```

The input data: 
[{'a': 'b', 'b': [1, 2, 3], 'c': {'d': [4, 5, 6]}}].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
['a': string (required), 'b': Array(long) (required), 'c': {d: Array(long) (required)} (required)]
outputs: 
None
params: 
None
```

python

```
# Pandas DF input

data = [
  {"a": "a", "b": ["a", "b", "c"], "c": {"d": 1, "e": 0.1}, "f": [{"g": "g"}, {"h": 1}]},
  {"b": ["a", "b"], "c": {"d": 2, "f": "f"}, "f": [{"g": "g"}]},
]
data = pd.DataFrame(data)

report_signature_info(data)
```

```

The input data: 
     a          b                   c                       f
0    a  [a, b, c]  {'d': 1, 'e': 0.1}  [{'g': 'g'}, {'h': 1}]
1  NaN     [a, b]  {'d': 2, 'f': 'f'}            [{'g': 'g'}].
The data is of type: <class 'pandas.core.frame.DataFrame'>.
The inferred signature is:

inputs: 
['a': string (optional), 'b': Array(string) (required), 'c': {d: long (required), e: double (optional), f: string (optional)} (required), 'f': Array({g: string (optional), h: long (optional)}) (required)]
outputs: 
None
params: 
None
```

### Signature Enforcement[​](#signature-enforcement "Direct link to Signature Enforcement")

In this part of the tutorial, we focus on the practical application of signature enforcement in MLflow. Signature enforcement is a powerful feature that ensures the data provided to a model aligns with the defined input schema. This step is crucial in preventing errors and inconsistencies that can arise from mismatched or incorrectly formatted data.

Through hands-on examples, we will observe how MLflow enforces the conformity of data to the expected signature at runtime. We'll use the `MyModel` class, a simple Python model, to demonstrate how MLflow checks the compatibility of input data against the model's signature. This process helps in safeguarding the model against incompatible or erroneous inputs, thereby enhancing the robustness and reliability of model predictions.

This section also highlights the importance of precise data representation in MLflow and the implications it has on model performance. By testing with different types of data, including those that do not conform to the expected schema, we will see how MLflow validates data and provides informative feedback. This aspect of signature enforcement is invaluable for debugging data issues and refining model inputs, making it a key skill for anyone involved in deploying machine learning models.

python

```
class MyModel(mlflow.pyfunc.PythonModel):
  def predict(self, context, model_input, params=None):
      return model_input
```

python

```
data = [{"a": ["a", "b", "c"], "b": "b", "c": {"d": "d"}}, {"a": ["a"], "c": {"d": "d", "e": "e"}}]

report_signature_info(data)
```

```

The input data: 
[{'a': ['a', 'b', 'c'], 'b': 'b', 'c': {'d': 'd'}}, {'a': ['a'], 'c': {'d': 'd', 'e': 'e'}}].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
['a': Array(string) (required), 'b': string (optional), 'c': {d: string (required), e: string (optional)} (required)]
outputs: 
None
params: 
None
```

python

```
# Generate a prediction that will serve as the model output example for signature inference
model_output = MyModel().predict(context=None, model_input=data)

with mlflow.start_run():
  model_info = mlflow.pyfunc.log_model(
      python_model=MyModel(),
      name="test_model",
      signature=infer_signature(model_input=data, model_output=model_output),
  )

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
prediction = loaded_model.predict(data)

prediction
```

```
/Users/benjamin.wilson/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils.
warnings.warn("Setuptools is replacing distutils.")
```

|   | a          | b   | c                    |
| - | ---------- | --- | -------------------- |
| 0 | \[a, b, c] | b   | {'d': 'd'}           |
| 1 | \[a]       | NaN | {'d': 'd', 'e': 'e'} |

We can check the inferred signature directly from the logged model information that is returned from the call to `log_model()`

python

```
model_info.signature
```

```
inputs: 
['a': Array(string) (required), 'b': string (optional), 'c': {d: string (required), e: string (optional)} (required)]
outputs: 
['a': Array(string) (required), 'b': string (optional), 'c': {d: string (required), e: string (optional)} (required)]
params: 
None
```

We can also quickly verify that the logged input signature matches the signature inference. While we're at it, we can generate the output signature as well.

> NOTE: it is recommended to log both the input and output signatures with your models.

python

```
report_signature_info(data, prediction)
```

```

The input data: 
[{'a': ['a', 'b', 'c'], 'b': 'b', 'c': {'d': 'd'}}, {'a': ['a'], 'c': {'d': 'd', 'e': 'e'}}].
The data is of type: <class 'list'>.
The inferred signature is:

inputs: 
['a': Array(string) (required), 'b': string (optional), 'c': {d: string (required), e: string (optional)} (required)]
outputs: 
['a': Array(string) (required), 'b': string (optional), 'c': {d: string (required), e: string (optional)} (required)]
params: 
None
```

python

```
# Using the model while not providing an optional input (note the output return structure and the non existent optional columns)

loaded_model.predict([{"a": ["a", "b", "c"], "c": {"d": "d"}}])
```

|   | a          | c          |
| - | ---------- | ---------- |
| 0 | \[a, b, c] | {'d': 'd'} |

python

```
# Using the model while omitting the input of required fields (this will raise an Exception from schema enforcement,
# stating that the required fields "a" and "c" are missing)

loaded_model.predict([{"b": "b"}])
```

```
---------------------------------------------------------------------------
```

```
MlflowException                           Traceback (most recent call last)
```

```
~/repos/mlflow-fork/mlflow/mlflow/pyfunc/__init__.py in predict(self, data, params)
  469             try:
--> 470                 data = _enforce_schema(data, input_schema)
  471             except Exception as e:
```

```
~/repos/mlflow-fork/mlflow/mlflow/models/utils.py in _enforce_schema(pf_input, input_schema)
  939                 message += f" Note that there were extra inputs: {extra_cols}"
--> 940             raise MlflowException(message)
  941     elif not input_schema.is_tensor_spec():
```

```
MlflowException: Model is missing inputs ['a', 'c'].
```

```

During handling of the above exception, another exception occurred:
```

```
MlflowException                           Traceback (most recent call last)
```

```
/var/folders/cd/n8n0rm2x53l_s0xv_j_xklb00000gp/T/ipykernel_97464/1628231496.py in <cell line: 4>()
    2 # stating that the required fields "a" and "c" are missing)
    3 
----> 4 loaded_model.predict([{"b": "b"}])
```

```
~/repos/mlflow-fork/mlflow/mlflow/pyfunc/__init__.py in predict(self, data, params)
  471             except Exception as e:
  472                 # Include error in message for backwards compatibility
--> 473                 raise MlflowException.invalid_parameter_value(
  474                     f"Failed to enforce schema of data '{data}' "
  475                     f"with schema '{input_schema}'. "
```

```
MlflowException: Failed to enforce schema of data '[{'b': 'b'}]' with schema '['a': Array(string) (required), 'b': string (optional), 'c': {d: string (required), e: string (optional)} (required)]'. Error: Model is missing inputs ['a', 'c'].
```

### Updating Signatures[​](#updating-signatures "Direct link to Updating Signatures")

This section of the tutorial addresses the dynamic nature of data and models, focusing on the crucial task of updating an MLflow model's signature. As datasets evolve and requirements change, it becomes necessary to modify the signature of a model to align with the new data structure or inputs. This ability to update a signature is key to maintaining the accuracy and relevance of your model over time.

We will demonstrate how to identify when a signature update is needed and walk through the process of creating and applying a new signature to an existing model. This section highlights the flexibility of MLflow in accommodating changes in data formats and structures without the need to re-save the entire model. However, for registered models in MLflow, updating the signature requires re-registering the model to reflect the changes in the registered version.

By exploring the steps to update a model's signature, you will learn how to update the model signature in the event that you manually defined a signature that is invalid or if you failed to define one while logging and need to update the model with a valid signature.

python

```
# Updating an existing model that wasn't saved with a signature


class MyTypeCheckerModel(mlflow.pyfunc.PythonModel):
  def predict(self, context, model_input, params=None):
      print(type(model_input))
      print(model_input)
      if not isinstance(model_input, (pd.DataFrame, list)):
          raise ValueError("The input must be a list.")
      return "Input is valid."


with mlflow.start_run():
  model_info = mlflow.pyfunc.log_model(
      python_model=MyTypeCheckerModel(),
      name="test_model",
  )

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

loaded_model.metadata.signature
```

python

```
test_data = [{"a": "we are expecting strings", "b": "and only strings"}, [1, 2, 3]]
loaded_model.predict(test_data)
```

```
<class 'list'>
[{'a': 'we are expecting strings', 'b': 'and only strings'}, [1, 2, 3]]
```

```
'Input is valid.'
```

### The Necessity of Schema Enforcement in MLflow[​](#the-necessity-of-schema-enforcement-in-mlflow "Direct link to The Necessity of Schema Enforcement in MLflow")

In this part of the tutorial, we address a common challenge in machine learning model deployment: the clarity and interpretability of error messages. Without schema enforcement, models can often return cryptic or misleading error messages. This occurs because, in the absence of a well-defined schema, the model attempts to process inputs that may not align with its expectations, leading to ambiguous or hard-to-diagnose errors.

#### Why Schema Enforcement Matters[​](#why-schema-enforcement-matters "Direct link to Why Schema Enforcement Matters")

Schema enforcement acts as a gatekeeper, ensuring that the data fed into a model precisely matches the expected format. This not only reduces the likelihood of runtime errors but also makes any errors that do occur much easier to understand and rectify. Without such enforcement, diagnosing issues becomes a time-consuming and complex task, often requiring deep dives into the model's internal logic.

#### Updating Model Signature for Clearer Error Messages[​](#updating-model-signature-for-clearer-error-messages "Direct link to Updating Model Signature for Clearer Error Messages")

To illustrate the value of schema enforcement, we will update the signature of a saved model to match an expected data structure. This process involves defining the expected data structure, using the `infer_signature` function to generate the appropriate signature, and then applying this signature to the model using `set_signature`. By doing so, we ensure that any future errors are more informative and aligned with the data structure we anticipate, simplifying troubleshooting and enhancing model reliability.

python

```
expected_data_structure = [{"a": "string", "b": "another string"}, {"a": "string"}]

signature = infer_signature(expected_data_structure, loaded_model.predict(expected_data_structure))

set_signature(model_info.model_uri, signature)
```

```
<class 'list'>
[{'a': 'string', 'b': 'another string'}, {'a': 'string'}]
```

python

```
loaded_with_signature = mlflow.pyfunc.load_model(model_info.model_uri)

loaded_with_signature.metadata.signature
```

```
inputs: 
['a': string (required), 'b': string (optional)]
outputs: 
[string (required)]
params: 
None
```

python

```
loaded_with_signature.predict(expected_data_structure)
```

```
<class 'pandas.core.frame.DataFrame'>
      a               b
0  string  another string
1  string             NaN
```

```
'Input is valid.'
```

#### Validating that schema enforcement will not permit a flawed input[​](#validating-that-schema-enforcement-will-not-permit-a-flawed-input "Direct link to Validating that schema enforcement will not permit a flawed input")

Now that we've set our signature correctly and updated the model definition, let's ensure that the previous flawed input type will raise a useful error message!

python

```
loaded_with_signature.predict(test_data)
```

```
---------------------------------------------------------------------------
```

```
AttributeError                            Traceback (most recent call last)
```

```
~/repos/mlflow-fork/mlflow/mlflow/pyfunc/__init__.py in predict(self, data, params)
  469             try:
--> 470                 data = _enforce_schema(data, input_schema)
  471             except Exception as e:
```

```
~/repos/mlflow-fork/mlflow/mlflow/models/utils.py in _enforce_schema(pf_input, input_schema)
  907         elif isinstance(pf_input, (list, np.ndarray, pd.Series)):
--> 908             pf_input = pd.DataFrame(pf_input)
  909 
```

```
~/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
  781                         columns = ensure_index(columns)
--> 782                     arrays, columns, index = nested_data_to_arrays(
  783                         # error: Argument 3 to "nested_data_to_arrays" has incompatible
```

```
~/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pandas/core/internals/construction.py in nested_data_to_arrays(data, columns, index, dtype)
  497 
--> 498     arrays, columns = to_arrays(data, columns, dtype=dtype)
  499     columns = ensure_index(columns)
```

```
~/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pandas/core/internals/construction.py in to_arrays(data, columns, dtype)
  831     elif isinstance(data[0], abc.Mapping):
--> 832         arr, columns = _list_of_dict_to_arrays(data, columns)
  833     elif isinstance(data[0], ABCSeries):
```

```
~/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pandas/core/internals/construction.py in _list_of_dict_to_arrays(data, columns)
  911         sort = not any(isinstance(d, dict) for d in data)
--> 912         pre_cols = lib.fast_unique_multiple_list_gen(gen, sort=sort)
  913         columns = ensure_index(pre_cols)
```

```
~/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pandas/_libs/lib.pyx in pandas._libs.lib.fast_unique_multiple_list_gen()
```

```
~/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pandas/core/internals/construction.py in <genexpr>(.0)
  909     if columns is None:
--> 910         gen = (list(x.keys()) for x in data)
  911         sort = not any(isinstance(d, dict) for d in data)
```

```
AttributeError: 'list' object has no attribute 'keys'
```

```

During handling of the above exception, another exception occurred:
```

```
MlflowException                           Traceback (most recent call last)
```

```
/var/folders/cd/n8n0rm2x53l_s0xv_j_xklb00000gp/T/ipykernel_97464/2586525788.py in <cell line: 1>()
----> 1 loaded_with_signature.predict(test_data)
```

```
~/repos/mlflow-fork/mlflow/mlflow/pyfunc/__init__.py in predict(self, data, params)
  471             except Exception as e:
  472                 # Include error in message for backwards compatibility
--> 473                 raise MlflowException.invalid_parameter_value(
  474                     f"Failed to enforce schema of data '{data}' "
  475                     f"with schema '{input_schema}'. "
```

```
MlflowException: Failed to enforce schema of data '[{'a': 'we are expecting strings', 'b': 'and only strings'}, [1, 2, 3]]' with schema '['a': string (required), 'b': string (optional)]'. Error: 'list' object has no attribute 'keys'
```

###

### Wrapping Up: Insights and Best Practices from the MLflow Signature Playground[​](#wrapping-up-insights-and-best-practices-from-the-mlflow-signature-playground "Direct link to Wrapping Up: Insights and Best Practices from the MLflow Signature Playground")

As we conclude our journey through the MLflow Signature Playground Notebook, we've gained invaluable insights into the intricacies of model signatures within the MLflow ecosystem. This tutorial has equipped you with the knowledge and practical skills needed to effectively manage and utilize model signatures, ensuring the robustness and accuracy of your machine learning models.

Key takeaways include the importance of accurately defining scalar types, the significance of enforcing and adhering to model signatures for data integrity, and the flexibility offered by MLflow in updating an invalid model signature. These concepts are not just theoretical but are fundamental to successful model deployment and management in real-world scenarios.

Whether you're a data scientist refining your models or a developer integrating machine learning into your applications, understanding and utilizing model signatures is crucial. We hope this tutorial has provided you with a solid foundation in MLflow signatures, empowering you to implement these best practices in your future ML projects.
