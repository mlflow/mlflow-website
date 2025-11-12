# DSPy Quickstart

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/dspy/notebooks/dspy_quickstart.ipynb)

[DSPy](https://dspy-docs.vercel.app/) simplifies building language model (LM) pipelines by replacing manual prompt engineering with structured "text transformation graphs." These graphs use flexible, learning modules that automate and optimize LM tasks like reasoning, retrieval, and answering complex questions.

## How does it work?[​](#how-does-it-work "Direct link to How does it work?")

At a high level, DSPy optimizes prompts, selects the best language model, and can even fine-tune the model using training data.

The process follows these three steps, common to most DSPy [optimizers](https://dspy.ai/learn/optimization/optimizers/):

1. **Candidate Generation**: DSPy finds all `Predict` modules in the program and generates variations of instructions and demonstrations (e.g., examples for prompts). This step creates a set of possible candidates for the next stage.
2. **Parameter Optimization**: DSPy then uses methods like random search, TPE, or Optuna to select the best candidate. Fine-tuning models can also be done at this stage.

## This Demo[​](#this-demo "Direct link to This Demo")

Below we create a simple program that demonstrates the power of DSPy. We will build a text classifier leveraging OpenAI. By the end of this tutorial, we will...

1. Define a [dspy.Signature](https://dspy.ai/learn/programming/signatures/) and [dspy.Module](https://dspy.ai/learn/programming/modules/) to perform text classification.
2. Leverage [dspy.SIMBA](https://dspy.ai/api/optimizers/SIMBA/) to compile our module so it's better at classifying our text.
3. Analyze internal steps with MLflow Tracing.
4. Log the compiled model with MLflow.
5. Load the logged model and perform inference.

python

```python
%pip install -U datasets openai "dspy>=3.0.3" "mlflow>=3.4.0"

```

## Setup[​](#setup "Direct link to Setup")

### Set Up LLM[​](#set-up-llm "Direct link to Set Up LLM")

After installing the relevant dependencies, let's set up access to an OpenAI LLM. Here, will leverage OpenAI's `gpt-4o-mini` model.

python

```python
# Set OpenAI API Key to the environment variable. You can also pass the token to dspy.LM()
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI Key:")

```

python

```python
import dspy

# Define your model. We will use OpenAI for simplicity
model_name = "gpt-4o-mini"

# Note that an OPENAI_API_KEY environment must be present. You can also pass the token to dspy.LM()
lm = dspy.LM(
  model=f"openai/{model_name}",
  max_tokens=500,
  temperature=0.1,
)
dspy.settings.configure(lm=lm)

```

### Create MLflow Experiment[​](#create-mlflow-experiment "Direct link to Create MLflow Experiment")

Create a new MLflow Experiment to track your DSPy models, metrics, parameters, and traces in one place. Although there is already a "default" experiment created in your workspace, it is highly recommended to create one for different tasks to organize experiment artifacts.

python

```python
import mlflow

mlflow.set_experiment("DSPy Quickstart")

```

### Turn on Auto Tracing with MLflow[​](#turn-on-auto-tracing-with-mlflow "Direct link to Turn on Auto Tracing with MLflow")

[MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html) is a powerful observability tool for monitoring and debugging what happens inside your DSPy modules, helping you identify potential bottlenecks or issues quickly. To enable DSPy tracing, you just need to call `mlflow.dspy.autolog` and that's it!

python

```python
mlflow.dspy.autolog()

```

### Set Up Data[​](#set-up-data "Direct link to Set Up Data")

Next, we will download the [Reuters 21578](https://huggingface.co/datasets/yangwang825/reuters-21578) dataset from Huggingface. We also write a utility to ensure that our train/test split has the same labels.

python

```python
import numpy as np
import pandas as pd
from datasets import load_dataset
from dspy.datasets.dataset import Dataset


def read_data_and_subset_to_categories() -> tuple[pd.DataFrame]:
  """
  Read the reuters-21578 dataset. Docs can be found in the url below:
  https://huggingface.co/datasets/yangwang825/reuters-21578
  """

  # Read train/test split
  dataset = load_dataset("yangwang825/reuters-21578")
  train = pd.DataFrame(dataset["train"])
  test = pd.DataFrame(dataset["test"])

  # Clean the labels
  label_map = {
      0: "acq",
      1: "crude",
      2: "earn",
      3: "grain",
      4: "interest",
      5: "money-fx",
      6: "ship",
      7: "trade",
  }

  train["label"] = train["label"].map(label_map)
  test["label"] = test["label"].map(label_map)

  return train, test


class CSVDataset(Dataset):
  def __init__(
      self, n_train_per_label: int = 20, n_test_per_label: int = 10, *args, **kwargs
  ) -> None:
      super().__init__(*args, **kwargs)
      self.n_train_per_label = n_train_per_label
      self.n_test_per_label = n_test_per_label

      self._create_train_test_split_and_ensure_labels()

  def _create_train_test_split_and_ensure_labels(self) -> None:
      """Perform a train/test split that ensure labels in `dev` are also in `train`."""
      # Read the data
      train_df, test_df = read_data_and_subset_to_categories()

      # Sample for each label
      train_samples_df = pd.concat(
          [group.sample(n=self.n_train_per_label) for _, group in train_df.groupby("label")]
      )
      test_samples_df = pd.concat(
          [group.sample(n=self.n_test_per_label) for _, group in test_df.groupby("label")]
      )

      # Set DSPy class variables
      self._train = train_samples_df.to_dict(orient="records")
      self._dev = test_samples_df.to_dict(orient="records")


# Limit to a small dataset to showcase the value of bootstrapping
dataset = CSVDataset(n_train_per_label=3, n_test_per_label=1)

# Create train and test sets containing DSPy
# Note that we must specify the expected input value name
train_dataset = [example.with_inputs("text") for example in dataset.train]
test_dataset = [example.with_inputs("text") for example in dataset.dev]
unique_train_labels = {example.label for example in dataset.train}

print(len(train_dataset), len(test_dataset))
print(f"Train labels: {unique_train_labels}")
print(train_dataset[0])

```

### Set up DSPy Signature and Module[​](#set-up-dspy-signature-and-module "Direct link to Set up DSPy Signature and Module")

Finally, we will define our task: text classification.

There are a variety of ways you can provide guidelines to DSPy signature behavior. Currently, DSPy allows users to specify:

1. A high-level goal via the class docstring.
2. A set of input fields, with optional metadata.
3. A set of output fields with optional metadata.

DSPy will then leverage this information to inform optimization.

In the below example, note that we simply provide the expected labels to `output` field in the `TextClassificationSignature` class. From this initial state, we'll look to use DSPy to learn to improve our classifier accuracy.

python

```python
class TextClassificationSignature(dspy.Signature):
  text = dspy.InputField()
  label = dspy.OutputField(
      desc=f"Label of predicted class. Possible labels are {unique_train_labels}"
  )


class TextClassifier(dspy.Module):
  def __init__(self):
      super().__init__()
      self.generate_classification = dspy.Predict(TextClassificationSignature)

  def forward(self, text: str):
      return self.generate_classification(text=text)

```

## Run it\![​](#run-it "Direct link to Run it!")

### Hello World[​](#hello-world "Direct link to Hello World")

Let's demonstrate predicting via the DSPy module and associated signature. The program has correctly learned our labels from the signature `desc` field and generates reasonable predictions.

python

```python
# Initilize our impact_improvement class
text_classifier = TextClassifier()

message = "I am interested in space"
print(text_classifier(text=message))

message = "I enjoy ice skating"
print(text_classifier(text=message))

```

### Review Traces[​](#review-traces "Direct link to Review Traces")

1. Open the MLflow UI and select the `"DSPy Quickstart"` experiment.
2. Go to the `"Traces"` tab to view the generated traces.

Now, you can observe how DSPy translates your query and interacts with the LLM. This feature is extremely valuable for debugging, iteratively refining components within your system, and monitoring models in production. While the module in this tutorial is relatively simple, the tracing feature becomes even more powerful as your model grows in complexity.

![MLflow DSPy Trace](/mlflow-website/docs/latest/assets/images/dspy-trace-bd339ce15bda9cbb5f88a48a24c2bbf4.png)

## Compilation[​](#compilation "Direct link to Compilation")

### Training[​](#training "Direct link to Training")

To train, we will leverage [SIMBA](https://dspy.ai/api/optimizers/SIMBA/), an optimizer that will take bootstrap samples from our training set and leverage a random search strategy to optimize our predictive accuracy.

Note that in the below example, we leverage a simple metric definition of exact match, as defined in `validate_classification`, but [dspy.Metrics](https://dspy.ai/learn/evaluation/metrics/) can contain complex and LM-based logic to properly evaluate our accuracy.

python

```python
from dspy import SIMBA


def validate_classification(example, prediction, trace=None) -> bool:
  return example.label == prediction.label


optimizer = SIMBA(
  metric=validate_classification,
  max_demos=2,
  bsize=12,
  num_threads=1,
)

compiled_pe = optimizer.compile(TextClassifier(), trainset=train_dataset)

```

### Compare Pre/Post Compiled Accuracy[​](#compare-prepost-compiled-accuracy "Direct link to Compare Pre/Post Compiled Accuracy")

Finally, let's explore how well our trained model can predict on unseen test data.

python

```python
def check_accuracy(classifier, test_data: pd.DataFrame = test_dataset) -> float:
  residuals = []
  predictions = []
  for example in test_data:
      prediction = classifier(text=example["text"])
      residuals.append(int(validate_classification(example, prediction)))
      predictions.append(prediction)
  return residuals, predictions


uncompiled_residuals, uncompiled_predictions = check_accuracy(TextClassifier())
print(f"Uncompiled accuracy: {np.mean(uncompiled_residuals)}")

compiled_residuals, compiled_predictions = check_accuracy(compiled_pe)
print(f"Compiled accuracy: {np.mean(compiled_residuals)}")

```

As shown above, our compiled accuracy is non-zero - our base LLM inferred meaning of the classification labels simply via our initial prompt. However, with DSPy training, the prompts, demonstrations, and input/output signatures have been updated to give our model to 100% accuracy on unseen data. That's a gain of 12 percentage points!

Let's take a look at each prediction in our test set.

python

```python
for uncompiled_residual, uncompiled_prediction in zip(uncompiled_residuals, uncompiled_predictions):
  is_correct = "Correct" if bool(uncompiled_residual) else "Incorrect"
  prediction = uncompiled_prediction.label
  print(f"{is_correct} prediction: {' ' * (12 - len(is_correct))}{prediction}")

```

python

```python
for compiled_residual, compiled_prediction in zip(compiled_residuals, compiled_predictions):
  is_correct = "Correct" if bool(compiled_residual) else "Incorrect"
  prediction = compiled_prediction.label
  print(f"{is_correct} prediction: {' ' * (12 - len(is_correct))}{prediction}")

```

## Log and Load the Model with MLflow[​](#log-and-load-the-model-with-mlflow "Direct link to Log and Load the Model with MLflow")

Now that we have a compiled model with higher classification accuracy, let's leverage MLflow to log this model and load it for inference.

python

```python
import mlflow

with mlflow.start_run():
  model_info = mlflow.dspy.log_model(
      compiled_pe,
      name="model",
      input_example="what is 2 + 2?",
  )

```

Open the MLflow UI again and check the complied model is recorded to a new MLflow Run. Now you can load the model back for inference using `mlflow.dspy.load_model` or `mlflow.pyfunc.load_model`.

💡 MLflow will remember the environment configuration stored in `dspy.settings`, such as the language model (LM) used during the experiment. This ensures excellent reproducibility for your experiment.

python

```python
# Define input text
print("
==============Input Text============")
text = test_dataset[0]["text"]
print(f"Text: {text}")

# Inference with original DSPy object
print("
--------------Original DSPy Prediction------------")
print(compiled_pe(text=text).label)

# Inference with loaded DSPy object
print("
--------------Loaded DSPy Prediction------------")
loaded_model_dspy = mlflow.dspy.load_model(model_info.model_uri)
print(loaded_model_dspy(text=text).label)

# Inference with MLflow PyFunc API
loaded_model_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
print("
--------------PyFunc Prediction------------")
print(loaded_model_pyfunc.predict(text)["label"])

```

## Next Steps[​](#next-steps "Direct link to Next Steps")

This example demonstrates how DSPy works. Below are some potential extensions for improving this project, both with DSPy and MLflow.

### DSPy[​](#dspy "Direct link to DSPy")

* Use real-world data for the classifier.
* Experiment with different optimizers.
* For more in-depth examples, check out the [tutorials](https://dspy.ai/tutorials/) and [documentation](https://dspy.ai/learn/).

### MLflow[​](#mlflow "Direct link to MLflow")

* Deploy the model using MLflow serving.
* Use MLflow to experiment with various optimization strategies.
* Track your DSPy experiments using [DSPy Optimizer Autologging](https://mlflow.org/docs/latest/genai/flavors/dspy/optimizer/).

Happy coding!
