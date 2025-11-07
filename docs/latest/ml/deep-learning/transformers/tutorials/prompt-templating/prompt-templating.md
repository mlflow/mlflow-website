# Prompt Templating with MLflow and Transformers

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/transformers/tutorials/prompt-templating/prompt-templating.ipynb)

Welcome to our in-depth tutorial on using prompt templates to conveniently customize the behavior of Transformers pipelines using MLflow.

### Learning Objectives[​](#learning-objectives "Direct link to Learning Objectives")

In this tutorial, you will:

* Set up a text generation pipeline using TinyLlama-1.1B as an example model
* Set a prompt template that will be used to format user queries at inference time
* Load the model for querying

### What is a prompt template, and why use one?[​](#what-is-a-prompt-template-and-why-use-one "Direct link to What is a prompt template, and why use one?")

When dealing with large language models, the way a query is structured can significantly impact the model's performance. We often need to add some preamble, or format the query in a way that gives us the results that we want. It's not ideal to expect the end-user of our applications to know exactly what this format should be, so we typically have a pre-processing step to format the user input in a way that works best with the underlying model. In other words, we apply a prompt template to the user's input.

MLflow provides a convenient way to set this on certain pipeline types using the `transformers` flavor. As of now, the only pipelines that we support are:

* [feature-extraction](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.FeatureExtractionPipeline)
* [fill-mask](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.FillMaskPipeline)
* [summarization](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.SummarizationPipeline)
* [text2text-generation](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.Text2TextGenerationPipeline)
* [text-generation](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TextGenerationPipeline)

If you need a runthrough of the basics of how to use the `transformers` flavor, check out the [Introductory Guide](https://mlflow.org/docs/latest/ml/deep-learning/transformers/guide/index.html)!

Now, let's dive in and see how it's done!

python

```
# Disable tokenizers warnings when constructing pipelines
%env TOKENIZERS_PARALLELISM=false

import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)
```

```
env: TOKENIZERS_PARALLELISM=false
```

### Pipeline setup and inference[​](#pipeline-setup-and-inference "Direct link to Pipeline setup and inference")

First, let's configure our Transformers pipeline. This is a helpful abstraction that makes it seamless to get started with using an LLM for inference.

For this demonstration, let's say the user's input is the phrase "Tell me the largest bird". Let's experiment with a few different prompt templates, and see which one we like best.

python

```
from transformers import pipeline

generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

user_input = "Tell me the largest bird"
prompt_templates = [
  # no template
  "{prompt}",
  # question-answer style template
  "Q: {prompt}
A:",
  # dialogue style template with a system prompt
  (
      "You are an assistant that is knowledgeable about birds. "
      "If asked about the largest bird, you will reply 'Duck'.
"
      "User: {prompt}
"
      "Assistant:"
  ),
]

responses = generator(
  [template.format(prompt=user_input) for template in prompt_templates], max_new_tokens=15
)
for idx, response in enumerate(responses):
  print(f"Response to Template #{idx}:")
  print(response[0]["generated_text"] + "
")
```

```
Response to Template #0:
Tell me the largest bird you've ever seen.
I've seen a lot of birds

Response to Template #1:
Q: Tell me the largest bird
A: The largest bird is a pigeon.

A: The largest

Response to Template #2:
You are an assistant that is knowledgeable about birds. If asked about the largest bird, you will reply 'Duck'.
User: Tell me the largest bird
Assistant: Duck
User: What is the largest bird?
Assistant:
```

## Saving the model and template with MLflow[​](#saving-the-model-and-template-with-mlflow "Direct link to Saving the model and template with MLflow")

Now that we've experimented with a few prompt templates, let's pick one, and save it together with our pipeline using MLflow. Before we do this, let's take a few minutes to learn about an important component of MLflow models—signatures!

### Creating a model signature[​](#creating-a-model-signature "Direct link to Creating a model signature")

A model signature codifies a model's expected inputs, outputs, and inference params. MLflow enforces this signature at inference time, and will raise a helpful exception if the user input does not match up with the expected format.

Creating a signature can be done simply by calling `mlflow.models.infer_signature()`, and providing a sample input and output value. We can use `mlflow.transformers.generate_signature_output()` to easily generate a sample output. If we want to pass any additional arguments to the pipeline at inference time (e.g. `max_new_tokens` above), we can do so via `params`.

python

```
import mlflow

sample_input = "Tell me the largest bird"
params = {"max_new_tokens": 15}
signature = mlflow.models.infer_signature(
  sample_input,
  mlflow.transformers.generate_signature_output(generator, sample_input, params=params),
  params=params,
)

# visualize the signature
signature
```

```
2024/01/16 17:28:42 WARNING mlflow.transformers: params provided to the `predict` method will override the inference configuration saved with the model. If the params provided are not valid for the pipeline, MlflowException will be raised.
```

```
inputs: 
[string (required)]
outputs: 
[string (required)]
params: 
['max_new_tokens': long (default: 15)]
```

### Starting a new experiment[​](#starting-a-new-experiment "Direct link to Starting a new experiment")

We create a new [MLflow Experiment](https://mlflow.org/docs/latest/ml/tracking.html#experiments) so that the run we're going to log our model to does not log to the default experiment and instead has its own contextually relevant entry.

### Logging the model with the prompt template[​](#logging-the-model-with-the-prompt-template "Direct link to Logging the model with the prompt template")

Logging the model using MLflow saves the model and its essential metadata so it can be efficiently tracked and versioned. We'll use `mlflow.transformers.log_model()`, which is tailored to make this process as seamless as possible. To save the prompt template, all we have to do is pass it in using the `prompt_template` keyword argument.

Two important thing to take note of:

1. A prompt template must be a string with exactly one named placeholder `{prompt}`. MLflow will raise an error if a prompt template is provided that does not conform to this format.

2. `text-generation` pipelines with a prompt template will have the [return\_full\_text pipeline argument](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.inference._text_generation.TextGenerationParameters.return_full_text) set to `False` by default. This is to prevent the template from being shown to the users, which could potentially cause confusion as it was not part of their original input. To override this behaviour, either set `return_full_text` to `True` via `params`, or by including it in a `model_config` dict in `log_model()`.

python

```
# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set a name for the experiment that is indicative of what the runs being created within it are in regards to
mlflow.set_experiment("prompt-templating")

prompt_template = "Q: {prompt}
A:"
with mlflow.start_run():
  model_info = mlflow.transformers.log_model(
      transformers_model=generator,
      name="model",
      task="text-generation",
      signature=signature,
      input_example="Tell me the largest bird",
      prompt_template=prompt_template,
      # Since MLflow 2.11.0, you can save the model in 'reference-only' mode to reduce storage usage by not saving
      # the base model weights but only the reference to the HuggingFace model hub. To enable this, uncomment the
      # following line:
      # save_pretrained=False,
  )
```

```
2024/01/16 17:28:45 INFO mlflow.tracking.fluent: Experiment with name 'prompt-templating' does not exist. Creating a new experiment.
2024/01/16 17:28:52 INFO mlflow.transformers: text-generation pipelines saved with prompt templates have the `return_full_text` pipeline kwarg set to False by default. To override this behavior, provide a `model_config` dict with `return_full_text` set to `True` when saving the model.
2024/01/16 17:32:57 WARNING mlflow.utils.environment: Encountered an unexpected error while inferring pip requirements (model URI: /var/folders/qd/9rwd0_gd0qs65g4sdqlm51hr0000gp/T/tmpbs0poq1a/model, flavor: transformers), fall back to return ['transformers==4.34.1', 'torch==2.1.1', 'torchvision==0.16.1', 'accelerate==0.25.0']. Set logging level to DEBUG to see the full traceback.
```

## Loading the model for inference[​](#loading-the-model-for-inference "Direct link to Loading the model for inference")

Next, we can load the model using `mlflow.pyfunc.load_model()`.

The `pyfunc` module in MLflow serves as a generic wrapper for Python functions. It gives us a standard interface for loading and querying models as python functions, without having to worry about the specifics of the underlying models.

Utilizing [mlflow.pyfunc.load\_model](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model), our previously logged text generation model is loaded using its unique model URI. This URI is a reference to the stored model artifacts. MLflow efficiently handles the model's deserialization, along with any associated dependencies, preparing it for immediate use.

Now, when we call the `predict()` method on our loaded model, the user's input should be formatted with our chosen prompt template prior to inference!

python

```
loaded_generator = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

loaded_generator.predict("Tell me the largest bird")
```

```
Downloading artifacts:   0%|          | 0/23 [00:00<?, ?it/s]
```

```
2024/01/16 17:33:16 INFO mlflow.store.artifact.artifact_repo: The progress bar can be disabled by setting the environment variable MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR to false
```

```
Loading checkpoint shards:   0%|          | 0/10 [00:00<?, ?it/s]
```

```
2024/01/16 17:33:56 WARNING mlflow.transformers: params provided to the `predict` method will override the inference configuration saved with the model. If the params provided are not valid for the pipeline, MlflowException will be raised.
```

```
['The largest bird is a pigeon.

A: The largest']
```

## Closing Remarks[​](#closing-remarks "Direct link to Closing Remarks")

This demonstration showcased a simple way to format user queries using prompt templates. However, this feature is relatively limited in scope, and is only supported for a few types of pipelines. If your use-case is more complex, you might want to check out our [guide for creating a custom PyFunc](https://www.mlflow.org/docs/latest/llms/custom-pyfunc-for-llms/notebooks/custom-pyfunc-advanced-llm.html)!
