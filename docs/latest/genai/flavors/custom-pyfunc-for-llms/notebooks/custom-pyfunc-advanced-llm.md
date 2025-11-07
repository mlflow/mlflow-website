# Serving LLMs with MLflow: Leveraging Custom PyFunc

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/custom-pyfunc-for-llms/notebooks/custom-pyfunc-advanced-llm.ipynb)

### Introduction[​](#introduction "Direct link to Introduction")

This tutorial guides you through saving and deploying Large Language Models (LLMs) using a custom `pyfunc` with MLflow, ideal for models not directly supported by MLflow's default transformers flavor.

### Learning Objectives[​](#learning-objectives "Direct link to Learning Objectives")

* Understand the need for custom `pyfunc` definitions in specific model scenarios.
* Learn to create a custom `pyfunc` to manage model dependencies and interface data.
* Gain insights into simplifying user interfaces in deployed environments with custom `pyfunc`.

#### The Challenge with Default Implementations[​](#the-challenge-with-default-implementations "Direct link to The Challenge with Default Implementations")

While MLflow's `transformers` flavor generally handles models from the HuggingFace Transformers library, some models or configurations might not align with this standard approach. In such cases, like ours, where the model cannot utilize the default `pipeline` type, we face a unique challenge of deploying these models using MLflow.

#### The Power of Custom PyFunc[​](#the-power-of-custom-pyfunc "Direct link to The Power of Custom PyFunc")

To address this, MLflow's custom `pyfunc` comes to the rescue. It allows us to:

* Handle model loading and its dependencies efficiently.
* Customize the inference process to suit specific model requirements.
* Adapt interface data to create a user-friendly environment in deployed applications.

Our focus will be on the practical application of a custom `pyfunc` to deploy LLMs effectively within MLflow's ecosystem.

By the end of this tutorial, you'll be equipped with the knowledge to tackle similar challenges in your machine learning projects, leveraging the full potential of MLflow for custom model deployments.

### Important Considerations Before Proceeding[​](#important-considerations-before-proceeding "Direct link to Important Considerations Before Proceeding")

#### Hardware Recommendations[​](#hardware-recommendations "Direct link to Hardware Recommendations")

This guide demonstrates the usage of a particularly large and intricate Large Language Model (LLM). Given its complexity:

* **GPU Requirement**: It's **strongly advised** to run this example on a system with a CUDA-capable GPU that possesses at least 64GB of VRAM.
* **CPU Caution**: While technically feasible, executing the model on a CPU can result in extremely prolonged inference times, potentially taking tens of minutes for a single prediction, even on top-tier CPUs. The final cell of this notebook is deliberately not executed due to the limitations with performance when running this model on a CPU-only system. However, with an appropriately powerful GPU, the total runtime of this notebook is \~8 minutes end to end.

#### Execution Recommendations[​](#execution-recommendations "Direct link to Execution Recommendations")

If you're considering running the code in this notebook:

* **Performance**: For a smoother experience and to truly harness the model's capabilities, use hardware aligned with the model's design.

* **Dependencies**: Ensure you've installed the recommended dependencies for optimal model performance. These are crucial for efficient model loading, initialization, attention computations, and inference processing:

bash

```
pip install xformers==0.0.20 einops==0.6.1 flash-attn==v1.0.3.post0 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
```

python

```
# Load necessary libraries

import accelerate
import torch
import transformers
from huggingface_hub import snapshot_download

import mlflow
```

```
/Users/benjamin.wilson/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pydantic/_internal/_fields.py:128: UserWarning: Field "model_server_url" has conflict with protected namespace "model_".

You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
warnings.warn(
/Users/benjamin.wilson/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pydantic/_internal/_config.py:317: UserWarning: Valid config keys have changed in V2:
* 'schema_extra' has been renamed to 'json_schema_extra'
warnings.warn(message, UserWarning)
```

#### Downloading the Model and Tokenizer[​](#downloading-the-model-and-tokenizer "Direct link to Downloading the Model and Tokenizer")

First, we need to download our model and tokenizer. Here's how we do it:

python

```
# Download the MPT-7B instruct model and tokenizer to a local directory cache
snapshot_location = snapshot_download(repo_id="mosaicml/mpt-7b-instruct", local_dir="mpt-7b")
```

```
Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]
```

```
Downloading README.md:   0%|          | 0.00/7.96k [00:00<?, ?B/s]
```

```
Downloading .gitattributes:   0%|          | 0.00/1.48k [00:00<?, ?B/s]
```

```
Downloading adapt_tokenizer.py:   0%|          | 0.00/1.72k [00:00<?, ?B/s]
```

```
Downloading attention.py:   0%|          | 0.00/21.6k [00:00<?, ?B/s]
```

```
Downloading config.json:   0%|          | 0.00/1.23k [00:00<?, ?B/s]
```

```
Downloading blocks.py:   0%|          | 0.00/2.84k [00:00<?, ?B/s]
```

```
Downloading custom_embedding.py:   0%|          | 0.00/292 [00:00<?, ?B/s]
```

```
Downloading configuration_mpt.py:   0%|          | 0.00/11.0k [00:00<?, ?B/s]
```

```
Downloading meta_init_context.py:   0%|          | 0.00/3.96k [00:00<?, ?B/s]
```

```
Downloading fc.py:   0%|          | 0.00/167 [00:00<?, ?B/s]
```

```
Downloading ffn.py:   0%|          | 0.00/1.75k [00:00<?, ?B/s]
```

```
Downloading generation_config.json:   0%|          | 0.00/112 [00:00<?, ?B/s]
```

```
Downloading (…)refixlm_converter.py:   0%|          | 0.00/10.5k [00:00<?, ?B/s]
```

```
Downloading modeling_mpt.py:   0%|          | 0.00/20.1k [00:00<?, ?B/s]
```

```
Downloading flash_attn_triton.py:   0%|          | 0.00/28.2k [00:00<?, ?B/s]
```

```
Downloading requirements.txt:   0%|          | 0.00/113 [00:00<?, ?B/s]
```

```
Downloading param_init_fns.py:   0%|          | 0.00/11.9k [00:00<?, ?B/s]
```

```
Downloading (…)model.bin.index.json:   0%|          | 0.00/16.0k [00:00<?, ?B/s]
```

```
Downloading (…)cial_tokens_map.json:   0%|          | 0.00/99.0 [00:00<?, ?B/s]
```

```
Downloading norm.py:   0%|          | 0.00/3.12k [00:00<?, ?B/s]
```

```
Downloading tokenizer.json:   0%|          | 0.00/2.11M [00:00<?, ?B/s]
```

```
Downloading (…)l-00001-of-00002.bin:   0%|          | 0.00/9.94G [00:00<?, ?B/s]
```

```
Downloading (…)l-00002-of-00002.bin:   0%|          | 0.00/3.36G [00:00<?, ?B/s]
```

```
Downloading tokenizer_config.json:   0%|          | 0.00/237 [00:00<?, ?B/s]
```

#### Defining the Custom PyFunc[​](#defining-the-custom-pyfunc "Direct link to Defining the Custom PyFunc")

Now, let's define our custom `pyfunc`. This will dictate how our model loads its dependencies and how it performs predictions. Notice how we've wrapped the intricacies of the model within this class.

python

```
class MPT(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
      """
      This method initializes the tokenizer and language model
      using the specified model snapshot directory.
      """
      # Initialize tokenizer and language model
      self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          context.artifacts["snapshot"], padding_side="left"
      )

      config = transformers.AutoConfig.from_pretrained(
          context.artifacts["snapshot"], trust_remote_code=True
      )
      # If you are running this in a system that has a sufficiently powerful GPU with available VRAM,
      # uncomment the configuration setting below to leverage triton.
      # Note that triton dramatically improves the inference speed performance

      # config.attn_config["attn_impl"] = "triton"

      self.model = transformers.AutoModelForCausalLM.from_pretrained(
          context.artifacts["snapshot"],
          config=config,
          torch_dtype=torch.bfloat16,
          trust_remote_code=True,
      )

      # NB: If you do not have a CUDA-capable device or have torch installed with CUDA support
      # this setting will not function correctly. Setting device to 'cpu' is valid, but
      # the performance will be very slow.
      self.model.to(device="cpu")
      # If running on a GPU-compatible environment, uncomment the following line:
      # self.model.to(device="cuda")

      self.model.eval()

  def _build_prompt(self, instruction):
      """
      This method generates the prompt for the model.
      """
      INSTRUCTION_KEY = "### Instruction:"
      RESPONSE_KEY = "### Response:"
      INTRO_BLURB = (
          "Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request."
      )

      return f"""{INTRO_BLURB}
      {INSTRUCTION_KEY}
      {instruction}
      {RESPONSE_KEY}
      """

  def predict(self, context, model_input, params=None):
      """
      This method generates prediction for the given input.
      """
      prompt = model_input["prompt"][0]

      # Retrieve or use default values for temperature and max_tokens
      temperature = params.get("temperature", 0.1) if params else 0.1
      max_tokens = params.get("max_tokens", 1000) if params else 1000

      # Build the prompt
      prompt = self._build_prompt(prompt)

      # Encode the input and generate prediction
      # NB: Sending the tokenized inputs to the GPU here explicitly will not work if your system does not have CUDA support.
      # If attempting to run this with GPU support, change 'cpu' to 'cuda' for maximum performance
      encoded_input = self.tokenizer.encode(prompt, return_tensors="pt").to("cpu")
      output = self.model.generate(
          encoded_input,
          do_sample=True,
          temperature=temperature,
          max_new_tokens=max_tokens,
      )

      # Removing the prompt from the generated text
      prompt_length = len(self.tokenizer.encode(prompt, return_tensors="pt")[0])
      generated_response = self.tokenizer.decode(
          output[0][prompt_length:], skip_special_tokens=True
      )

      return {"candidates": [generated_response]}
```

### Building the Prompt[​](#building-the-prompt "Direct link to Building the Prompt")

One key aspect of our custom `pyfunc` is the construction of a model prompt. Instead of the end-user having to understand and construct this prompt, our custom `pyfunc` takes care of it. This ensures that regardless of the intricacies of the model's requirements, the end-user interface remains simple and consistent.

Review the method `_build_prompt()` inside our class above to see how custom input processing logic can be added to a custom pyfunc to support required translations of user-input data into a format that is compatible with the wrapped model instance.

python

```
import numpy as np
import pandas as pd

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

# Define input and output schema
input_schema = Schema(
  [
      ColSpec(DataType.string, "prompt"),
  ]
)
output_schema = Schema([ColSpec(DataType.string, "candidates")])

parameters = ParamSchema(
  [
      ParamSpec("temperature", DataType.float, np.float32(0.1), None),
      ParamSpec("max_tokens", DataType.integer, np.int32(1000), None),
  ]
)

signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=parameters)


# Define input example
input_example = pd.DataFrame({"prompt": ["What is machine learning?"]})
```

#### Set the experiment that we're going to be logging our custom model to[​](#set-the-experiment-that-were-going-to-be-logging-our-custom-model-to "Direct link to Set the experiment that we're going to be logging our custom model to")

If the experiment doesn't already exist, MLflow will create a new experiment with this name and will alert you that it has created a new experiment.

python

```
# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

# mlflow.set_tracking_uri("http://127.0.0.1:8080")

mlflow.set_experiment(experiment_name="mpt-7b-instruct-evaluation")
```

```
2023/11/29 17:33:23 INFO mlflow.tracking.fluent: Experiment with name 'mpt-7b-instruct-evaluation' does not exist. Creating a new experiment.
```

```
<Experiment: artifact_location='file:///Users/benjamin.wilson/repos/mlflow-fork/mlflow/docs/source/llms/custom-pyfunc-for-llms/notebooks/mlruns/265930820950682761', creation_time=1701297203895, experiment_id='265930820950682761', last_update_time=1701297203895, lifecycle_stage='active', name='mpt-7b-instruct-evaluation', tags={}>
```

python

```
# Get the current base version of torch that is installed, without specific version modifiers
torch_version = torch.__version__.split("+")[0]

# Start an MLflow run context and log the MPT-7B model wrapper along with the param-included signature to
# allow for overriding parameters at inference time
with mlflow.start_run():
  model_info = mlflow.pyfunc.log_model(
      name="mpt-7b-instruct",
      python_model=MPT(),
      # NOTE: the artifacts dictionary mapping is critical! This dict is used by the load_context() method in our MPT() class.
      artifacts={"snapshot": snapshot_location},
      pip_requirements=[
          f"torch=={torch_version}",
          f"transformers=={transformers.__version__}",
          f"accelerate=={accelerate.__version__}",
          "einops",
          "sentencepiece",
      ],
      input_example=input_example,
      signature=signature,
  )
```

```
Downloading artifacts:   0%|          | 0/24 [00:00<?, ?it/s]
```

```
2023/11/29 17:33:24 INFO mlflow.store.artifact.artifact_repo: The progress bar can be disabled by setting the environment variable MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR to false
/Users/benjamin.wilson/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils.
warnings.warn("Setuptools is replacing distutils.")
```

#### Load the saved model[​](#load-the-saved-model "Direct link to Load the saved model")

python

```
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
```

```
/Users/benjamin.wilson/.cache/huggingface/modules/transformers_modules/mpt-7b/configuration_mpt.py:97: UserWarning: alibi is turned on, setting `learned_pos_emb` to `False.`
warnings.warn(f'alibi is turned on, setting `learned_pos_emb` to `False.`')
```

#### Test the model for inference[​](#test-the-model-for-inference "Direct link to Test the model for inference")

python

```
# The execution of this is commented out for the purposes of runtime on CPU.
# If you are running this on a system with a sufficiently powerful GPU, you may uncomment and interface with the model!

# loaded_model.predict(pd.DataFrame(
#     {"prompt": ["What is machine learning?"]}), params={"temperature": 0.6}
# )
```

### Conclusion[​](#conclusion "Direct link to Conclusion")

Through this tutorial, we've seen the power and flexibility of MLflow's custom `pyfunc`. By understanding the specific needs of our model and defining a custom `pyfunc` to cater to those needs, we can ensure a seamless deployment process and a user-friendly interface.
