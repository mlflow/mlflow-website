# Fine-Tuning Open-Source LLM using QLoRA with MLflow and PEFT

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft.ipynb)

## Overview[​](#overview "Direct link to Overview")

Many powerful open-source LLMs have emerged and are easily accessible. However, they are not designed to be deployed to your production environment out-of-the-box; instead, you have to **fine-tune** them for your specific tasks, such as a chatbot, content generation, etc. One challenge, though, is that training LLMs is usually very expensive. Even if your dataset for fine-tuning is small, the backpropagation step needs to compute gradients for billions of parameters. For example, fully fine-tuning the Llama7B model requires 112GB of VRAM, i.e. at least two 80GB A100 GPUs. Fortunately, there are many research efforts on how to reduce the cost of LLM fine-tuning.

In this tutorial, we will demonstrate how to build a powerful **text-to-SQL** generator by fine-tuning the Mistral 7B model with **a single 24GB VRAM GPU**.

### What You Will Learn[​](#what-you-will-learn "Direct link to What You Will Learn")

1. Hands-on learning of the typical LLM fine-tuning process.
2. Understand how to use **QLoRA** and **PEFT** to overcome the GPU memory limitation for fine-tuning.
3. Manage the model training cycle using **MLflow** to log the model artifacts, hyperparameters, metrics, and prompts.
4. How to save prompt template and inference parameters (e.g. max\_token\_length) in MLflow to simplify prediction interface.

### Key Actors[​](#key-actors "Direct link to Key Actors")

In this tutorial, you will learn about the techniques and methods behind efficient LLM fine-tuning by actually running the code. There are more detailed explanations for each cell below, but let's start with a brief preview of a few main important libraries/methods used in this tutorial.

* [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) model is a pretrained text-generation model with 7 billion parameters, developed by [mistral.ai](https://mistral.ai/). The model employs various optimization techniques such as Group-Query Attention, Sliding-Window Attention, Byte-fallback BPE tokenizer, and outperforms the Llama 2 13B on benchmarks with fewer parameters.
* [QLoRA](https://github.com/artidoro/qlora) is a novel method that allows us to fine-tune large foundational models with limited GPU resources. It reduces the number of trainable parameters by learning pairs of rank-decomposition matrices and also applies 4-bit quantization to the frozen pretrained model to further reduce the memory footprint.
* [PEFT](https://huggingface.co/docs/peft/en/index) is a library developed by HuggingFace🤗, that enables developers to easily integrate various optimization methods with pretrained models available on the HuggingFace Hub. With PEFT, you can apply QLoRA to the pretrained model with a few lines of configurations and run fine-tuning just like the normal Transformers model training.
* [MLflow](https://mlflow.org/) manages an exploding number of configurations, assets, and metrics during the LLM training on your behalf. MLflow is natively integrated with Transformers and PEFT, and plays a crucial role in organizing the fine-tuning cycle.

## 1. Environment Set up[​](#1-environment-set-up "Direct link to 1. Environment Set up")

### Hardware Requirement[​](#hardware-requirement "Direct link to Hardware Requirement")

Please ensure your GPU has at least 20GB of VRAM available. This notebook has been tested on a single NVIDIA A10G GPU with 24GB of VRAM.

python

```python
%sh nvidia-smi

```

### Install Python Libraries[​](#install-python-libraries "Direct link to Install Python Libraries")

This tutorial utilizes the following Python libraries:

* [mlflow](https://pypi.org/project/mlflow/) - for tracking parameters, metrics, and saving trained models. Version **2.11.0 or later** is required to log PEFT models with MLflow.
* [transformers](https://pypi.org/project/transformers/) - for defining the model, tokenizer, and trainer.
* [peft](https://pypi.org/project/peft/) - for creating a LoRA adapter on top of the Transformer model.
* [bitsandbytes](https://pypi.org/project/bitsandbytes/) - for loading the base model with 4-bit quantization for QLoRA.
* [accelerate](https://pypi.org/project/accelerate/) - a dependency required by bitsandbytes.
* [datasets](https://pypi.org/project/datasets/) - for loading the training dataset from the HuggingFace hub.

**Note**: Restarting the Python kernel may be necessary after installing these dependencies.

The notebook has been tested with `mlflow==2.11.0`, `transformers==4.35.2`, `peft==0.8.2`, `bitsandbytes==0.42.0`, `accelerate==0.27.2`, and `datasets==2.17.1`.

python

```python
%pip install mlflow>=2.11.0
%pip install transformers peft accelerate bitsandbytes datasets -q -U

```

## 2. Dataset Preparation[​](#2-dataset-preparation "Direct link to 2. Dataset Preparation")

### Load Dataset from HuggingFace Hub[​](#load-dataset-from-huggingface-hub "Direct link to Load Dataset from HuggingFace Hub")

We will use the `b-mc2/sql-create-context` dataset from the [Hugging Face Hub](https://huggingface.co/datasets/b-mc2/sql-create-context) for this tutorial. This dataset comprises 78.6k pairs of natural language queries and their corresponding SQL statements, making it ideal for training a text-to-SQL model. The dataset includes three columns:

* `question`: A natural language question posed regarding the data.
* `context`: Additional information about the data, such as the schema for the table being queried.
* `answer`: The SQL query that represents the expected output.

python

```python
import pandas as pd
from datasets import load_dataset
from IPython.display import HTML, display

dataset_name = "b-mc2/sql-create-context"
dataset = load_dataset(dataset_name, split="train")


def display_table(dataset_or_sample):
  # A helper fuction to display a Transformer dataset or single sample contains multi-line string nicely
  pd.set_option("display.max_colwidth", None)
  pd.set_option("display.width", None)
  pd.set_option("display.max_rows", None)

  if isinstance(dataset_or_sample, dict):
      df = pd.DataFrame(dataset_or_sample, index=[0])
  else:
      df = pd.DataFrame(dataset_or_sample)

  html = df.to_html().replace("\n", "<br>")
  styled_html = f"""<style> .dataframe th, .dataframe tbody td {{ text-align: left; padding-right: 30px; }} </style> {html}"""
  display(HTML(styled_html))


display_table(dataset.select(range(3)))

```

|   | question                                                                      | context                                                                                | answer                                                      |
| - | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 0 | How many heads of the departments are older than 56 ?                         | CREATE TABLE head (age INTEGER)                                                        | SELECT COUNT(\*) FROM head WHERE age > 56                   |
| 1 | List the name, born state and age of the heads of departments ordered by age. | CREATE TABLE head (name VARCHAR, born\_state VARCHAR, age VARCHAR)                     | SELECT name, born\_state, age FROM head ORDER BY age        |
| 2 | List the creation year, name and budget of each department.                   | CREATE TABLE department (creation VARCHAR, name VARCHAR, budget\_in\_billions VARCHAR) | SELECT creation, name, budget\_in\_billions FROM department |

### Split Train and Test Dataset[​](#split-train-and-test-dataset "Direct link to Split Train and Test Dataset")

The `b-mc2/sql-create-context` dataset consists of a single split, "train". We will separate 20% of this as test samples.

python

```python
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(f"Training dataset contains {len(train_dataset)} text-to-SQL pairs")
print(f"Test dataset contains {len(test_dataset)} text-to-SQL pairs")

```

### Define Prompt Template[​](#define-prompt-template "Direct link to Define Prompt Template")

The Mistral 7B model is a text comprehension model, so we have to construct a text prompt that incorporates the user's question, context, and our system instructions. The new `prompt` column in the dataset will contain the text prompt to be fed into the model during training. It is important to note that we also include the expected response within the prompt, allowing the model to be trained in a self-supervised manner.

python

```python
PROMPT_TEMPLATE = """You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.

### Table:
{context}

### Question:
{question}

### Response:
{output}"""


def apply_prompt_template(row):
  prompt = PROMPT_TEMPLATE.format(
      question=row["question"],
      context=row["context"],
      output=row["answer"],
  )
  return {"prompt": prompt}


train_dataset = train_dataset.map(apply_prompt_template)
display_table(train_dataset.select(range(1)))

```

|   | question                                                                     | context                                                                                                                | answer                                                                                                                    | prompt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| - | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0 | Which Perth has Gold Coast yes, Sydney yes, Melbourne yes, and Adelaide yes? | CREATE TABLE table\_name\_56 (perth VARCHAR, adelaide VARCHAR, melbourne VARCHAR, gold\_coast VARCHAR, sydney VARCHAR) | SELECT perth FROM table\_name\_56 WHERE gold\_coast = "yes" AND sydney = "yes" AND melbourne = "yes" AND adelaide = "yes" | You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.<br /><br />### Table:<br />CREATE TABLE table\_name\_56 (perth VARCHAR, adelaide VARCHAR, melbourne VARCHAR, gold\_coast VARCHAR, sydney VARCHAR)<br /><br />### Question:<br />Which Perth has Gold Coast yes, Sydney yes, Melbourne yes, and Adelaide yes?<br /><br />### Response:<br />SELECT perth FROM table\_name\_56 WHERE gold\_coast = "yes" AND sydney = "yes" AND melbourne = "yes" AND adelaide = "yes" |

### Padding the Training Dataset[​](#padding-the-training-dataset "Direct link to Padding the Training Dataset")

As a final step of dataset preparation, we need to apply **padding** to the training dataset. Padding ensures that all input sequences in a batch are of the same length.

A crucial point to note is the need to *add padding to the left*. This approach is adopted because the model generates tokens autoregressively, meaning it continues from the last token. Adding padding to the right would cause the model to generate new tokens from these padding tokens, resulting in the output sequence including padding tokens in the middle.

* Padding to right

text

```text
Today |  is  |   a    |  cold  |  <pad>  ==generate=>  "Today is a cold <pad> day"
 How  |  to  | become |  <pad> |  <pad>  ==generate=>  "How to become a <pad> <pad> great engineer".

```

* Padding to left:

text

```text
<pad> |  Today  |  is  |  a   |  cold     ==generate=>  "<pad> Today is a cold day"
<pad> |  <pad>  |  How |  to  |  become   ==generate=>  "<pad> <pad> How to become a great engineer".

```

python

```python
from transformers import AutoTokenizer

base_model_id = "mistralai/Mistral-7B-v0.1"

# You can use a different max length if your custom dataset has shorter/longer input sequences.
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(
  base_model_id,
  model_max_length=MAX_LENGTH,
  padding_side="left",
  add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_and_pad_to_fixed_length(sample):
  result = tokenizer(
      sample["prompt"],
      truncation=True,
      max_length=MAX_LENGTH,
      padding="max_length",
  )
  result["labels"] = result["input_ids"].copy()
  return result


tokenized_train_dataset = train_dataset.map(tokenize_and_pad_to_fixed_length)

assert all(len(x["input_ids"]) == MAX_LENGTH for x in tokenized_train_dataset)

display_table(tokenized_train_dataset.select(range(1)))

```

## 3. Load the Base Model (with 4-bit quantization)[​](#3-load-the-base-model-with-4-bit-quantization "Direct link to 3. Load the Base Model (with 4-bit quantization)")

Next, we'll load the Mistral 7B model, which will serve as our base model for fine-tuning. This model can be loaded from the HuggingFace Hub repository [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) using the Transformers' `from_pretrained()` API. However, here we are also providing a `quantization_config` parameter.

This parameter embodies the key technique of [QLoRA](https://github.com/artidoro/qlora) that significantly reduces memory usage during fine-tuning. The following paragraph details the method and the implications of this configuration. However, feel free to skip if it appears complex. After all, we rarely need to modify the `quantization_config` values ourselves :)

**How It Works**

In short, QLoRA is a combination of **Q**uantization and **LoRA**. To grasp its functionality, it's simpler to begin with LoRA. [LoRA (Low Rank Adaptation)](https://github.com/microsoft/LoRA) is a preceding method for resource-efficient fine-tuning, by reducing the number of trainable parameters through matrix decomposition. Let `W'` represent the final weight matrix from fine-tuning. In LoRA, `W'` is approximated by the sum of the original weight and its update, i.e., `W + ΔW`, then decomposing the delta part into two low-dimensional matrices, i.e., `ΔW ≈ AB`. Suppose `W` is `m`x`m`, and we select a smaller `r` for the rank of `A` and `B`, where `A` is `m`x`r` and `B` is `r`x`m`. Now, the original trainable parameters, which are quadratic in size of `W` (i.e., `m^2`), after decomposition, become `2mr`. Empirically, we can choose a much smaller number for `r`, e.g., 32, 64, compared to the full weight matrix size, therefore this significantly reduces the number of parameters to train.

[QLoRA](https://github.com/artidoro/qlora) extends LoRA, employing the same strategy for matrix decomposition. However, it further reduces memory usage by applying 4-bit quantization to the frozen pretrained model `W`. According to their research, the largest memory usage during LoRA fine-tuning is the backpropagation through the frozen parameters `W` to compute gradients for the adaptors `A` and `B`. Thus, quantizing `W` to 4-bit significantly reduces the overall memory consumption. This is achieved with the `load_in_4bit=True` setting shown below.

Moreover, QLoRA introduces additional techniques to optimize resource usage without significantly impacting model performance. For more technical details, please refer to [the paper](https://arxiv.org/pdf/2305.14314.pdf), but we implement them by setting the following quantization configurations in bitsandbytes:

* The 4-bit NormalFloat type is specified by `bnb_4bit_quant_type="nf4"`.
* Double quantization is activated by `bnb_4bit_use_double_quant=True`.
* QLoRA re-quantizes the 4-bit weights back to a higher precision when computing the gradients for `A` and `B`, to prevent performance degradation. This datatype is specified by `bnb_4bit_compute_dtype=torch.bfloat16`.

python

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
  # Load the model with 4-bit quantization
  load_in_4bit=True,
  # Use double quantization
  bnb_4bit_use_double_quant=True,
  # Use 4-bit Normal Float for storing the base model weights in GPU memory
  bnb_4bit_quant_type="nf4",
  # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
  bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=quantization_config)

```

### How Does the Base Model Perform?[​](#how-does-the-base-model-perform "Direct link to How Does the Base Model Perform?")

First, let's assess the performance of the vanilla Mistral model on the SQL generation task before any fine-tuning. As expected, the model does not produce correct SQL queries; instead, it generates random answers in natural language. This outcome indicates the necessity of fine-tuning the model for our specific task.

python

```python
import transformers

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
pipeline = transformers.pipeline(model=model, tokenizer=tokenizer, task="text-generation")

sample = test_dataset[1]
prompt = PROMPT_TEMPLATE.format(
  context=sample["context"], question=sample["question"], output=""
)  # Leave the answer part blank

with torch.no_grad():
  response = pipeline(prompt, max_new_tokens=256, repetition_penalty=1.15, return_full_text=False)

display_table({"prompt": prompt, "generated_query": response[0]["generated_text"]})

```

|   | prompt                                                                                                                                                                                                                                                                                                                                                                                             | generated\_query                                                                                                                                                                                                                                                                                                                                                                      |
| - | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 | You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.<br /><br />### Table:<br />CREATE TABLE table\_name\_61 (game INTEGER, opponent VARCHAR, record VARCHAR)<br /><br />### Question:<br />What is the lowest numbered game against Phoenix with a record of 29-17?<br /><br />### Response:<br /> | <br />A: The lowest numbered game against Phoenix was played on 03/04/2018. The score was PHO 115 - DAL 106.<br />What is the highest numbered game against Phoenix?<br />A: The highest numbered game against Phoenix was played on 03/04/2018. The score was PHO 115 - DAL 106.<br />Which players have started at Point Guard for Dallas in a regular season game against Phoenix? |

## 4. Define a PEFT Model[​](#4-define-a-peft-model "Direct link to 4. Define a PEFT Model")

As discussed earlier, QLoRA stands for **Quantization** + **LoRA**. Having applied the quantization part, we now proceed with the LoRA aspect. Although the mathematics behind LoRA is intricate, [PEFT](https://huggingface.co/docs/peft/en/index) helps us by simplifying the process of adapting LoRA to the pretrained Transformer model.

In the next cell, we create a [LoraConfig](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py) with various settings for LoRA. Contrary to the earlier `quantization_config`, these hyperparameters might need optimization to achieve the best model performance for your specific task. **MLflow** facilitates this process by tracking these hyperparameters, the associated model, and its outcomes.

At the end of the cell, we display the number of trainable parameters during fine-tuning, and their percentage relative to the total model parameters. Here, we are training only 1.16% of the total 7 billion parameters.

python

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Enabling gradient checkpointing, to make the training further efficient
model.gradient_checkpointing_enable()
# Set up the model for quantization-aware training e.g. casting layers, parameter freezing, etc.
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
  task_type="CAUSAL_LM",
  # This is the rank of the decomposed matrices A and B to be learned during fine-tuning. A smaller number will save more GPU memory but might result in worse performance.
  r=32,
  # This is the coefficient for the learned ΔW factor, so the larger number will typically result in a larger behavior change after fine-tuning.
  lora_alpha=64,
  # Drop out ratio for the layers in LoRA adaptors A and B.
  lora_dropout=0.1,
  # We fine-tune all linear layers in the model. It might sound a bit large, but the trainable adapter size is still only **1.16%** of the whole model.
  target_modules=[
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj",
      "lm_head",
  ],
  # Bias parameters to train. 'none' is recommended to keep the original model performing equally when turning off the adapter.
  bias="none",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

```

**That's it!!!** PEFT has made the LoRA setup super easy.

An additional bonus is that the PEFT model exposes the same interfaces as a Transformers model. This means that everything from here on is quite similar to the standard model training process using Transformers.

## 5. Kick-off a Training Job[​](#5-kick-off-a-training-job "Direct link to 5. Kick-off a Training Job")

Similar to conventional Transformers training, we'll first set up a Trainer object to organize the training iterations. There are numerous hyperparameters to configure, but MLflow will manage them on your behalf.

To enable MLflow logging, you can specify `report_to="mlflow"` and name your training trial with the `run_name` parameter. This action initiates an [MLflow run](https://mlflow.org/docs/latest/ml/tracking.html#runs) that automatically logs training metrics, hyperparameters, configurations, and the trained model.

python

```python
from datetime import datetime

import transformers
from transformers import TrainingArguments

import mlflow

# Comment-out this line if you are running the tutorial on Databricks
mlflow.set_experiment("MLflow PEFT Tutorial")

training_args = TrainingArguments(
  # Set this to mlflow for logging your training
  report_to="mlflow",
  # Name the MLflow run
  run_name=f"Mistral-7B-SQL-QLoRA-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
  # Replace with your output destination
  output_dir="YOUR_OUTPUT_DIR",
  # For the following arguments, refer to https://huggingface.co/docs/transformers/main_classes/trainer
  per_device_train_batch_size=2,
  gradient_accumulation_steps=4,
  gradient_checkpointing=True,
  optim="paged_adamw_8bit",
  bf16=True,
  learning_rate=2e-5,
  lr_scheduler_type="constant",
  max_steps=500,
  save_steps=100,
  logging_steps=100,
  warmup_steps=5,
  # https://discuss.huggingface.co/t/training-llama-with-lora-on-multiple-gpus-may-exist-bug/47005/3
  ddp_find_unused_parameters=False,
)

trainer = transformers.Trainer(
  model=peft_model,
  train_dataset=tokenized_train_dataset,
  data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
  args=training_args,
)

# use_cache=True is incompatible with gradient checkpointing.
peft_model.config.use_cache = False

```

The training duration may span several hours, contingent upon your hardware specifications. Nonetheless, the primary objective of this tutorial is to acquaint you with the process of fine-tuning using PEFT and MLflow, rather than to cultivate a highly performant SQL generator. If you don't care much about the model performance, you may specify a smaller number of steps or interrupt the following cell to proceed with the rest of the notebook.

python

```python
trainer.train()

```

\[500/500 45:41, Epoch 0/1]

| Step | Training Loss |
| ---- | ------------- |
| 100  | 0.681700      |
| 200  | 0.522400      |
| 300  | 0.507300      |
| 400  | 0.494800      |
| 500  | 0.474600      |

## 6. Save the PEFT Model to MLflow[​](#6-save-the-peft-model-to-mlflow "Direct link to 6. Save the PEFT Model to MLflow")

Hooray! We have successfully fine-tuned the Mistral 7B model into an SQL generator. Before concluding the training, one final step is to save the trained PEFT model to MLflow.

### Set Prompt Template and Default Inference Parameters (optional)[​](#set-prompt-template-and-default-inference-parameters-optional "Direct link to Set Prompt Template and Default Inference Parameters (optional)")

LLMs prediction behavior is not only defined by the model weights, but also largely controlled by the prompt and inference paramters such as `max_token_length`, `repetition_penalty`. Therefore, it is highly advisable to save those metadata along with the model, so that you can expect the consistent behavior when loading the model later.

#### Prompt Template[​](#prompt-template "Direct link to Prompt Template")

The user prompt itself is free text, but you can harness the input by applying a 'template'. MLflow Transformer flavor supports saving a prompt template with the model, and apply it automatically before the prediction. This also allows you to hide the system prompt from model clients. To save the prompt template, we have to define a single string that contains `{prompt}` variable, and pass it to the `prompt_template` argument of [mlflow.transformers.log\_model](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html#mlflow.transformers.log_model) API. Refer to [Saving Prompt Templates with Transformer Pipelines](https://mlflow.org/docs/latest/ml/deep-learning/transformers/guide/index.html#saving-prompt-templates-with-transformer-pipelines) for more detailed usage of this feature.

python

```python
# Basically the same format as we applied to the dataset. However, the template only accepts {prompt} variable so both table and question need to be fed in there.
prompt_template = """You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.

{prompt}

### Response:
"""

```

#### Inference Parameters[​](#inference-parameters "Direct link to Inference Parameters")

Inference parameters can be saved with MLflow model as a part of [Model Signature](https://mlflow.org/docs/latest/ml/model/signatures.html). The signature defines model input and output format with additional parameters passed to the model prediction, and you can let MLflow to infer it from some sample input using [mlflow.models.infer\_signature](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.infer_signature) API. If you pass the concrete value for parameters, MLflow treats them as default values and apply them at the inference if they are not provided by users. For more details about the Model Signature, please refer to the [MLflow documentation](https://mlflow.org/docs/latest/ml/model/signatures.html).

python

```python
from mlflow.models import infer_signature

sample = train_dataset[1]

# MLflow infers schema from the provided sample input/output/params
signature = infer_signature(
  model_input=sample["prompt"],
  model_output=sample["answer"],
  # Parameters are saved with default values if specified
  params={"max_new_tokens": 256, "repetition_penalty": 1.15, "return_full_text": False},
)
signature

```

### Save the PEFT Model to MLflow[​](#save-the-peft-model-to-mlflow "Direct link to Save the PEFT Model to MLflow")

Finally, we will call [mlflow.transformers.log\_model](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html#mlflow.transformers.log_model) API to log the model to MLflow. A few critical points to remember when logging a PEFT model to MLflow are:

1. **MLflow logs the Transformer model as a [Pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines).** A pipeline bundles a model with its tokenizer (or other components, depending on the task type) and simplifies the prediction steps into an easy-to-use interface, making it an excellent tool for ensuring reproducibility. In the code below, we pass the model and tokenizer as a dictionary, then MLflow automatically deduces the correct pipeline type and saves it.
2. **MLflow does not save the base model weight for the PEFT model**. When executing `mlflow.transformers.log_model`, MLflow only saves the small number of trained parameters, i.e., the PEFT adapter. For the base model, MLflow instead records a reference to the HuggingFace hub (repository name and commit hash), and downloads the base model weights on the fly when loading the PEFT model. This approach significantly reduces storage usage and logging latency; for instance, the logged artifacts size in this tutorial is less than 1GB, while the full Mistral 7B model is about 20GB.
3. **Save a tokenizer without padding**. During fine-tuning, we applied padding to the dataset to standardize the sequence length in a batch. However, padding is no longer necessary at inference, so we save a different tokenizer without padding. This ensures the loaded model can be used for inference immediately.

**Note**: Currently, manual logging is required for the PEFT adapter and config, while other information, such as dataset, metrics, Trainer parameters, etc., are automatically logged. However, this process may be automated in future versions of MLflow and Transformers.

python

```python
import mlflow

# Get the ID of the MLflow Run that was automatically created above
last_run_id = mlflow.last_active_run().info.run_id

# Save a tokenizer without padding because it is only needed for training
tokenizer_no_pad = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True)

# If you interrupt the training, uncomment the following line to stop the MLflow run
# mlflow.end_run()

with mlflow.start_run(run_id=last_run_id):
  mlflow.log_params(peft_config.to_dict())
  mlflow.transformers.log_model(
      transformers_model={"model": trainer.model, "tokenizer": tokenizer_no_pad},
      prompt_template=prompt_template,
      signature=signature,
      name="model",  # This is a relative path to save model files within MLflow run
  )

```

### What's Logged to MLflow?[​](#whats-logged-to-mlflow "Direct link to What's Logged to MLflow?")

Let's briefly review what is logged/saved to MLflow as a result of your training. To access the MLflow UI, run `mlflow server` commands and open `https://localhost:PORT` (PORT is 5000 by default). Select the experiment "MLflow PEFT Tutorial" (or the notebook name when running on Databricks) on the left side. Then click on the latest MLflow Run named `Mistral-7B-SQL-QLoRA-2024-...` to view the Run details.

#### Parameters[​](#parameters "Direct link to Parameters")

The `Parameters` section displays hundreds of parameters specified for the Trainer, LoraConfig, and BitsAndBytesConfig, such as `learning_rate`, `r`, `bnb_4bit_quant_type`. It also includes default parameters that were not explicitly specified, which is crucial for ensuring reproducibility, especially if the library's default values change.

#### Metrics[​](#metrics "Direct link to Metrics")

The `Metrics` section presents the model metrics collected during the run, such as `train_loss`. You can visualize these metrics with various types of graphs in the "Chart" tab.

#### Artifacts[​](#artifacts "Direct link to Artifacts")

The `Artifacts` section displays the files/directories saved in MLflow as a result of training. For Transformers PEFT training, you should see the following files/directories:

text

```text

    model/
      ├─ peft/
      │  ├─ adapter_config.json       # JSON file of the LoraConfig
      │  ├─ adapter_module.safetensor # The weight file of the LoRA adapter
      │  └─ README.md                 # Empty README file generated by Transformers
      │
      ├─ LICENSE.txt                  # License information about the base model (Mistral-7B-0.1)
      ├─ MLModel                      # Contains various metadata about your model
      ├─ conda.yaml                   # Dependencies to create conda environment
      ├─ model_card.md                # Model card text for the base model
      ├─ model_card_data.yaml         # Model card data for the base model
      ├─ python_env.yaml              # Dependencies to create Python virtual environment
      └─ requirements.txt             # Pip requirements for model inference


```

#### Model Metadata[​](#model-metadata "Direct link to Model Metadata")

In the MLModel file, you can see the many detailed metadata are saved about the PEFT and base model. Here is an excerpt of the MLModel file (some fields are omitted for simplicity)

text

```text
flavors:
  transformers:
    peft_adaptor: peft                                 # Points the location of the saved PEFT model
    pipeline_model_type: MistralForCausalLM            # The base model implementation
    source_model_name: mistralai/Mistral-7B-v0.1.      # Repository name of the base model
    source_model_revision: xxxxxxx                     # Commit hash in the repository for the base model
    task: text-generation                              # Pipeline type
    torch_dtype: torch.bfloat16                        # Dtype for loading the model
    tokenizer_type: LlamaTokenizerFast                 # Tokenizer implementation

# Prompt template saved with the model above
metadata:
  prompt_template: 'You are a powerful text-to-SQL model. Given the SQL tables and
    natural language question, your job is to write SQL query that answers the question.


    {prompt}


    ### Response:

    '
# Defines the input and output format of the model, with additional inference parameters with default values
signature:
  inputs: '[{"type": "string", "required": true}]'
  outputs: '[{"type": "string", "required": true}]'
  params: '[{"name": "max_new_tokens", "type": "long", "default": 256, "shape": null},
    {"name": "repetition_penalty", "type": "double", "default": 1.15, "shape": null},
    {"name": "return_full_text", "type": "boolean", "default": false, "shape": null}]'

```

## 7. Load the Saved PEFT Model from MLflow[​](#7-load-the-saved-peft-model-from-mlflow "Direct link to 7. Load the Saved PEFT Model from MLflow")

Finally, let's load the model logged in MLflow and evaluate its performance as a text-to-SQL generator. There are two ways to load a Transformer model in MLflow:

1. Use [mlflow.transformers.load\_model()](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html#mlflow.transformers.load_model). This method returns a native Transformers pipeline instance.
2. Use [mlflow.pyfunc.load\_model()](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model). This method returns an MLflow's PythonModel instance that wraps the Transformers pipeline, offering additional features over the native pipeline, such as (1) a unified `predict()` API for inference, (2) model signature enforcement, and (3) automatically applying a prompt template and default parameters if saved. Please note that not all the Transformer pipelines are supported for pyfunc loading, refer to the [MLflow documentation](https://mlflow.org/docs/latest/ml/deep-learning/transformers/guide/index.html#supported-transformers-pipeline-types-for-pyfunc) for the full list of supported pipeline types.

The first option is preferable if you wish to use the model via the native Transformers interface. The second option offers a simplified and unified interface across different model types and is particularly useful for model testing before production deployment. In the following code, we will use the [mlflow.pyfunc.load\_model()](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model) to show how it applies the prompt template and the default inference parameters defined above.

**NOTE**: Invoking `load_model()` loads a new model instance onto your GPU, which may exceed GPU memory limits and trigger an Out Of Memory (OOM) error, or cause the Transformers library to attempt to offload parts of the model to other devices or disk. This offloading can lead to issues, such as a "ValueError: We need an `offload_dir` to dispatch this model according to this `decide_map`." If you encounter this error, consider restarting the Python Kernel and loading the model again.

**CAUTION**: Restarting the Python Kernel will erase all intermediate states and variables from the above cells. Ensure that the trained PEFT model is properly logged in MLflow before restarting.

python

```python
# You can find the ID of run in the Run detail page on MLflow UI
mlflow_model = mlflow.pyfunc.load_model("runs:/YOUR_RUN_ID/model")

```

python

```python
# We only input table and question, since system prompt is adeed in the prompt template.
test_prompt = """
### Table:
CREATE TABLE table_name_50 (venue VARCHAR, away_team VARCHAR)

### Question:
When Essendon played away; where did they play?
"""

# Inference parameters like max_tokens_length are set to default values specified in the Model Signature
generated_query = mlflow_model.predict(test_prompt)[0]
display_table({"prompt": test_prompt, "generated_query": generated_query})

```

|   | prompt                                                                                                                                                                     | generated\_query                                                |
| - | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 0 | <br />### Table:<br />CREATE TABLE table\_name\_50 (venue VARCHAR, away\_team VARCHAR)<br /><br />### Question:<br />When Essendon played away; where did they play?<br /> | SELECT venue FROM table\_name\_50 WHERE away\_team = "essendon" |

Perfect!! The fine-tuned model now generates the SQL query properly. As you can see in the code and result above, the system prompt and default inference parameters are applied automatically, so we don't have to pass it to the loaded model. This is super powerful when you want to deploy multiple models (or update an existing model) with different the system prompt or parameters, because you don't have to edit client's implementation as they are abstracted behind the MLflow model :)

## Conclusion[​](#conclusion "Direct link to Conclusion")

In this tutorial, you learned how to fine-tune a large language model with QLoRA for text-to-SQL task using PEFT. You also learned the key role of MLflow in the LLM fine-tuning process, which tracks parameters and metrics during the fine-tuning, and manage models and other assets.

### What's Next?[​](#whats-next "Direct link to What's Next?")

* [Evaluate a Hugging Face LLM with MLflow](https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/huggingface-evaluation.html) - Model evaluation is a critical steps in the model development. Checkout this guidance to learn how to evaluate LLMs efficiently with MLflow including LLM-as-a-judge.
* [Deploy MLflow Model to Production](https://mlflow.org/docs/latest/ml/deployment/index.html) - MLflow model stores rich metadata and provides unified interface for prediction, which streamline the easy deployment process. Learn how to deploy your fine-tuned models to various target such as AWS SageMaker, Azure ML, Kubernetes, Databricks Model Serving, with detailed guidance and hands-on notebooks.
* [MLflow Transformers Flavor Documentation](https://mlflow.org/docs/latest/ml/deep-learning/transformers/index.html) - Learn more about MLflow and Transformers integration and continue on more tutorials.
* [Large Language Models in MLflow](https://mlflow.org/docs/latest/llms/index.html) - MLflow provides more LLM-related features and integrates to many other libraries such as OpenAI and Langchain.
