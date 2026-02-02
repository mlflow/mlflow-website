# MLflow Transformers Flavor

The MLflow Transformers flavor provides native integration with the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library, supporting model logging, loading, and inference for NLP, audio, vision, and multimodal tasks.

## Key Features[​](#key-features "Direct link to Key Features")

* **Pipeline and Component Logging**: Save complete pipelines or individual model components
* **PyFunc Integration**: Deploy models with standardized inference interfaces
* **PEFT Support**: Native support for parameter-efficient fine-tuning (LoRA, QLoRA, etc.)
* **Prompt Templates**: Save and manage prompt templates with pipelines
* **Automatic Metadata Logging**: Model cards and metadata logged automatically
* **Flexible Inference Configuration**: Customize model behavior via `model_config` and signature parameters

## Installation[​](#installation "Direct link to Installation")

bash

```bash
pip install mlflow transformers

```

## Basic Usage[​](#basic-usage "Direct link to Basic Usage")

### Logging a Pipeline[​](#logging-a-pipeline "Direct link to Logging a Pipeline")

python

```python
import mlflow
from transformers import pipeline

# Create a text generation pipeline
text_gen = pipeline("text-generation", model="gpt2")

# Log the pipeline
with mlflow.start_run():
    mlflow.transformers.log_model(
        transformers_model=text_gen,
        name="model",
    )

```

### Loading and Inference[​](#loading-and-inference "Direct link to Loading and Inference")

python

```python
# Load as native transformers
model = mlflow.transformers.load_model("runs:/<run_id>/model")
result = model("Hello, how are you?")

# Load as PyFunc
pyfunc_model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
result = pyfunc_model.predict("Hello, how are you?")

```

## Autologging with HuggingFace Trainer[​](#autologging-with-huggingface-trainer "Direct link to Autologging with HuggingFace Trainer")

When using the HuggingFace `Trainer` class for fine-tuning, you can enable automatic logging to MLflow by setting `report_to="mlflow"` in the `TrainingArguments`:

python

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    report_to="mlflow",  # Enable MLflow logging
    # ... other training arguments
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # ... other trainer arguments
)

trainer.train()

```

This automatically logs training metrics, hyperparameters, and model checkpoints to your active MLflow run.

## Tutorials[​](#tutorials "Direct link to Tutorials")

### Quickstart[​](#quickstart "Direct link to Quickstart")

[Text Generation with Transformers](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/text-generation/text-generation.md)

[Introductory quickstart for using Transformers with MLflow](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/text-generation/text-generation.md)

### Fine-Tuning[​](#fine-tuning "Direct link to Fine-Tuning")

[Fine-tuning a Foundation Model](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-fine-tuning.md)

[Track fine-tuning experiments and log optimized models](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-fine-tuning.md)[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft.md)

[Fine-tuning with PEFT](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft.md)

[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft.md)

[Memory-efficient fine-tuning using PEFT (QLoRA) techniques](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft.md)

### Advanced Use Cases[​](#advanced-use-cases "Direct link to Advanced Use Cases")

[Audio Transcription](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/audio-transcription/whisper.md)

[Use Whisper models for audio transcription](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/audio-transcription/whisper.md)[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/translation/component-translation.md)

[Translation](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/translation/component-translation.md)

[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/translation/component-translation.md)

[Component-based model logging for translation tasks](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/translation/component-translation.md)[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/conversational-model.md)

[Conversational Pipelines](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/conversational-model.md)

[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/conversational-model.md)

[Stateful chat with conversational pipelines](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/conversational-model.md)[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/pyfunc-chat-model.md)

[OpenAI-Compatible Chatbot](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/pyfunc-chat-model.md)

[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/pyfunc-chat-model.md)

[Build and serve an OpenAI-compatible chatbot](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/pyfunc-chat-model.md)[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/prompt-templating/prompt-templating.md)

[Prompt Templating](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/prompt-templating/prompt-templating.md)

[](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/prompt-templating/prompt-templating.md)

[Optimize LLM outputs with prompt templates](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/prompt-templating/prompt-templating.md)

## Important Considerations[​](#important-considerations "Direct link to Important Considerations")

### PyFunc Limitations[​](#pyfunc-limitations "Direct link to PyFunc Limitations")

* Not all pipeline types are supported for PyFunc inference
* Some outputs (e.g., additional scores, references) may not be captured
* Audio and text LLMs are supported; vision and multimodal models require native loading
* See the [guide](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#loading-a-transformers-model-as-a-python-function) for supported pipeline types

### Input/Output Types[​](#inputoutput-types "Direct link to Input/Output Types")

Input and output formats for PyFunc may differ from native pipelines. Ensure compatibility with your data processing workflows.

### Model Configuration[​](#model-configuration "Direct link to Model Configuration")

Parameters in `ModelSignature` override those in `model_config` when both are provided.

## Working with Large Models[​](#working-with-large-models "Direct link to Working with Large Models")

For models with billions of parameters, MLflow provides optimization techniques to reduce memory usage and speed up logging. See the [large models guide](/mlflow-website/docs/latest/ml/deep-learning/transformers/large-models.md).

## Tasks[​](#tasks "Direct link to Tasks")

The `task` parameter determines input/output format. MLflow supports native Transformers tasks plus advanced tasks like `llm/v1/chat` and `llm/v1/completions` for OpenAI-compatible inference. See the [tasks guide](/mlflow-website/docs/latest/ml/deep-learning/transformers/task.md).

## Learn More[​](#learn-more "Direct link to Learn More")

### [Detailed Guide](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md)

[Comprehensive documentation covering pipelines, PyFunc, signatures, PEFT, and more](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md)

[Learn more →](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md)

### [Transformers Documentation](https://huggingface.co/docs/transformers/index)

[Official Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)

[Learn more →](https://huggingface.co/docs/transformers/index)
