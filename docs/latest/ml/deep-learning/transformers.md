# MLflow Transformers Flavor

## Introduction[​](#introduction "Direct link to Introduction")

**Transformers** by 🤗 [Hugging Face](https://huggingface.co/docs/transformers/index) represents a cornerstone in the realm of machine learning, offering state-of-the-art capabilities for a multitude of frameworks including [PyTorch](https://pytorch.org), [TensorFlow](https://www.tensorflow.org), and [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html). This library has become the de facto standard for natural language processing (NLP) and audio transcription processing. It also provides a compelling and advanced set of options for computer vision and multimodal AI tasks. Transformers achieves all of this by providing pre-trained models and accessible high-level APIs that are not only powerful but also versatile and easy to implement.

For instance, one of the cornerstones of the simplicity of the transformers library is the [pipeline API](https://huggingface.co/transformers/main_classes/pipelines.html), an encapsulation of the most common NLP tasks into a single API call. This API allows users to perform a variety of tasks based on the specified task without having to worry about the underlying model or the preprocessing steps.

![Transformers Pipeline Architecture](/mlflow-website/docs/latest/assets/images/transformers-pipeline-architecture-7a4ac0c60f92b89ff3b0252c1789fe59.png)

Transformers Pipeline Architecture for the Whisper Model

The integration of the Transformers library with MLflow enhances the management of machine learning workflows, from experiment tracking to model deployment. This combination offers a robust and efficient pathway for incorporating advanced NLP and AI capabilities into your applications.

**Key Features of the Transformers Library**:

* **Access to Pre-trained Models**: A vast collection of [pre-trained models](https://huggingface.co/models) for various tasks, minimizing training time and resources.
* **Task Versatility**: Support for multiple modalities including text, image, and speech processing tasks.
* **Framework Interoperability**: Compatibility with PyTorch, TensorFlow, JAX, ONNX, and TorchScript.
* **Community Support**: An active community for collaboration and support, accessible via forums and the Hugging Face Hub.

**MLflow's Transformers Flavor**:

MLflow supports the use of the Transformers package by providing:

* **Simplified Experiment Tracking**: Efficient logging of parameters, metrics, and models during the [fine-tuning process](https://huggingface.co/docs/transformers/main_classes/trainer).
* **Effortless Model Deployment**: Streamlined deployment to various production environments.
* **Library Integration**: Integration with HuggingFace libraries like [Accelerate](https://huggingface.co/docs/accelerate/index), [PEFT](https://huggingface.co/docs/peft/en/index) for model optimization.
* **Prompt Management**: [Save prompt templates](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#saving-prompt-templates-with-transformer-pipelines) with transformers pipelines to optimize inference with less boilerplate.

**Example Use Case:**

For an illustration of fine-tuning a model and logging the results with MLflow, refer to the [fine-tuning tutorials](#transformers-finetuning-tutorials). These tutorial demonstrate the process of fine-tuning a pretrained foundational model into the application-specific model such as a spam classifier, SQL generator. MLflow plays a pivotal role in tracking the fine-tuning process, including datasets, hyperparameters, performance metrics, and the final model artifacts. The image below shows the result of the tutorial within the MLflow UI.

![Fine-tuning a Transformers Model with MLflow](/mlflow-website/docs/latest/assets/images/transformers-fine-tuning-546c747f338376a20679ae21511fc07e.png)

Fine-tuning a Transformers Model with MLflow

### Deployment Made Easy[​](#deployment-made-easy "Direct link to Deployment Made Easy")

Once a model is trained, it needs to be [deployed for inference](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#example-of-loading-a-transformers-model-as-a-python-function). MLflow's integration with Transformers simplifies this by providing functions such as [`mlflow.transformers.load_model()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.transformers.html#mlflow.transformers.load_model) and [`mlflow.pyfunc.load_model()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model), which allow for easy model serving. As part of the feature support for enhanced inference with transformers, MLflow provides mechanisms to enable the use of [inference arguments](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#scalability-for-inference) that can reduce the computational overhead and lower the memory requirements for deployment.

## Getting Started with the MLflow Transformers Flavor - Tutorials and Guides[​](#getting-started-with-the-mlflow-transformers-flavor---tutorials-and-guides "Direct link to Getting Started with the MLflow Transformers Flavor - Tutorials and Guides")

Below, you will find a number of guides that focus on different use cases using *transformers* that leverage MLflow's APIs for tracking and inference capabilities.

### Introductory Quickstart to using Transformers with MLflow[​](#introductory-quickstart-to-using-transformers-with-mlflow "Direct link to Introductory Quickstart to using Transformers with MLflow")

If this is your first exposure to transformers or use transformers extensively but are new to MLflow, this is a great place to start.

[Quickstart: Text Generation with Transformers](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/text-generation/text-generation.md)

[Learn how to leverage the transformers integration with MLflow in this **introductory quickstart**.](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/text-generation/text-generation.md)

### Transformers Fine-Tuning Tutorials with MLflow[​](#transformers-finetuning-tutorials "Direct link to Transformers Fine-Tuning Tutorials with MLflow")

Fine-tuning a model is a common task in machine learning workflows. These tutorials are designed to showcase how to fine-tune a model using the transformers library with harnessing MLflow's APIs for tracking experiment configurations and results.

[Fine tuning a transformers Foundation Model](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-fine-tuning.md)

[Learn how to fine-tune a transformers model using MLflow to keep track of the training process and to log a use-case-specific tuned pipeline.](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-fine-tuning.md)

[Fine tuning LLMs efficiently using PEFT and MLflow](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft.md)

[Learn how to fine-tune a large foundational models with significantly reduced memory usage using PEFT (QLoRA) and MLflow.](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft.md)

### Use Case Tutorials for Transformers with MLflow[​](#use-case-tutorials-for-transformers-with-mlflow "Direct link to Use Case Tutorials for Transformers with MLflow")

Interested in learning about how to leverage transformers for tasks other than basic text generation? Want to learn more about the breadth of problems that you can solve with transformers and MLflow?

These more advanced tutorials are designed to showcase different applications of the transformers model architecture and how to leverage MLflow to track and deploy these models.

[Audio Transcription with Transformers](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/audio-transcription/whisper.md)

[Learn how to leverage the Whisper Model with MLflow to generate accurate audio transcriptions.](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/audio-transcription/whisper.md)

[Translation with Transformers](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/translation/component-translation.md)

[Learn about the options for saving and loading transformers models in MLflow for customization of your workflows with a fun translation example!](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/translation/component-translation.md)

[Chat with Transformers](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/conversational-model.md)

[Learn the basics of stateful chat Conversational Pipelines with Transformers and MLflow.](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/conversational-model.md)

[Building and Serving an OpenAI-Compatible Chatbot](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/pyfunc-chat-model.md)

[Learn how to build an OpenAI-compatible chatbot using a local Transformers model and MLflow, and serve it with minimal configuration.](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/conversational/pyfunc-chat-model.md)

[Prompt templating with Transformers Pipelines](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/prompt-templating/prompt-templating.md)

[Learn how to set prompt templates on Transformers Pipelines to optimize your LLM's outputs, and simplify the end-user experience.](/mlflow-website/docs/latest/ml/deep-learning/transformers/tutorials/prompt-templating/prompt-templating.md)

## Important Details to be aware of with the transformers flavor[​](#important-details-to-be-aware-of-with-the-transformers-flavor "Direct link to Important Details to be aware of with the transformers flavor")

When working with the transformers flavor in MLflow, there are several important considerations to keep in mind:

* **PyFunc Limitations**: Not all output from a Transformers pipeline may be captured when using the python\_function flavor. For example, if additional references or scores are required from the output, the native implementation should be used instead. Also not all the pipeline types are supported for pyfunc. Please refer to [Loading a Transformers Model as a Python Function](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#loading-a-transformers-model-as-a-python-function) for the supported pipeline types and their input and output format.
* **Supported Pipeline Types**: Not all Transformers pipeline types are currently supported for use with the python\_function flavor. In particular, new model architectures may not be supported until the transformers library has a designated pipeline type in its supported pipeline implementations.
* **Input and Output Types**: The input and output types for the python\_function implementation may differ from those expected from the native pipeline. Users need to ensure compatibility with their data processing workflows.
* **Model Configuration**: When saving or logging models, the model\_config can be used to set certain parameters. However, if both model\_config and a ModelSignature with parameters are saved, the default parameters in ModelSignature will override those in model\_config.
* **Audio and Vision Models**: Audio and text-based large language models are supported for use with pyfunc, while other types like computer vision and multi-modal models are only supported for native type loading.
* **Prompt Templates**: Prompt templating is currently supported for a few pipeline types. For a full list of supported pipelines, and more information about the feature, see [this link](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#saving-prompt-templates-with-transformer-pipelines).

## Logging Large Models[​](#logging-large-models "Direct link to Logging Large Models")

By default, MLflow consumes certain memory footprint and storage space for logging models. This can be a concern when working with large foundational models with billions of parameters. To address this, MLflow provides a few optimization techniques to reduce resource consumption during logging and speed up the logging process. Please refer to the [Working with Large Models in MLflow Transformers flavor](/mlflow-website/docs/latest/ml/deep-learning/transformers/large-models.md) guide to learn more about these tips.

## Working with `tasks` for Transformer Pipelines[​](#working-with-tasks-for-transformer-pipelines "Direct link to working-with-tasks-for-transformer-pipelines")

In MLflow Transformers flavor, `task` plays a crucial role in determining the input and output format of the model. Please refer to the [Tasks in MLflow Transformers](/mlflow-website/docs/latest/ml/deep-learning/transformers/task.md) guide on how to use the native Transformers task types, and leverage the advanced tasks such as `llm/v1/chat` and `llm/v1/completions` for OpenAI-compatible inference.

## [Detailed Documentation](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md)[​](#detailed-documentation "Direct link to detailed-documentation")

To learn more about the nuances of the *transformers* flavor in MLflow, delve into [the comprehensive guide](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md), which covers:

* [Pipelines vs. Component Logging](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#pipelines-vs-component-logging): Explore the different approaches for saving model components or complete pipelines and understand the nuances of loading these models for various use cases.
* [Transformers Model as a Python Function](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#loading-a-transformers-model-as-a-python-function) : Familiarize yourself with the various `transformers` pipeline types compatible with the pyfunc model flavor. Understand the standardization of input and output formats in the pyfunc model implementation for the flavor, ensuring seamless integration with JSON and Pandas DataFrames.
* [Prompt Template](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#saving-prompt-templates-with-transformer-pipelines): Learn how to save a prompt template with transformers pipelines to optimize inference with less boilerplate.
* [Model Config and Model Signature Params for Inference](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#using-model-config-and-model-signature-params-for-inference): Learn how to leverage `model_config` and `ModelSignature` for flexible and customized model loading and inference.
* [Automatic Metadata and ModelCard Logging](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#automatic-metadata-and-modelcard-logging): Discover the automatic logging features for model cards and other metadata, enhancing model documentation and transparency.
* [Model Signature Inference](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#automatic-signature-inference) : Learn about MLflow's capability within the `transformers` flavor to automatically infer and attach model signatures, facilitating easier model deployment.
* [Overriding Pytorch dtype](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#scalability-for-inference) : Gain insights into optimizing `transformers` models for inference, focusing on memory optimization and data type configurations.
* [Input Data Types for Audio Pipelines](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#input-data-types-for-audio-pipelines): Understand the specific requirements for handling audio data in transformers pipelines, including the handling of different input types like str, bytes, and np.ndarray.
* [PEFT Models in MLflow Transformers flavor](/mlflow-website/docs/latest/ml/deep-learning/transformers/guide.md#peft-models-in-mlflow-transformers-flavor): PEFT (Parameter-Efficient Fine-Tuning) is natively supported in MLflow, enabling various optimization techniques like LoRA, QLoRA, and more for reducing fine-tuning cost significantly. Check out the guide and tutorials to learn more about how to leverage PEFT with MLflow.

## Learn more about Transformers[​](#learn-more-about-transformers "Direct link to Learn more about Transformers")

Interested in learning more about how to leverage transformers for your machine learning workflows?

🤗 Hugging Face has a fantastic NLP course. Check it out and see how to leverage [Transformers, Datasets, Tokenizers, and Accelerate](https://huggingface.co/learn/nlp-course/chapter1/1).
