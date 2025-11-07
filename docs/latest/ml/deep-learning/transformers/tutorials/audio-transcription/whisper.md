# Introduction to MLflow and OpenAI's Whisper

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/transformers/tutorials/audio-transcription/whisper.ipynb)

Discover the integration of [OpenAI's Whisper](https://huggingface.co/openai), an [ASR system](https://en.wikipedia.org/wiki/Speech_recognition), with MLflow in this tutorial.

### What You Will Learn in This Tutorial[​](#what-you-will-learn-in-this-tutorial "Direct link to What You Will Learn in This Tutorial")

* Establish an audio transcription **pipeline** using the Whisper model.
* **Log** and manage Whisper models with MLflow.
* Infer and understand Whisper model **signatures**.
* **Load** and interact with Whisper models stored in MLflow.
* Utilize MLflow's **pyfunc** for Whisper model serving and transcription tasks.

#### What is Whisper?[​](#what-is-whisper "Direct link to What is Whisper?")

Whisper, developed by OpenAI, is a versatile ASR model trained for high-accuracy speech-to-text conversion. It stands out due to its training on diverse accents and environments, available via the Transformers library for easy use.

#### Why MLflow with Whisper?[​](#why-mlflow-with-whisper "Direct link to Why MLflow with Whisper?")

Integrating MLflow with Whisper enhances ASR model management:

* **Experiment Tracking**: Facilitates tracking of model configurations and performance for optimal results.
* **Model Management**: Centralizes different versions of Whisper models, enhancing organization and accessibility.
* **Reproducibility**: Ensures consistency in transcriptions by tracking all components required for reproducing model behavior.
* **Deployment**: Streamlines the deployment of Whisper models in various production settings, ensuring efficient application.

Interested in learning more about Whisper? To read more about the significant breakthroughs in transcription capabilities that Whisper brought to the field of ASR, you can [read the white paper](https://arxiv.org/abs/2212.04356) and see more about the active development and [read more about the progress](https://openai.com/research/whisper) at OpenAI's research website.

Ready to enhance your speech-to-text capabilities? Let's explore automatic speech recognition using MLflow and Whisper!

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

### Setting Up the Environment and Acquiring Audio Data[​](#setting-up-the-environment-and-acquiring-audio-data "Direct link to Setting Up the Environment and Acquiring Audio Data")

Initial steps for transcription using [Whisper](https://github.com/openai/whisper): acquiring [audio](https://www.nasa.gov/audio-and-ringtones/) and setting up MLflow.

Before diving into the audio transcription process with OpenAI's Whisper, there are a few preparatory steps to ensure everything is in place for a smooth and effective transcription experience.

#### Audio Acquisition[​](#audio-acquisition "Direct link to Audio Acquisition")

The first step is to acquire an audio file to work with. For this tutorial, we use a publicly available audio file from NASA. This sample audio provides a practical example to demonstrate Whisper's transcription capabilities.

#### Model and Pipeline Initialization[​](#model-and-pipeline-initialization "Direct link to Model and Pipeline Initialization")

We load the Whisper model, along with its tokenizer and feature extractor, from the Transformers library. These components are essential for processing the audio data and converting it into a format that the Whisper model can understand and transcribe. Next, we create a transcription pipeline using the Whisper model. This pipeline simplifies the process of feeding audio data into the model and obtaining the transcription.

#### MLflow Environment Setup[​](#mlflow-environment-setup "Direct link to MLflow Environment Setup")

In addition to the model and audio data setup, we initialize our MLflow environment. MLflow is used to track and manage our experiments, offering an organized way to document the transcription process and results.

The following code block covers these initial setup steps, providing the foundation for our audio transcription task with the Whisper model.

python

```
import requests
import transformers

import mlflow

# Acquire an audio file that is in the public domain
resp = requests.get(
  "https://www.nasa.gov/wp-content/uploads/2015/01/590325main_ringtone_kennedy_WeChoose.mp3"
)
resp.raise_for_status()
audio = resp.content

# Set the task that our pipeline implementation will be using
task = "automatic-speech-recognition"

# Define the model instance
architecture = "openai/whisper-large-v3"

# Load the components and necessary configuration for Whisper ASR from the Hugging Face Hub
model = transformers.WhisperForConditionalGeneration.from_pretrained(architecture)
tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture)
feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture)
model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]

# Instantiate our pipeline for ASR using the Whisper model
audio_transcription_pipeline = transformers.pipeline(
  task=task, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
)
```

```
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
```

### Formatting the Transcription Output[​](#formatting-the-transcription-output "Direct link to Formatting the Transcription Output")

In this section, we introduce a utility function that is used solely for the purpose of enhancing the readability of the transcription output within this Jupyter notebook demo. It is important to note that this function is designed for demonstration purposes and should not be included in production code or used for any other purpose beyond this tutorial.

The `format_transcription` function takes a long string of transcribed text and formats it by splitting it into sentences and inserting newline characters. This makes the output easier to read when printed in the notebook environment.

python

```
def format_transcription(transcription):
  """
  Function for formatting a long string by splitting into sentences and adding newlines.
  """
  # Split the transcription into sentences, ensuring we don't split on abbreviations or initials
  sentences = [
      sentence.strip() + ("." if not sentence.endswith(".") else "")
      for sentence in transcription.split(". ")
      if sentence
  ]

  # Join the sentences with a newline character
  return "
".join(sentences)
```

### Executing the Transcription Pipeline[​](#executing-the-transcription-pipeline "Direct link to Executing the Transcription Pipeline")

Perform audio transcription using the Whisper pipeline and review the output.

After setting up the Whisper model and audio transcription pipeline, our next step is to process an audio file to extract its transcription. This part of the tutorial is crucial as it demonstrates the practical application of the Whisper model in converting spoken language into written text.

#### Transcription Process[​](#transcription-process "Direct link to Transcription Process")

The code block below feeds an audio file into the pipeline, which then produces the transcription. The `format_transcription` function, defined earlier, enhances readability by formatting the output with sentence splits and newline characters.

#### Importance of Pre-Save Testing[​](#importance-of-pre-save-testing "Direct link to Importance of Pre-Save Testing")

Testing the transcription pipeline before saving the model in MLflow is vital. This step verifies that the model works as expected, ensuring accuracy and reliability. Such validation avoids issues post-deployment and confirms that the model performs consistently with the training data it was exposed to. It also provides a benchmark to compare against the output after the model is loaded back from MLflow, ensuring consistency in performance.

Execute the following code to transcribe the audio and assess the quality and accuracy of the transcription provided by the Whisper model.

python

```
# Verify that our pipeline is capable of processing an audio file and transcribing it
transcription = audio_transcription_pipeline(audio)

print(format_transcription(transcription["text"]))
```

```
We choose to go to the moon in this decade and do the other things.
Not because they are easy, but because they are hard.
3, 2, 1, 0.
All engines running.
Liftoff.
We have a liftoff.
32 minutes past the hour.
Liftoff on Apollo 11.
```

### Model Signature and Configuration[​](#model-signature-and-configuration "Direct link to Model Signature and Configuration")

Generate a model signature for Whisper to understand its input and output data requirements.

The model signature is critical for defining the schema for the Whisper model's inputs and outputs, clarifying the data types and structures expected. This step ensures the model processes inputs correctly and outputs structured data.

#### Handling Different Audio Formats[​](#handling-different-audio-formats "Direct link to Handling Different Audio Formats")

While the default signature covers binary audio data, the `transformers` flavor accommodates multiple formats, including numpy arrays and URL-based inputs. This flexibility allows Whisper to transcribe from various sources, although URL-based transcription isn't demonstrated here.

#### Model Configuration[​](#model-configuration "Direct link to Model Configuration")

Setting the model configuration involves parameters like *chunk* and *stride* lengths for audio processing. These settings are adjustable to suit different transcription needs, enhancing Whisper's performance for specific scenarios.

Run the next code block to infer the model's signature and configure key parameters, aligning Whisper's functionality with your project's requirements.

python

```
# Specify parameters and their defaults that we would like to be exposed for manipulation during inference time
model_config = {
  "chunk_length_s": 20,
  "stride_length_s": [5, 3],
}

# Define the model signature by using the input and output of our pipeline, as well as specifying our inference parameters that will allow for those parameters to
# be overridden at inference time.
signature = mlflow.models.infer_signature(
  audio,
  mlflow.transformers.generate_signature_output(audio_transcription_pipeline, audio),
  params=model_config,
)

# Visualize the signature
signature
```

```
inputs: 
[binary]
outputs: 
[string]
params: 
['chunk_length_s': long (default: 20), 'stride_length_s': long (default: [5, 3]) (shape: (-1,))]
```

### Creating an experiment[​](#creating-an-experiment "Direct link to Creating an experiment")

We create a new MLflow Experiment so that the run we're going to log our model to does not log to the default experiment and instead has its own contextually relevant entry.

python

```
# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

# mlflow.set_tracking_uri("http://127.0.0.1:8080")

mlflow.set_experiment("Whisper Transcription ASR")
```

```
<Experiment: artifact_location='file:///Users/benjamin.wilson/repos/mlflow-fork/mlflow/docs/source/llms/transformers/tutorials/audio-transcription/mlruns/864092483920291025', creation_time=1701294423466, experiment_id='864092483920291025', last_update_time=1701294423466, lifecycle_stage='active', name='Whisper Transcription ASR', tags={}>
```

### Logging the Model with MLflow[​](#logging-the-model-with-mlflow "Direct link to Logging the Model with MLflow")

Learn how to log the Whisper model and its configurations with MLflow.

Logging the Whisper model in MLflow is a critical step for capturing essential information for model reproduction, sharing, and deployment. This process involves:

#### Key Components of Model Logging[​](#key-components-of-model-logging "Direct link to Key Components of Model Logging")

* **Model Information**: Includes the model, its signature, and an input example.
* **Model Configuration**: Any specific parameters set for the model, like *chunk length* or *stride length*.

#### Using MLflow's `log_model` Function[​](#using-mlflows-log_model-function "Direct link to using-mlflows-log_model-function")

This function is utilized within an MLflow run to log the model and its configurations. It ensures that all necessary components for model usage are recorded.

Executing the code in the next cell will log the Whisper model in the current MLflow experiment. This includes storing the model in a specified artifact path and documenting the default configurations that will be applied during inference.

python

```
# Log the pipeline
with mlflow.start_run():
  model_info = mlflow.transformers.log_model(
      transformers_model=audio_transcription_pipeline,
      name="whisper_transcriber",
      signature=signature,
      input_example=audio,
      model_config=model_config,
      # Since MLflow 2.11.0, you can save the model in 'reference-only' mode to reduce storage usage by not saving
      # the base model weights but only the reference to the HuggingFace model hub. To enable this, uncomment the
      # following line:
      # save_pretrained=False,
  )
```

### Loading and Using the Model Pipeline[​](#loading-and-using-the-model-pipeline "Direct link to Loading and Using the Model Pipeline")

Explore how to load and use the Whisper model pipeline from MLflow.

After logging the Whisper model in MLflow, the next crucial step is to load and use it for inference. This process ensures that our logged model operates as intended and can be effectively used for tasks like audio transcription.

#### Loading the Model[​](#loading-the-model "Direct link to Loading the Model")

The model is loaded in its native format using MLflow's `load_model` function. This step verifies that the model can be retrieved and used seamlessly after being logged in MLflow.

#### Using the Loaded Model[​](#using-the-loaded-model "Direct link to Using the Loaded Model")

Once loaded, the model is ready for inference. We demonstrate this by passing an MP3 audio file to the model and obtaining its transcription. This test is a practical demonstration of the model's capabilities post-logging.

This step is a form of validation before moving to more complex deployment scenarios. Ensuring that the model functions correctly in its native format helps in troubleshooting and streamlines the deployment process, especially for large and complex models like Whisper.

python

```
# Load the pipeline in its native format
loaded_transcriber = mlflow.transformers.load_model(model_uri=model_info.model_uri)

# Perform transcription with the native pipeline implementation
transcription = loaded_transcriber(audio)

print(f"
Whisper native output transcription:
{format_transcription(transcription['text'])}")
```

```
2023/11/30 12:51:43 INFO mlflow.transformers: 'runs:/f7503a09d20f4fb481544968b5ed28dd/whisper_transcriber' resolved as 'file:///Users/benjamin.wilson/repos/mlflow-fork/mlflow/docs/source/llms/transformers/tutorials/audio-transcription/mlruns/864092483920291025/f7503a09d20f4fb481544968b5ed28dd/artifacts/whisper_transcriber'
```

```
Loading checkpoint shards:   0%|          | 0/13 [00:00<?, ?it/s]
```

```
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
```

```

Whisper native output transcription:
We choose to go to the moon in this decade and do the other things.
Not because they are easy, but because they are hard.
3, 2, 1, 0.
All engines running.
Liftoff.
We have a liftoff.
32 minutes past the hour.
Liftoff on Apollo 11.
```

### Using the Pyfunc Flavor for Inference[​](#using-the-pyfunc-flavor-for-inference "Direct link to Using the Pyfunc Flavor for Inference")

Learn how MLflow's `pyfunc` flavor facilitates flexible model deployment.

MLflow's `pyfunc` flavor provides a generic interface for model inference, offering flexibility across various machine learning frameworks and deployment environments. This feature is beneficial for deploying models where the original framework may not be available, or a more adaptable interface is required.

#### Loading and Predicting with Pyfunc[​](#loading-and-predicting-with-pyfunc "Direct link to Loading and Predicting with Pyfunc")

The code below illustrates how to load the Whisper model as a `pyfunc` and use it for prediction. This method highlights MLflow's capability to adapt and deploy models in diverse scenarios.

#### Output Format Considerations[​](#output-format-considerations "Direct link to Output Format Considerations")

Note the difference in the output format when using `pyfunc` compared to the native format. The `pyfunc` output conforms to standard pyfunc output signatures, typically represented as a `List[str]` type, aligning with broader MLflow standards for model outputs.

python

```
# Load the saved transcription pipeline as a generic python function
pyfunc_transcriber = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

# Ensure that the pyfunc wrapper is capable of transcribing passed-in audio
pyfunc_transcription = pyfunc_transcriber.predict([audio])

# Note: the pyfunc return type if `return_timestamps` is set is a JSON encoded string.
print(f"
Pyfunc output transcription:
{format_transcription(pyfunc_transcription[0])}")
```

```
Loading checkpoint shards:   0%|          | 0/13 [00:00<?, ?it/s]
```

```
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
2023/11/30 12:52:02 WARNING mlflow.transformers: params provided to the `predict` method will override the inference configuration saved with the model. If the params provided are not valid for the pipeline, MlflowException will be raised.
```

```

Pyfunc output transcription:
We choose to go to the moon in this decade and do the other things.
Not because they are easy, but because they are hard.
3, 2, 1, 0.
All engines running.
Liftoff.
We have a liftoff.
32 minutes past the hour.
Liftoff on Apollo 11.
```

### Tutorial Roundup[​](#tutorial-roundup "Direct link to Tutorial Roundup")

Throughout this tutorial, we've explored how to:

* Set up an audio transcription pipeline using the OpenAI Whisper model.
* Format and prepare audio data for transcription.
* Log, load, and use the model with MLflow, leveraging both the native and pyfunc flavors for inference.
* Format the output for readability and practical use in a Jupyter Notebook environment.

We've seen the benefits of using MLflow for managing the machine learning lifecycle, including experiment tracking, model versioning, reproducibility, and deployment. By integrating MLflow with the Transformers library, we've streamlined the process of working with state-of-the-art NLP models, making it easier to track, manage, and deploy cutting-edge NLP applications.
