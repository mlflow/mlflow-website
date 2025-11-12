# Introduction to Sentence Transformers and MLflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/quickstart/sentence-transformers-quickstart.ipynb)

Welcome to our tutorial on leveraging **Sentence Transformers** with **MLflow** for advanced natural language processing and model management.

### Learning Objectives[​](#learning-objectives "Direct link to Learning Objectives")

* Set up a pipeline for sentence embeddings with `sentence-transformers`.
* Log models and configurations using MLflow.
* Understand and apply model signatures in MLflow to `sentence-transformers`.
* Deploy and use models for inference with MLflow's features.

#### What are Sentence Transformers?[​](#what-are-sentence-transformers "Direct link to What are Sentence Transformers?")

Sentence Transformers, an extension of the Hugging Face Transformers library, are designed for generating semantically rich sentence embeddings. They utilize models like BERT and RoBERTa, fine-tuned for tasks such as semantic search and text clustering, producing high-quality sentence-level embeddings.

#### Benefits of Integrating MLflow with Sentence Transformers[​](#benefits-of-integrating-mlflow-with-sentence-transformers "Direct link to Benefits of Integrating MLflow with Sentence Transformers")

Combining MLflow with Sentence Transformers enhances NLP projects by:

* Streamlining experiment management and logging.
* Offering better control over model versions and configurations.
* Ensuring reproducibility of results and model predictions.
* Simplifying the deployment process in production environments.

This integration empowers efficient tracking, management, and deployment of NLP applications.

python

```python
# Disable tokenizers warnings when constructing pipelines
%env TOKENIZERS_PARALLELISM=false

import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

```

### Setting Up the Environment for Sentence Embedding[​](#setting-up-the-environment-for-sentence-embedding "Direct link to Setting Up the Environment for Sentence Embedding")

Begin your journey with Sentence Transformers and MLflow by establishing the core working environment.

#### Key Steps for Initialization[​](#key-steps-for-initialization "Direct link to Key Steps for Initialization")

* Import necessary libraries: `SentenceTransformer` and `mlflow`.
* Initialize the `"all-MiniLM-L6-v2"` Sentence Transformer model.

#### Model Initialization[​](#model-initialization "Direct link to Model Initialization")

The compact and efficient `"all-MiniLM-L6-v2"` model is chosen for its effectiveness in generating meaningful sentence embeddings. Explore more models at the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=sentence-similarity\&sort=trending).

#### Purpose of the Model[​](#purpose-of-the-model "Direct link to Purpose of the Model")

This model excels in transforming sentences into semantically rich embeddings, applicable in various NLP tasks like semantic search and clustering.

python

```python
from sentence_transformers import SentenceTransformer

import mlflow

model = SentenceTransformer("all-MiniLM-L6-v2")

```

### Defining the Model Signature with MLflow[​](#defining-the-model-signature-with-mlflow "Direct link to Defining the Model Signature with MLflow")

Defining the model signature is a crucial step in setting up our Sentence Transformer model for consistent and expected behavior during inference.

#### Steps for Signature Definition[​](#steps-for-signature-definition "Direct link to Steps for Signature Definition")

* **Prepare Example Sentences**: Define example sentences to demonstrate the model's input and output formats.
* **Generate Model Signature**: Use the `mlflow.models.infer_signature` function with the model's input and output to automatically define the signature.

#### Importance of the Model Signature[​](#importance-of-the-model-signature "Direct link to Importance of the Model Signature")

* **Clarity in Data Formats**: Ensures clear documentation of the data types and structures the model expects and produces.
* **Model Deployment and Usage**: Crucial for deploying models to production, ensuring the model receives inputs in the correct format and produces expected outputs.
* **Error Prevention**: Helps in preventing errors during model inference by enforcing consistent data formats.

**NOTE**: The `List[str]` input type is equivalent at inference time to `str`. The MLflow flavor uses a `ColSpec[str]` definition for the input type.

python

```python
example_sentences = ["A sentence to encode.", "Another sentence to encode."]

# Infer the signature of the custom model by providing an input example and the resultant prediction output.
# We're not including any custom inference parameters in this example, but you can include them as a third argument
# to infer_signature(), as you will see in the advanced tutorials for Sentence Transformers.
signature = mlflow.models.infer_signature(
  model_input=example_sentences,
  model_output=model.encode(example_sentences),
)

# Visualize the signature
signature

```

### Creating an experiment[​](#creating-an-experiment "Direct link to Creating an experiment")

We create a new MLflow Experiment so that the run we're going to log our model to does not log to the default experiment and instead has its own contextually relevant entry.

python

```python
# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

# mlflow.set_tracking_uri("http://127.0.0.1:8080")

mlflow.set_experiment("Introduction to Sentence Transformers")

```

### Logging the Sentence Transformer Model with MLflow[​](#logging-the-sentence-transformer-model-with-mlflow "Direct link to Logging the Sentence Transformer Model with MLflow")

Logging the model in MLflow is essential for tracking, version control, and deployment, following the initialization and signature definition of our Sentence Transformer model.

#### Steps for Logging the Model[​](#steps-for-logging-the-model "Direct link to Steps for Logging the Model")

* **Start an MLflow Run**: Initiate a new run with `mlflow.start_run()`, grouping all logging operations.
* **Log the Model**: Use `mlflow.sentence_transformers.log_model` to log the model, providing the model object, artifact path, signature, and an input example.

#### Importance of Model Logging[​](#importance-of-model-logging "Direct link to Importance of Model Logging")

* **Model Management**: Facilitates the model's lifecycle management from training to deployment.
* **Reproducibility and Tracking**: Enables tracking of model versions and ensures reproducibility.
* **Ease of Deployment**: Simplifies deployment by allowing models to be easily deployed for inference.

python

```python
with mlflow.start_run():
  logged_model = mlflow.sentence_transformers.log_model(
      model=model,
      name="sbert_model",
      signature=signature,
      input_example=example_sentences,
  )

```

### Loading the Model and Testing Inference[​](#loading-the-model-and-testing-inference "Direct link to Loading the Model and Testing Inference")

After logging the Sentence Transformer model in MLflow, we demonstrate how to load and test it for real-time inference.

#### Loading the Model as a PyFunc[​](#loading-the-model-as-a-pyfunc "Direct link to Loading the Model as a PyFunc")

* **Why PyFunc**: Load the logged model using `mlflow.pyfunc.load_model` for seamless integration into Python-based services or applications.
* **Model URI**: Use the `logged_model.model_uri` to accurately locate and load the model from MLflow.

#### Conducting Inference Tests[​](#conducting-inference-tests "Direct link to Conducting Inference Tests")

* **Test Sentences**: Define sentences to test the model's embedding generation capabilities.
* **Performing Predictions**: Use the model's `predict` method with test sentences to obtain embeddings.
* **Printing Embedding Lengths**: Verify embedding generation by checking the length of embedding arrays, corresponding to the dimensionality of each sentence representation.

#### Importance of Inference Testing[​](#importance-of-inference-testing "Direct link to Importance of Inference Testing")

* **Model Validation**: Confirm the model's expected behavior and data processing capability upon loading.
* **Deployment Readiness**: Validate the model's readiness for real-time integration into application services.

python

```python
inference_test = ["I enjoy pies of both apple and cherry.", "I prefer cookies."]

# Load our custom model by providing the uri for where the model was logged.
loaded_model_pyfunc = mlflow.pyfunc.load_model(logged_model.model_uri)

# Perform a quick test to ensure that our loaded model generates the correct output
embeddings_test = loaded_model_pyfunc.predict(inference_test)

# Verify that the output is a list of lists of floats (our expected output format)
print(f"The return structure length is: {len(embeddings_test)}")

for i, embedding in enumerate(embeddings_test):
  print(f"The size of embedding {i + 1} is: {len(embeddings_test[i])}")

```

### Displaying Samples of Generated Embeddings[​](#displaying-samples-of-generated-embeddings "Direct link to Displaying Samples of Generated Embeddings")

Examine the content of embeddings to verify their quality and understand the model's output.

#### Inspecting the Embedding Samples[​](#inspecting-the-embedding-samples "Direct link to Inspecting the Embedding Samples")

* **Purpose of Sampling**: Inspect a sample of the entries in each embedding to understand the vector representations generated by the model.
* **Printing Embedding Samples**: Print the first 10 entries of each embedding vector using `embedding[:10]` to get a glimpse into the model's output.

#### Why Sampling is Important[​](#why-sampling-is-important "Direct link to Why Sampling is Important")

* **Quality Check**: Sampling provides a quick way to verify the embeddings' quality and ensures they are meaningful and non-degenerate.
* **Understanding Model Output**: Seeing parts of the embedding vectors offers an intuitive understanding of the model's output, beneficial for debugging and development.

python

```python
for i, embedding in enumerate(embeddings_test):
  print(f"The sample of the first 10 entries in embedding {i + 1} is: {embedding[:10]}")

```

### Native Model Loading in MLflow for Extended Functionality[​](#native-model-loading-in-mlflow-for-extended-functionality "Direct link to Native Model Loading in MLflow for Extended Functionality")

Explore the full range of Sentence Transformer functionalities with MLflow's support for native model loading.

#### Why Support Native Loading?[​](#why-support-native-loading "Direct link to Why Support Native Loading?")

* **Access to Native Functionalities**: Native loading unlocks all the features of the Sentence Transformer model, essential for advanced NLP tasks.
* **Loading the Model Natively**: Use `mlflow.sentence_transformers.load_model` to load the model with its full capabilities, enhancing flexibility and efficiency.

#### Generating Embeddings Using Native Model[​](#generating-embeddings-using-native-model "Direct link to Generating Embeddings Using Native Model")

* **Model Encoding**: Employ the model's native `encode` method to generate embeddings, taking advantage of optimized functionality.
* **Importance of Native Encoding**: Native encoding ensures the utilization of the model's full embedding generation capabilities, suitable for large-scale or complex NLP applications.

python

```python
# Load the saved model as a native Sentence Transformers model (unlike above, where we loaded as a generic python function)
loaded_model_native = mlflow.sentence_transformers.load_model(logged_model.model_uri)

# Use the native model to generate embeddings by calling encode() (unlike for the generic python function which uses the single entrypoint of `predict`)
native_embeddings = loaded_model_native.encode(inference_test)

for i, embedding in enumerate(native_embeddings):
  print(
      f"The sample of the native library encoding call for embedding {i + 1} is: {embedding[:10]}"
  )

```

### Conclusion: Embracing the Power of Sentence Transformers with MLflow[​](#conclusion-embracing-the-power-of-sentence-transformers-with-mlflow "Direct link to Conclusion: Embracing the Power of Sentence Transformers with MLflow")

As we reach the end of our Introduction to Sentence Transformers tutorial, we have successfully navigated the basics of integrating the Sentence Transformers library with MLflow. This foundational knowledge sets the stage for more advanced and specialized applications in the field of Natural Language Processing (NLP).

#### Recap of Key Learnings[​](#recap-of-key-learnings "Direct link to Recap of Key Learnings")

1. **Integration Basics**: We covered the essential steps of loading and logging a Sentence Transformer model using MLflow. This process demonstrated the simplicity and effectiveness of integrating cutting-edge NLP tools within MLflow's ecosystem.

2. **Signature and Inference**: Through the creation of a model signature and the execution of inference tasks, we showcased how to operationalize the Sentence Transformer model, ensuring that it's ready for real-world applications.

3. **Model Loading and Prediction**: We explored two ways of loading the model - as a PyFunc model and using the native Sentence Transformers loading mechanism. This dual approach highlighted the versatility of MLflow in accommodating different model interaction methods.

4. **Embeddings Exploration**: By generating and examining sentence embeddings, we glimpsed the transformative potential of transformer models in capturing semantic information from text.

#### Looking Ahead[​](#looking-ahead "Direct link to Looking Ahead")

* **Expanding Horizons**: While this tutorial focused on the foundational aspects of Sentence Transformers and MLflow, there's a whole world of advanced applications waiting to be explored. From semantic similarity analysis to paraphrase mining, the potential use cases are vast and varied.

* **Continued Learning**: We strongly encourage you to delve into the other tutorials in this series, which dive deeper into more intriguing use cases like similarity analysis, semantic search, and paraphrase mining. These tutorials will provide you with a broader understanding and more practical applications of Sentence Transformers in various NLP tasks.

#### Final Thoughts[​](#final-thoughts "Direct link to Final Thoughts")

The journey into NLP with Sentence Transformers and MLflow is just beginning. With the skills and insights gained from this tutorial, you are well-equipped to explore more complex and exciting applications. The integration of advanced NLP models with MLflow's robust management and deployment capabilities opens up new avenues for innovation and exploration in the field of language understanding and beyond.

Thank you for joining us on this introductory journey, and we look forward to seeing how you apply these tools and concepts in your NLP endeavors!
