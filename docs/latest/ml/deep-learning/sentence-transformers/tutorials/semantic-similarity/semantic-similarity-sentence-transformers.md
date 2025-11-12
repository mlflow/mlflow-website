# Introduction to Advanced Semantic Similarity Analysis with Sentence Transformers and MLflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/semantic-similarity/semantic-similarity-sentence-transformers.ipynb)

Dive into advanced Semantic Similarity Analysis using Sentence Transformers and MLflow in this comprehensive tutorial.

### Learning Objectives[​](#learning-objectives "Direct link to Learning Objectives")

* Configure `sentence-transformers` for semantic similarity analysis.
* Explore custom `PythonModel` implementation in MLflow.
* Log models and manage configurations with MLflow.
* Deploy and apply models for inference using MLflow's features.

#### Unveiling the Power of Sentence Transformers for NLP[​](#unveiling-the-power-of-sentence-transformers-for-nlp "Direct link to Unveiling the Power of Sentence Transformers for NLP")

Sentence Transformers, specialized adaptations of transformer models, excel in producing semantically rich sentence embeddings. Ideal for semantic search and similarity analysis, these models bring a deeper semantic understanding to NLP tasks.

#### MLflow: Pioneering Flexible Model Management and Deployment[​](#mlflow-pioneering-flexible-model-management-and-deployment "Direct link to MLflow: Pioneering Flexible Model Management and Deployment")

MLflow's integration with Sentence Transformers introduces enhanced experiment tracking and flexible model management, crucial for NLP projects. Learn to implement a custom `PythonModel` within MLflow, extending functionalities for unique requirements.

Throughout this tutorial, you'll gain hands-on experience in managing and deploying sophisticated NLP models with MLflow, enhancing your skills in semantic similarity analysis and model lifecycle management.

python

```python
# Disable tokenizers warnings when constructing pipelines
%env TOKENIZERS_PARALLELISM=false

import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

```

### Implementing a Custom SimilarityModel with MLflow[​](#implementing-a-custom-similaritymodel-with-mlflow "Direct link to Implementing a Custom SimilarityModel with MLflow")

Discover how to create a custom `SimilarityModel` class using MLflow's `PythonModel` to assess semantic similarity between sentences.

#### Overview of SimilarityModel[​](#overview-of-similaritymodel "Direct link to Overview of SimilarityModel")

The `SimilarityModel` is a tailored Python class that leverages MLflow's flexible `PythonModel` interface. It is specifically designed to encapsulate the intricacies of computing semantic similarity between sentence pairs using sophisticated sentence embeddings.

#### Key Components of the Custom Model[​](#key-components-of-the-custom-model "Direct link to Key Components of the Custom Model")

* **Importing Libraries**: Essential libraries from MLflow, data handling, and Sentence Transformers are imported to facilitate model functionality.

* **Custom PythonModel - SimilarityModel**:

  <!-- -->

  * The `load_context` method focuses on efficient and safe model loading, crucial for handling complex models like Sentence Transformers.
  * The `predict` method, equipped with input type checking and error handling, ensures that the model delivers accurate cosine similarity scores, reflecting semantic correlations.

#### Significance of Custom SimilarityModel[​](#significance-of-custom-similaritymodel "Direct link to Significance of Custom SimilarityModel")

* **Flexibility and Customization**: The model's design allows for specialized handling of inputs and outputs, aligning perfectly with unique requirements of semantic similarity tasks.
* **Robust Error Handling**: Detailed input type checking guarantees a user-friendly experience, preventing common input errors and ensuring the predictability of model behavior.
* **Efficient Model Loading**: The strategic use of the `load_context` method for model initialization circumvents serialization challenges, ensuring a smooth operational flow.
* **Targeted Functionality**: The custom `predict` method directly computes similarity scores, showcasing the model's capability to deliver task-specific, actionable insights.

This custom `SimilarityModel` exemplifies the adaptability of MLflow's `PythonModel` in crafting bespoke NLP solutions, setting a precedent for similar endeavors in various machine learning projects.

python

```python
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel


class SimilarityModel(PythonModel):
  def load_context(self, context):
      """Load the model context for inference."""
      from sentence_transformers import SentenceTransformer

      try:
          self.model = SentenceTransformer.load(context.artifacts["model_path"])
      except Exception as e:
          raise ValueError(f"Error loading model: {e}")

  def predict(self, context, model_input, params):
      """Predict method for comparing similarity between two sentences."""
      from sentence_transformers import util

      if isinstance(model_input, pd.DataFrame):
          if model_input.shape[1] != 2:
              raise ValueError("DataFrame input must have exactly two columns.")
          sentence_1 = model_input.iloc[0, 0]
          sentence_2 = model_input.iloc[0, 1]
      elif isinstance(model_input, dict):
          sentence_1 = model_input.get("sentence_1")
          sentence_2 = model_input.get("sentence_2")
          if sentence_1 is None or sentence_2 is None:
              raise ValueError(
                  "Both 'sentence_1' and 'sentence_2' must be provided in the input dictionary."
              )
      else:
          raise TypeError(
              f"Unexpected type for model_input: {type(model_input)}. Must be either a Dict or a DataFrame."
          )

      embedding_1 = self.model.encode(sentence_1)
      embedding_2 = self.model.encode(sentence_2)

      return np.array(util.cos_sim(embedding_1, embedding_2).tolist())

```

### Preparing the Sentence Transformer Model and Signature[​](#preparing-the-sentence-transformer-model-and-signature "Direct link to Preparing the Sentence Transformer Model and Signature")

Explore the essential steps for setting up the Sentence Transformer model for logging and deployment with MLflow.

#### Loading and Saving the Pre-trained Model[​](#loading-and-saving-the-pre-trained-model "Direct link to Loading and Saving the Pre-trained Model")

* **Model Initialization**: A pre-trained Sentence Transformer model, `"all-MiniLM-L6-v2"`, is loaded for its efficiency in generating high-quality embeddings suitable for diverse NLP tasks.
* **Model Saving**: The model is saved locally to `/tmp/sbert_model` to facilitate easy access by MLflow, a prerequisite for model logging in the platform.

#### Preparing Input Example and Artifacts[​](#preparing-input-example-and-artifacts "Direct link to Preparing Input Example and Artifacts")

* **Input Example Creation**: A DataFrame with sample sentences is prepared, representing typical model inputs and aiding in defining the model's input format.
* **Defining Artifacts**: The saved model's file path is specified as an artifact in MLflow, an essential step for associating the model with MLflow runs.

#### Generating Test Output for Signature[​](#generating-test-output-for-signature "Direct link to Generating Test Output for Signature")

* **Test Output Calculation**: The cosine similarity between sentence embeddings is computed, providing a practical example of the model's output.
* **Signature Inference**: MLflow's `infer_signature` function is utilized to generate a signature that encapsulates the expected input and output formats, reinforcing the model's operational schema.

#### Importance of These Steps[​](#importance-of-these-steps "Direct link to Importance of These Steps")

* **Model Readiness**: These preparatory steps ensure the model is primed for efficient logging and seamless deployment within the MLflow ecosystem.
* **Input-Output Contract**: The established signature acts as a clear contract, defining the model's input-output dynamics, pivotal for maintaining consistency and accuracy in deployment scenarios.

Having meticulously prepared the Sentence Transformer model and its signature, we are now well-equipped to advance towards its integration and management in MLflow.

python

```python
# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create an input example DataFrame
input_example = pd.DataFrame([{"sentence_1": "I like apples", "sentence_2": "I like oranges"}])

# Save the model in the /tmp directory
model_directory = "/tmp/sbert_model"
model.save(model_directory)

# Define artifacts with the absolute path
artifacts = {"model_path": model_directory}

# Generate test output for signature
test_output = np.array(
  util.cos_sim(
      model.encode(input_example["sentence_1"][0]), model.encode(input_example["sentence_2"][0])
  ).tolist()
)

# Define the signature associated with the model
signature = infer_signature(input_example, test_output)

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

mlflow.set_experiment("Semantic Similarity")

```

### Logging the Custom Model with MLflow[​](#logging-the-custom-model-with-mlflow "Direct link to Logging the Custom Model with MLflow")

Learn how to log the custom SimilarityModel with MLflow for effective model management and deployment.

#### Creating a Path for the PyFunc Model[​](#creating-a-path-for-the-pyfunc-model "Direct link to Creating a Path for the PyFunc Model")

We establish `pyfunc_path`, a temporary storage location for the Python model. This path is crucial for MLflow to serialize and store the model effectively.

#### Logging the Model in MLflow[​](#logging-the-model-in-mlflow "Direct link to Logging the Model in MLflow")

* **Initiating MLflow Run**: An MLflow run is started, encapsulating all model logging processes within a structured framework.
* **Model Logging Details**: The model is identified as `"similarity"`, providing a clear reference for future model retrieval and analysis. An instance of `SimilarityModel` is logged, encapsulating the Sentence Transformer model and similarity prediction logic. An illustrative DataFrame demonstrates the expected model input format, aiding in user comprehension and model usability. The inferred signature, detailing the input-output schema, is included, reinforcing the correct usage of the model. The artifacts dictionary specifies the location of the serialized Sentence Transformer model, crucial for model reconstruction. Dependencies like `sentence_transformers` and `numpy` are listed, ensuring the model's functional integrity in varied deployment environments.

#### Significance of Model Logging[​](#significance-of-model-logging "Direct link to Significance of Model Logging")

* **Model Tracking and Versioning**: Logging facilitates comprehensive tracking and effective versioning, enhancing model lifecycle management.
* **Reproducibility and Deployment**: The logged model, complete with its dependencies, input example, and signature, becomes easily reproducible and deployable, promoting consistent application across environments.

Having logged our `SimilarityModel` in MLflow, it stands ready for advanced applications such as comparative analysis, version management, and deployment for practical inference use cases.

python

```python
pyfunc_path = "/tmp/sbert_pyfunc"

with mlflow.start_run() as run:
  model_info = mlflow.pyfunc.log_model(
      name="similarity",
      python_model=SimilarityModel(),
      input_example=input_example,
      signature=signature,
      artifacts=artifacts,
      pip_requirements=["sentence_transformers", "numpy"],
  )

```

### Model Inference and Testing Similarity Prediction[​](#model-inference-and-testing-similarity-prediction "Direct link to Model Inference and Testing Similarity Prediction")

Demonstrate the use of the `SimilarityModel` to compute semantic similarity between sentences after logging it with MLflow.

#### Loading the Model for Inference[​](#loading-the-model-for-inference "Direct link to Loading the Model for Inference")

* **Loading with MLflow**: Utilize `mlflow.pyfunc.load_model` with the model's URI to load the custom `SimilarityModel` for inference.
* **Model Readiness**: The loaded model, named `loaded_dynamic`, is equipped with the logic defined in the `SimilarityModel` and is ready to compute similarities.

#### Preparing Data for Similarity Prediction[​](#preparing-data-for-similarity-prediction "Direct link to Preparing Data for Similarity Prediction")

* **Creating Input Data**: Construct a DataFrame, `similarity_data`, with pairs of sentences for which similarity will be computed, showcasing the model's input flexibility.

#### Computing and Displaying Similarity Score[​](#computing-and-displaying-similarity-score "Direct link to Computing and Displaying Similarity Score")

* **Predicting Similarity**: Invoke the `predict` method on `loaded_dynamic` with `similarity_data` to calculate the cosine similarity between sentence embeddings.
* **Interpreting the Result**: The resulting `similarity_score` numerically represents the semantic similarity, offering immediate insights into the model's output.

#### Importance of This Testing[​](#importance-of-this-testing "Direct link to Importance of This Testing")

* **Model Validation**: Confirm the custom model's expected behavior when predicting on new data, ensuring its validity.
* **Practical Application**: Highlight the model's practical utility in real-world scenarios, demonstrating its capability in semantic similarity analysis.

python

```python
# Load our custom semantic similarity model implementation by providing the uri that the model was logged to
loaded_dynamic = mlflow.pyfunc.load_model(model_info.model_uri)

# Create an evaluation test DataFrame
similarity_data = pd.DataFrame([{"sentence_1": "I like apples", "sentence_2": "I like oranges"}])

# Verify that the model generates a reasonable prediction
similarity_score = loaded_dynamic.predict(similarity_data)

print(f"The similarity between these sentences is: {similarity_score}")

```

### Evaluating Semantic Similarity with Distinct Text Pairs[​](#evaluating-semantic-similarity-with-distinct-text-pairs "Direct link to Evaluating Semantic Similarity with Distinct Text Pairs")

Explore the model's capability to discern varying degrees of semantic similarity with carefully chosen text pairs.

#### Selection of Text Pairs[​](#selection-of-text-pairs "Direct link to Selection of Text Pairs")

* **Low Similarity Pair**: Diverse themes in sentences predict a low similarity score, showcasing the model's ability to recognize contrasting semantic contents.
* **High Similarity Pair**: Sentences with similar themes and tones anticipate a high similarity score, demonstrating the model's semantic parallel detection.

#### sBERT Model's Role in Similarity Calculation[​](#sbert-models-role-in-similarity-calculation "Direct link to sBERT Model's Role in Similarity Calculation")

* **Semantic Understanding**: Utilizing sBERT to encode semantic essence into vectors.
* **Cosine Similarity**: Calculating similarity scores to quantify semantic closeness.

#### Computing and Displaying Similarity Scores[​](#computing-and-displaying-similarity-scores "Direct link to Computing and Displaying Similarity Scores")

* **Predicting for Low Similarity Pair**: Observing the model's interpretation of semantically distant sentences.
* **Predicting for High Similarity Pair**: Assessing the model's ability to detect semantic similarities in contextually related sentences.

#### Why This Matters[​](#why-this-matters "Direct link to Why This Matters")

* **Model Validation**: These tests affirm the model's nuanced language understanding and semantic relationship quantification.
* **Practical Implications**: Insights from the model's processing of semantic content inform applications in content recommendation, information retrieval, and text comparison.

python

```python
low_similarity = {
  "sentence_1": "The explorer stood at the edge of the dense rainforest, "
  "contemplating the journey ahead. The untamed wilderness was "
  "a labyrinth of exotic plants and unknown dangers, a challenge "
  "for even the most seasoned adventurer, brimming with the "
  "prospect of new discoveries and uncharted territories.",
  "sentence_2": "To install the software, begin by downloading the latest "
  "version from the official website. Once downloaded, run the "
  "installer and follow the on-screen instructions. Ensure that "
  "your system meets the minimum requirements and agree to the "
  "license terms to complete the installation process successfully.",
}

high_similarity = {
  "sentence_1": "Standing in the shadow of the Great Pyramids of Giza, I felt a "
  "profound sense of awe. The towering structures, a testament to "
  "ancient ingenuity, rose majestically against the clear blue sky. "
  "As I walked around the base of the pyramids, the intricate "
  "stonework and sheer scale of these wonders of the ancient world "
  "left me speechless, enveloped in a deep sense of history.",
  "sentence_2": "My visit to the Great Pyramids of Giza was an unforgettable "
  "experience. Gazing upon these monumental structures, I was "
  "captivated by their grandeur and historical significance. Each "
  "step around these ancient marvels filled me with a deep "
  "appreciation for the architectural prowess of a civilization long "
  "gone, yet still speaking through these timeless monuments.",
}

# Validate that semantically unrelated texts return a low similarity score
low_similarity_score = loaded_dynamic.predict(low_similarity)

print(f"The similarity score for the 'low_similarity' pair is: {low_similarity_score}")

# Validate that semantically similar texts return a high similarity score
high_similarity_score = loaded_dynamic.predict(high_similarity)

print(f"The similarity score for the 'high_similarity' pair is: {high_similarity_score}")

```

### Conclusion: Harnessing the Power of Custom MLflow Python Functions in NLP[​](#conclusion-harnessing-the-power-of-custom-mlflow-python-functions-in-nlp "Direct link to Conclusion: Harnessing the Power of Custom MLflow Python Functions in NLP")

As we conclude this tutorial, let's recap the significant strides we've made in understanding and applying advanced NLP techniques using Sentence Transformers and MLflow.

#### Key Takeaways from the Tutorial[​](#key-takeaways-from-the-tutorial "Direct link to Key Takeaways from the Tutorial")

* **Versatile NLP Modeling**: We explored how to harness the advanced capabilities of Sentence Transformers for semantic similarity analysis, a critical task in many NLP applications.
* **Custom MLflow Python Function**: The implementation of the custom `SimilarityModel` in MLflow demonstrated the power and flexibility of using Python functions to extend and adapt the functionality of pre-trained models to suit specific project needs.
* **Model Management and Deployment**: We delved into the process of logging, managing, and deploying these models with MLflow, showcasing how MLflow streamlines these aspects of the machine learning lifecycle.
* **Practical Semantic Analysis**: Through hands-on examples, we demonstrated the model's ability to discern varying degrees of semantic similarity between sentence pairs, validating its effectiveness in real-world semantic analysis tasks.

#### The Power and Flexibility of MLflow's Python Functions[​](#the-power-and-flexibility-of-mlflows-python-functions "Direct link to The Power and Flexibility of MLflow's Python Functions")

* **Customization for Specific Needs**: One of the tutorial's highlights is the demonstration of how MLflow's `PythonModel` can be customized. This customization is not only powerful but also necessary for tailoring models to specific NLP tasks that go beyond standard model functionalities.
* **Adaptability and Extension**: The `PythonModel` framework in MLflow provides a solid foundation for implementing a wide range of NLP models. Its adaptability allows for the extension of base model functionalities, such as transforming a sentence embedding model into a semantic similarity comparison tool.

#### Empowering Advanced NLP Applications[​](#empowering-advanced-nlp-applications "Direct link to Empowering Advanced NLP Applications")

* **Ease of Modification**: The tutorial showcased that modifying the provided `PythonModel` implementation for different flavors in MLflow can be done with relative ease, empowering you to create models that align precisely with your project's requirements.
* **Wide Applicability**: Whether it's semantic search, content recommendation, or automated text comparison, the approach outlined in this tutorial can be adapted to a broad spectrum of NLP tasks, opening doors to innovative applications in the field.

#### Moving Forward[​](#moving-forward "Direct link to Moving Forward")

Armed with the knowledge and skills acquired in this tutorial, you are now well-equipped to apply these advanced NLP techniques in your projects. The seamless integration of Sentence Transformers with MLflow's robust model management and deployment capabilities paves the way for developing sophisticated, efficient, and effective NLP solutions.

Thank you for joining us on this journey through advanced NLP modeling with Sentence Transformers and MLflow. We hope this tutorial has inspired you to explore further and innovate in your NLP endeavors!

Happy Modeling!
