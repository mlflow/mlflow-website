# Advanced Paraphrase Mining with Sentence Transformers and MLflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.ipynb)

Embark on an enriching journey through advanced paraphrase mining using Sentence Transformers, enhanced by MLflow.

### Learning Objectives[​](#learning-objectives "Direct link to Learning Objectives")

* Apply `sentence-transformers` for advanced paraphrase mining.
* Develop a custom `PythonModel` in MLflow tailored for this task.
* Effectively manage and track models within the MLflow ecosystem.
* Deploy paraphrase mining models using MLflow's deployment capabilities.

#### Exploring Paraphrase Mining[​](#exploring-paraphrase-mining "Direct link to Exploring Paraphrase Mining")

Discover the process of identifying semantically similar but textually distinct sentences, a key aspect in various NLP applications such as document summarization and chatbot development.

#### The Role of Sentence Transformers in Paraphrase Mining[​](#the-role-of-sentence-transformers-in-paraphrase-mining "Direct link to The Role of Sentence Transformers in Paraphrase Mining")

Learn how Sentence Transformers, specialized for generating rich sentence embeddings, are used to capture deep semantic meanings and compare textual content.

#### MLflow: Simplifying Model Management and Deployment[​](#mlflow-simplifying-model-management-and-deployment "Direct link to MLflow: Simplifying Model Management and Deployment")

Delve into how MLflow streamlines the process of managing and deploying NLP models, with a focus on efficient tracking and customizable model implementations.

Join us to develop a nuanced understanding of paraphrase mining and master the art of managing and deploying NLP models with MLflow.

python

```python
import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

```

### Introduction to the Paraphrase Mining Model[​](#introduction-to-the-paraphrase-mining-model "Direct link to Introduction to the Paraphrase Mining Model")

Initiate the Paraphrase Mining Model, integrating Sentence Transformers and MLflow for advanced NLP tasks.

#### Overview of the Model Structure[​](#overview-of-the-model-structure "Direct link to Overview of the Model Structure")

* **Loading Model and Corpus `load_context` Method**: Essential for loading the Sentence Transformer model and the text corpus for paraphrase identification.
* **Paraphrase Mining Logic `predict` Method**: Integrates custom logic for input validation and paraphrase mining, offering customizable parameters.
* **Sorting and Filtering Matches `_sort_and_filter_matches` Helper Method**: Ensures relevant and unique paraphrase identification by sorting and filtering based on similarity scores.

#### Key Features[​](#key-features "Direct link to Key Features")

* **Advanced NLP Techniques**: Utilizes Sentence Transformers for semantic text understanding.
* **Custom Logic Integration**: Demonstrates flexibility in model behavior customization.
* **User Customization Options**: Allows end users to adjust match criteria for various use cases.
* **Efficiency in Processing**: Pre-encodes the corpus for efficient paraphrase mining operations.
* **Robust Error Handling**: Incorporates validations for reliable model performance.

#### Practical Implications[​](#practical-implications "Direct link to Practical Implications")

This model provides a powerful tool for paraphrase detection in diverse applications, exemplifying the effective use of custom models within the MLflow framework.

python

```python
import warnings

import pandas as pd
from sentence_transformers import SentenceTransformer, util

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel


class ParaphraseMiningModel(PythonModel):
  def load_context(self, context):
      """Load the model context for inference, including the customer feedback corpus."""
      try:
          # Load the pre-trained sentence transformer model
          self.model = SentenceTransformer.load(context.artifacts["model_path"])

          # Load the customer feedback corpus from the specified file
          corpus_file = context.artifacts["corpus_file"]
          with open(corpus_file) as file:
              self.corpus = file.read().splitlines()

      except Exception as e:
          raise ValueError(f"Error loading model and corpus: {e}")

  def _sort_and_filter_matches(
      self,
      query: str,
      paraphrase_pairs: list[tuple[float, int, int]],
      similarity_threshold: float,
  ):
      """Sort and filter the matches by similarity score."""

      # Convert to list of tuples and sort by score
      sorted_matches = sorted(paraphrase_pairs, key=lambda x: x[1], reverse=True)

      # Filter and collect paraphrases for the query, avoiding duplicates
      query_paraphrases = {}
      for score, i, j in sorted_matches:
          if score < similarity_threshold:
              continue

          paraphrase = self.corpus[j] if self.corpus[i] == query else self.corpus[i]
          if paraphrase == query:
              continue

          if paraphrase not in query_paraphrases or score > query_paraphrases[paraphrase]:
              query_paraphrases[paraphrase] = score

      return sorted(query_paraphrases.items(), key=lambda x: x[1], reverse=True)

  def predict(self, context, model_input, params=None):
      """Predict method to perform paraphrase mining over the corpus."""

      # Validate and extract the query input
      if isinstance(model_input, pd.DataFrame):
          if model_input.shape[1] != 1:
              raise ValueError("DataFrame input must have exactly one column.")
          query = model_input.iloc[0, 0]
      elif isinstance(model_input, dict):
          query = model_input.get("query")
          if query is None:
              raise ValueError("The input dictionary must have a key named 'query'.")
      else:
          raise TypeError(
              f"Unexpected type for model_input: {type(model_input)}. Must be either a Dict or a DataFrame."
          )

      # Determine the minimum similarity threshold
      similarity_threshold = params.get("similarity_threshold", 0.5) if params else 0.5

      # Add the query to the corpus for paraphrase mining
      extended_corpus = self.corpus + [query]

      # Perform paraphrase mining
      paraphrase_pairs = util.paraphrase_mining(
          self.model, extended_corpus, show_progress_bar=False
      )

      # Convert to list of tuples and sort by score
      sorted_paraphrases = self._sort_and_filter_matches(
          query, paraphrase_pairs, similarity_threshold
      )

      # Warning if no paraphrases found
      if not sorted_paraphrases:
          warnings.warn("No paraphrases found above the similarity threshold.", UserWarning)

      return {sentence[0]: str(sentence[1]) for sentence in sorted_paraphrases}

```

### Preparing the Corpus for Paraphrase Mining[​](#preparing-the-corpus-for-paraphrase-mining "Direct link to Preparing the Corpus for Paraphrase Mining")

Set up the foundation for paraphrase mining by creating and preparing a diverse corpus.

#### Corpus Creation[​](#corpus-creation "Direct link to Corpus Creation")

* Define a `corpus` comprising a range of sentences from various topics, including space exploration, AI, gardening, and more. This diversity enables the model to identify paraphrases across a broad spectrum of subjects.

#### Writing the Corpus to a File[​](#writing-the-corpus-to-a-file "Direct link to Writing the Corpus to a File")

* The corpus is saved to a file named `feedback.txt`, mirroring a common practice in large-scale data handling.
* This step also prepares the corpus for efficient processing within the Paraphrase Mining Model.

#### Significance of the Corpus[​](#significance-of-the-corpus "Direct link to Significance of the Corpus")

The corpus serves as the key dataset for the model to find semantically similar sentences. Its variety ensures the model's adaptability and effectiveness across diverse use cases.

python

```python
corpus = [
  "Exploring ancient cities in Europe offers a glimpse into history.",
  "Modern AI technologies are revolutionizing industries.",
  "Healthy eating contributes significantly to overall well-being.",
  "Advancements in renewable energy are combating climate change.",
  "Learning a new language opens doors to different cultures.",
  "Gardening is a relaxing hobby that connects you with nature.",
  "Blockchain technology could redefine digital transactions.",
  "Homemade Italian pasta is a delight to cook and eat.",
  "Practicing yoga daily improves both physical and mental health.",
  "The art of photography captures moments in time.",
  "Baking bread at home has become a popular quarantine activity.",
  "Virtual reality is creating new experiences in gaming.",
  "Sustainable travel is becoming a priority for eco-conscious tourists.",
  "Reading books is a great way to unwind and learn.",
  "Jazz music provides a rich tapestry of sound and rhythm.",
  "Marathon training requires discipline and perseverance.",
  "Studying the stars helps us understand our universe.",
  "The rise of electric cars is an important environmental development.",
  "Documentary films offer deep insights into real-world issues.",
  "Crafting DIY projects can be both fun and rewarding.",
  "The history of ancient civilizations is fascinating to explore.",
  "Exploring the depths of the ocean reveals a world of marine wonders.",
  "Learning to play a musical instrument can be a rewarding challenge.",
  "Artificial intelligence is shaping the future of personalized medicine.",
  "Cycling is not only a great workout but also eco-friendly transportation.",
  "Home automation with IoT devices is enhancing living experiences.",
  "Understanding quantum computing requires a grasp of complex physics.",
  "A well-brewed cup of coffee is the perfect start to the day.",
  "Urban farming is gaining popularity as a sustainable food source.",
  "Meditation and mindfulness can lead to a more balanced life.",
  "The popularity of podcasts has revolutionized audio storytelling.",
  "Space exploration continues to push the boundaries of human knowledge.",
  "Wildlife conservation is essential for maintaining biodiversity.",
  "The fusion of technology and fashion is creating new trends.",
  "E-learning platforms have transformed the educational landscape.",
  "Dark chocolate has surprising health benefits when enjoyed in moderation.",
  "Robotics in manufacturing is leading to more efficient production.",
  "Creating a personal budget is key to financial well-being.",
  "Hiking in nature is a great way to connect with the outdoors.",
  "3D printing is innovating the way we create and manufacture objects.",
  "Sommeliers can identify a wine's characteristics with just a taste.",
  "Mind-bending puzzles and riddles are great for cognitive exercise.",
  "Social media has a profound impact on communication and culture.",
  "Urban sketching captures the essence of city life on paper.",
  "The ethics of AI is a growing field in tech philosophy.",
  "Homemade skincare remedies are becoming more popular.",
  "Virtual travel experiences can provide a sense of adventure at home.",
  "Ancient mythology still influences modern storytelling and literature.",
  "Building model kits is a hobby that requires patience and precision.",
  "The study of languages opens windows into different worldviews.",
  "Professional esports has become a major global phenomenon.",
  "The mysteries of the universe are unveiled through space missions.",
  "Astronauts' experiences in space stations offer unique insights into life beyond Earth.",
  "Telescopic observations bring distant galaxies within our view.",
  "The study of celestial bodies helps us understand the cosmos.",
  "Space travel advancements could lead to interplanetary exploration.",
  "Observing celestial events provides valuable data for astronomers.",
  "The development of powerful rockets is key to deep space exploration.",
  "Mars rover missions are crucial in searching for extraterrestrial life.",
  "Satellites play a vital role in our understanding of Earth's atmosphere.",
  "Astrophysics is central to unraveling the secrets of space.",
  "Zero gravity environments in space pose unique challenges and opportunities.",
  "Space tourism might soon become a reality for many.",
  "Lunar missions have contributed significantly to our knowledge of the moon.",
  "The International Space Station is a hub for groundbreaking space research.",
  "Studying comets and asteroids reveals information about the early solar system.",
  "Advancements in space technology have implications for many scientific fields.",
  "The possibility of life on other planets continues to intrigue scientists.",
  "Black holes are among the most mysterious phenomena in space.",
  "The history of space exploration is filled with remarkable achievements.",
  "Future space missions could unlock the mysteries of dark matter.",
]

# Write out the corpus to a file
corpus_file = "/tmp/feedback.txt"
with open(corpus_file, "w") as file:
  for sentence in corpus:
      file.write(sentence + "
")

```

### Setting Up the Paraphrase Mining Model[​](#setting-up-the-paraphrase-mining-model "Direct link to Setting Up the Paraphrase Mining Model")

Prepare the Sentence Transformer model for integration with MLflow to harness its paraphrase mining capabilities.

#### Loading the Sentence Transformer Model[​](#loading-the-sentence-transformer-model "Direct link to Loading the Sentence Transformer Model")

* Initialize the `all-MiniLM-L6-v2` Sentence Transformer model, ideal for generating sentence embeddings suitable for paraphrase mining.

#### Preparing the Input Example[​](#preparing-the-input-example "Direct link to Preparing the Input Example")

* Create a DataFrame as an input example to illustrate the type of query the model will handle, aiding in defining the model's input structure.

#### Saving the Model[​](#saving-the-model "Direct link to Saving the Model")

* Save the model to `/tmp/paraphrase_search_model` for portability and ease of loading during deployment with MLflow.

#### Defining Artifacts and Corpus Path[​](#defining-artifacts-and-corpus-path "Direct link to Defining Artifacts and Corpus Path")

* Specify paths to the saved model and corpus as artifacts in MLflow, crucial for model logging and reproduction.

#### Generating Test Output for Signature[​](#generating-test-output-for-signature "Direct link to Generating Test Output for Signature")

* Generate a sample output, illustrating the model's expected output format for paraphrase mining.

#### Creating the Model Signature[​](#creating-the-model-signature "Direct link to Creating the Model Signature")

* Use MLflow's `infer_signature` to define the model's input and output schema, adding the `similarity_threshold` parameter for inference flexibility.

python

```python
# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create an input example DataFrame
input_example = pd.DataFrame({"query": ["This product works well. I'm satisfied."]})

# Save the model in the /tmp directory
model_directory = "/tmp/paraphrase_search_model"
model.save(model_directory)

# Define the path for the corpus file
corpus_file = "/tmp/feedback.txt"

# Define the artifacts (paths to the model and corpus file)
artifacts = {"model_path": model_directory, "corpus_file": corpus_file}

# Generate test output for signature
# Sample output for paraphrase mining could be a list of tuples (paraphrase, score)
test_output = [{"This product is satisfactory and functions as expected.": "0.8"}]

# Define the signature associated with the model
# The signature includes the structure of the input and the expected output, as well as any parameters that
# we would like to expose for overriding at inference time (including their default values if they are not overridden).
signature = infer_signature(
  model_input=input_example, model_output=test_output, params={"similarity_threshold": 0.5}
)

# Visualize the signature, showing our overridden inference parameter and its default.
signature

```

### Creating an experiment[​](#creating-an-experiment "Direct link to Creating an experiment")

We create a new MLflow Experiment so that the run we're going to log our model to does not log to the default experiment and instead has its own contextually relevant entry.

python

```python
# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

# mlflow.set_tracking_uri("http://127.0.0.1:8080")

mlflow.set_experiment("Paraphrase Mining")

```

### Logging the Paraphrase Mining Model with MLflow[​](#logging-the-paraphrase-mining-model-with-mlflow "Direct link to Logging the Paraphrase Mining Model with MLflow")

Log the custom Paraphrase Mining Model with MLflow, a key step for model management and deployment.

#### Initiating an MLflow Run[​](#initiating-an-mlflow-run "Direct link to Initiating an MLflow Run")

* Start an MLflow run to create a comprehensive record of model logging and tracking within the MLflow framework.

#### Logging the Model in MLflow[​](#logging-the-model-in-mlflow "Direct link to Logging the Model in MLflow")

* Use MLflow's Python model logging function to integrate the custom model into the MLflow ecosystem.
* Provide a unique name for the model for easy identification in MLflow.
* Log the instantiated Paraphrase Mining Model, along with an input example, model signature, artifacts, and Python dependencies.

#### Outcomes and Benefits of Model Logging[​](#outcomes-and-benefits-of-model-logging "Direct link to Outcomes and Benefits of Model Logging")

* Register the model within MLflow for streamlined management and deployment, enhancing its accessibility and trackability.
* Ensure model reproducibility and version control across deployment environments.

python

```python
with mlflow.start_run() as run:
  model_info = mlflow.pyfunc.log_model(
      name="paraphrase_model",
      python_model=ParaphraseMiningModel(),
      input_example=input_example,
      signature=signature,
      artifacts=artifacts,
      pip_requirements=["sentence_transformers"],
  )

```

### Model Loading and Paraphrase Mining Prediction[​](#model-loading-and-paraphrase-mining-prediction "Direct link to Model Loading and Paraphrase Mining Prediction")

Illustrate the real-world application of the Paraphrase Mining Model by loading it with MLflow and executing a prediction.

#### Loading the Model for Inference[​](#loading-the-model-for-inference "Direct link to Loading the Model for Inference")

* Utilize MLflow's `load_model` function to retrieve and prepare the model for inference.
* Locate and load the model using its unique URI within the MLflow registry.

#### Executing a Paraphrase Mining Prediction[​](#executing-a-paraphrase-mining-prediction "Direct link to Executing a Paraphrase Mining Prediction")

* Make a prediction using the model's `predict` method, applying the paraphrase mining logic embedded in the model class.
* Pass a representative query with a set `similarity_threshold` to find matching paraphrases in the corpus.

#### Interpreting the Model Output[​](#interpreting-the-model-output "Direct link to Interpreting the Model Output")

* Review the list of semantically similar sentences to the query, highlighting the model's paraphrase identification capabilities.
* Analyze the similarity scores to understand the degree of semantic relatedness between the query and corpus sentences.

#### Conclusion[​](#conclusion "Direct link to Conclusion")

This demonstration validates the Paraphrase Mining Model's effectiveness in real-world scenarios, underscoring its utility in content recommendation, information retrieval, and conversational AI.

python

```python
# Load our model by supplying the uri that was used to save the model artifacts
loaded_dynamic = mlflow.pyfunc.load_model(model_info.model_uri)

# Perform a quick validation that our loaded model is performing adequately
loaded_dynamic.predict(
  {"query": "Space exploration is fascinating."}, params={"similarity_threshold": 0.65}
)

```

### Conclusion: Insights and Potential Enhancements[​](#conclusion-insights-and-potential-enhancements "Direct link to Conclusion: Insights and Potential Enhancements")

As we wrap up this tutorial, let's reflect on our journey through the implementation of a Paraphrase Mining Model using Sentence Transformers and MLflow. We've successfully built and deployed a model capable of identifying semantically similar sentences, showcasing the flexibility and power of MLflow's `PythonModel` implementation.

#### Key Takeaways[​](#key-takeaways "Direct link to Key Takeaways")

* We learned how to integrate advanced NLP techniques, specifically paraphrase mining, with MLflow. This integration not only enhances model management but also simplifies deployment and scalability.
* The flexibility of the `PythonModel` implementation in MLflow was a central theme. We saw firsthand how it allows for the incorporation of custom logic into the model's predict function, catering to specific NLP tasks like paraphrase mining.
* Through our custom model, we explored the dynamics of sentence embeddings, semantic similarity, and the nuances of language understanding. This understanding is crucial in a wide range of applications, from content recommendation to conversational AI.

#### Ideas for Enhancing the Paraphrase Mining Model[​](#ideas-for-enhancing-the-paraphrase-mining-model "Direct link to Ideas for Enhancing the Paraphrase Mining Model")

While our model serves as a robust starting point, there are several enhancements that could be made within the `predict` function to make it more powerful and feature-rich:

1. **Contextual Filters**: Introduce filters based on contextual clues or specific keywords to refine the search results further. This feature would allow users to narrow down paraphrases to those most relevant to their particular context or subject matter.

2. **Sentiment Analysis Integration**: Incorporate sentiment analysis to group paraphrases by their emotional tone. This would be especially useful in applications like customer feedback analysis, where understanding sentiment is as important as content.

3. **Multi-Lingual Support**: Expand the model to support paraphrase mining in multiple languages. This enhancement would significantly broaden the model's applicability in global or multi-lingual contexts.

#### Scalability with Vector Databases[​](#scalability-with-vector-databases "Direct link to Scalability with Vector Databases")

* Moving beyond a static text file as a corpus, a more scalable and real-world approach would involve connecting the model to an external vector database or in-memory store.
* Pre-calculated embeddings could be stored and updated in such databases, accommodating real-time content generation without requiring model redeployment. This approach would dramatically improve the model's scalability and responsiveness in real-world applications.

#### Final Thoughts[​](#final-thoughts "Direct link to Final Thoughts")

The journey through building and deploying the Paraphrase Mining Model has been both enlightening and practical. We've seen how MLflow's `PythonModel` offers a flexible canvas for crafting custom NLP solutions, and how sentence transformers can be leveraged to delve deep into the semantics of language.

This tutorial is just the beginning. There's a vast potential for further exploration and innovation in paraphrase mining and NLP as a whole. We encourage you to build upon this foundation, experiment with enhancements, and continue pushing the boundaries of what's possible with MLflow and advanced NLP techniques.
