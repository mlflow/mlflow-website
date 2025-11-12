# Advanced Semantic Search with Sentence Transformers and MLflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers.ipynb)

Embark on a hands-on journey exploring Advanced Semantic Search using Sentence Transformers and MLflow.

### What You Will Learn[​](#what-you-will-learn "Direct link to What You Will Learn")

* Implement advanced semantic search with `sentence-transformers`.
* Customize MLflow's `PythonModel` for unique project requirements.
* Manage and log models within MLflow's ecosystem.
* Deploy complex models for practical applications using MLflow.

#### Understanding Semantic Search[​](#understanding-semantic-search "Direct link to Understanding Semantic Search")

Semantic search transcends keyword matching, using language nuances and context to find relevant results. This advanced approach reflects human language understanding, considering the varied meanings of words in different scenarios.

#### Harnessing Power of Sentence Transformers for Search[​](#harnessing-power-of-sentence-transformers-for-search "Direct link to Harnessing Power of Sentence Transformers for Search")

Sentence Transformers, specialized for context-rich sentence embeddings, transform search queries and text corpora into semantic vectors. This enables the identification of semantically similar entries, a cornerstone of semantic search.

#### MLflow: A Vanguard in Model Management and Deployment[​](#mlflow-a-vanguard-in-model-management-and-deployment "Direct link to MLflow: A Vanguard in Model Management and Deployment")

MLflow enhances NLP projects with efficient experiment logging and customizable model environments. It brings efficiency to experiment tracking and adds a layer of customization, vital for unique NLP tasks.

Join us in this tutorial to master advanced semantic search techniques and discover how MLflow can revolutionize your approach to NLP model deployment and management.

python

```python
import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

```

### Understanding the Semantic Search Model with MLflow and Sentence Transformers[​](#understanding-the-semantic-search-model-with-mlflow-and-sentence-transformers "Direct link to Understanding the Semantic Search Model with MLflow and Sentence Transformers")

Delve into the intricacies of the `SemanticSearchModel`, a custom implementation for semantic search using MLflow and Sentence Transformers.

#### MLflow and Custom PyFunc Models[​](#mlflow-and-custom-pyfunc-models "Direct link to MLflow and Custom PyFunc Models")

MLflow's custom Python function (`pyfunc`) models provide a flexible and deployable solution for integrating complex logic, ideal for our `SemanticSearchModel`.

#### The Model's Core Functionalities[​](#the-models-core-functionalities "Direct link to The Model's Core Functionalities")

* **Context Loading**: Essential for initializing the Sentence Transformer model and preparing the corpus for semantic comparison.
* **Predict Method**: The central function for semantic search, encompassing input validation, query encoding, and similarity computation.

#### Detailed Breakdown of Predict Method[​](#detailed-breakdown-of-predict-method "Direct link to Detailed Breakdown of Predict Method")

* **Input Validation**: Ensures proper format and extraction of the query sentence.
* **Query Encoding**: Converts the query into an embedding for comparison.
* **Cosine Similarity Computation**: Determines the relevance of each corpus entry to the query.
* **Top Results Extraction**: Identifies the most relevant entries based on similarity scores.
* **Relevancy Filtering**: Filters results based on a minimum relevancy threshold, enhancing practical usability.
* **Warning Mechanism**: Issues a warning if all top results are below the relevancy threshold, ensuring a result is always provided.

#### Conclusion[​](#conclusion "Direct link to Conclusion")

This semantic search model exemplifies the integration of NLP with MLflow, showcasing flexibility, user-friendliness, and practical application in modern machine learning workflows.

python

```python
import warnings

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel


class SemanticSearchModel(PythonModel):
  def load_context(self, context):
      """Load the model context for inference, including the corpus from a file."""
      try:
          # Load the pre-trained sentence transformer model
          self.model = SentenceTransformer.load(context.artifacts["model_path"])

          # Load the corpus from the specified file
          corpus_file = context.artifacts["corpus_file"]
          with open(corpus_file) as file:
              self.corpus = file.read().splitlines()

          # Encode the corpus and convert it to a tensor
          self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)

      except Exception as e:
          raise ValueError(f"Error loading model and corpus: {e}")

  def predict(self, context, model_input, params=None):
      """Predict method to perform semantic search over the corpus."""

      if isinstance(model_input, pd.DataFrame):
          if model_input.shape[1] != 1:
              raise ValueError("DataFrame input must have exactly one column.")
          model_input = model_input.iloc[0, 0]
      elif isinstance(model_input, dict):
          model_input = model_input.get("sentence")
          if model_input is None:
              raise ValueError("The input dictionary must have a key named 'sentence'.")
      else:
          raise TypeError(
              f"Unexpected type for model_input: {type(model_input)}. Must be either a Dict or a DataFrame."
          )

      # Encode the query
      query_embedding = self.model.encode(model_input, convert_to_tensor=True)

      # Compute cosine similarity scores
      cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

      # Determine the number of top results to return
      top_k = params.get("top_k", 3) if params else 3  # Default to 3 if not specified

      minimum_relevancy = (
          params.get("minimum_relevancy", 0.2) if params else 0.2
      )  # Default to 0.2 if not specified

      # Get the top_k most similar sentences from the corpus
      top_results = np.argsort(cos_scores, axis=0)[-top_k:]

      # Prepare the initial results list
      initial_results = [
          (self.corpus[idx], cos_scores[idx].item()) for idx in reversed(top_results)
      ]

      # Filter the results based on the minimum relevancy threshold
      filtered_results = [result for result in initial_results if result[1] >= minimum_relevancy]

      # If all results are below the threshold, issue a warning and return the top result
      if not filtered_results:
          warnings.warn(
              "All top results are below the minimum relevancy threshold. "
              "Returning the highest match instead.",
              RuntimeWarning,
          )
          return [initial_results[0]]
      else:
          return filtered_results

```

### Building and Preparing the Semantic Search Corpus[​](#building-and-preparing-the-semantic-search-corpus "Direct link to Building and Preparing the Semantic Search Corpus")

Explore constructing and preparing the corpus for the semantic search model, a critical component for search functionality.

#### Simulating a Real-World Use Case[​](#simulating-a-real-world-use-case "Direct link to Simulating a Real-World Use Case")

We create a simplified corpus of synthetic blog posts to demonstrate the model's core functionality, replicating a scaled-down version of a typical real-world scenario.

#### Key Steps in Corpus Preparation[​](#key-steps-in-corpus-preparation "Direct link to Key Steps in Corpus Preparation")

* **Corpus Creation**: Formation of a list representing individual blog post entries.
* **Writing to a File**: Saving the corpus to a text file, mimicking the process of data extraction and preprocessing in a real application.

#### Efficient Data Handling for Scalability[​](#efficient-data-handling-for-scalability "Direct link to Efficient Data Handling for Scalability")

Our model encodes the corpus into embeddings for rapid comparison, demonstrating an efficient approach suitable for scaling to larger datasets.

#### Production Considerations[​](#production-considerations "Direct link to Production Considerations")

* **Storing Embeddings**: Discusses options for efficient storage and retrieval of embeddings, crucial in large-scale applications.
* **Scalability**: Highlights the importance of scalable storage systems for handling extensive datasets and complex queries.
* **Updating the Corpus**: Outlines strategies for managing and updating the corpus in dynamic, evolving use cases.

#### Realizing the Semantic Search Concept[​](#realizing-the-semantic-search-concept "Direct link to Realizing the Semantic Search Concept")

This setup, while simplified, reflects the essential steps for developing a robust and scalable semantic search system, combining NLP techniques with efficient data management. In a real production use-case, the processing of a corpus (creating embeddings) would be an external process to that which is running the semantic search. The corpus example below is intended to showcase functionality solely for the purposes of demonstration.

python

```python
corpus = [
  "Perfecting a Sourdough Bread Recipe: The Joy of Baking. Baking sourdough bread "
  "requires patience, skill, and a good understanding of yeast fermentation. Each "
  "loaf is unique, telling its own story of the baker's journey.",
  "The Mars Rover's Discoveries: Unveiling the Red Planet. NASA's Mars rover has "
  "sent back stunning images and data, revealing the planet's secrets. These "
  "discoveries may hold the key to understanding Mars' history.",
  "The Art of Growing Herbs: Enhancing Your Culinary Skills. Growing your own "
  "herbs can transform your cooking, adding fresh and vibrant flavors. Whether it's "
  "basil, thyme, or rosemary, each herb has its own unique characteristics.",
  "AI in Software Development: Transforming the Tech Landscape. The rapid "
  "advancements in artificial intelligence are reshaping how we approach software "
  "development. From automation to machine learning, the possibilities are endless.",
  "Backpacking Through Europe: A Journey of Discovery. Traveling across Europe by "
  "backpack allows one to immerse in diverse cultures and landscapes. It's an "
  "adventure that combines the thrill of exploration with personal growth.",
  "Shakespeare's Timeless Influence: Reshaping Modern Storytelling. The works of "
  "William Shakespeare continue to inspire and influence contemporary literature. "
  "His mastery of language and deep understanding of human nature are unparalleled.",
  "The Rise of Renewable Energy: A Sustainable Future. Embracing renewable energy "
  "is crucial for achieving a sustainable and environmentally friendly lifestyle. "
  "Solar, wind, and hydro power are leading the way in this green revolution.",
  "The Magic of Jazz: An Exploration of Sound and Harmony. Jazz music, known for "
  "its improvisation and complex harmonies, has a rich and diverse history. It "
  "evokes a range of emotions, often reflecting the soul of the musician.",
  "Yoga for Mind and Body: The Benefits of Regular Practice. Engaging in regular "
  "yoga practice can significantly improve flexibility, strength, and mental "
  "well-being. It's a holistic approach to health, combining physical and spiritual "
  "aspects.",
  "The Egyptian Pyramids: Monuments of Ancient Majesty. The ancient Egyptian "
  "pyramids, monumental tombs for pharaohs, are marvels of architectural "
  "ingenuity. They stand as a testament to the advanced skills of ancient builders.",
  "Vegan Cuisine: A World of Flavor. Exploring vegan cuisine reveals a world of "
  "nutritious and delicious possibilities. From hearty soups to delectable desserts, "
  "plant-based dishes are diverse and satisfying.",
  "Extraterrestrial Life: The Endless Search. The quest to find life beyond Earth "
  "continues to captivate scientists and the public alike. Advances in space "
  "technology are bringing us closer to answering this age-old question.",
  "The Art of Plant Pruning: Promoting Healthy Growth. Regular pruning is essential "
  "for maintaining healthy and vibrant plants. It's not just about cutting back, but "
  "understanding each plant's growth patterns and needs.",
  "Cybersecurity in the Digital Age: Protecting Our Data. With the rise of digital "
  "technology, cybersecurity has become a critical concern. Protecting sensitive "
  "information from cyber threats is an ongoing challenge for individuals and "
  "businesses alike.",
  "The Great Wall of China: A Historical Journey. Visiting the Great Wall offers "
  "more than just breathtaking views; it's a journey through history. This ancient "
  "structure tells stories of empires, invasions, and human resilience.",
  "Mystery Novels: Crafting Suspense and Intrigue. A great mystery novel captivates "
  "the reader with intricate plots and unexpected twists. It's a genre that combines "
  "intellectual challenge with entertainment.",
  "Conserving Endangered Species: A Global Effort. Protecting endangered species "
  "is a critical task that requires international collaboration. From rainforests to "
  "oceans, every effort counts in preserving our planet's biodiversity.",
  "Emotions in Classical Music: A Symphony of Feelings. Classical music is not just "
  "an auditory experience; it's an emotional journey. Each composition tells a story, "
  "conveying feelings from joy to sorrow, tranquility to excitement.",
  "CrossFit: A Test of Strength and Endurance. CrossFit is more than just a fitness "
  "regimen; it's a lifestyle that challenges your physical and mental limits. It "
  "combines various disciplines to create a comprehensive workout.",
  "The Renaissance: An Era of Artistic Genius. The Renaissance marked a period of "
  "extraordinary artistic and scientific achievements. It was a time when creativity "
  "and innovation flourished, reshaping the course of history.",
  "Exploring International Cuisines: A Culinary Adventure. Discovering international "
  "cuisines is an adventure for the palate. Each dish offers a glimpse into the "
  "culture and traditions of its origin.",
  "Astronaut Training: Preparing for the Unknown. Becoming an astronaut involves "
  "rigorous training to prepare for the extreme conditions of space. It's a journey "
  "that tests both physical endurance and mental resilience.",
  "Sustainable Gardening: Nurturing the Environment. Sustainable gardening is not "
  "just about growing plants; it's about cultivating an ecosystem. By embracing "
  "environmentally friendly practices, gardeners can have a positive impact on the "
  "planet.",
  "The Smartphone Revolution: Changing Communication. Smartphones have transformed "
  "how we communicate, offering unprecedented connectivity and convenience. This "
  "technology continues to evolve, shaping our daily interactions.",
  "Experiencing African Safaris: Wildlife and Wilderness. An African safari is an "
  "unforgettable experience that brings you face-to-face with the wonders of "
  "wildlife. It's a journey that connects you with the raw beauty of nature.",
  "Graphic Novels: A Blend of Art and Story. Graphic novels offer a unique medium "
  "where art and narrative intertwine to tell compelling stories. They challenge "
  "traditional forms of storytelling, offering visual and textual richness.",
  "Addressing Ocean Pollution: A Call to Action. The increasing levels of pollution "
  "in our oceans are a pressing environmental concern. Protecting marine life and "
  "ecosystems requires concerted global efforts.",
  "The Origins of Hip Hop: A Cultural Movement. Hip hop music, originating from the "
  "streets of New York, has grown into a powerful cultural movement. Its beats and "
  "lyrics reflect the experiences and voices of a community.",
  "Swimming: A Comprehensive Workout. Swimming offers a full-body workout that is "
  "both challenging and refreshing. It's an exercise that enhances cardiovascular "
  "health, builds muscle, and improves endurance.",
  "The Fall of the Berlin Wall: A Historical Turning Point. The fall of the Berlin "
  "Wall was not just a physical demolition; it was a symbol of political and social "
  "change. This historic event marked the end of an era and the beginning of a new "
  "chapter in world history.",
]

# Write the corpus to a file
corpus_file = "/tmp/search_corpus.txt"
with open(corpus_file, "w") as file:
  for sentence in corpus:
      file.write(sentence + "
")

```

### Model Preparation and Configuration in MLflow[​](#model-preparation-and-configuration-in-mlflow "Direct link to Model Preparation and Configuration in MLflow")

Explore the steps to prepare and configure the Sentence Transformer model for integration with MLflow, essential for deployment readiness.

#### Loading and Saving the Sentence Transformer Model[​](#loading-and-saving-the-sentence-transformer-model "Direct link to Loading and Saving the Sentence Transformer Model")

* **Model Initialization**: Loading the `"all-MiniLM-L6-v2"` model, known for its balance in performance and speed, suitable for semantic search tasks.
* **Model Storage**: Saving the model to a directory, essential for later deployment via MLflow. The choice of `/tmp/search_model` is for tutorial convenience so that your current working directory is not filled with the model files. You can change this to any location of your choosing.

#### Preparing Model Artifacts and Signature[​](#preparing-model-artifacts-and-signature "Direct link to Preparing Model Artifacts and Signature")

* **Artifacts Dictionary**: Creating a dictionary with paths to model and corpus file, guiding MLflow to the components that are required to initialize the custom model object.
* **Input Example and Test Output**: Defining sample input and output to illustrate the model's expected data formats.
* **Model Signature**: Using `infer_signature` for automatic signature generation, encompassing input, output, and operational parameters.

#### Importance of the Model Signature[​](#importance-of-the-model-signature "Direct link to Importance of the Model Signature")

The signature ensures data consistency between training and deployment, enhancing model usability and reducing error potential. Having a signature specified ensures that type validation occurs at inference time, preventing unexpected behavior with invalid type conversions that could render incorrect or confusing inference results.

#### Conclusion[​](#conclusion-1 "Direct link to Conclusion")

This comprehensive preparation process guarantees the model is deployment-ready, with all dependencies and operational requirements explicitly defined.

python

```python
# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create an input example DataFrame
input_example = ["Something I want to find matches for."]

# Save the model in the /tmp directory
model_directory = "/tmp/search_model"
model.save(model_directory)

artifacts = {"model_path": model_directory, "corpus_file": corpus_file}

# Generate test output for signature
test_output = ["match 1", "match 2", "match 3"]

# Define the signature associated with the model
signature = infer_signature(
  input_example, test_output, params={"top_k": 3, "minimum_relevancy": 0.2}
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

mlflow.set_experiment("Semantic Similarity")

```

### Logging the Model with MLflow[​](#logging-the-model-with-mlflow "Direct link to Logging the Model with MLflow")

Discover the process of logging the model in MLflow, a crucial step for managing and deploying the model within the MLflow framework.

#### Starting an MLflow Run[​](#starting-an-mlflow-run "Direct link to Starting an MLflow Run")

* **Context Management**: Initiating an MLflow run using `with mlflow.start_run()`, essential for tracking and managing model-related operations.

#### Logging the Model[​](#logging-the-model "Direct link to Logging the Model")

* **Model Logging**: Utilizing `mlflow.pyfunc.log_model` to log the custom `SemanticSearchModel`, including key arguments like model name, instance, input example, signature, artifacts, and requirements.

#### Outcome of Model Logging[​](#outcome-of-model-logging "Direct link to Outcome of Model Logging")

* **Model Registration**: Ensures the model is registered with all necessary components in MLflow, ready for deployment.
* **Reproducibility and Traceability**: Facilitates consistent model deployment and tracks versioning and associated data.

#### Conclusion[​](#conclusion-2 "Direct link to Conclusion")

Completing this critical step transitions the model from development to a deployment-ready state, encapsulated within the MLflow ecosystem.

python

```python
with mlflow.start_run() as run:
  model_info = mlflow.pyfunc.log_model(
      name="semantic_search",
      python_model=SemanticSearchModel(),
      input_example=input_example,
      signature=signature,
      artifacts=artifacts,
      pip_requirements=["sentence_transformers", "numpy"],
  )

```

### Model Inference and Prediction Demonstration[​](#model-inference-and-prediction-demonstration "Direct link to Model Inference and Prediction Demonstration")

Observe the practical application of our semantic search model, demonstrating its ability to respond to user queries with relevant predictions.

#### Loading the Model for Inference[​](#loading-the-model-for-inference "Direct link to Loading the Model for Inference")

* **Model Loading**: Utilizing `mlflow.pyfunc.load_model` to load the model, preparing it to process semantic search queries.

#### Making a Prediction[​](#making-a-prediction "Direct link to Making a Prediction")

* **Running a Query**: Passing a sample query to the loaded model, demonstrating its semantic search capability.

#### Understanding the Prediction Output[​](#understanding-the-prediction-output "Direct link to Understanding the Prediction Output")

* **Output Format**: Analysis of the prediction output, showcasing the model's semantic understanding through relevance scores.
* **Example Results**: Illustrating the model's results, including relevance scores for various query-related entries.

#### Conclusion[​](#conclusion-3 "Direct link to Conclusion")

This demonstration underscores the model's efficacy in semantic search, highlighting its potential in recommendation and knowledge retrieval applications.

python

```python
# Load our model as a PyFuncModel.
# Note that unlike the example shown in the Introductory Tutorial, there is no 'native' flavor for PyFunc models.
# This model cannot be loaded with `mlflow.sentence_transformers.load_model()` because it is not in the native model format.
loaded_dynamic = mlflow.pyfunc.load_model(model_info.model_uri)

# Make sure that it generates a reasonable output
loaded_dynamic.predict(["I'd like some ideas for a meal to cook."])

```

### Advanced Query Handling with Customizable Parameters and Warning Mechanism[​](#advanced-query-handling-with-customizable-parameters-and-warning-mechanism "Direct link to Advanced Query Handling with Customizable Parameters and Warning Mechanism")

Explore the model's advanced features, including customizable search parameters and a unique warning mechanism for optimal user experience.

#### Executing a Customized Prediction with Warnings[​](#executing-a-customized-prediction-with-warnings "Direct link to Executing a Customized Prediction with Warnings")

* **Customized Query with Challenging Parameters**: Testing the model's ability to discern highly relevant content with a high relevancy threshold query.
* **Triggering the Warning**: A mechanism to alert users when search criteria are too restrictive, enhancing user feedback.

#### Understanding the Model's Response[​](#understanding-the-models-response "Direct link to Understanding the Model's Response")

* **Result in Challenging Scenarios**: Analyzing the model's response to stringent search criteria, including cases where the relevancy threshold is not met.

#### Implications and Best Practices[​](#implications-and-best-practices "Direct link to Implications and Best Practices")

* **Balancing Relevancy and Coverage**: Discussing the importance of setting appropriate relevancy thresholds to ensure a balance between precision and result coverage.
* **User Feedback for Corpus Improvement**: Utilizing warnings as feedback for refining the corpus and enhancing the search system.

#### Conclusion[​](#conclusion-4 "Direct link to Conclusion")

This advanced feature set demonstrates the model's adaptability and the importance of fine-tuning search parameters for a dynamic and responsive search experience.

python

```python
# Verify that the fallback logic works correctly by returning the 'best, closest' result, even though the parameters submitted should return no results.
# We are also validating that the warning is issued, alerting us to the fact that this behavior is occurring.
loaded_dynamic.predict(
  ["Latest stories on computing"], params={"top_k": 10, "minimum_relevancy": 0.4}
)

```

### Conclusion: Crafting Custom Logic with MLflow's PythonModel[​](#conclusion-crafting-custom-logic-with-mlflows-pythonmodel "Direct link to Conclusion: Crafting Custom Logic with MLflow's PythonModel")

As we wrap up this tutorial, let's reflect on the key learnings and the powerful capabilities of MLflow's `PythonModel` in crafting custom logic for real-world applications, particularly when integrating advanced libraries like `sentence-transformers`.

#### Key Takeaways[​](#key-takeaways "Direct link to Key Takeaways")

1. **Flexibility of PythonModel**:

   * The `PythonModel` in MLflow offers unparalleled flexibility in defining custom logic. Throughout this tutorial, we leveraged this to build a semantic search model tailored to our specific requirements.
   * This flexibility proves invaluable when dealing with complex use cases that go beyond standard model implementations.

2. **Integration with Sentence Transformers**:

   * We seamlessly integrated the `sentence-transformers` library within our MLflow model. This demonstrated how advanced NLP capabilities can be embedded within custom models to handle sophisticated tasks like semantic search.
   * The use of transformer models for generating embeddings showcased how cutting-edge NLP techniques could be applied in practical scenarios.

3. **Customization and User Experience**:

   * Our model not only performed the core task of semantic search but also allowed for customizable search parameters (`top_k` and `minimum_relevancy`). This level of customization is crucial for aligning the model's output with varying user needs.
   * The inclusion of a warning mechanism further enriched the model by providing valuable feedback, enhancing the user experience.

4. **Real-World Application and Scalability**:

   * While our tutorial focused on a controlled dataset, the principles and methodologies apply to much larger, real-world datasets. The discussion around using vector databases and in-memory databases like Redis or Elasticsearch for scalability highlighted how the model could be adapted for large-scale applications.

#### Empowering Real-World Applications[​](#empowering-real-world-applications "Direct link to Empowering Real-World Applications")

* The combination of MLflow's `PythonModel` and advanced libraries like `sentence-transformers` simplifies the creation of sophisticated, real-world applications.
* The ability to encapsulate complex logic, manage dependencies, and ensure model portability makes MLflow an invaluable tool in the modern data scientist's toolkit.

#### Moving Forward[​](#moving-forward "Direct link to Moving Forward")

* As we conclude, remember that the journey doesn't end here. The concepts and techniques explored in this tutorial lay the groundwork for further exploration and innovation in the field of NLP and beyond.
* We encourage you to take these learnings, experiment with your datasets, and continue pushing the boundaries of what's possible with MLflow and advanced NLP technologies.

Thank you for joining us on this enlightening journey through semantic search with Sentence Transformers and MLflow!
