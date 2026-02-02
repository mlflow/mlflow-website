"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["231"],{41983(e,n,i){i.r(n),i.d(n,{metadata:()=>t,default:()=>g,frontMatter:()=>c,contentTitle:()=>d,toc:()=>u,assets:()=>h});var t=JSON.parse('{"id":"deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers-ipynb","title":"Advanced Semantic Search with Sentence Transformers and MLflow","description":"Download this notebook","source":"@site/docs/classic-ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers-ipynb.mdx","sourceDirName":"deep-learning/sentence-transformers/tutorials/semantic-search","slug":"/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers","permalink":"/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers","draft":false,"unlisted":false,"editUrl":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers.ipynb","tags":[],"version":"current","frontMatter":{"custom_edit_url":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers.ipynb","slug":"semantic-search-sentence-transformers"},"sidebar":"classicMLSidebar","previous":{"title":"Semantic Similarity","permalink":"/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/semantic-similarity/semantic-similarity-sentence-transformers"},"next":{"title":"Paraphrase Mining","permalink":"/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers"}}'),r=i(74848),a=i(28453),s=i(75940),o=i(75453);i(66354);var l=i(42676);let c={custom_edit_url:"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers.ipynb",slug:"semantic-search-sentence-transformers"},d="Advanced Semantic Search with Sentence Transformers and MLflow",h={},u=[{value:"What You Will Learn",id:"what-you-will-learn",level:3},{value:"Understanding Semantic Search",id:"understanding-semantic-search",level:4},{value:"Harnessing Power of Sentence Transformers for Search",id:"harnessing-power-of-sentence-transformers-for-search",level:4},{value:"MLflow: A Vanguard in Model Management and Deployment",id:"mlflow-a-vanguard-in-model-management-and-deployment",level:4},{value:"Understanding the Semantic Search Model with MLflow and Sentence Transformers",id:"understanding-the-semantic-search-model-with-mlflow-and-sentence-transformers",level:3},{value:"MLflow and Custom PyFunc Models",id:"mlflow-and-custom-pyfunc-models",level:4},{value:"The Model&#39;s Core Functionalities",id:"the-models-core-functionalities",level:4},{value:"Detailed Breakdown of Predict Method",id:"detailed-breakdown-of-predict-method",level:4},{value:"Conclusion",id:"conclusion",level:4},{value:"Building and Preparing the Semantic Search Corpus",id:"building-and-preparing-the-semantic-search-corpus",level:3},{value:"Simulating a Real-World Use Case",id:"simulating-a-real-world-use-case",level:4},{value:"Key Steps in Corpus Preparation",id:"key-steps-in-corpus-preparation",level:4},{value:"Efficient Data Handling for Scalability",id:"efficient-data-handling-for-scalability",level:4},{value:"Production Considerations",id:"production-considerations",level:4},{value:"Realizing the Semantic Search Concept",id:"realizing-the-semantic-search-concept",level:4},{value:"Model Preparation and Configuration in MLflow",id:"model-preparation-and-configuration-in-mlflow",level:3},{value:"Loading and Saving the Sentence Transformer Model",id:"loading-and-saving-the-sentence-transformer-model",level:4},{value:"Preparing Model Artifacts and Signature",id:"preparing-model-artifacts-and-signature",level:4},{value:"Importance of the Model Signature",id:"importance-of-the-model-signature",level:4},{value:"Conclusion",id:"conclusion-1",level:4},{value:"Creating an experiment",id:"creating-an-experiment",level:3},{value:"Logging the Model with MLflow",id:"logging-the-model-with-mlflow",level:3},{value:"Starting an MLflow Run",id:"starting-an-mlflow-run",level:4},{value:"Logging the Model",id:"logging-the-model",level:4},{value:"Outcome of Model Logging",id:"outcome-of-model-logging",level:4},{value:"Conclusion",id:"conclusion-2",level:4},{value:"Model Inference and Prediction Demonstration",id:"model-inference-and-prediction-demonstration",level:3},{value:"Loading the Model for Inference",id:"loading-the-model-for-inference",level:4},{value:"Making a Prediction",id:"making-a-prediction",level:4},{value:"Understanding the Prediction Output",id:"understanding-the-prediction-output",level:4},{value:"Conclusion",id:"conclusion-3",level:4},{value:"Advanced Query Handling with Customizable Parameters and Warning Mechanism",id:"advanced-query-handling-with-customizable-parameters-and-warning-mechanism",level:3},{value:"Executing a Customized Prediction with Warnings",id:"executing-a-customized-prediction-with-warnings",level:4},{value:"Understanding the Model&#39;s Response",id:"understanding-the-models-response",level:4},{value:"Implications and Best Practices",id:"implications-and-best-practices",level:4},{value:"Conclusion",id:"conclusion-4",level:4},{value:"Conclusion: Crafting Custom Logic with MLflow&#39;s PythonModel",id:"conclusion-crafting-custom-logic-with-mlflows-pythonmodel",level:3},{value:"Key Takeaways",id:"key-takeaways",level:4},{value:"Empowering Real-World Applications",id:"empowering-real-world-applications",level:4},{value:"Moving Forward",id:"moving-forward",level:4}];function m(e){let n={code:"code",h1:"h1",h3:"h3",h4:"h4",header:"header",li:"li",ol:"ol",p:"p",strong:"strong",ul:"ul",...(0,a.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.header,{children:(0,r.jsx)(n.h1,{id:"advanced-semantic-search-with-sentence-transformers-and-mlflow",children:"Advanced Semantic Search with Sentence Transformers and MLflow"})}),"\n",(0,r.jsx)(l.O,{href:"https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers.ipynb",children:"Download this notebook"}),"\n",(0,r.jsx)(n.p,{children:"Embark on a hands-on journey exploring Advanced Semantic Search using Sentence Transformers and MLflow."}),"\n",(0,r.jsx)(n.h3,{id:"what-you-will-learn",children:"What You Will Learn"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["Implement advanced semantic search with ",(0,r.jsx)(n.code,{children:"sentence-transformers"}),"."]}),"\n",(0,r.jsxs)(n.li,{children:["Customize MLflow's ",(0,r.jsx)(n.code,{children:"PythonModel"})," for unique project requirements."]}),"\n",(0,r.jsx)(n.li,{children:"Manage and log models within MLflow's ecosystem."}),"\n",(0,r.jsx)(n.li,{children:"Deploy complex models for practical applications using MLflow."}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"understanding-semantic-search",children:"Understanding Semantic Search"}),"\n",(0,r.jsx)(n.p,{children:"Semantic search transcends keyword matching, using language nuances and context to find relevant results. This advanced approach reflects human language understanding, considering the varied meanings of words in different scenarios."}),"\n",(0,r.jsx)(n.h4,{id:"harnessing-power-of-sentence-transformers-for-search",children:"Harnessing Power of Sentence Transformers for Search"}),"\n",(0,r.jsx)(n.p,{children:"Sentence Transformers, specialized for context-rich sentence embeddings, transform search queries and text corpora into semantic vectors. This enables the identification of semantically similar entries, a cornerstone of semantic search."}),"\n",(0,r.jsx)(n.h4,{id:"mlflow-a-vanguard-in-model-management-and-deployment",children:"MLflow: A Vanguard in Model Management and Deployment"}),"\n",(0,r.jsx)(n.p,{children:"MLflow enhances NLP projects with efficient experiment logging and customizable model environments. It brings efficiency to experiment tracking and adds a layer of customization, vital for unique NLP tasks."}),"\n",(0,r.jsx)(n.p,{children:"Join us in this tutorial to master advanced semantic search techniques and discover how MLflow can revolutionize your approach to NLP model deployment and management."}),"\n",(0,r.jsx)(s.d,{executionCount:1,children:`import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)`}),"\n",(0,r.jsx)(n.h3,{id:"understanding-the-semantic-search-model-with-mlflow-and-sentence-transformers",children:"Understanding the Semantic Search Model with MLflow and Sentence Transformers"}),"\n",(0,r.jsxs)(n.p,{children:["Delve into the intricacies of the ",(0,r.jsx)(n.code,{children:"SemanticSearchModel"}),", a custom implementation for semantic search using MLflow and Sentence Transformers."]}),"\n",(0,r.jsx)(n.h4,{id:"mlflow-and-custom-pyfunc-models",children:"MLflow and Custom PyFunc Models"}),"\n",(0,r.jsxs)(n.p,{children:["MLflow's custom Python function (",(0,r.jsx)(n.code,{children:"pyfunc"}),") models provide a flexible and deployable solution for integrating complex logic, ideal for our ",(0,r.jsx)(n.code,{children:"SemanticSearchModel"}),"."]}),"\n",(0,r.jsx)(n.h4,{id:"the-models-core-functionalities",children:"The Model's Core Functionalities"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Context Loading"}),": Essential for initializing the Sentence Transformer model and preparing the corpus for semantic comparison."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Predict Method"}),": The central function for semantic search, encompassing input validation, query encoding, and similarity computation."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"detailed-breakdown-of-predict-method",children:"Detailed Breakdown of Predict Method"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Input Validation"}),": Ensures proper format and extraction of the query sentence."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Query Encoding"}),": Converts the query into an embedding for comparison."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Cosine Similarity Computation"}),": Determines the relevance of each corpus entry to the query."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Top Results Extraction"}),": Identifies the most relevant entries based on similarity scores."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Relevancy Filtering"}),": Filters results based on a minimum relevancy threshold, enhancing practical usability."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Warning Mechanism"}),": Issues a warning if all top results are below the relevancy threshold, ensuring a result is always provided."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"conclusion",children:"Conclusion"}),"\n",(0,r.jsx)(n.p,{children:"This semantic search model exemplifies the integration of NLP with MLflow, showcasing flexibility, user-friendliness, and practical application in modern machine learning workflows."}),"\n",(0,r.jsx)(s.d,{executionCount:2,children:`import warnings

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
          return filtered_results`}),"\n",(0,r.jsx)(n.h3,{id:"building-and-preparing-the-semantic-search-corpus",children:"Building and Preparing the Semantic Search Corpus"}),"\n",(0,r.jsx)(n.p,{children:"Explore constructing and preparing the corpus for the semantic search model, a critical component for search functionality."}),"\n",(0,r.jsx)(n.h4,{id:"simulating-a-real-world-use-case",children:"Simulating a Real-World Use Case"}),"\n",(0,r.jsx)(n.p,{children:"We create a simplified corpus of synthetic blog posts to demonstrate the model's core functionality, replicating a scaled-down version of a typical real-world scenario."}),"\n",(0,r.jsx)(n.h4,{id:"key-steps-in-corpus-preparation",children:"Key Steps in Corpus Preparation"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Corpus Creation"}),": Formation of a list representing individual blog post entries."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Writing to a File"}),": Saving the corpus to a text file, mimicking the process of data extraction and preprocessing in a real application."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"efficient-data-handling-for-scalability",children:"Efficient Data Handling for Scalability"}),"\n",(0,r.jsx)(n.p,{children:"Our model encodes the corpus into embeddings for rapid comparison, demonstrating an efficient approach suitable for scaling to larger datasets."}),"\n",(0,r.jsx)(n.h4,{id:"production-considerations",children:"Production Considerations"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Storing Embeddings"}),": Discusses options for efficient storage and retrieval of embeddings, crucial in large-scale applications."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Scalability"}),": Highlights the importance of scalable storage systems for handling extensive datasets and complex queries."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Updating the Corpus"}),": Outlines strategies for managing and updating the corpus in dynamic, evolving use cases."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"realizing-the-semantic-search-concept",children:"Realizing the Semantic Search Concept"}),"\n",(0,r.jsx)(n.p,{children:"This setup, while simplified, reflects the essential steps for developing a robust and scalable semantic search system, combining NLP techniques with efficient data management. In a real production use-case, the processing of a corpus (creating embeddings) would be an external process to that which is running the semantic search. The corpus example below is intended to showcase functionality solely for the purposes of demonstration."}),"\n",(0,r.jsx)(s.d,{executionCount:3,children:`corpus = [
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
")`}),"\n",(0,r.jsx)(n.h3,{id:"model-preparation-and-configuration-in-mlflow",children:"Model Preparation and Configuration in MLflow"}),"\n",(0,r.jsx)(n.p,{children:"Explore the steps to prepare and configure the Sentence Transformer model for integration with MLflow, essential for deployment readiness."}),"\n",(0,r.jsx)(n.h4,{id:"loading-and-saving-the-sentence-transformer-model",children:"Loading and Saving the Sentence Transformer Model"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Model Initialization"}),": Loading the ",(0,r.jsx)(n.code,{children:'"all-MiniLM-L6-v2"'})," model, known for its balance in performance and speed, suitable for semantic search tasks."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Model Storage"}),": Saving the model to a directory, essential for later deployment via MLflow. The choice of ",(0,r.jsx)(n.code,{children:"/tmp/search_model"})," is for tutorial convenience so that your current working directory is not filled with the model files. You can change this to any location of your choosing."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"preparing-model-artifacts-and-signature",children:"Preparing Model Artifacts and Signature"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Artifacts Dictionary"}),": Creating a dictionary with paths to model and corpus file, guiding MLflow to the components that are required to initialize the custom model object."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Input Example and Test Output"}),": Defining sample input and output to illustrate the model's expected data formats."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Model Signature"}),": Using ",(0,r.jsx)(n.code,{children:"infer_signature"})," for automatic signature generation, encompassing input, output, and operational parameters."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"importance-of-the-model-signature",children:"Importance of the Model Signature"}),"\n",(0,r.jsx)(n.p,{children:"The signature ensures data consistency between training and deployment, enhancing model usability and reducing error potential. Having a signature specified ensures that type validation occurs at inference time, preventing unexpected behavior with invalid type conversions that could render incorrect or confusing inference results."}),"\n",(0,r.jsx)(n.h4,{id:"conclusion-1",children:"Conclusion"}),"\n",(0,r.jsx)(n.p,{children:"This comprehensive preparation process guarantees the model is deployment-ready, with all dependencies and operational requirements explicitly defined."}),"\n",(0,r.jsx)(s.d,{executionCount:4,children:`# Load a pre-trained sentence transformer model
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
signature`}),"\n",(0,r.jsx)(o.p,{children:`inputs: 
[string]
outputs: 
[string]
params: 
['top_k': long (default: 3), 'minimum_relevancy': double (default: 0.2)]`}),"\n",(0,r.jsx)(n.h3,{id:"creating-an-experiment",children:"Creating an experiment"}),"\n",(0,r.jsx)(n.p,{children:"We create a new MLflow Experiment so that the run we're going to log our model to does not log to the default experiment and instead has its own contextually relevant entry."}),"\n",(0,r.jsx)(s.d,{executionCount:5,children:`# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

# mlflow.set_tracking_uri("http://127.0.0.1:8080")

mlflow.set_experiment("Semantic Similarity")`}),"\n",(0,r.jsx)(o.p,{children:"<Experiment: artifact_location='file:///Users/benjamin.wilson/repos/mlflow-fork/mlflow/docs/source/llms/sentence-transformers/tutorials/semantic-search/mlruns/405641275158666585', creation_time=1701278766302, experiment_id='405641275158666585', last_update_time=1701278766302, lifecycle_stage='active', name='Semantic Similarity', tags={}>"}),"\n",(0,r.jsx)(n.h3,{id:"logging-the-model-with-mlflow",children:"Logging the Model with MLflow"}),"\n",(0,r.jsx)(n.p,{children:"Discover the process of logging the model in MLflow, a crucial step for managing and deploying the model within the MLflow framework."}),"\n",(0,r.jsx)(n.h4,{id:"starting-an-mlflow-run",children:"Starting an MLflow Run"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Context Management"}),": Initiating an MLflow run using ",(0,r.jsx)(n.code,{children:"with mlflow.start_run()"}),", essential for tracking and managing model-related operations."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"logging-the-model",children:"Logging the Model"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Model Logging"}),": Utilizing ",(0,r.jsx)(n.code,{children:"mlflow.pyfunc.log_model"})," to log the custom ",(0,r.jsx)(n.code,{children:"SemanticSearchModel"}),", including key arguments like model name, instance, input example, signature, artifacts, and requirements."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"outcome-of-model-logging",children:"Outcome of Model Logging"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Model Registration"}),": Ensures the model is registered with all necessary components in MLflow, ready for deployment."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Reproducibility and Traceability"}),": Facilitates consistent model deployment and tracks versioning and associated data."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"conclusion-2",children:"Conclusion"}),"\n",(0,r.jsx)(n.p,{children:"Completing this critical step transitions the model from development to a deployment-ready state, encapsulated within the MLflow ecosystem."}),"\n",(0,r.jsx)(s.d,{executionCount:6,children:`with mlflow.start_run() as run:
  model_info = mlflow.pyfunc.log_model(
      name="semantic_search",
      python_model=SemanticSearchModel(),
      input_example=input_example,
      signature=signature,
      artifacts=artifacts,
      pip_requirements=["sentence_transformers", "numpy"],
  )`}),"\n",(0,r.jsx)(o.p,{children:"Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]"}),"\n",(0,r.jsx)(o.p,{isStderr:!0,children:"2023/11/30 15:57:53 INFO mlflow.store.artifact.artifact_repo: The progress bar can be disabled by setting the environment variable MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR to false"}),"\n",(0,r.jsx)(o.p,{children:"Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]"}),"\n",(0,r.jsx)(n.h3,{id:"model-inference-and-prediction-demonstration",children:"Model Inference and Prediction Demonstration"}),"\n",(0,r.jsx)(n.p,{children:"Observe the practical application of our semantic search model, demonstrating its ability to respond to user queries with relevant predictions."}),"\n",(0,r.jsx)(n.h4,{id:"loading-the-model-for-inference",children:"Loading the Model for Inference"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Model Loading"}),": Utilizing ",(0,r.jsx)(n.code,{children:"mlflow.pyfunc.load_model"})," to load the model, preparing it to process semantic search queries."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"making-a-prediction",children:"Making a Prediction"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Running a Query"}),": Passing a sample query to the loaded model, demonstrating its semantic search capability."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"understanding-the-prediction-output",children:"Understanding the Prediction Output"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Output Format"}),": Analysis of the prediction output, showcasing the model's semantic understanding through relevance scores."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Example Results"}),": Illustrating the model's results, including relevance scores for various query-related entries."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"conclusion-3",children:"Conclusion"}),"\n",(0,r.jsx)(n.p,{children:"This demonstration underscores the model's efficacy in semantic search, highlighting its potential in recommendation and knowledge retrieval applications."}),"\n",(0,r.jsx)(s.d,{executionCount:7,children:`# Load our model as a PyFuncModel.
# Note that unlike the example shown in the Introductory Tutorial, there is no 'native' flavor for PyFunc models.
# This model cannot be loaded with \`mlflow.sentence_transformers.load_model()\` because it is not in the native model format.
loaded_dynamic = mlflow.pyfunc.load_model(model_info.model_uri)

# Make sure that it generates a reasonable output
loaded_dynamic.predict(["I'd like some ideas for a meal to cook."])`}),"\n",(0,r.jsx)(o.p,{children:`[('Exploring International Cuisines: A Culinary Adventure. Discovering international cuisines is an adventure for the palate. Each dish offers a glimpse into the culture and traditions of its origin.',
0.43857115507125854),
('Vegan Cuisine: A World of Flavor. Exploring vegan cuisine reveals a world of nutritious and delicious possibilities. From hearty soups to delectable desserts, plant-based dishes are diverse and satisfying.',
0.34688490629196167),
("The Art of Growing Herbs: Enhancing Your Culinary Skills. Growing your own herbs can transform your cooking, adding fresh and vibrant flavors. Whether it's basil, thyme, or rosemary, each herb has its own unique characteristics.",
0.22686949372291565)]`}),"\n",(0,r.jsx)(n.h3,{id:"advanced-query-handling-with-customizable-parameters-and-warning-mechanism",children:"Advanced Query Handling with Customizable Parameters and Warning Mechanism"}),"\n",(0,r.jsx)(n.p,{children:"Explore the model's advanced features, including customizable search parameters and a unique warning mechanism for optimal user experience."}),"\n",(0,r.jsx)(n.h4,{id:"executing-a-customized-prediction-with-warnings",children:"Executing a Customized Prediction with Warnings"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Customized Query with Challenging Parameters"}),": Testing the model's ability to discern highly relevant content with a high relevancy threshold query."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Triggering the Warning"}),": A mechanism to alert users when search criteria are too restrictive, enhancing user feedback."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"understanding-the-models-response",children:"Understanding the Model's Response"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Result in Challenging Scenarios"}),": Analyzing the model's response to stringent search criteria, including cases where the relevancy threshold is not met."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"implications-and-best-practices",children:"Implications and Best Practices"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Balancing Relevancy and Coverage"}),": Discussing the importance of setting appropriate relevancy thresholds to ensure a balance between precision and result coverage."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"User Feedback for Corpus Improvement"}),": Utilizing warnings as feedback for refining the corpus and enhancing the search system."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"conclusion-4",children:"Conclusion"}),"\n",(0,r.jsx)(n.p,{children:"This advanced feature set demonstrates the model's adaptability and the importance of fine-tuning search parameters for a dynamic and responsive search experience."}),"\n",(0,r.jsx)(s.d,{executionCount:8,children:`# Verify that the fallback logic works correctly by returning the 'best, closest' result, even though the parameters submitted should return no results.
# We are also validating that the warning is issued, alerting us to the fact that this behavior is occurring.
loaded_dynamic.predict(
  ["Latest stories on computing"], params={"top_k": 10, "minimum_relevancy": 0.4}
)`}),"\n",(0,r.jsx)(o.p,{isStderr:!0,children:`/var/folders/cd/n8n0rm2x53l_s0xv_j_xklb00000gp/T/ipykernel_55915/1325605132.py:71: RuntimeWarning: All top results are below the minimum relevancy threshold. Returning the highest match instead.
warnings.warn(`}),"\n",(0,r.jsx)(o.p,{children:`[('AI in Software Development: Transforming the Tech Landscape. The rapid advancements in artificial intelligence are reshaping how we approach software development. From automation to machine learning, the possibilities are endless.',
0.2533860206604004)]`}),"\n",(0,r.jsx)(n.h3,{id:"conclusion-crafting-custom-logic-with-mlflows-pythonmodel",children:"Conclusion: Crafting Custom Logic with MLflow's PythonModel"}),"\n",(0,r.jsxs)(n.p,{children:["As we wrap up this tutorial, let's reflect on the key learnings and the powerful capabilities of MLflow's ",(0,r.jsx)(n.code,{children:"PythonModel"})," in crafting custom logic for real-world applications, particularly when integrating advanced libraries like ",(0,r.jsx)(n.code,{children:"sentence-transformers"}),"."]}),"\n",(0,r.jsx)(n.h4,{id:"key-takeaways",children:"Key Takeaways"}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Flexibility of PythonModel"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["The ",(0,r.jsx)(n.code,{children:"PythonModel"})," in MLflow offers unparalleled flexibility in defining custom logic. Throughout this tutorial, we leveraged this to build a semantic search model tailored to our specific requirements."]}),"\n",(0,r.jsx)(n.li,{children:"This flexibility proves invaluable when dealing with complex use cases that go beyond standard model implementations."}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Integration with Sentence Transformers"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["We seamlessly integrated the ",(0,r.jsx)(n.code,{children:"sentence-transformers"})," library within our MLflow model. This demonstrated how advanced NLP capabilities can be embedded within custom models to handle sophisticated tasks like semantic search."]}),"\n",(0,r.jsx)(n.li,{children:"The use of transformer models for generating embeddings showcased how cutting-edge NLP techniques could be applied in practical scenarios."}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Customization and User Experience"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["Our model not only performed the core task of semantic search but also allowed for customizable search parameters (",(0,r.jsx)(n.code,{children:"top_k"})," and ",(0,r.jsx)(n.code,{children:"minimum_relevancy"}),"). This level of customization is crucial for aligning the model's output with varying user needs."]}),"\n",(0,r.jsx)(n.li,{children:"The inclusion of a warning mechanism further enriched the model by providing valuable feedback, enhancing the user experience."}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Real-World Application and Scalability"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:"While our tutorial focused on a controlled dataset, the principles and methodologies apply to much larger, real-world datasets. The discussion around using vector databases and in-memory databases like Redis or Elasticsearch for scalability highlighted how the model could be adapted for large-scale applications."}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"empowering-real-world-applications",children:"Empowering Real-World Applications"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["The combination of MLflow's ",(0,r.jsx)(n.code,{children:"PythonModel"})," and advanced libraries like ",(0,r.jsx)(n.code,{children:"sentence-transformers"})," simplifies the creation of sophisticated, real-world applications."]}),"\n",(0,r.jsx)(n.li,{children:"The ability to encapsulate complex logic, manage dependencies, and ensure model portability makes MLflow an invaluable tool in the modern data scientist's toolkit."}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"moving-forward",children:"Moving Forward"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:"As we conclude, remember that the journey doesn't end here. The concepts and techniques explored in this tutorial lay the groundwork for further exploration and innovation in the field of NLP and beyond."}),"\n",(0,r.jsx)(n.li,{children:"We encourage you to take these learnings, experiment with your datasets, and continue pushing the boundaries of what's possible with MLflow and advanced NLP technologies."}),"\n"]}),"\n",(0,r.jsx)(n.p,{children:"Thank you for joining us on this enlightening journey through semantic search with Sentence Transformers and MLflow!"})]})}function g(e={}){let{wrapper:n}={...(0,a.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(m,{...e})}):m(e)}},75453(e,n,i){i.d(n,{p:()=>r});var t=i(74848);let r=({children:e,isStderr:n})=>(0,t.jsx)("pre",{style:{margin:0,borderRadius:0,background:"none",fontSize:"0.85rem",flexGrow:1,padding:"var(--padding-sm)"},children:e})},75940(e,n,i){i.d(n,{d:()=>a});var t=i(74848),r=i(37449);let a=({children:e,executionCount:n})=>(0,t.jsx)("div",{style:{flexGrow:1,minWidth:0,marginTop:"var(--padding-md)",width:"100%"},children:(0,t.jsx)(r.A,{className:"codeBlock_oJcR",language:"python",children:e})})},42676(e,n,i){i.d(n,{O:()=>s});var t=i(74848),r=i(96540);let a="3.9.1.dev0";function s({children:e,href:n}){let i=(0,r.useCallback)(async e=>{if(e.preventDefault(),window.gtag)try{window.gtag("event","notebook-download",{href:n})}catch{}a.includes("dev")||(n=n.replace(/\/master\//,`/v${a}/`));let i=await fetch(n),t=await i.blob(),r=window.URL.createObjectURL(t),s=document.createElement("a");s.style.display="none",s.href=r,s.download=n.split("/").pop(),document.body.appendChild(s),s.click(),window.URL.revokeObjectURL(r),document.body.removeChild(s)},[n]);return(0,t.jsx)("a",{className:"button button--primary",style:{marginBottom:"1rem",display:"block",width:"min-content"},href:n,download:!0,onClick:i,children:e})}},66354(e,n,i){i.d(n,{Q:()=>r});var t=i(74848);let r=({children:e})=>(0,t.jsx)("div",{style:{flexGrow:1,minWidth:0,fontSize:"0.8rem",width:"100%"},children:e})},52915(e,n,i){i.d(n,{A:()=>h});var t=i(74848);i(96540);var r=i(34164),a=i(71643),s=i(66697),o=i(92949),l=i(64560),c=i(47819);function d({language:e}){return(0,t.jsxs)("div",{className:(0,r.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,t.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,t.jsx)(c.A,{})]})}function h({className:e}){let{metadata:n}=(0,a.Ph)(),i=n.language||"text";return(0,t.jsxs)(s.A,{as:"div",className:(0,r.A)(e,n.className),children:[n.title&&(0,t.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,t.jsx)(o.A,{children:n.title})}),(0,t.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,t.jsx)(d,{language:i}),(0,t.jsx)(l.A,{})]})]})}}}]);