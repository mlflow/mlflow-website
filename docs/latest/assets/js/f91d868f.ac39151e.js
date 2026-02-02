"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["148"],{49721(e,n,i){i.r(n),i.d(n,{metadata:()=>a,default:()=>m,frontMatter:()=>d,contentTitle:()=>c,toc:()=>p,assets:()=>h});var a=JSON.parse('{"id":"deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers-ipynb","title":"Advanced Paraphrase Mining with Sentence Transformers and MLflow","description":"Download this notebook","source":"@site/docs/classic-ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers-ipynb.mdx","sourceDirName":"deep-learning/sentence-transformers/tutorials/paraphrase-mining","slug":"/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers","permalink":"/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers","draft":false,"unlisted":false,"editUrl":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.ipynb","tags":[],"version":"current","frontMatter":{"custom_edit_url":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.ipynb","slug":"paraphrase-mining-sentence-transformers"},"sidebar":"classicMLSidebar","previous":{"title":"Semantic Search","permalink":"/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers"},"next":{"title":"spaCy","permalink":"/mlflow-website/docs/latest/ml/deep-learning/spacy/"}}'),t=i(74848),r=i(28453),s=i(75940),o=i(75453);i(66354);var l=i(42676);let d={custom_edit_url:"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.ipynb",slug:"paraphrase-mining-sentence-transformers"},c="Advanced Paraphrase Mining with Sentence Transformers and MLflow",h={},p=[{value:"Learning Objectives",id:"learning-objectives",level:3},{value:"Exploring Paraphrase Mining",id:"exploring-paraphrase-mining",level:4},{value:"The Role of Sentence Transformers in Paraphrase Mining",id:"the-role-of-sentence-transformers-in-paraphrase-mining",level:4},{value:"MLflow: Simplifying Model Management and Deployment",id:"mlflow-simplifying-model-management-and-deployment",level:4},{value:"Introduction to the Paraphrase Mining Model",id:"introduction-to-the-paraphrase-mining-model",level:3},{value:"Overview of the Model Structure",id:"overview-of-the-model-structure",level:4},{value:"Key Features",id:"key-features",level:4},{value:"Practical Implications",id:"practical-implications",level:4},{value:"Preparing the Corpus for Paraphrase Mining",id:"preparing-the-corpus-for-paraphrase-mining",level:3},{value:"Corpus Creation",id:"corpus-creation",level:4},{value:"Writing the Corpus to a File",id:"writing-the-corpus-to-a-file",level:4},{value:"Significance of the Corpus",id:"significance-of-the-corpus",level:4},{value:"Setting Up the Paraphrase Mining Model",id:"setting-up-the-paraphrase-mining-model",level:3},{value:"Loading the Sentence Transformer Model",id:"loading-the-sentence-transformer-model",level:4},{value:"Preparing the Input Example",id:"preparing-the-input-example",level:4},{value:"Saving the Model",id:"saving-the-model",level:4},{value:"Defining Artifacts and Corpus Path",id:"defining-artifacts-and-corpus-path",level:4},{value:"Generating Test Output for Signature",id:"generating-test-output-for-signature",level:4},{value:"Creating the Model Signature",id:"creating-the-model-signature",level:4},{value:"Creating an experiment",id:"creating-an-experiment",level:3},{value:"Logging the Paraphrase Mining Model with MLflow",id:"logging-the-paraphrase-mining-model-with-mlflow",level:3},{value:"Initiating an MLflow Run",id:"initiating-an-mlflow-run",level:4},{value:"Logging the Model in MLflow",id:"logging-the-model-in-mlflow",level:4},{value:"Outcomes and Benefits of Model Logging",id:"outcomes-and-benefits-of-model-logging",level:4},{value:"Model Loading and Paraphrase Mining Prediction",id:"model-loading-and-paraphrase-mining-prediction",level:3},{value:"Loading the Model for Inference",id:"loading-the-model-for-inference",level:4},{value:"Executing a Paraphrase Mining Prediction",id:"executing-a-paraphrase-mining-prediction",level:4},{value:"Interpreting the Model Output",id:"interpreting-the-model-output",level:4},{value:"Conclusion",id:"conclusion",level:4},{value:"Conclusion: Insights and Potential Enhancements",id:"conclusion-insights-and-potential-enhancements",level:3},{value:"Key Takeaways",id:"key-takeaways",level:4},{value:"Ideas for Enhancing the Paraphrase Mining Model",id:"ideas-for-enhancing-the-paraphrase-mining-model",level:4},{value:"Scalability with Vector Databases",id:"scalability-with-vector-databases",level:4},{value:"Final Thoughts",id:"final-thoughts",level:4}];function u(e){let n={code:"code",h1:"h1",h3:"h3",h4:"h4",header:"header",li:"li",ol:"ol",p:"p",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.header,{children:(0,t.jsx)(n.h1,{id:"advanced-paraphrase-mining-with-sentence-transformers-and-mlflow",children:"Advanced Paraphrase Mining with Sentence Transformers and MLflow"})}),"\n",(0,t.jsx)(l.O,{href:"https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.ipynb",children:"Download this notebook"}),"\n",(0,t.jsx)(n.p,{children:"Embark on an enriching journey through advanced paraphrase mining using Sentence Transformers, enhanced by MLflow."}),"\n",(0,t.jsx)(n.h3,{id:"learning-objectives",children:"Learning Objectives"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["Apply ",(0,t.jsx)(n.code,{children:"sentence-transformers"})," for advanced paraphrase mining."]}),"\n",(0,t.jsxs)(n.li,{children:["Develop a custom ",(0,t.jsx)(n.code,{children:"PythonModel"})," in MLflow tailored for this task."]}),"\n",(0,t.jsx)(n.li,{children:"Effectively manage and track models within the MLflow ecosystem."}),"\n",(0,t.jsx)(n.li,{children:"Deploy paraphrase mining models using MLflow's deployment capabilities."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"exploring-paraphrase-mining",children:"Exploring Paraphrase Mining"}),"\n",(0,t.jsx)(n.p,{children:"Discover the process of identifying semantically similar but textually distinct sentences, a key aspect in various NLP applications such as document summarization and chatbot development."}),"\n",(0,t.jsx)(n.h4,{id:"the-role-of-sentence-transformers-in-paraphrase-mining",children:"The Role of Sentence Transformers in Paraphrase Mining"}),"\n",(0,t.jsx)(n.p,{children:"Learn how Sentence Transformers, specialized for generating rich sentence embeddings, are used to capture deep semantic meanings and compare textual content."}),"\n",(0,t.jsx)(n.h4,{id:"mlflow-simplifying-model-management-and-deployment",children:"MLflow: Simplifying Model Management and Deployment"}),"\n",(0,t.jsx)(n.p,{children:"Delve into how MLflow streamlines the process of managing and deploying NLP models, with a focus on efficient tracking and customizable model implementations."}),"\n",(0,t.jsx)(n.p,{children:"Join us to develop a nuanced understanding of paraphrase mining and master the art of managing and deploying NLP models with MLflow."}),"\n",(0,t.jsx)(s.d,{executionCount:1,children:`import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)`}),"\n",(0,t.jsx)(n.h3,{id:"introduction-to-the-paraphrase-mining-model",children:"Introduction to the Paraphrase Mining Model"}),"\n",(0,t.jsx)(n.p,{children:"Initiate the Paraphrase Mining Model, integrating Sentence Transformers and MLflow for advanced NLP tasks."}),"\n",(0,t.jsx)(n.h4,{id:"overview-of-the-model-structure",children:"Overview of the Model Structure"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsxs)(n.strong,{children:["Loading Model and Corpus ",(0,t.jsx)(n.code,{children:"load_context"})," Method"]}),": Essential for loading the Sentence Transformer model and the text corpus for paraphrase identification."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsxs)(n.strong,{children:["Paraphrase Mining Logic ",(0,t.jsx)(n.code,{children:"predict"})," Method"]}),": Integrates custom logic for input validation and paraphrase mining, offering customizable parameters."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsxs)(n.strong,{children:["Sorting and Filtering Matches ",(0,t.jsx)(n.code,{children:"_sort_and_filter_matches"})," Helper Method"]}),": Ensures relevant and unique paraphrase identification by sorting and filtering based on similarity scores."]}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"key-features",children:"Key Features"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Advanced NLP Techniques"}),": Utilizes Sentence Transformers for semantic text understanding."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Custom Logic Integration"}),": Demonstrates flexibility in model behavior customization."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"User Customization Options"}),": Allows end users to adjust match criteria for various use cases."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Efficiency in Processing"}),": Pre-encodes the corpus for efficient paraphrase mining operations."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Robust Error Handling"}),": Incorporates validations for reliable model performance."]}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"practical-implications",children:"Practical Implications"}),"\n",(0,t.jsx)(n.p,{children:"This model provides a powerful tool for paraphrase detection in diverse applications, exemplifying the effective use of custom models within the MLflow framework."}),"\n",(0,t.jsx)(s.d,{executionCount:2,children:`import warnings

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

      return {sentence[0]: str(sentence[1]) for sentence in sorted_paraphrases}`}),"\n",(0,t.jsx)(n.h3,{id:"preparing-the-corpus-for-paraphrase-mining",children:"Preparing the Corpus for Paraphrase Mining"}),"\n",(0,t.jsx)(n.p,{children:"Set up the foundation for paraphrase mining by creating and preparing a diverse corpus."}),"\n",(0,t.jsx)(n.h4,{id:"corpus-creation",children:"Corpus Creation"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["Define a ",(0,t.jsx)(n.code,{children:"corpus"})," comprising a range of sentences from various topics, including space exploration, AI, gardening, and more. This diversity enables the model to identify paraphrases across a broad spectrum of subjects."]}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"writing-the-corpus-to-a-file",children:"Writing the Corpus to a File"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["The corpus is saved to a file named ",(0,t.jsx)(n.code,{children:"feedback.txt"}),", mirroring a common practice in large-scale data handling."]}),"\n",(0,t.jsx)(n.li,{children:"This step also prepares the corpus for efficient processing within the Paraphrase Mining Model."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"significance-of-the-corpus",children:"Significance of the Corpus"}),"\n",(0,t.jsx)(n.p,{children:"The corpus serves as the key dataset for the model to find semantically similar sentences. Its variety ensures the model's adaptability and effectiveness across diverse use cases."}),"\n",(0,t.jsx)(s.d,{executionCount:3,children:`corpus = [
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
")`}),"\n",(0,t.jsx)(n.h3,{id:"setting-up-the-paraphrase-mining-model",children:"Setting Up the Paraphrase Mining Model"}),"\n",(0,t.jsx)(n.p,{children:"Prepare the Sentence Transformer model for integration with MLflow to harness its paraphrase mining capabilities."}),"\n",(0,t.jsx)(n.h4,{id:"loading-the-sentence-transformer-model",children:"Loading the Sentence Transformer Model"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["Initialize the ",(0,t.jsx)(n.code,{children:"all-MiniLM-L6-v2"})," Sentence Transformer model, ideal for generating sentence embeddings suitable for paraphrase mining."]}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"preparing-the-input-example",children:"Preparing the Input Example"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Create a DataFrame as an input example to illustrate the type of query the model will handle, aiding in defining the model's input structure."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"saving-the-model",children:"Saving the Model"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["Save the model to ",(0,t.jsx)(n.code,{children:"/tmp/paraphrase_search_model"})," for portability and ease of loading during deployment with MLflow."]}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"defining-artifacts-and-corpus-path",children:"Defining Artifacts and Corpus Path"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Specify paths to the saved model and corpus as artifacts in MLflow, crucial for model logging and reproduction."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"generating-test-output-for-signature",children:"Generating Test Output for Signature"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Generate a sample output, illustrating the model's expected output format for paraphrase mining."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"creating-the-model-signature",children:"Creating the Model Signature"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["Use MLflow's ",(0,t.jsx)(n.code,{children:"infer_signature"})," to define the model's input and output schema, adding the ",(0,t.jsx)(n.code,{children:"similarity_threshold"})," parameter for inference flexibility."]}),"\n"]}),"\n",(0,t.jsx)(s.d,{executionCount:4,children:`# Load a pre-trained sentence transformer model
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
signature`}),"\n",(0,t.jsx)(o.p,{children:`inputs: 
['query': string]
outputs: 
['This product is satisfactory and functions as expected.': string]
params: 
['similarity_threshold': double (default: 0.5)]`}),"\n",(0,t.jsx)(n.h3,{id:"creating-an-experiment",children:"Creating an experiment"}),"\n",(0,t.jsx)(n.p,{children:"We create a new MLflow Experiment so that the run we're going to log our model to does not log to the default experiment and instead has its own contextually relevant entry."}),"\n",(0,t.jsx)(s.d,{executionCount:5,children:`# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

# mlflow.set_tracking_uri("http://127.0.0.1:8080")

mlflow.set_experiment("Paraphrase Mining")`}),"\n",(0,t.jsx)(o.p,{children:"<Experiment: artifact_location='file:///Users/benjamin.wilson/repos/mlflow-fork/mlflow/docs/source/llms/sentence-transformers/tutorials/paraphrase-mining/mlruns/380691166097743403', creation_time=1701282619556, experiment_id='380691166097743403', last_update_time=1701282619556, lifecycle_stage='active', name='Paraphrase Mining', tags={}>"}),"\n",(0,t.jsx)(n.h3,{id:"logging-the-paraphrase-mining-model-with-mlflow",children:"Logging the Paraphrase Mining Model with MLflow"}),"\n",(0,t.jsx)(n.p,{children:"Log the custom Paraphrase Mining Model with MLflow, a key step for model management and deployment."}),"\n",(0,t.jsx)(n.h4,{id:"initiating-an-mlflow-run",children:"Initiating an MLflow Run"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Start an MLflow run to create a comprehensive record of model logging and tracking within the MLflow framework."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"logging-the-model-in-mlflow",children:"Logging the Model in MLflow"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Use MLflow's Python model logging function to integrate the custom model into the MLflow ecosystem."}),"\n",(0,t.jsx)(n.li,{children:"Provide a unique name for the model for easy identification in MLflow."}),"\n",(0,t.jsx)(n.li,{children:"Log the instantiated Paraphrase Mining Model, along with an input example, model signature, artifacts, and Python dependencies."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"outcomes-and-benefits-of-model-logging",children:"Outcomes and Benefits of Model Logging"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Register the model within MLflow for streamlined management and deployment, enhancing its accessibility and trackability."}),"\n",(0,t.jsx)(n.li,{children:"Ensure model reproducibility and version control across deployment environments."}),"\n"]}),"\n",(0,t.jsx)(s.d,{executionCount:6,children:`with mlflow.start_run() as run:
  model_info = mlflow.pyfunc.log_model(
      name="paraphrase_model",
      python_model=ParaphraseMiningModel(),
      input_example=input_example,
      signature=signature,
      artifacts=artifacts,
      pip_requirements=["sentence_transformers"],
  )`}),"\n",(0,t.jsx)(o.p,{children:"Downloading artifacts:   0%|          | 0/11 [00:00<?, ?it/s]"}),"\n",(0,t.jsx)(o.p,{isStderr:!0,children:"2023/11/30 15:41:39 INFO mlflow.store.artifact.artifact_repo: The progress bar can be disabled by setting the environment variable MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR to false"}),"\n",(0,t.jsx)(o.p,{children:"Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]"}),"\n",(0,t.jsx)(n.h3,{id:"model-loading-and-paraphrase-mining-prediction",children:"Model Loading and Paraphrase Mining Prediction"}),"\n",(0,t.jsx)(n.p,{children:"Illustrate the real-world application of the Paraphrase Mining Model by loading it with MLflow and executing a prediction."}),"\n",(0,t.jsx)(n.h4,{id:"loading-the-model-for-inference",children:"Loading the Model for Inference"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["Utilize MLflow's ",(0,t.jsx)(n.code,{children:"load_model"})," function to retrieve and prepare the model for inference."]}),"\n",(0,t.jsx)(n.li,{children:"Locate and load the model using its unique URI within the MLflow registry."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"executing-a-paraphrase-mining-prediction",children:"Executing a Paraphrase Mining Prediction"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["Make a prediction using the model's ",(0,t.jsx)(n.code,{children:"predict"})," method, applying the paraphrase mining logic embedded in the model class."]}),"\n",(0,t.jsxs)(n.li,{children:["Pass a representative query with a set ",(0,t.jsx)(n.code,{children:"similarity_threshold"})," to find matching paraphrases in the corpus."]}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"interpreting-the-model-output",children:"Interpreting the Model Output"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Review the list of semantically similar sentences to the query, highlighting the model's paraphrase identification capabilities."}),"\n",(0,t.jsx)(n.li,{children:"Analyze the similarity scores to understand the degree of semantic relatedness between the query and corpus sentences."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"conclusion",children:"Conclusion"}),"\n",(0,t.jsx)(n.p,{children:"This demonstration validates the Paraphrase Mining Model's effectiveness in real-world scenarios, underscoring its utility in content recommendation, information retrieval, and conversational AI."}),"\n",(0,t.jsx)(s.d,{executionCount:7,children:`# Load our model by supplying the uri that was used to save the model artifacts
loaded_dynamic = mlflow.pyfunc.load_model(model_info.model_uri)

# Perform a quick validation that our loaded model is performing adequately
loaded_dynamic.predict(
  {"query": "Space exploration is fascinating."}, params={"similarity_threshold": 0.65}
)`}),"\n",(0,t.jsx)(o.p,{children:`{'Studying the stars helps us understand our universe.': '0.8207424879074097',
'The history of space exploration is filled with remarkable achievements.': '0.7770636677742004',
'Exploring ancient cities in Europe offers a glimpse into history.': '0.7461957335472107',
'Space travel advancements could lead to interplanetary exploration.': '0.7090306282043457',
'Space exploration continues to push the boundaries of human knowledge.': '0.6893945932388306',
'The mysteries of the universe are unveiled through space missions.': '0.6830739974975586',
'The study of celestial bodies helps us understand the cosmos.': '0.671358048915863'}`}),"\n",(0,t.jsx)(n.h3,{id:"conclusion-insights-and-potential-enhancements",children:"Conclusion: Insights and Potential Enhancements"}),"\n",(0,t.jsxs)(n.p,{children:["As we wrap up this tutorial, let's reflect on our journey through the implementation of a Paraphrase Mining Model using Sentence Transformers and MLflow. We've successfully built and deployed a model capable of identifying semantically similar sentences, showcasing the flexibility and power of MLflow's ",(0,t.jsx)(n.code,{children:"PythonModel"})," implementation."]}),"\n",(0,t.jsx)(n.h4,{id:"key-takeaways",children:"Key Takeaways"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"We learned how to integrate advanced NLP techniques, specifically paraphrase mining, with MLflow. This integration not only enhances model management but also simplifies deployment and scalability."}),"\n",(0,t.jsxs)(n.li,{children:["The flexibility of the ",(0,t.jsx)(n.code,{children:"PythonModel"})," implementation in MLflow was a central theme. We saw firsthand how it allows for the incorporation of custom logic into the model's predict function, catering to specific NLP tasks like paraphrase mining."]}),"\n",(0,t.jsx)(n.li,{children:"Through our custom model, we explored the dynamics of sentence embeddings, semantic similarity, and the nuances of language understanding. This understanding is crucial in a wide range of applications, from content recommendation to conversational AI."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"ideas-for-enhancing-the-paraphrase-mining-model",children:"Ideas for Enhancing the Paraphrase Mining Model"}),"\n",(0,t.jsxs)(n.p,{children:["While our model serves as a robust starting point, there are several enhancements that could be made within the ",(0,t.jsx)(n.code,{children:"predict"})," function to make it more powerful and feature-rich:"]}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsxs)(n.p,{children:[(0,t.jsx)(n.strong,{children:"Contextual Filters"}),": Introduce filters based on contextual clues or specific keywords to refine the search results further. This feature would allow users to narrow down paraphrases to those most relevant to their particular context or subject matter."]}),"\n"]}),"\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsxs)(n.p,{children:[(0,t.jsx)(n.strong,{children:"Sentiment Analysis Integration"}),": Incorporate sentiment analysis to group paraphrases by their emotional tone. This would be especially useful in applications like customer feedback analysis, where understanding sentiment is as important as content."]}),"\n"]}),"\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsxs)(n.p,{children:[(0,t.jsx)(n.strong,{children:"Multi-Lingual Support"}),": Expand the model to support paraphrase mining in multiple languages. This enhancement would significantly broaden the model's applicability in global or multi-lingual contexts."]}),"\n"]}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"scalability-with-vector-databases",children:"Scalability with Vector Databases"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Moving beyond a static text file as a corpus, a more scalable and real-world approach would involve connecting the model to an external vector database or in-memory store."}),"\n",(0,t.jsx)(n.li,{children:"Pre-calculated embeddings could be stored and updated in such databases, accommodating real-time content generation without requiring model redeployment. This approach would dramatically improve the model's scalability and responsiveness in real-world applications."}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"final-thoughts",children:"Final Thoughts"}),"\n",(0,t.jsxs)(n.p,{children:["The journey through building and deploying the Paraphrase Mining Model has been both enlightening and practical. We've seen how MLflow's ",(0,t.jsx)(n.code,{children:"PythonModel"})," offers a flexible canvas for crafting custom NLP solutions, and how sentence transformers can be leveraged to delve deep into the semantics of language."]}),"\n",(0,t.jsx)(n.p,{children:"This tutorial is just the beginning. There's a vast potential for further exploration and innovation in paraphrase mining and NLP as a whole. We encourage you to build upon this foundation, experiment with enhancements, and continue pushing the boundaries of what's possible with MLflow and advanced NLP techniques."})]})}function m(e={}){let{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(u,{...e})}):u(e)}},75453(e,n,i){i.d(n,{p:()=>t});var a=i(74848);let t=({children:e,isStderr:n})=>(0,a.jsx)("pre",{style:{margin:0,borderRadius:0,background:"none",fontSize:"0.85rem",flexGrow:1,padding:"var(--padding-sm)"},children:e})},75940(e,n,i){i.d(n,{d:()=>r});var a=i(74848),t=i(37449);let r=({children:e,executionCount:n})=>(0,a.jsx)("div",{style:{flexGrow:1,minWidth:0,marginTop:"var(--padding-md)",width:"100%"},children:(0,a.jsx)(t.A,{className:"codeBlock_oJcR",language:"python",children:e})})},42676(e,n,i){i.d(n,{O:()=>s});var a=i(74848),t=i(96540);let r="3.9.1.dev0";function s({children:e,href:n}){let i=(0,t.useCallback)(async e=>{if(e.preventDefault(),window.gtag)try{window.gtag("event","notebook-download",{href:n})}catch{}r.includes("dev")||(n=n.replace(/\/master\//,`/v${r}/`));let i=await fetch(n),a=await i.blob(),t=window.URL.createObjectURL(a),s=document.createElement("a");s.style.display="none",s.href=t,s.download=n.split("/").pop(),document.body.appendChild(s),s.click(),window.URL.revokeObjectURL(t),document.body.removeChild(s)},[n]);return(0,a.jsx)("a",{className:"button button--primary",style:{marginBottom:"1rem",display:"block",width:"min-content"},href:n,download:!0,onClick:i,children:e})}},66354(e,n,i){i.d(n,{Q:()=>t});var a=i(74848);let t=({children:e})=>(0,a.jsx)("div",{style:{flexGrow:1,minWidth:0,fontSize:"0.8rem",width:"100%"},children:e})},52915(e,n,i){i.d(n,{A:()=>h});var a=i(74848);i(96540);var t=i(34164),r=i(71643),s=i(66697),o=i(92949),l=i(64560),d=i(47819);function c({language:e}){return(0,a.jsxs)("div",{className:(0,t.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,a.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,a.jsx)(d.A,{})]})}function h({className:e}){let{metadata:n}=(0,r.Ph)(),i=n.language||"text";return(0,a.jsxs)(s.A,{as:"div",className:(0,t.A)(e,n.className),children:[n.title&&(0,a.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,a.jsx)(o.A,{children:n.title})}),(0,a.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,a.jsx)(c,{language:i}),(0,a.jsx)(l.A,{})]})]})}}}]);