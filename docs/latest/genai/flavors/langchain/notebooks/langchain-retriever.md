# Introduction to RAG with MLflow and LangChain

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/langchain/notebooks/langchain-retriever.ipynb)

## Tutorial Overview[​](#tutorial-overview "Direct link to Tutorial Overview")

Welcome to this tutorial, where we explore the integration of Retrieval Augmented Generation (RAG) with MLflow and LangChain. Our focus is on demonstrating how to create advanced RAG systems and showcasing the unique capabilities enabled by MLflow in these applications.

### Understanding RAG and how to develop one with MLflow[​](#understanding-rag-and-how-to-develop-one-with-mlflow "Direct link to Understanding RAG and how to develop one with MLflow")

Retrieval Augmented Generation (RAG) combines the power of language model generation with information retrieval, allowing language models to access and incorporate external data. This approach significantly enriches the model's responses with detailed and context-specific information.

MLflow is instrumental in this process. As an open-source platform, it facilitates the logging, tracking, and deployment of complex models, including RAG chains. With MLflow, integrating LangChain becomes more streamlined, enhancing the development, evaluation, and deployment processes of RAG models.

> NOTE: In this tutorial, we'll be using GPT-3.5 as our base language model. It's important to note that the results obtained from a RAG system will differ from those obtained by interfacing directly with GPT models. RAG's unique approach of combining external data retrieval with language model generation creates more nuanced and contextually rich responses.

![](https://i.imgur.com/uwo1PCj.png)

## Learning Outcomes[​](#learning-outcomes "Direct link to Learning Outcomes")

By the end of this tutorial, you will learn:

* How to establish a RAG chain using LangChain and MLflow.
* Techniques for scraping and processing documents to feed into a RAG system.
* Best practices for deploying and using RAG models to answer complex queries.
* Understanding the practical implications and differences in responses when using RAG in comparison to direct language model interactions.

## Setting up our Retriever Dependencies[​](#setting-up-our-retriever-dependencies "Direct link to Setting up our Retriever Dependencies")

In order to have a place to store our vetted data (the information that we're going to be retrieving), we're going to use a Vector Database. The framework that we're choosing to use (due to its simplicity, capabilities, and free-to-use characteristics) is FAISS, from **Meta**.

## FAISS Installation for the Tutorial[​](#faiss-installation-for-the-tutorial "Direct link to FAISS Installation for the Tutorial")

### Understanding FAISS[​](#understanding-faiss "Direct link to Understanding FAISS")

For this tutorial, we will be utilizing [FAISS](https://github.com/facebookresearch/faiss/wiki) (Facebook AI Similarity Search, developed and maintained by the [Meta AI research group](https://ai.meta.com/tools/faiss/)), an efficient similarity search and clustering library. It's a highly useful library that easily handles large datasets and is capable of performing operations such as nearest neighbor search, which are critical in Retrieval Augmented Generation (RAG) systems. There are numerous other vector database solutions that can perform similar functionality; we are using FAISS in this tutorial due to its simplicity, ease of use, and fantastic performance.

## Notebook compatibility[​](#notebook-compatibility "Direct link to Notebook compatibility")

With rapidly changing libraries such as `langchain`, examples can become outdated rather quickly and will no longer work. For the purposes of demonstration, here are the critical dependencies that are recommended to use to effectively run this notebook:

| Package            | Version    |
| ------------------ | ---------- |
| langchain          | **0.1.16** |
| lanchain-community | **0.0.33** |
| langchain-openai   | **0.0.8**  |
| openai             | **1.12.0** |
| tiktoken           | **0.6.0**  |
| mlflow             | **2.12.1** |
| faiss-cpu          | **1.7.4**  |

If you attempt to execute this notebook with different versions, it may function correctly, but it is recommended to use the precise versions above to ensure that your code executes properly.

### Installing Requirements[​](#installing-requirements "Direct link to Installing Requirements")

Before proceeding with the tutorial, ensure that you have FAISS and [Beautiful Soup](https://pypi.org/project/beautifulsoup4/) installed via `pip`. The version specifiers for other packages are guaranteed to work with this notebook. Other versions of these packages may not function correctly due to breaking changes their APIs.

bash

```bash
    pip install beautifulsoup4 faiss-cpu==1.7.4 langchain==0.1.16 langchain-community==0.0.33 langchain-openai==0.0.8 openai==1.12.0 tiktoken==0.6.0

```

> NOTE: If you'd like to run this using your GPU, you can install `faiss-gpu` instead.

python

```python
import os
import shutil
import tempfile

import requests
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings

import mlflow

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

```

> **NOTE: If you'd like to use Azure OpenAI with LangChain, you need to install `openai>=1.10.0` and `langchain-openai>=0.0.6`, as well as to specify the following credentials and parameters:**

python

```python
from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings

# Set this to `azure`
os.environ["OPENAI_API_TYPE"] = "azure"
# The API version you want to use: set this to `2023-05-15` for the released version.
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
assert "AZURE_OPENAI_ENDPOINT" in os.environ, (
  "Please set the AZURE_OPENAI_ENDPOINT environment variable. It is the base URL for your Azure OpenAI resource. You can find this in the Azure portal under your Azure OpenAI resource."
)
assert "OPENAI_API_KEY" in os.environ, (
  "Please set the OPENAI_API_KEY environment variable. It is the API key for your Azure OpenAI resource. You can find this in the Azure portal under your Azure OpenAI resource."
)

azure_openai_llm = AzureOpenAI(
  deployment_name="<your-deployment-name>",
  model_name="gpt-4o-mini",
)
azure_openai_embeddings = AzureOpenAIEmbeddings(
  azure_deployment="<your-deployment-name>",
)

```

## Scraping Federal Documents for RAG Processing[​](#scraping-federal-documents-for-rag-processing "Direct link to Scraping Federal Documents for RAG Processing")

In this section of the tutorial, we will demonstrate how to scrape content from federal document webpages for use in our RAG system. We'll be focusing on extracting transcripts from specific sections of webpages, which will then be used to feed our Retrieval Augmented Generation (RAG) model. This process is crucial for providing the RAG system with relevant external data.

### Function Overview[​](#function-overview "Direct link to Function Overview")

* The function `fetch_federal_document` is designed to scrape and return the transcript of specific federal documents.
* It takes two arguments: `url` (the webpage URL) and `div_class` (the class of the div element containing the transcript).
* The function handles web requests, parses HTML content, and extracts the desired transcript text.

This step is integral to building a RAG system that relies on external, context-specific data. By effectively fetching and processing this data, we can enrich our model's responses with accurate information directly sourced from authoritative documents.

> **NOTE**: In a real-world scenario, you would have your specific text data located on disk somewhere (either locally or on your cloud provider) and the process of loading the embedded data into a vector search database would be entirely external to this active fetching displayed below. We're simply showing the entire process here for demonstration purposes to show the entire end-to-end workflow for interfacing with a RAG model.

python

```python
def fetch_federal_document(url, div_class):
  """
  Scrapes the transcript of the Act Establishing Yellowstone National Park from the given URL.

  Args:
  url (str): URL of the webpage to scrape.

  Returns:
  str: The transcript text of the Act.
  """
  # Sending a request to the URL
  response = requests.get(url)
  if response.status_code == 200:
      # Parsing the HTML content of the page
      soup = BeautifulSoup(response.text, "html.parser")

      # Finding the transcript section by its HTML structure
      transcript_section = soup.find("div", class_=div_class)
      if transcript_section:
          transcript_text = transcript_section.get_text(separator="
", strip=True)
          return transcript_text
      else:
          return "Transcript section not found."
  else:
      return f"Failed to retrieve the webpage. Status code: {response.status_code}"

```

## Document Fetching and FAISS Database Creation[​](#document-fetching-and-faiss-database-creation "Direct link to Document Fetching and FAISS Database Creation")

In this next part, we focus on two key processes:

1. **Document Fetching**:

   * We use `fetch_and_save_documents` to retrieve documents from specified URLs.
   * This function takes a list of URLs and a file path as inputs.
   * Each document fetched from the URLs is appended to a single file at the given path.

2. **FAISS Database Creation**:

   * `create_faiss_database` is responsible for creating a FAISS database from the documents saved in the previous step.
   * The function leverages `TextLoader` to load the text, `CharacterTextSplitter` for document splitting, and `OpenAIEmbeddings` for generating embeddings.
   * The resulting FAISS database, which facilitates efficient similarity searches, is saved to a specified directory and returned for further use.

These functions streamline the process of gathering relevant documents and setting up a FAISS database, essential for implementing advanced Retrieval-Augmented Generation (RAG) applications in MLflow. By modularizing these steps, we ensure code reusability and maintainability.

python

```python
def fetch_and_save_documents(url_list, doc_path):
  """
  Fetches documents from given URLs and saves them to a specified file path.

  Args:
      url_list (list): List of URLs to fetch documents from.
      doc_path (str): Path to the file where documents will be saved.
  """
  for url in url_list:
      document = fetch_federal_document(url, "col-sm-9")
      with open(doc_path, "a") as file:
          file.write(document)


def create_faiss_database(document_path, database_save_directory, chunk_size=500, chunk_overlap=10):
  """
  Creates and saves a FAISS database using documents from the specified file.

  Args:
      document_path (str): Path to the file containing documents.
      database_save_directory (str): Directory where the FAISS database will be saved.
      chunk_size (int, optional): Size of each document chunk. Default is 500.
      chunk_overlap (int, optional): Overlap between consecutive chunks. Default is 10.

  Returns:
      FAISS database instance.
  """
  # Load documents from the specified file
  document_loader = TextLoader(document_path)
  raw_documents = document_loader.load()

  # Split documents into smaller chunks with specified size and overlap
  document_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  document_chunks = document_splitter.split_documents(raw_documents)

  # Generate embeddings for each document chunk
  embedding_generator = OpenAIEmbeddings()
  faiss_database = FAISS.from_documents(document_chunks, embedding_generator)

  # Save the FAISS database to the specified directory
  faiss_database.save_local(database_save_directory)

  return faiss_database

```

## Setting Up the Working Environment and FAISS Database[​](#setting-up-the-working-environment-and-faiss-database "Direct link to Setting Up the Working Environment and FAISS Database")

This section of the tutorial deals with the setup for our Retrieval-Augmented Generation (RAG) application. We'll establish the working environment and create the necessary FAISS database:

1. **Temporary Directory Creation**:

   * A temporary directory is created using `tempfile.mkdtemp()`. This directory serves as a workspace for storing our documents and the FAISS database.

2. **Document Path and FAISS Index Directory**:

   * Paths for storing the fetched documents and FAISS database are defined within this temporary directory.

3. **Document Fetching**:

   * We have a list of URLs (`url_listings`) containing the documents we need to fetch.
   * `fetch_and_save_documents` function is used to retrieve and save the documents from these URLs into a single file located at `doc_path`.

4. **FAISS Database Creation**:

   * The `create_faiss_database` function is then called to create a FAISS database from the saved documents, using the default `chunk_size` and `chunk_overlap` values.
   * This database (`vector_db`) is crucial for the RAG process, as it enables efficient similarity searches on the loaded documents.

By the end of this process, we have all documents consolidated in a single location and a FAISS database ready to be used for retrieval purposes in our MLflow-enabled RAG application.

python

```python
temporary_directory = tempfile.mkdtemp()

doc_path = os.path.join(temporary_directory, "docs.txt")
persist_dir = os.path.join(temporary_directory, "faiss_index")

url_listings = [
  "https://www.archives.gov/milestone-documents/act-establishing-yellowstone-national-park#transcript",
  "https://www.archives.gov/milestone-documents/sherman-anti-trust-act#transcript",
]

fetch_and_save_documents(url_listings, doc_path)

vector_db = create_faiss_database(doc_path, persist_dir)

```

## Establishing RetrievalQA Chain and Logging with MLflow[​](#establishing-retrievalqa-chain-and-logging-with-mlflow "Direct link to Establishing RetrievalQA Chain and Logging with MLflow")

In this final setup phase, we focus on creating the RetrievalQA chain and integrating it with MLflow:

1. **Initializing the RetrievalQA Chain**:

   * The `RetrievalQA` chain is initialized using the OpenAI language model (`llm`) and the retriever from our previously created FAISS database (`vector_db.as_retriever()`).
   * This chain will use the OpenAI model for generating responses and the FAISS retriever for document-based information retrieval.

2. **Loader Function for Retrieval**:

   * A `load_retriever` function is defined to load the retriever from the FAISS database saved in the specified directory.
   * This function is crucial for reloading the retriever when the model is used later.

3. **Logging the Model with MLflow**:

   * The RetrievalQA chain is logged using `mlflow.langchain.log_model`.
   * This process includes specifying the `artifact_path`, the `loader_fn` for the retriever, and the `persist_dir` where the FAISS database is stored.
   * Logging the model with MLflow ensures it is tracked and can be easily retrieved for future use.

Through these steps, we successfully integrate a complex RAG application with MLflow, showcasing its capability to handle advanced NLP tasks.

python

```python
mlflow.set_experiment("Legal RAG")

retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=vector_db.as_retriever())


# Log the retrievalQA chain
def load_retriever(persist_directory):
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.load_local(
      persist_directory,
      embeddings,
      allow_dangerous_deserialization=True,  # This is required to load the index from MLflow
  )
  return vectorstore.as_retriever()


with mlflow.start_run() as run:
  model_info = mlflow.langchain.log_model(
      retrievalQA,
      name="retrieval_qa",
      loader_fn=load_retriever,
      persist_dir=persist_dir,
  )

```

> **IMPORTANT**: In order to load a stored vectorstore instance such as our FAISS instance above, we need to specify within the load function the argument `allow_dangeous_deserialization` to `True` in order for the load to succeed. This is due to a safety warning that was introduced in `langchain` for loading objects that have been serialized using `pickle` or `cloudpickle`. While this issue of remote code execution is not a risk with using MLflow, as the serialization and deserialization happens entirely via API and within your environment, the argument must be set in order to prevent an Exception from being thrown at load time.

## Our RAG application in the MLflow UI[​](#our-rag-application-in-the-mlflow-ui "Direct link to Our RAG application in the MLflow UI")

![Our Model in the UI](https://i.imgur.com/u9zdkmM.png)

### Testing our RAG Model[​](#testing-our-rag-model "Direct link to Testing our RAG Model")

Now that we have the model stored in MLflow, we can load it back as a `pyfunc` and see how well it answers a few critically important questions about these acts of Congress in America.

python

```python
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

```

python

```python
def print_formatted_response(response_list, max_line_length=80):
  """
  Formats and prints responses with a maximum line length for better readability.

  Args:
  response_list (list): A list of strings representing responses.
  max_line_length (int): Maximum number of characters in a line. Defaults to 80.
  """
  for response in response_list:
      words = response.split()
      line = ""
      for word in words:
          if len(line) + len(word) + 1 <= max_line_length:
              line += word + " "
          else:
              print(line)
              line = word + " "
      print(line)

```

## Let's make sure that this thing works[​](#lets-make-sure-that-this-thing-works "Direct link to Let's make sure that this thing works")

Let's try out our Retriever model by sending it a fairly simple but purposefully vague question.

python

```python
answer1 = loaded_model.predict([{"query": "What does the document say about trespassers?"}])

print_formatted_response(answer1)

```

## Understanding the RetrievalQA Response[​](#understanding-the-retrievalqa-response "Direct link to Understanding the RetrievalQA Response")

With this model, our approach combines text retrieval from a database with language model generation to answer specific queries.

### How It Works:[​](#how-it-works "Direct link to How It Works:")

* **RetrievalQA Model**: This model, a part of the LangChain suite, is designed to first retrieve relevant information from a predefined database and then use a language model to generate a response based on this information.
* **Database Integration**: In this example, we've created a FAISS database from historical documents, such as the Act Establishing Yellowstone National Park. This database is used by the RetrievalQA model to find relevant sections of text.
* **Query Processing**: When we execute `loaded_model.predict([{"query": "What does the document say about trespassers?"}])`, the model first searches the database for parts of the document that are most relevant to the query about trespassers.

### Why Is This Response Different?[​](#why-is-this-response-different "Direct link to Why Is This Response Different?")

* **Context-Specific Answers**: Unlike a direct query to GPT-3.5, which might generate an answer based on its training data without specific context, the RetrievalQA model provides a response directly derived from the specific documents in the database.
* **Accurate and Relevant**: The response is more accurate and contextually relevant because it's based on the actual content of the specific document being queried.
* **No Generalization**: There's less generalization or assumption in the response. The RetrievalQA model is not "guessing" based on its training; it's providing information directly sourced from the document.

### Key Takeaway:[​](#key-takeaway "Direct link to Key Takeaway:")

* This methodology demonstrates how MLflow and LangChain facilitate complex RAG use cases, where direct interaction with historical or specific texts yields more precise answers than generic language model predictions.
* The tutorial highlights how leveraging RAG can be particularly useful in scenarios where responses need to be grounded in specific texts or documents, showcasing a powerful blend of retrieval and generation capabilities.

## Analyzing the Bridle-Path Query Response[​](#analyzing-the-bridle-path-query-response "Direct link to Analyzing the Bridle-Path Query Response")

This section of the tutorial showcases an interesting aspect of the RetrievalQA model's capabilities, particularly in handling queries that involve both specific information retrieval and additional context generation.

### Query and Response Breakdown:[​](#query-and-response-breakdown "Direct link to Query and Response Breakdown:")

* **Query**: The user asks, "What is a bridle-path and can I use one at Yellowstone?"
* **Response**: The model responds by explaining what a bridle-path is and confirms that bridle-paths can be used at Yellowstone based on the act.

### Understanding the Response Dynamics:[​](#understanding-the-response-dynamics "Direct link to Understanding the Response Dynamics:")

1. **Combining Document Data with LLM Context**:

   * The query about bridle-paths isn't directly answered in the act establishing Yellowstone.
   * The model uses its language understanding capabilities to provide a definition of a bridle-path.
   * It then merges this information with the context it retrieves from the FAISS database about the act, particularly regarding the construction of roads and bridle-paths in the park.

2. **Enhanced Contextual Understanding**:

   * The RetrievalQA model demonstrates an ability to supplement direct information from the database with additional context through its language model.
   * This approach provides a more comprehensive answer that aligns with the user's query, showing a blend of document-specific data and general knowledge.

3. **Why This Is Notable**:

   * Unlike a standard LLM response, the RetrievalQA model doesn't solely rely on its training data for general responses.
   * It effectively integrates specific document information with broader contextual understanding, offering a more nuanced answer.

### Key Takeaway:[​](#key-takeaway-1 "Direct link to Key Takeaway:")

* This example highlights how MLflow and LangChain, through the RetrievalQA model, facilitate a sophisticated response mechanism. The model not only retrieves relevant document information but also intelligently fills in gaps with its own language understanding capabilities.
* Such a response mechanism is particularly useful when dealing with queries that require both specific document references and additional contextual information, showcasing the advanced capabilities of RAG in practical applications.

python

```python
answer2 = loaded_model.predict(
  [{"query": "What is a bridle-path and can I use one at Yellowstone?"}]
)

print_formatted_response(answer2)

```

## A most serious question[​](#a-most-serious-question "Direct link to A most serious question")

In this section of our tutorial, we delve into a whimsically ridiculous query and how our model tackles it with a blend of accuracy and a hint of humor.

### Query Overview:[​](#query-overview "Direct link to Query Overview:")

* **The Query**: "Can I buy Yellowstone from the Federal Government to set up a buffalo-themed day spa?"
* **The Response**: The model, with a straight face, responds, "No, you cannot buy Yellowstone from the Federal Government to set up a buffalo-themed day spa."

### A Peek into the Model's Thought Process:[​](#a-peek-into-the-models-thought-process "Direct link to A Peek into the Model's Thought Process:")

1. **Direct and No-Nonsense Response**:

   * Despite the query's comedic undertone, the model gives a direct and clear-cut response. It's like the model is saying, "Nice try, but no, you can't do that."
   * This highlights the model's ability to remain factual, even when faced with a question that's clearly more humorous than serious.

2. **Understanding Legal Boundaries**:

   * The response respects the legal and regulatory sanctity of national parks. It seems our model takes the protection of national treasures like Yellowstone pretty seriously!
   * The model's training on legal and general knowledge assists in delivering a response that's accurate, albeit the question being a facetious one.

3. **Contrast with Traditional LLM Responses**:

   * A traditional LLM might have given a more generic answer. In contrast, our model, equipped with context-specific data, promptly debunks the whimsical idea of buying a national park for a spa.

### A Dash of Humor in Learning:[​](#a-dash-of-humor-in-learning "Direct link to A Dash of Humor in Learning:")

* The query, while absurd, serves as an amusing example of the model's capability to provide contextually relevant answers to even the most far-fetched questions.
* It's a reminder that learning can be both informative and entertaining. In this case, the model plays the role of a straight-faced comedian, addressing a wildly imaginative business proposal with a firm yet comical "No."

So, while you can't buy Yellowstone for your buffalo-themed spa dreams, you can certainly enjoy the park's natural beauty... just as a visitor, not as a spa owner!

python

```python
answer3 = loaded_model.predict(
  [
      {
          "query": "Can I buy Yellowstone from the Federal Government to set up a buffalo-themed day spa?"
      }
  ]
)

print_formatted_response(answer3)

```

## Maintaining Composure: Answering Another Whimsical Query[​](#maintaining-composure-answering-another-whimsical-query "Direct link to Maintaining Composure: Answering Another Whimsical Query")

In this part of our tutorial, we explore another amusing question about leasing land in Yellowstone for a buffalo-themed day spa. Let's see how our model, with unflappable composure, responds to this quirky inquiry.

### Query and Response:[​](#query-and-response "Direct link to Query and Response:")

* **The Query**: "Can I lease a small parcel of land from the Federal Government for a small buffalo-themed day spa for visitors to the park?"
* **The Response**: "No, you cannot lease a small parcel of land from the Federal Government for a small buffalo-themed day spa for visitors to the park..."

### Insights into the Model's Response:[​](#insights-into-the-models-response "Direct link to Insights into the Model's Response:")

1. **Factual and Unwavering**:

   * Despite the continued outlandish line of questioning, our model remains as cool as a cucumber. It patiently explains the limitations and actual purposes of leasing land in Yellowstone.
   * The response cites Section 2 of the act, adding legal precision to its rebuttal.

2. **A Lawyer's Patience Tested?**:

   * Imagine if this question was posed to an actual lawyer. By now, they might be rubbing their temples! But our model is unfazed and continues to provide factual answers.
   * This showcases the model's ability to handle repetitive and unusual queries without losing its 'cool'.

In conclusion, while our model firmly closes the door on the buffalo-themed day spa dreams, it does so with informative grace, demonstrating its ability to stay on course no matter how imaginative the queries get.

python

```python
answer4 = loaded_model.predict(
  [
      {
          "query": "Can I lease a small parcel of land from the Federal Government for a small "
          "buffalo-themed day spa for visitors to the park?"
      }
  ]
)

print_formatted_response(answer4)

```

## Another Attempt at the Buffalo-Themed Day Spa Dream[​](#another-attempt-at-the-buffalo-themed-day-spa-dream "Direct link to Another Attempt at the Buffalo-Themed Day Spa Dream")

Once more, we pose an imaginative question to our model, this time adding a hotel to the buffalo-themed day spa scenario. The reason for this modification is to evaluate whether the RAG application can discern a nuanced element of the intentionally vague wording of the two acts that we've loaded. Let's see the response to determine if it can resolve both bits of information!

### Quick Takeaway:[​](#quick-takeaway "Direct link to Quick Takeaway:")

* The model, sticking to its informative nature, clarifies the leasing aspects based on the Act's provisions.
* It interestingly connects the query to another act related to trade and commerce, showing its ability to cross-reference related legal documents.
* This response demonstrates the model's capacity to provide detailed, relevant information, even when faced with quirky and hypothetical scenarios.

python

```python
answer5 = loaded_model.predict(
  [
      {
          "query": "Can I lease a small parcel of land from the Federal Government for a small "
          "buffalo-themed day spa and hotel for visitors to stay in and relax at while visiting the park?"
      }
  ]
)
print_formatted_response(answer5)

```

## Well, what can I do then?[​](#well-what-can-i-do-then "Direct link to Well, what can I do then?")

### Takeaway:[​](#takeaway "Direct link to Takeaway:")

* The response reassuringly confirms that one can enjoy Yellowstone's natural beauty, provided park rules and regulations are respected.
* This illustrates the model's ability to provide straightforward, practical advice in response to simple, real-world questions. It clearly has the context of the original act and is able to infer what is permissible (enjoying the reserved land).

python

```python
answer6 = loaded_model.predict(
  [{"query": "Can I just go to the park and peacefully enjoy the natural splendor?"}]
)

print_formatted_response(answer6)

```

## Evaluating the RetrievalQA Model's Legal Context Integration[​](#evaluating-the-retrievalqa-models-legal-context-integration "Direct link to Evaluating the RetrievalQA Model's Legal Context Integration")

This section of the tutorial showcases the RetrievalQA model's sophisticated ability to integrate and interpret context from multiple legal documents. The model is challenged with a query that requires synthesizing information from distinct sources. This test is particularly interesting for its demonstration of the model's proficiency in:

1. **Contextual Integration**: The model adeptly pulls in relevant legal details from different documents, illustrating its capacity to navigate through multiple sources of information.

2. **Legal Interpretation**: It interprets the legal implications related to the query, highlighting the model's understanding of complex legal language and concepts.

3. **Cross-Document Inference**: The model's ability to discern and extract the most pertinent information from a pool of multiple documents is a testament to its advanced capabilities in multi-document scenarios.

This evaluation provides a clear example of the model's potential in handling intricate queries that necessitate a deep and nuanced understanding of diverse data sources.

python

```python
answer7 = loaded_model.predict(
  [
      {
          "query": "Can I start a buffalo themed day spa outside of Yellowstone National Park and stifle any competition?"
      }
  ]
)

print_formatted_response(answer7)

```

**Cleanup: Removing Temporary Directory**:

* After we're done asking our Retriever Model a bunch of silly questions, the temporary directory created earlier is cleaned up using `shutil.rmtree`.

python

```python
# Clean up our temporary directory that we created with our FAISS instance
shutil.rmtree(temporary_directory)

```

## Conclusion: Mastering RAG with MLflow[​](#conclusion-mastering-rag-with-mlflow "Direct link to Conclusion: Mastering RAG with MLflow")

In this tutorial, we explored the depths of Retrieval Augmented Generation (RAG) applications, enabled and simplified by MLflow. Here's a recap of our journey and the key takeaways:

1. **Ease of Developing RAG Applications**: We learned how MLflow facilitates the development of RAG applications by streamlining the process of integrating large language models with external data sources. Our hands-on experience demonstrated the process of fetching, processing, and embedding documents into a FAISS database, all managed within the MLflow framework.

2. **Advanced Query Handling**: Through our tests, we observed how the MLflow-wrapped LangChain RAG model adeptly handled complex queries, drawing from multiple documents to provide context-rich and accurate responses. This showcased the potential of RAG models in processing and understanding queries that require multi-source data integration.

3. **MLflow's Role in Deployment and Management**: MLflow's robustness was evident in how it simplifies the logging, deployment, and management of complex models. Its ability to track experiments, manage artifacts, and ease the deployment process highlights its indispensability in the machine learning lifecycle.

4. **Practical Application Insights**: Our queries, while humorous at times, served to illustrate the practical capabilities of RAG models. From legal interpretations to hypothetical scenarios, we saw how these models could be applied in real-world situations, providing insightful and contextually relevant responses.

5. **The Future of RAG and MLflow**: This tutorial underscored the potential of combining RAG with MLflow's streamlined management capabilities. As RAG continues to evolve, MLflow stands out as a crucial tool for harnessing its power, making advanced NLP applications more accessible and efficient.

In summary, our journey through this tutorial has not only equipped us with the knowledge to develop and deploy RAG applications effectively but also opened our eyes to the vast possibilities that lie ahead in the realm of advanced NLP, all made more attainable through MLflow.

## What's next?[​](#whats-next "Direct link to What's next?")

If you'd like to learn more about how MLflow and LangChain integrate, see the other [advanced tutorials for MLflow's LangChain flavor](https://www.mlflow.org/docs/latest/llms/langchain/index.html#advanced-tutorials).
