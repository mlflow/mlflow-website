# Advanced Tutorial: Embeddings Support with OpenAI in MLflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/openai/notebooks/openai-embeddings-generation.ipynb)

Welcome to this advanced guide on implementing OpenAI embeddings within the MLflow framework. This tutorial delves into the configuration and utilization of OpenAI's powerful embeddings, a key component in modern machine learning models.

### Understanding Embeddings[​](#understanding-embeddings "Direct link to Understanding Embeddings")

Embeddings are a form of representation learning where words, phrases, or even entire documents are converted into vectors in a high-dimensional space. These vectors capture semantic meaning, enabling models to understand and process language more effectively. Embeddings are extensively used in natural language processing (NLP) for tasks like text classification, sentiment analysis, and language translation.

### How Embeddings Work[​](#how-embeddings-work "Direct link to How Embeddings Work")

Embeddings work by mapping textual data to vectors such that the distance and direction between vectors represent relationships between the words or phrases. For example, in a well-trained embedding space, synonyms are located closer together, while unrelated terms are farther apart. This spatial arrangement allows algorithms to recognize context and semantics, enhancing their ability to interpret and respond to natural language.

### In This Tutorial[​](#in-this-tutorial "Direct link to In This Tutorial")

* **Embedding Endpoint Configuration**: Setting up and utilizing OpenAI's embedding endpoints in MLflow.
* **Real-world Application**: Practical example of comparing the text content of various web pages to one another to determine the amount of similarity in their contextually-specific content.
* **Efficiency and Precision Enhancements**: Techniques for improving model performance using OpenAI embeddings.

By the end of this tutorial, you'll have a thorough understanding of how to integrate and leverage OpenAI embeddings in your MLflow projects, harnessing the power of advanced NLP techniques. You'll also see a real-world application of using text embeddings of documents to compare their similarity. This use case is particularly useful for web content development as a critical task when performing search engine optimization (SEO) to ensure that site page contents are not too similar to one another (which could result in a downgrade in page rankings).

### Required packages[​](#required-packages "Direct link to Required packages")

In order to run this tutorial, you will need to install `beautifulsoup4` from PyPI.

Let's dive into the world of embeddings and explore their transformative impact on machine learning models!

python

```
import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)
```

python

```
import os

import numpy as np
import openai
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema, TensorSpec

assert "OPENAI_API_KEY" in os.environ, " OPENAI_API_KEY environment variable must be set"
```

### Integrating OpenAI Model with MLflow for Document Similarity[​](#integrating-openai-model-with-mlflow-for-document-similarity "Direct link to Integrating OpenAI Model with MLflow for Document Similarity")

In this tutorial segment, we demonstrate the process of setting up and utilizing an OpenAI embedding model within MLflow for document similarity tasks.

#### Key Steps[​](#key-steps "Direct link to Key Steps")

1. **Setting an MLflow Experiment**: We begin by setting the experiment context in MLflow, specifically for document similarity, using `mlflow.set_experiment("Documentation Similarity")`.

2. **Logging the Model in MLflow**: We initiate an MLflow run and log metadata and access configuration parameters to communicate with a specific OpenAI endpoint. The OpenAI endpoint that we've chosen here points to the model "text-embedding-ada-002", chosen for its robust embedding capabilities. During this step, we detail these access configurations, the embedding task, input/output schemas, and parameters like batch size.

3. **Loading the Logged Model for Use**: After logging the MLflow model, we proceed to load it using MLflow's `pyfunc` module. This is a critical step for applying the model to perform document similarity tasks within the MLflow ecosystem.

These steps are essential for integrating access to OpenAI's embedding model into MLflow, facilitating advanced NLP operations like document similarity analysis.

python

```
mlflow.set_experiment("Documenatation Similarity")

with mlflow.start_run():
  model_info = mlflow.openai.log_model(
      model="text-embedding-ada-002",
      task=openai.embeddings,
      name="model",
      signature=ModelSignature(
          inputs=Schema([ColSpec(type="string", name=None)]),
          outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
          params=ParamSchema([ParamSpec(name="batch_size", dtype="long", default=1024)]),
      ),
  )

# Load the model in pyfunc format
model = mlflow.pyfunc.load_model(model_info.model_uri)
```

### Webpage Text Extraction for Embedding Analysis[​](#webpage-text-extraction-for-embedding-analysis "Direct link to Webpage Text Extraction for Embedding Analysis")

This section of the tutorial introduces functions designed to extract and prepare text from webpages, a crucial step before applying embedding models for analysis.

#### Overview of Functions[​](#overview-of-functions "Direct link to Overview of Functions")

1. **insert\_space\_after\_tags**:

   * Adds a space after specific HTML tags in a BeautifulSoup object for better text readability.

2. **extract\_text\_from\_url**:

   * Extracts text from a specified webpage section using its URL and a target ID. Filters and organizes the text from tags like `<h>`, `<li>`, and `<p>`, excluding certain irrelevant sections.

These functions are integral to preprocessing web content, ensuring that the text fed into the embedding model is clean, relevant, and well-structured.

python

```
def insert_space_after_tags(soup, tags):
  """
  Insert a space after each tag specified in the provided BeautifulSoup object.

  Args:
      soup: BeautifulSoup object representing the parsed HTML.
      tags: List of tag names (as strings) after which space should be inserted.
  """
  for tag_name in tags:
      for tag in soup.find_all(tag_name):
          tag.insert_after(" ")


def extract_text_from_url(url, id):
  """
  Extract and return text content from a specific section of a webpage.
  """
  try:
      response = requests.get(url)
      response.raise_for_status()  # Raises HTTPError for bad requests (4XX, 5XX)
  except requests.exceptions.RequestException as e:
      return f"Request failed: {e}"

  soup = BeautifulSoup(response.text, "html.parser")
  target_div = soup.find("div", {"class": "section", "id": id})
  if not target_div:
      return "Target element not found."

  insert_space_after_tags(target_div, ["strong", "a"])

  content_tags = target_div.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "li", "p"])
  filtered_tags = [
      tag
      for tag in content_tags
      if not (
          (tag.name == "li" and tag.find("p") and tag.find("a", class_="reference external"))
          or (tag.name == "p" and tag.find_parent("ul"))
          or (tag.get_text(strip=True).lower() == "note")
      )
  ]

  return "
".join(tag.get_text(separator=" ", strip=True) for tag in filtered_tags)
```

#### Detailed Workflow:[​](#detailed-workflow "Direct link to Detailed Workflow:")

* The function `extract_text_from_url` first fetches the webpage content using the `requests` library.
* It then parses the HTML content using BeautifulSoup.
* Specific HTML tags are targeted for text extraction, ensuring that the content is relevant and well-structured for embedding analysis.
* The `insert_space_after_tags` function is called within `extract_text_from_url` to improve text readability post-extraction.

### Measuring Similarity and Distance Between Embeddings[​](#measuring-similarity-and-distance-between-embeddings "Direct link to Measuring Similarity and Distance Between Embeddings")

In this next part of the tutorial, we utilize two functions from `sklearn` to measure the similarity and distance between document embeddings, essential for evaluating and comparing text-based machine learning models.

#### Function Overviews[​](#function-overviews "Direct link to Function Overviews")

1. **cosine\_similarity**:

   * **Purpose**: Calculates the cosine similarity between two embedding vectors.
   * **How It Works**: This function computes similarity by finding the cosine of the angle between the two vectors, a common method for assessing how similar two documents are in terms of their content.
   * **Relevance**: Very useful in NLP, especially for tasks like document retrieval and clustering, where the goal is to find documents with similar content.

2. **euclidean\_distances**:

   * **Purpose**: Computes the Euclidean distance between two embedding vectors.
   * **Functionality**: Similar to `cosine_similarity` this function calculates the Euclidean distance, which is the "straight line" distance between the two points in the embedding space. This measure is useful for understanding how different two documents are.
   * **Relevance within NLP**: Offers a more intuitive physical distance metric, useful for tasks like document classification and anomaly detection.

These functions are crucial for analyzing and comparing the outputs of embedding models, providing insights into the relationships between different text data in terms of similarity and distinction.

### Comparing Webpages Using Embeddings[​](#comparing-webpages-using-embeddings "Direct link to Comparing Webpages Using Embeddings")

This section of the tutorial introduces a function, `compare_pages`, designed to compare the content of two webpages using embedding models. This function is key for understanding how similar or different two given webpages are in terms of their textual content.

#### Function Overview[​](#function-overview "Direct link to Function Overview")

* **Function Name**: `compare_pages`

* **Purpose**: Compares two webpages and returns a similarity score based on their content.

* **Parameters**:

  <!-- -->

  * `url1` and `url2`: URLs of the webpages to be compared.
  * `id1` and `id2`: Target IDs for the main text content divs on each page.

#### How It Works[​](#how-it-works "Direct link to How It Works")

1. **Text Extraction**: The function starts by extracting text from the specified sections of each webpage using the `extract_text_from_url` function.
2. **Embedding Prediction**: It then uses the previously loaded OpenAI model to generate embeddings for the extracted texts.
3. **Similarity and Distance Measurement**: The function calculates both the cosine similarity and Euclidean distance between the two embeddings. These metrics provide a quantifiable measure of how similar or dissimilar the webpage contents are.
4. **Result**: Returns a tuple containing the cosine similarity score and the Euclidean distance. If text extraction fails, it returns an error message.

#### Practical Application[​](#practical-application "Direct link to Practical Application")

This function is particularly useful in scenarios where comparing the content of different webpages is necessary, such as in content curation, plagiarism detection, or similarity analysis for SEO purposes.

By leveraging the power of embeddings and similarity metrics, `compare_pages` provides a robust method for quantitatively assessing webpage content similarities and differences.

python

```
def compare_pages(url1, url2, id1, id2):
  """
  Compare two webpages and return the similarity score.

  Args:
      url1: URL of the first webpage.
      url2: URL of the second webpage.
      id1: The target id for the div containing the main text content of the first page
      id2: The target id for the div containing the main text content of the second page

  Returns:
      A tuple of floats representing the similarity score for cosine similarity and euclidean distance.
  """
  text1 = extract_text_from_url(url1, id1)
  text2 = extract_text_from_url(url2, id2)

  if text1 and text2:
      embedding1 = model.predict([text1])
      embedding2 = model.predict([text2])

      return (
          cosine_similarity(embedding1, embedding2),
          euclidean_distances(embedding1, embedding2),
      )
  else:
      return "Failed to retrieve content."
```

### Similarity Analysis Between MLflow Documentation Pages[​](#similarity-analysis-between-mlflow-documentation-pages "Direct link to Similarity Analysis Between MLflow Documentation Pages")

In this tutorial segment, we demonstrate the practical application of the `compare_pages` function by comparing two specific pages from the MLflow documentation. Our goal is to assess how similar the content of the main Large Language Models (LLMs) page is to the LLM Evaluation page within the 2.8.1 release of MLflow.

#### Process Overview[​](#process-overview "Direct link to Process Overview")

* **Target Webpages**:

  <!-- -->

  * The main LLMs page: [LLMs page for MLflow 2.8.1 release](https://www.mlflow.org/docs/2.8.1/llms/index.html)
  * The LLM Evaluation page: [LLM Evaluation for MLflow 2.8.1](https://www.mlflow.org/docs/2.8.1/llms/llm-evaluate/index.html)

* **Content IDs**: We use 'llms' for the main LLMs page and 'mlflow-llm-evaluate' for the LLM Evaluation page to target specific content sections.

* **Comparison Execution**: The `compare_pages` function is called with these URLs and content IDs to perform the analysis.

#### Results[​](#results "Direct link to Results")

* **Cosine Similarity and Euclidean Distance**: The function returns two key metrics:

  <!-- -->

  * Cosine Similarity: Measures the cosine of the angle between the embedding vectors of the two pages. A higher value indicates greater similarity.
  * Euclidean Distance: Represents the 'straight-line' distance between the two points in the embedding space, with lower values indicating closer similarity.

#### Interpretation[​](#interpretation "Direct link to Interpretation")

The results show a high degree of cosine similarity (0.8792), suggesting that the content of the two pages is quite similar in terms of context and topics covered. The Euclidean distance of 0.4914, while relatively low, offers a complementary perspective, indicating some level of distinctiveness in the content.

#### Conclusion[​](#conclusion "Direct link to Conclusion")

This analysis highlights the effectiveness of using embeddings and similarity metrics for comparing webpage content. In practical terms, it helps in understanding the overlap and differences in documentation, aiding in content optimization, redundancy reduction, and ensuring comprehensive coverage of topics.

python

```
# Get the similarity between the main LLMs page in the MLflow Docs and the LLM Evaluation page for the 2.8.1 release of MLflow

llm_cosine, llm_euclid = compare_pages(
  url1="https://www.mlflow.org/docs/2.8.1/llms/index.html",
  url2="https://www.mlflow.org/docs/2.8.1/llms/llm-evaluate/index.html",
  id1="llms",
  id2="mlflow-llm-evaluate",
)

print(
  f"The cosine similarity between the LLMs page and the LLM Evaluation page is: {llm_cosine} and the euclidean distance is: {llm_euclid}"
)
```

```
The cosine similarity between the LLMs page and the LLM Evaluation page is: [[0.879243]] and the euclidean distance is: [[0.49144073]]
```

### Brief Overview of Similarity Between MLflow LLMs and Plugins Pages[​](#brief-overview-of-similarity-between-mlflow-llms-and-plugins-pages "Direct link to Brief Overview of Similarity Between MLflow LLMs and Plugins Pages")

This section demonstrates a quick similarity analysis between the MLflow Large Language Models (LLMs) page and the Plugins page from the 2.8.1 release.

#### Analysis Execution[​](#analysis-execution "Direct link to Analysis Execution")

* **Pages Compared**:

  <!-- -->

  * LLMs page: [LLMs page for MLflow 2.8.1 release](https://www.mlflow.org/docs/2.8.1/llms/index.html)
  * Plugins page: [Plugins page for MLflow 2.8.1 release](https://www.mlflow.org/docs/2.8.1/plugins.html)

* **IDs Used**: 'llms' for the LLMs page and 'mflow-plugins' for the Plugins page.

* **Function**: `compare_pages` is utilized for the comparison.

#### Results[​](#results-1 "Direct link to Results")

* **Cosine Similarity**: 0.6806, indicating moderate similarity in content.
* **Euclidean Distance**: 0.7992, suggesting a noticeable difference in the context and topics covered by the two pages.

The results reflect a moderate level of similarity between the LLMs and Plugins pages, with a significant degree of distinctiveness in their content. This analysis is useful for understanding the relationship and content overlap between different sections of the MLflow documentation.

python

```
# Get the similarity between the main LLMs page in the MLflow Docs and the Plugins page for the 2.8.1 release of MLflow

plugins_cosine, plugins_euclid = compare_pages(
  url1="https://www.mlflow.org/docs/2.8.1/llms/index.html",
  url2="https://www.mlflow.org/docs/2.8.1/plugins.html",
  id1="llms",
  id2="mflow-plugins",
)

print(
  f"The cosine similarity between the LLMs page and the MLflow Projects page is: {plugins_cosine} and the euclidean distance is: {plugins_euclid}"
)
```

```
The cosine similarity between the LLMs page and the MLflow Projects page is: [[0.68062298]] and the euclidean distance is: [[0.79922088]]
```

### Tutorial Recap: Leveraging OpenAI Embeddings in MLflow[​](#tutorial-recap-leveraging-openai-embeddings-in-mlflow "Direct link to Tutorial Recap: Leveraging OpenAI Embeddings in MLflow")

As we conclude this tutorial, let's recap the key concepts and techniques we've explored regarding the use of OpenAI embeddings within the MLflow framework.

#### Key Takeaways[​](#key-takeaways "Direct link to Key Takeaways")

1. **Integrating OpenAI Models in MLflow**:

   * We learned how to log and load OpenAI's "text-embedding-ada-002" model within MLflow, an essential step for utilizing these embeddings in machine learning workflows.

2. **Text Extraction and Preprocessing**:

   * The tutorial introduced methods for extracting and preprocessing text from webpages, ensuring the data is clean and structured for embedding analysis.

3. **Calculating Similarity and Distance**:

   * We delved into functions for measuring cosine similarity and Euclidean distance between document embeddings, vital for comparing textual content.

4. **Real-World Application: Webpage Content Comparison**:

   * Practical application of these concepts was demonstrated through the comparison of different MLflow documentation pages. We analyzed the similarity and differences in their content using the embeddings generated by the OpenAI model.

5. **Interpreting Results**:

   * The tutorial provided insights into interpreting the results of similarity and distance metrics, highlighting their relevance in understanding content relationships.

#### Conclusion[​](#conclusion-1 "Direct link to Conclusion")

This advanced tutorial aimed to enhance your skills in applying OpenAI embeddings in MLflow, focusing on real-world applications like document similarity analysis. By integrating these powerful NLP tools, we've showcased how to extract more value and insights from textual data, a crucial aspect of modern machine learning projects.

We hope this guide has been informative and instrumental in advancing your understanding and application of OpenAI embeddings within the MLflow framework.

### What's Next?[​](#whats-next "Direct link to What's Next?")

To continue your learning journey, see the additional [advanced tutorials for MLflow's OpenAI flavor](https://www.mlflow.org/docs/latest/genai/flavors/openai/index.html#advanced-tutorials).
