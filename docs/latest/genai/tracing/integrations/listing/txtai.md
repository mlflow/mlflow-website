# Tracing txtai

![txtai Tracing via autolog](/mlflow-website/docs/latest/assets/images/txtai-rag-tracing-507199b924f1c7ed180e0d940758e2dc.png)

[txtai](https://github.com/neuml/txtai?tab=readme-ov-file) is an all-in-one embeddings database for semantic search, LLM orchestration and language model workflows.

[MLflow Tracing](/mlflow-website/docs/latest/genai/tracing.md) provides automatic tracing capability for txtai. Auto tracing for txtai can be enabled by calling the `mlflow.txtai.autolog` function, MLflow will capture traces for LLM invocation, embeddings, vector search, and log them to the active MLflow Experiment.

To get started, install the [MLflow txtai extension](https://github.com/neuml/mlflow-txtai/tree/master):

bash

```
pip install mlflow-txtai
```

Then, enable autologging in your Python code:

python

```
import mlflow

mlflow.txtai.autolog()
```

### Examples[​](#examples "Direct link to Examples")

* Simple Example
* RAG
* Agent

The simplest example to show the tracing integration is to instrument a [Textractor pipeline](https://neuml.github.io/txtai/pipeline/data/textractor/).

python

```
import mlflow
from txtai.pipeline import Textractor

# Enable MLflow auto-tracing for txtai
mlflow.txtai.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("txtai")

# Define and run a simple Textractor pipeline.
textractor = Textractor()
textractor("https://github.com/neuml/txtai")
```

![txtai Textractor Tracing via autolog](/mlflow-website/docs/latest/assets/images/txtai-textractor-tracing-15f2e1b268fc3fc4921c06e5e9a87cd8.png)

You can easily trace a [RAG pipeline](https://neuml.github.io/txtai/pipeline/text/rag/).

python

```
import mlflow
from txtai import Embeddings, RAG

# Enable MLflow auto-tracing for txtai
mlflow.txtai.autolog()

wiki = Embeddings()
wiki.load(provider="huggingface-hub", container="neuml/txtai-wikipedia-slim")

# Define prompt template
template = """
Answer the following question using only the context below. Only include information
specifically discussed.

question: {question}
context: {context} """

# Create RAG pipeline
rag = RAG(
    wiki,
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    system="You are a friendly assistant. You answer questions from users.",
    template=template,
    context=10,
)

rag("Tell me about the Roman Empire", maxlength=2048)
```

![txtai Rag Tracing via autolog](/mlflow-website/docs/latest/assets/images/txtai-rag-tracing-507199b924f1c7ed180e0d940758e2dc.png)

You can effortlessly trace the internals of a [txtai agent](https://neuml.github.io/txtai/agent/) designed to research questions on astronomy.

python

```
import mlflow
from txtai import Agent, Embeddings

# Enable MLflow auto-tracing for txtai
mlflow.txtai.autolog()


def search(query):
    """
    Searches a database of astronomy data.

    Make sure to call this tool only with a string input, never use JSON.

    Args:
        query: concepts to search for using similarity search

    Returns:
        list of search results with for each match
    """

    return embeddings.search(
        "SELECT id, text, distance FROM txtai WHERE similar(:query)",
        10,
        parameters={"query": query},
    )


embeddings = Embeddings()
embeddings.load(provider="huggingface-hub", container="neuml/txtai-astronomy")

agent = Agent(
    tools=[search],
    llm="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    max_iterations=10,
)

researcher = """
{command}

Do the following.
 - Search for results related to the topic.
 - Analyze the results
 - Continue querying until conclusive answers are found
 - Write a Markdown report
"""

agent(
    researcher.format(
        command="""
Write a detailed list with explanations of 10 candidate stars that could potentially be habitable to life.
"""
    ),
    maxlength=16000,
)
```

![txtai Agent Tracing via autolog](/mlflow-website/docs/latest/assets/images/txtai-agent-tracing-f69f47a9de38c40814ea985887782850.png)

### More Information[​](#more-information "Direct link to More Information")

For more examples and guidance on using txtai with MLflow, please refer to the [MLflow txtai extension documentation](https://github.com/neuml/mlflow-txtai/tree/master)
