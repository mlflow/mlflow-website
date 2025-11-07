# Log Prompts with Models

Prompts are often used as a part of GenAI applications. Managing the association between prompts and models is crucial for tracking the evolution of models and ensuring consistency across different environments. MLflow Prompt Registry is integrated with MLflow's model tracking capability, allowing you to track which prompts (and versions) are used by your models and applications.

## Basic Usage[​](#basic-usage "Direct link to Basic Usage")

To log a model with associated prompts, use the `prompts` parameter in the `log_model` method. The `prompts` parameter accepts a list of prompt URLs or prompt objects that are associated with the model. The associated prompts are displayed in the MLflow UI for the model run.

text

```
import mlflow

with mlflow.start_run():
    mlflow.<flavor>.log_model(
        model,
        ...
        # Specify a list of prompt URLs or prompt objects.
        prompts=["prompts:/summarization-prompt/2"]
    )
```

warning

The `prompts` parameter for associating prompts with models is only supported for GenAI flavors such as OpenAI, LangChain, LlamaIndex, DSPy, etc. Please refer to the [GenAI flavors](/mlflow-website/docs/latest/genai/flavors.md) for the full list of supported flavors.

## Example 1: Logging Prompts with LangChain[​](#example-1-logging-prompts-with-langchain "Direct link to Example 1: Logging Prompts with LangChain")

### 1. Create a prompt[​](#1-create-a-prompt "Direct link to 1. Create a prompt")

If you haven't already created a prompt, follow [the instructions in this page](/mlflow-website/docs/latest/genai/prompt-registry/create-and-edit-prompts.md) to create a new prompt.

### 2. Define a Chain using the registered prompts[​](#2-define-a-chain-using-the-registered-prompts "Direct link to 2. Define a Chain using the registered prompts")

python

```
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load registered prompt
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt/2")

# Create LangChain prompt object
langchain_prompt = ChatPromptTemplate.from_messages(
    [
        (
            # IMPORTANT: Convert prompt template from double to single curly braces format
            "system",
            prompt.to_single_brace_format(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# Define the LangChain chain
llm = ChatOpenAI()
chain = langchain_prompt | llm

# Invoke the chain
response = chain.invoke({"num_sentences": 1, "sentences": "This is a test sentence."})
print(response)
```

### 3. Log the Chain to MLflow[​](#3-log-the-chain-to-mlflow "Direct link to 3. Log the Chain to MLflow")

Then log the chain to MLflow and specify the prompt URL in the `prompts` parameter:

python

```
with mlflow.start_run(run_name="summarizer-model"):
    mlflow.langchain.log_model(
        chain, name="model", prompts=["prompts:/summarization-prompt/2"]
    )
```

Now you can view the associated prompts to the model in MLflow UI:

![Associated Prompts](/mlflow-website/docs/latest/assets/images/prompt-logged-model-5db16e3e39f3bbd4ca3138c96dff045e.png)

Moreover, you can view the list of models (runs) that use a specific prompt in the prompt details page:

![Associated Prompts](/mlflow-website/docs/latest/assets/images/prompt-logged-model-links-b4c073f2c15e203ec34d7ef35c840112.png)

## Example 2: Automatic Prompt Logging with Models-from-Code[​](#example-2-automatic-prompt-logging-with-models-from-code "Direct link to Example 2: Automatic Prompt Logging with Models-from-Code")

[Models-from-Code](/mlflow-website/docs/latest/ml/model/models-from-code.md) is a feature that allows you to define and log models in code. Logging a model with code brings several benefits, such as portability, readability, avoiding serialization, and more.

Combining with MLflow Prompt Registry, the feature unlocks even more flexibility to manage prompt versions. Notably, if your model code uses a prompt from MLflow Prompt Registry, MLflow **automatically** logs it with the model for you.

In the following example, we use LangGraph to define a very simple chat bot using the registered prompt.

### 1. Create a prompt[​](#1-create-a-prompt-1 "Direct link to 1. Create a prompt")

text

```
import mlflow

# Register a new prompt
prompt = mlflow.genai.register_prompt(
    name="chat-prompt",
    template="You are an expert in programming. Please answer the user's question about programming.",
)
```

### 2. Define a Graph using the registered prompt[​](#2-define-a-graph-using-the-registered-prompt "Direct link to 2. Define a Graph using the registered prompt")

Create a Python script `chatbot.py` with the following content.

tip

If you are using Jupyter notebook, you can uncomment the `%writefile` magic command and run the following code in a cell to generate the script.

python

```
# %%writefile chatbot.py

import mlflow
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: list


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
system_prompt = mlflow.genai.load_prompt("prompts:/chat-prompt/1")


def add_system_message(state: State):
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt.to_single_brace_format(),
            },
            *state["messages"],
        ]
    }


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("add_system_message", add_system_message)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "add_system_message")
graph_builder.add_edge("add_system_message", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

mlflow.models.set_model(graph)
```

### 3. Log the Graph to MLflow[​](#3-log-the-graph-to-mlflow "Direct link to 3. Log the Graph to MLflow")

Specify the file path to the script in the `model` parameter:

python

```
with mlflow.start_run():
    model_info = mlflow.langchain.log_model(
        lc_model="./chatbot.py",
        name="graph",
    )
```

We didn't specify the `prompts` parameter this time, but MLflow automatically logs the prompt loaded within the script to the logged model. Now you can view the associated prompt in MLflow UI:

![Associated Prompts](/mlflow-website/docs/latest/assets/images/prompt-logged-graph-0b4cd4a6c6e28f2afeffea5a9bcee188.png)

### 4. Load the graph back and invoke[​](#4-load-the-graph-back-and-invoke "Direct link to 4. Load the graph back and invoke")

Finally, let's load the graph back and invoke it to see the chatbot in action.

python

```
# Enable MLflow tracing for LangChain to view the prompt passed to LLM.
mlflow.langchain.autolog()

# Load the graph
graph = mlflow.langchain.load_model(model_info.model_uri)

graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the difference between multi-threading and multi-processing?",
            }
        ]
    }
)
```

![Chatbot](/mlflow-website/docs/latest/assets/images/prompt-logged-trace-f531e466499e24d2b8541d655237ea91.png)
