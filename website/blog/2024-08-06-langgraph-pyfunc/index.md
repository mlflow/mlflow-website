---
title: LangGraph with Custom PyFunc
tags: [genai, mlops]
slug: mlflow
authors: [michael-berk, mlflow-maintainers]
thumbnail: /img/blog/release-candidates.png
---

In this blog, we'll guide you through creating a LangGraph chatbot within an MLflow custom PyFunc. By combining MLflow with LangGraph's ability to create and manage cyclical graphs, you can create powerful stateful, multi-actor applications in a scalable fashion.

Throughout this post we will demonstrate how to leverage MLflow's ChatModel to create a serializable and servable MLflow model which can easily be tracked, versioned, and deployed on a variety of servers.

### What is a Custom PyFunc?

While MLflow strives to cover many popular machine learning libraries, there has been a proliferation of open source packages. If users want MLflow's myriad benefits paired with a package that doesn't have native support, users can create a [custom PyFunc model](https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/index.html or https://mlflow.org/blog/custom-pyfunc).
Custom PyFunc models allow you to integrate any Python code, providing flexibility in defining GenAI apps and AI models. These models can be easily logged, managed, and deployed using the typical MLflow APIs, enhancing flexibility and portability in machine learning workflows.

Within the category of custom PyFunc models, MLflow supports a specialized model called [ChatModel](https://mlflow.org/docs/latest/llms/transformers/tutorials/conversational/pyfunc-chat-model.html). It extends the base PyFunc functionality to specifically support messages. For this demo, we will use ChatModel to create a LangGraph chatbot.

### What is LangGraph?

[LangGraph](https://langchain-ai.github.io/langgraph/) is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits:

- **Cycles and Branching**: Implement loops and conditionals in your apps.
- **Persistence**: Automatically save state after each step in the graph. Pause and resume the graph execution at any point to support error recovery, human-in-the-loop workflows, time travel and more.
- **Human-in-the-Loop**: Interrupt graph execution to approve or edit next action planned by the agent.
- **Streaming Support**: Stream outputs as they are produced by each node (including token streaming).
- **Integration with LangChain**: LangGraph integrates seamlessly with LangChain and LangSmith (but does not require them).

LangGraph allows you to define flows that involve cycles, essential for most agentic architectures, differentiating it from DAG-based solutions. As a very low-level framework, it provides fine-grained control over both the flow and state of your application, crucial for creating reliable agents. Additionally, LangGraph includes built-in persistence, enabling advanced human-in-the-loop and memory features.

LangGraph is inspired by Pregel and Apache Beam. The public interface draws inspiration from NetworkX. LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.

For a full walkthrough, check out the [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/) and for more on the fundamentals of design with LangGraph, check out the [conceptual guides](https://langchain-ai.github.io/langgraph/concepts/#human-in-the-loop).

## 1 - Setup

First, we must install the required dependencies. We will use OpenAI for our LLM in this example, but using LangChain with LangGraph makes it easy to substitute any alternative supported LLM or LLM provider.

```python
%%capture
%pip install langgraph==0.2.3 langsmith==0.1.98 mlflow>=2.15.1
%pip install -U typing_extensions
%pip install langchain_openai==0.1.21
```

Next, let's get our relevant secrets. `getpass`, as demonstrated in the [LangGraph quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/#setup) is a great way to insert your keys into an interactive jupyter environment.

```python
import os

# Set required environment variables for authenticating to OpenAI and LangSmith
# Check additional MLflow tutorials for examples of authentication if needed
# https://mlflow.org/docs/latest/llms/openai/guide/index.html#direct-openai-service-usage
assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."
assert "LANGSMITH_API_KEY" in os.environ, "Please set the LANGSMITH_API_KEY environment variable."
```

## 2 - Custom Utilities

While this is a demo, it's good practice to separate reusable utilities into a separate file/directory. Below we create three general utilities that theoretically would valuable when building additional MLflow + LangGraph implementations.

Note that we use the magic `%%writefile` command to create a new file in a jupyter notebook context. If you're running this outside of an interactive notebook, simply create the file below, omitting the `%%writefile {FILE_NAME}.py` line.

```python
%%writefile langgraph_utils.py
# omit this line if directly creating this file; this command is purely for running within Jupyter

import os
from typing import Union, List, Dict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
)
from mlflow.types.llm import ChatMessage


def validate_langgraph_environment_variables():
    """Ensure that required secrets and project environment variables are present."""

    # Validate enviornment variable secrets are present
    required_secrets = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]

    if missing_keys := [key for key in required_secrets if not os.environ.get(key)]:
        raise ValueError(f"The following keys are missing: {missing_keys}")

    # Add project environent variables if not present
    os.environ["LANCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.environ.get(
        "LANGCHAIN_TRACING_V2", "LangGraph MLflow Tutorial"
    )


def _format_mlflow_chat_message_for_langraph_message(
    chat_message: ChatMessage,
) -> Dict:
    mlflow_role_to_langgraph_type = {
        "user": "human",
        "assistant": "ai",
        "system": "system",
    }

    if role_clean := mlflow_role_to_langgraph_type.get(chat_message.role):
        return {"type": role_clean, "data": {"content": chat_message.content}}
    else:
        raise ValueError(f"Incorrect role specified: {chat_message.role}")


def mlflow_chat_message_to_langgraph_message(
    chat_message: List[ChatMessage],
) -> List[Union[AIMessage, HumanMessage, SystemMessage]]:
    """Convert MLflow messages (list of mlflow.types.llm.ChatMessage) to LangGraph messages.

    This utility is required because LangGraph messages have a different structure and type
    than MLflow ChatMessage. If we pass the payload coming into our `predict()` method directly
    into the LangGraph graph, we'll get an error.
    """
    # NOTE: This is a simplified example for demonstration purposes
    if isinstance(chat_message, list):
        list_of_parsed_dicts = [
            _format_mlflow_chat_message_for_langraph_message(d) for d in chat_message
        ]
        return messages_from_dict(list_of_parsed_dicts)
    else:
        raise ValueError(f"Invalid _dict type: {type(chat_message)}")

```

By the end of this step, you should see a new file in your current directory with the name `langgraph_utils.py`.

Note that it's best practice to add unit tests and properly organize your project into logically structured directories.

## 3 - Custom PyFunc ChatModel

Great! Now that we have some reusable utilities located in `./langgraph_utils.py`, we are ready to declare a custom PyFunc and log the model. However, before writing more code, let's provide some quick background on the **Model from Code** feature.

### 3.1 - Create our Model-From-Code File

Historically, MLflow's process of saving a custom `pyfunc` model uses a mechanism that has some frustrating drawbacks: `cloudpickle`. Prior to the release of support for saving a model as a Python script in MLflow 2.12.2 (known as the [models from code](https://mlflow.org/docs/latest/models.html#models-from-code) feature), logging a defined `pyfunc` involved pickling an instance of that model. Along with the pickled model artifact, MLflow will store the signature, which can be passed or inferred from the `model_input` parameter. It will also log inferred model dependencies to help you serve the model in a new environment.

Pickle is an easy-to-use serialization mechanism, but it has a variety of limitations:

- **Limited Support for Some Data Types**: `cloudpickle` may struggle with serializing certain complex or low-level data types, such as file handles, sockets, or objects containing these types, which can lead to errors or incorrect deserialization.
- **Version Compatibility Issues**: Serialized objects with `cloudpickle` may not be deserializable across different versions of `cloudpickle` or Python, making long-term storage or sharing between different environments risky.
- **Recursion Depth for Nested Dependencies**: `cloudpickle` can serialize objects with nested dependencies (e.g., functions within functions, or objects that reference other objects). However, deeply nested dependencies can hit the recursion depth limit imposed by Python's interpreter.
- **Mutable Object States that Cannot be Serialized**: `cloudpickle` struggles to serialize certain mutable objects whose states change during runtime, especially if these objects contain non-serializable elements like open file handles, thread locks, or custom C extensions. Even if `cloudpickle` can serialize the object structure, it may fail to capture the exact state or may not be able to deserialize the state accurately, leading to potential data loss or incorrect behavior upon deserialization.

To get around this issue, we must perform the following steps:

1. Create an additional Python file in our directory.
2. In that file, create a function that creates a [CompiledStateGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot), which is DAG-based stateful chatbot.
3. Also in that file, create a [MLflow custom PyFunc](https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/index.html). Note that in our case, we're using a [custom ChatModel](https://mlflow.org/docs/latest/llms/transformers/tutorials/conversational/pyfunc-chat-model.html#Customizing-the-model).
4. Also in that file, set the custom ChatModel to be accessible by [MLflow model from code](https://mlflow.org/docs/latest/models.html#models-from-code) via the [mlflow.models.set_model()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.set_model) command.
5. In a different file, log the **path** to the file created in steps 1-3 instead of the model object.

By passing a Python file, we simply can load the model from that Python code, thereby bypassing all the headaches associated with serialization and `cloudpickle`.

```python
%%writefile graph_chain.py
# omit this line if directly creating this file; this command is purely for running within Jupyter

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

# Our custom utilities
from langgraph_utils import (
    mlflow_chat_message_to_langgraph_message,
    validate_langgraph_environment_variables,
)

import mlflow
from mlflow.types.llm import ChatMessage, ChatParams, ChatResponse

import random
from typing import Annotated, List
from typing_extensions import TypedDict


def load_graph() -> CompiledStateGraph:
    """Create example chatbot from LangGraph Quickstart."""

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)
    llm = ChatOpenAI()

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


class LangGraphChatModel(mlflow.pyfunc.ChatModel):
    def load_context(self, context):
        self.graph = load_graph()

    def predict(
        self, context, messages: List[ChatMessage], params: ChatParams
    ) -> ChatResponse:

        # Format mlflow ChatMessage as LangGraph messages
        messages = mlflow_chat_message_to_langgraph_message(messages)

        # Query the model
        response = self.graph.invoke({"messages": messages})

        # Extract the response text
        text = response["messages"][-1].content

        # NB: chat session ID should be handled on the client side. Here we
        # create a placeholder for demonstration purposes. Furthermore, if you
        # need persistance between model sessions, it's a good idea to
        # write your session history to a database.
        id = f"some_meaningful_id_{random.randint(0, 100)}"

        # Format the response to be compatible with MLflow
        response = {
            "id": id,
            "model": "MyChatModel",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

        return ChatResponse(**response)


# Set our model to be accessible by MLflow model from code
mlflow.models.set_model(LangGraphChatModel())
```

### 3.2 - Log our Model-From-Code

After creating this ChatModel implementation in we leverage the standard MLflow APIs to log the model. However, as noted above, instead of passing a model object, we pass the path `str` to the file containing our `mlflow.models.set_model()` command.

```python
import mlflow

# Save the model
with mlflow.start_run() as run:
    # Log the model to the mlflow tracking server
    mlflow.pyfunc.log_model(
        python_model="graph_chain.py", # Path to our custom model
        artifact_path="langgraph_model",
    )

    # Store the run id for later loading
    run_id = run.info.run_id
```

## 4 - Use the Logged Model

Now that we have successfully logged a model, we can load it and leverage it for inference.

In the code below, we demonstrate that our chain has chatbot functionality!

```python
import mlflow

# Load the model
# NOTE: you need the run_id from the above step or another model URI format
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/langgraph_model")

# Show inference and message history
print("-------- Message 1 -----------")
message = "What's my name?"
payload = {"messages": [{"role": "user", "content": message}]}
response = loaded_model.predict(payload)

print(f"User: {message}")
print(f"Agent: {response['choices'][-1]['message']['content']}")

# print("\n-------- Message 2 -----------")
message = "My name is Morpheus."
message_history = [choice['message'] for choice in response['choices']]
payload = {"messages": message_history + [{"role": "user", "content": message}]}
response = loaded_model.predict(payload)

print(f"User: {message}")
print(f"Agent: {response['choices'][-1]['message']['content']}")

# # print("\n-------- Message 3 -----------")
message = "What's my name?"
message_history = [choice['message'] for choice in response['choices']]
payload = {"messages": message_history + [{"role": "user", "content": message}]}
response = loaded_model.predict(payload)

print(f"User: {message}")
print(f"Agent: {response['choices'][-1]['message']['content']}")
```

Ouput:

```text
-------- Message 1 -----------
User: What's my name?
Agent: I'm sorry, I don't know your name. Can you please tell me?

-------- Message 2 -----------
User: My name is Morpheus.
Agent: Nice to meet you, Morpheus! How can I assist you today?

-------- Message 3 -----------
User: What's my name?
Agent: Your name is Morpheus!
```

## 5 - Summary

There are many logical extensions of the this tutorial, however the MLflow components can remain largely unchanged. Some examples include persisting chat history to a database, implementing a more complex langgraph object, productionizing this solution, and much more!

To summarize, here's what was covered in this tutorial:

- Creating a simple LangGraph chain.
- Declaring a custom MLflow PyFunc ChatModel that wraps the above LangGraph chain with pre/post-processing logic.
- Leveraging MLflow [model from code](https://mlflow.org/docs/latest/models.html#models-from-code) functionality to log our Custom PyFunc.
- Loading the Custom PyFunc via the standard MLflow APIs.

Happy coding!
