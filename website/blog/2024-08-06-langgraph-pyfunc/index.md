---
title: LangGraph with Custom PyFunc 
tags: [genai, mlops]
slug: mlflow-tracing
authors: [mlflow-maintainers]
thumbnail: img/blog/release-candidates.png
---

In this blog, we'll guide you through creating a LangGraph chatbot within an MLflow custom PyFunc.

### What is a Custom PyFunc?

While MLflow strives to cover many popular machine learning libraries, there has been a proliferation of open source packages. If users want MLflows myriad benefits paired with a package that doesn't have native support, users can create a custom PyFunc model. 
Custom PyFunc models allow you to integrate any Python code, providing flexibility in defining GenAI apps and AI mdoels. These models can be easily logged, managed, and deployed using the typical MLflow APIs, enhancing flexibility and portability in machine learning workflows.

### What is LangGraph?

[LangGraph](https://langchain-ai.github.io/langgraph/) is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: 
* **Cycles and Branching**: Implement loops and conditionals in your apps.
* **Persistence**: Automatically save state after each step in the graph. Pause and resume the graph execution at any point to support error recovery, human-in-the-loop workflows, time travel and more.
* **Human-in-the-Loop**: Interrupt graph execution to approve or edit next action planned by the agent.
* **Streaming Support**: Stream outputs as they are produced by each node (including token streaming).
* **Integration with LangChain**: LangGraph integrates seamlessly with LangChain and LangSmith (but does not require them).

LangGraph allows you to define flows that involve cycles, essential for most agentic architectures, differentiating it from DAG-based solutions. As a very low-level framework, it provides fine-grained control over both the flow and state of your application, crucial for creating reliable agents. Additionally, LangGraph includes built-in persistence, enabling advanced human-in-the-loop and memory features.

LangGraph is inspired by Pregel and Apache Beam. The public interface draws inspiration from NetworkX. LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.

For a full walkthrough, check out the [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/) and for more on the fundamentals of design with LangGraph, check out the [conceptual guides](https://langchain-ai.github.io/langgraph/concepts/#human-in-the-loop).

# The Code

### Setup
First, we must install the required dependencies. We will use OpenAI for our LLM in this example, but using LangChain with LangGraph makes it easy to substitute any alternative supported LLM or LLM provider.

```python
%%capture
%pip install langgraph==0.2.3 langsmith==0.1.98 mlflow>=2.15.1
%pip install -U typing_extensions
%pip install langchain_openai==0.1.21
```

Next, let's get our relevant secrets.

```python
import os

# Set required environment variables for authenticating to OpenAI and LangSmith
# Check additional MLflow tutorials for examples of authentication if needed
# https://mlflow.org/docs/latest/llms/openai/guide/index.html#direct-openai-service-usage
assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."
assert "LANGSMITH_API_KEY" in os.environ, "Please set the LANGSMITH_API_KEY environment variable."
```

### Custom Utilities
Historically, MLflow's process of saving a custom `pyfunc` model uses a mechanism that has some frustrating drawbacks: `cloudpickle`. Prior to the release of support for saving a model as a python script in MLflow 2.12.2 (known as the [models from code](https://mlflow.org/docs/latest/models.html#models-from-code) feature), logging a defined `pyfunc` involved pickling an instance of that model. Along with the pickled model artifact, MLflow will store the signature, which can be passed or inferred from the `model_input` parameter. It will also log inferred model dependencies to help you serve the model in a new environment.

Pickle is an easy-to-use serialization mechanism, but it has a variety of limitations: 
* **Limited Support for Some Data Types**: `cloudpickle` may struggle with serializing certain complex or low-level data types, such as file handles, sockets, or objects containing these types, which can lead to errors or incorrect deserialization.
* **Version Compatibility Issues**: Serialized objects with `cloudpickle` may not be deserializable across different versions of `cloudpickle` or Python, making long-term storage or sharing between different environments risky.
* **Recursion Depth for Nested Dependencies**: `cloudpickle` can serialize objects with nested dependencies (e.g., functions within functions, or objects that reference other objects). However, deeply nested dependencies can hit the recursion depth limit imposed by Python's interpreter.
* **Mutable Object States that Cannot be Serialized**: `cloudpickle` struggles to serialize certain mutable objects whose states change during runtime, especially if these objects contain non-serializable elements like open file handles, thread locks, or custom C extensions. Even if `cloudpickle` can serialize the object structure, it may fail to capture the exact state or may not be able to deserialize the state accurately, leading to potential data loss or incorrect behavior upon deserialization.

To get around this issue, we will leverage the [code_paths](https://mlflow.org/docs/latest/model/dependencies.html?highlight=code_paths#saving-extra-code-with-an-mlflow-model-manual-declaration) argument to specify a custom dependency. Instead of serializing the LangGraph chain in binary format, we leverage python to instantiate the object. On the backend, MLflow simply runs this python file with the specified package dependencies.

Below, we use the magic `%%writefile` command to create a new file in a jupyter notebook context. If you're running this outside of an interactive notebook, simply create the file below, omitting the `%%writefile {FILE_NAME}.py` line.

```python
%%writefile langgraph_utils.py
# omit this line if directly creating this file; this command is purely for running within Jupyter

import os

def validate_langgraph_environment_variables():
    """Ensure that required secrets and project environment variables are present."""
    
    # Validate enviornment variable secrets are present
    required_secrets = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]

    if missing_keys := [key for key in required_secrets if not os.environ.get(key)]:
        raise ValueError(f"The following keys are missing: {missing_keys}")

    # Add project environent variables if not present
    os.environ["LANCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.environ.get(
        "LANGCHAIN_TRACING_V2", 
        "LangGraph MLflow Tutorial"
    )

```


By the end of this step, you should see a new file in your current directory with the name `langgraph_utils.py`.

### Custom PyFunc
Great! Now that the custom dependency is created at `./langgraph_utils.py`, we are ready to declare a custom PyFunc and log the model.

At a high level, we perform the following steps.
1. Create a LangGraph `CompiledStateGraph` via the `_load_graph()` function. 
2. Wrap this LangGraph `CompiledStateGraph` with an [MLflow custom PyFunc](https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/index.html).
3. Set the model to be accessible by [MLflow model from code](https://mlflow.org/docs/latest/models.html#models-from-code) via the [mlflow.models.set_model()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.set_model) command.


```python
%%writefile graph_chain.py
# omit this line if directly creating this file; this command is purely for running within Jupyter

import mlflow 
from mlflow.pyfunc import PythonModel
from mlflow.models.utils import _convert_llm_input_data

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from typing import Annotated
from typing_extensions import TypedDict

# Our custom utility
from langgraph_utils import validate_langgraph_environment_variables

def _load_graph() -> CompiledStateGraph:
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

class LangGraphCustomChain(PythonModel):
    messages: list = []

    def __init__(self):
        # Validate required secret keys are present
        validate_langgraph_environment_variables()

        # Create the compiled graph to be used for prediction
        self.graph = _load_graph()

    def _format_input(self, text_input) -> dict:
        """Convert a string to a user message."""
        # Use an MLflow utility to clean the input
        converted_input = _convert_llm_input_data(text_input)

        # Handle lists (production logic would be more complex)
        if isinstance(converted_input, list):
            converted_input = next(iter(converted_input))

        # Format as dict of messages and return
        return {"messages": self.messages + [HumanMessage(converted_input)]}

    def _parse_output(self, message_output: dict) -> str:
        """Get the last message from the output."""
        # Append to message history
        self.messages = message_output["messages"]

        # Return the last message str
        return self.messages[-1].content

    def predict(self, context, input_data: str, params: dict = None):
        # Format the string input as messages
        state = self._format_input(input_data)

        # Run inference on our loaded chain
        output_messages = self.graph.invoke(state)

        # Update messages and return the string
        return self._parse_output(output_messages)


mlflow.models.set_model(LangGraphCustomChain())
```

### Custom PyFunc
After creating this class, we leverage the standard MLflow APIs to log the model. However, instead of passing a model object, we pass the path `str` to the file containing our `mlflow.models.set_model()` command. 

```python
import mlflow

# Save the model
with mlflow.start_run() as run:
    # Log the model to the mlflow tracking server
    mlflow.pyfunc.log_model(
        python_model="graph_chain.py", # Path to our custom model
        artifact_path="langgraph_model",
        input_example=["hi"],
    )

    run_id = run.info.run_id
```

### Use the Logged Model
Now that we have successfully logged a model, we can load it and leverage it for inference. 

In the code below, we demonstrate that our chat chain has memory!

```python
# Load the model
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/langgraph_model")

# Show inference and message history
print("-------- Message 1 -----------")
message = "What's my name?"
print(f"User: {message}")
print(f"Agent: {loaded_model.predict(message)}")

print("\n-------- Message 2 -----------")
message = "My name is Morpheus."
print(f"User: {message}")
print(f"Agent: {loaded_model.predict(message)}")

print("\n-------- Message 3 -----------")
message = "What's my name?"
print(f"User: {message}")
print(f"Agent: {loaded_model.predict(message)}")
```

Ouput:
```text
-------- Message 1 -----------
User: What's my name?
Agent: I'm sorry, but I do not have the ability to know your name unless you provide it to me. How can I assist you today?

-------- Message 2 -----------
User: My name is Morpheus?
Agent: Hello Morpheus! How can I assist you today?

-------- Message 3 -----------
User: What's my name?
Agent: Your name is Morpheus. How can I assist you today?
```

### Conclusion
There are many logical extensions of the this tutorial, however the MLflow components can remain largely unchanged. 

To summarize, here's what was covered in this tutorial:
* Creating a simple LangGraph chain.
* Declaring a custom MLflow PyFunc that wraps the above LangGraph chain with pre/post-processing logic. 
* Leveraging MLflow [model from code](https://mlflow.org/docs/latest/models.html#models-from-code) functionality to log our Custom PyFunc.
* Loading the Custom PyFunc via the standard MLflow APIs.

Happy coding!