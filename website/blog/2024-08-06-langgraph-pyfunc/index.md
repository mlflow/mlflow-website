---
title: LangGraph with Custom PyFunc 
tags: [genai, mlops]
slug: mlflow-tracing
authors: [mlflow-maintainers]
thumbnail: img/blog/release-candidates.png
---

In this blog, we'll guide you through creating a LangGraph chatbot within an MLflow custom PyFunc.

### What is a Custom PyFunc?

While MLflow strives to cover the many popular machine learning libraries, there are libraries and pieces of functionality within supported libraries that are not natively supported. For these cases, when users want MLflow functionality they can create a custom PyFunc model. 
Custom PyFunc models allow you to integrate any Python code, providing flexibility in defining complex and niche machine learning models. These models can be easily logged, managed, and deployed using the typical MLflow APIs, enhancing flexibility and portability in machine learning workflows.

### What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures, differentiating it from DAG-based solutions. As a very low-level framework, it provides fine-grained control over both the flow and state of your application, crucial for creating reliable agents. Additionally, LangGraph includes built-in persistence, enabling advanced human-in-the-loop and memory features.
LangGraph is inspired by Pregel and Apache Beam. The public interface draws inspiration from NetworkX. LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.

# The Code

### Setup
First, we must install the required dependencies. We will use OpenAI for our LLM, but LangChain paired with LangGraph makes it easy to substitute your desired LLM.

```python
%%capture --no-stderr
%pip install -U langgraph langsmith mlflow
%pip install --upgrade typing_extensions

# Used for this tutorial; not a requirement for LangGraph
%pip install -U langchain_openai
```

Next, let's get our relevant secrets.

```python
import os
import getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")
```

### Custom Dependencies
Before building our custom PyFunc model, let's first review what PyFunc does. When you specify a model to log, MLflow will try to leverage `cloudpickle` to store the serialized model. Along with this model artifact, MLflow will store the signature, which can be passed or inferred from the `model_input` parameter. It will also log inferred model dependencies to help you serve the model in a new environment.

However, as with many other packages, pickling objects is often not supported by LangGraph chains. 

To get around this issue, we will leverage the [code_paths](https://mlflow.org/docs/latest/model/dependencies.html?highlight=code_paths#saving-extra-code-with-an-mlflow-model-manual-declaration) argument to specify a custom dependency. Instead of serializing the entire LangGraph chain, we'll look to reload it from a python file. 

Below, we use the magic `%%writefile` command to create a new file in a jupyter notebook context. If you're running this outside of an interactive notebook, simply create the file below, omitting the `%%writefile graph_chain.py` line.

```python
%%writefile graph_chain.py

from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
llm = ChatOpenAI()


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
compiled_graph = graph_builder.compile()
```

By the end of this step, you should see another file in your current directory with the name `graph_chain.py`.

### Custom PyFunc
Great! Now that the custom dependency is created at `./graph_chain.py`, we are ready to declare a custom PyFunc and log the model.

To quickly summarize the code below, we look to create a class that inherits from `mlflow.pyfunc.PythonModel`. Within that class we create a `predict` method that is standard to MLflow. We also create a variety of helpers to facilitate chatbot functionality. One notable call-out is that in the `predict` method, we import out custom `compiled_graph` object from the `graph_chain` python module created in the above step. That will dynamically instantiate our chain without using pickle!

After creating this class, we leverage the standard APIs to log the model. 

```python
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models.utils import _convert_llm_input_data
from langchain_core.messages import HumanMessage


class LangGraphCustomChain(PythonModel):
    messages: list = []
    graph = None

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

    def load_context(self, context):
        # Validate enviornment variable secrets are present
        required_secrets = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]

        if missing_keys := [key for key in required_secrets if not os.environ.get(key)]:
            raise ValueError(f"The following keys are missing: {missing_keys}")

        # Add project environent variables if not present
        os.environ["LANCHAIN_TRACING_V2"] = (
            "true"
            if not os.environ.get("LANGCHAIN_TRACING_V2")
            else os.environ.get("LANGCHAIN_TRACING_V2")
        )
        os.environ["LANGCHAIN_PROJECT"] = (
            "LangGraph MLflow Tutorial"
            if not os.environ.get("LANGCHAIN_TRACING_V2")
            else os.environ.get("LANGCHAIN_TRACING_V2")
        )

    def predict(self, context, input_data: str, params: dict = None):
        # Import from the `graph_chain.py` module specified in the `code_paths` argument
        from graph_chain import compiled_graph

        self.graph = compiled_graph

        # Format the string input as messages
        state = self._format_input(input_data)

        # Run inference on our loaded chain
        output_messages = self.graph.invoke(state)

        # Update messages and return the string
        return self._parse_output(output_messages)


# Save the model
with mlflow.start_run() as run:
    # Log the model to the mlflow tracking server
    mlflow.pyfunc.log_model(
        artifact_path="langgraph_model",
        python_model=LangGraphCustomChain(),
        input_example=["hi"],
        code_paths=["./graph_chain.py"],
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
* Declaring an MVP LangGraph chain within an additional python file.
* Creating a custom PyFunc class that uses the above chain to create a stateful chatbot.
* Logging and loading the above objects via the standard MLflow APIs.

Happy coding!