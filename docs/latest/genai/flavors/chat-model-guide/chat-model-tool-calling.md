# Build a tool-calling model with mlflow\.pyfunc.ChatModel

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/chat-model-guide/chat-model-tool-calling.ipynb)

Welcome to the notebook tutorial on building a simple tool calling model using the [mlflow.pyfunc.ChatModel](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatModel) wrapper. ChatModel is a subclass of MLflow's highly customizable [PythonModel](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel), which was specifically designed to make creating GenAI workflows easier.

Briefly, here are some of the benefits of using ChatModel:

1. No need to define a complex signature! Chat models often accept complex inputs with many levels of nesting, and this can be cumbersome to define yourself.
2. Support for JSON / dict inputs (no need to wrap inputs or convert to Pandas DataFrame)
3. Includes the use of Dataclasses for defining expected inputs / outputs for a simplified development experience

For a more in-depth exploration of ChatModel, please check out the [detailed guide](https://mlflow.org/docs/latest/llms/chat-model-guide/index.html).

In this tutorial, we'll be building a simple OpenAI wrapper that makes use of the tool calling support (released in MLflow 2.17.0).

### Environment setup[​](#environment-setup "Direct link to Environment setup")

First, let's set up the environment. We'll need the OpenAI Python SDK, as well as MLflow >= 2.17.0. We'll also need to set our OpenAI API key in order to use the SDK.

python

```python
%pip install 'mlflow>=2.17.0' 'openai>=1.0' -qq

```

python

```python
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

```

### Step 1: Creating the tool definition[​](#step-1-creating-the-tool-definition "Direct link to Step 1: Creating the tool definition")

Let's begin to define our model! As mentioned in the introduction, we'll be subclassing `mlflow.pyfunc.ChatModel`. For this example, we'll build a toy model that uses a tool to retrieve the weather for a given city.

The first step is to create a tool definition that we can pass to OpenAI. We do this by using [mlflow.types.llm.FunctionToolDefinition](https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.FunctionToolDefinition) to describe the parameters that our tool accepts. The format of this dataclass is aligned with the OpenAI spec:

python

```python
import mlflow
from mlflow.types.llm import (
  FunctionToolDefinition,
  ParamProperty,
  ToolParamsSchema,
)


class WeatherModel(mlflow.pyfunc.ChatModel):
  def __init__(self):
      # a sample tool definition. we use the `FunctionToolDefinition`
      # class to describe the name and expected params for the tool.
      # for this example, we're defining a simple tool that returns
      # the weather for a given city.
      weather_tool = FunctionToolDefinition(
          name="get_weather",
          description="Get weather information",
          parameters=ToolParamsSchema(
              {
                  "city": ParamProperty(
                      type="string",
                      description="City name to get weather information for",
                  ),
              }
          ),
          # make sure to call `to_tool_definition()` to convert the `FunctionToolDefinition`
          # to a `ToolDefinition` object. this step is necessary to normalize the data format,
          # as multiple types of tools (besides just functions) might be available in the future.
      ).to_tool_definition()

      # OpenAI expects tools to be provided as a list of dictionaries
      self.tools = [weather_tool.to_dict()]

```

### Step 2: Implementing the tool[​](#step-2-implementing-the-tool "Direct link to Step 2: Implementing the tool")

Now that we have a definition for the tool, we need to actually implement it. For the purposes of this tutorial, we're just going to mock a response, but the implementation can be arbitrary—you might make an API call to an actual weather service, for example.

python

```python
class WeatherModel(mlflow.pyfunc.ChatModel):
  def __init__(self):
      weather_tool = FunctionToolDefinition(
          name="get_weather",
          description="Get weather information",
          parameters=ToolParamsSchema(
              {
                  "city": ParamProperty(
                      type="string",
                      description="City name to get weather information for",
                  ),
              }
          ),
      ).to_tool_definition()

      self.tools = [weather_tool.to_dict()]

      def get_weather(self, city: str) -> str:
          # in a real-world scenario, the implementation might be more complex
          return f"It's sunny in {city}, with a temperature of 20C"

```

### Step 3: Implementing the `predict` method[​](#step-3-implementing-the-predict-method "Direct link to step-3-implementing-the-predict-method")

The next thing we need to do is define a `predict()` function that accepts the following arguments:

1. `context`: [PythonModelContext](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModelContext) (not used in this tutorial)
2. `messages`: List\[[ChatMessage](https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.ChatMessage)]. This is the chat input that the model uses for generation.
3. `params`: [ChatParams](https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.ChatParams). These are commonly used params used to configure the chat model, e.g. `temperature`, `max_tokens`, etc. This is where the tool specifications can be found.

This is the function that will ultimately be called during inference.

For the implementation, we'll simply forward the user's input to OpenAI, and provide the `get_weather` tool as an option for the LLM to use if it chooses to do so. If we receive a tool call request, we'll call the `get_weather()` function and return the response back to OpenAI. We'll need to use what we've defined in the previous two steps in order to do this.

python

```python
import json

from openai import OpenAI

import mlflow
from mlflow.types.llm import (
  ChatMessage,
  ChatParams,
  ChatResponse,
)


class WeatherModel(mlflow.pyfunc.ChatModel):
  def __init__(self):
      weather_tool = FunctionToolDefinition(
          name="get_weather",
          description="Get weather information",
          parameters=ToolParamsSchema(
              {
                  "city": ParamProperty(
                      type="string",
                      description="City name to get weather information for",
                  ),
              }
          ),
      ).to_tool_definition()

      self.tools = [weather_tool.to_dict()]

  def get_weather(self, city: str) -> str:
      return "It's sunny in {}, with a temperature of 20C".format(city)

  # the core method that needs to be implemented. this function
  # will be called every time a user sends messages to our model
  def predict(self, context, messages: list[ChatMessage], params: ChatParams):
      # instantiate the OpenAI client
      client = OpenAI()

      # convert the messages to a format that the OpenAI API expects
      messages = [m.to_dict() for m in messages]

      # call the OpenAI API
      response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=messages,
          # pass the tools in the request
          tools=self.tools,
      )

      # if OpenAI returns a tool_calling response, then we call
      # our tool. otherwise, we just return the response as is
      tool_calls = response.choices[0].message.tool_calls
      if tool_calls:
          print("Received a tool call, calling the weather tool...")

          # for this example, we only provide the model with one tool,
          # so we can assume the tool call is for the weather tool. if
          # we had more, we'd need to check the name of the tool that
          # was called
          city = json.loads(tool_calls[0].function.arguments)["city"]
          tool_call_id = tool_calls[0].id

          # call the tool and construct a new chat message
          tool_response = ChatMessage(
              role="tool", content=self.get_weather(city), tool_call_id=tool_call_id
          ).to_dict()

          # send another request to the API, making sure to append
          # the assistant's tool call along with the tool response.
          messages.append(response.choices[0].message)
          messages.append(tool_response)
          response = client.chat.completions.create(
              model="gpt-4o-mini",
              messages=messages,
              tools=self.tools,
          )

      # return the result as a ChatResponse, as this
      # is the expected output of the predict method
      return ChatResponse.from_dict(response.to_dict())

```

### Step 4 (optional, but recommended): Enable tracing for the model[​](#step-4-optional-but-recommended-enable-tracing-for-the-model "Direct link to Step 4 (optional, but recommended): Enable tracing for the model")

This step is optional, but highly recommended to improve observability in your app. We'll be using [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html) to log the inputs and outputs of our model's internal functions, so we can easily debug when things go wrong. Agent-style tool calling models can make many layers of function calls during the lifespan of a single request, so tracing is invaluable in helping us understand what's going on at each step.

Integrating tracing is easy, we simply decorate the functions we're interested in (`get_weather()` and `predict()`) with `@mlflow.trace`! MLflow Tracing also has integrations with many popular GenAI frameworks, such as LangChain, OpenAI, LlamaIndex, and more. For the full list, check out this [documentation page](https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing). In this tutorial, we're using the OpenAI SDK to make API calls, so we can enable tracing for this by calling `mlflow.openai.autolog()`.

To view the traces in the UI, run `mlflow ui` in a separate terminal shell, and navigate to the `Traces` tab after using the model for inference below.

python

```python
from mlflow.entities.span import (
  SpanType,
)

# automatically trace OpenAI SDK calls
mlflow.openai.autolog()


class WeatherModel(mlflow.pyfunc.ChatModel):
  def __init__(self):
      weather_tool = FunctionToolDefinition(
          name="get_weather",
          description="Get weather information",
          parameters=ToolParamsSchema(
              {
                  "city": ParamProperty(
                      type="string",
                      description="City name to get weather information for",
                  ),
              }
          ),
      ).to_tool_definition()

      self.tools = [weather_tool.to_dict()]

  @mlflow.trace(span_type=SpanType.TOOL)
  def get_weather(self, city: str) -> str:
      return "It's sunny in {}, with a temperature of 20C".format(city)

  @mlflow.trace(span_type=SpanType.AGENT)
  def predict(self, context, messages: list[ChatMessage], params: ChatParams):
      client = OpenAI()

      messages = [m.to_dict() for m in messages]

      response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=messages,
          tools=self.tools,
      )

      tool_calls = response.choices[0].message.tool_calls
      if tool_calls:
          print("Received a tool call, calling the weather tool...")

          city = json.loads(tool_calls[0].function.arguments)["city"]
          tool_call_id = tool_calls[0].id

          tool_response = ChatMessage(
              role="tool", content=self.get_weather(city), tool_call_id=tool_call_id
          ).to_dict()

          messages.append(response.choices[0].message)
          messages.append(tool_response)
          response = client.chat.completions.create(
              model="gpt-4o-mini",
              messages=messages,
              tools=self.tools,
          )

      return ChatResponse.from_dict(response.to_dict())

```

### Step 5: Logging the model[​](#step-5-logging-the-model "Direct link to Step 5: Logging the model")

Finally, we need to log the model. This saves the model as an artifact in MLflow Tracking, and allows us to load and serve it later on.

(Note: this is a fundamental pattern in MLflow. To learn more, check out the [Quickstart guide](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html)!)

In order to do this, we need to do a few things:

1. Define an input example to inform users about the input we expect
2. Instantiate the model
3. Call `mlflow.pyfunc.log_model()` with the above as arguments

Take note of the Model URI printed out at the end of the cell—we'll need it when serving the model later!

python

```python
# messages to use as input examples
messages = [
  {"role": "system", "content": "Please use the provided tools to answer user queries."},
  {"role": "user", "content": "What's the weather in Singapore?"},
]

input_example = {
  "messages": messages,
}

# instantiate the model
model = WeatherModel()

# log the model
with mlflow.start_run():
  model_info = mlflow.pyfunc.log_model(
      name="weather-model",
      python_model=model,
      input_example=input_example,
  )

  print("Successfully logged the model at the following URI: ", model_info.model_uri)

```

### Using the model for inference[​](#using-the-model-for-inference "Direct link to Using the model for inference")

Now that the model is logged, our work is more or less done! In order to use the model for inference, let's load it back using `mlflow.pyfunc.load_model()`.

python

```python
import mlflow

# Load the previously logged ChatModel
tool_model = mlflow.pyfunc.load_model(model_info.model_uri)

system_prompt = {
  "role": "system",
  "content": "Please use the provided tools to answer user queries.",
}

messages = [
  system_prompt,
  {"role": "user", "content": "What's the weather in Singapore?"},
]

# Call the model's predict method
response = tool_model.predict({"messages": messages})
print(response["choices"][0]["message"]["content"])

messages = [
  system_prompt,
  {"role": "user", "content": "What's the weather in San Francisco?"},
]

# Generating another response
response = tool_model.predict({"messages": messages})
print(response["choices"][0]["message"]["content"])

```

### Serving the model[​](#serving-the-model "Direct link to Serving the model")

MLflow also allows you to serve models, using the `mlflow models serve` CLI tool. In another terminal shell, run the following from the same folder as this notebook:

sh

```sh
$ export OPENAI_API_KEY=<YOUR OPENAI API KEY>
$ mlflow models serve -m <MODEL_URI>

```

This will start serving the model on `http://127.0.0.1:5000`, and the model can be queried via POST request to the `/invocations` route.

python

```python
import requests

messages = [
  system_prompt,
  {"role": "user", "content": "What's the weather in Tokyo?"},
]

response = requests.post("http://127.0.0.1:5000/invocations", json={"messages": messages})
response.raise_for_status()
response.json()

```

### Conclusion[​](#conclusion "Direct link to Conclusion")

In this tutorial, we covered how to use MLflow's `ChatModel` class to create a convenient OpenAI wrapper that supports tool calling. Though the use-case was simple, the concepts covered here can be easily extended to support more complex functionality.

If you're looking to dive deeper into building quality GenAI apps, you might be also be interested in checking out [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html), an observability tool you can use to trace the execution of arbitrary functions (such as your tool calls, for example).
