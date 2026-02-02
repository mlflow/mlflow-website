"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["5486"],{18655(e,t,o){o.r(t),o.d(t,{metadata:()=>l,default:()=>u,frontMatter:()=>c,contentTitle:()=>d,toc:()=>m,assets:()=>h});var l=JSON.parse('{"id":"flavors/chat-model-guide/chat-model-tool-calling-ipynb","title":"Build a tool-calling model with mlflow.pyfunc.ChatModel","description":"Download this notebook","source":"@site/docs/genai/flavors/chat-model-guide/chat-model-tool-calling-ipynb.mdx","sourceDirName":"flavors/chat-model-guide","slug":"/flavors/chat-model-guide/chat-model-tool-calling","permalink":"/mlflow-website/docs/latest/genai/flavors/chat-model-guide/chat-model-tool-calling","draft":false,"unlisted":false,"editUrl":"https://github.com/mlflow/mlflow/edit/master/docs/docs/genai/flavors/chat-model-guide/chat-model-tool-calling.ipynb","tags":[],"version":"current","frontMatter":{"custom_edit_url":"https://github.com/mlflow/mlflow/edit/master/docs/docs/genai/flavors/chat-model-guide/chat-model-tool-calling.ipynb","slug":"chat-model-tool-calling"},"sidebar":"genAISidebar","previous":{"title":"Tutorial: Custom GenAI Models using ChatModel","permalink":"/mlflow-website/docs/latest/genai/flavors/chat-model-guide/"},"next":{"title":"Building with ResponsesAgent","permalink":"/mlflow-website/docs/latest/genai/flavors/responses-agent-intro"}}'),n=o(74848),s=o(28453),i=o(75940),a=o(75453);o(66354);var r=o(42676);let c={custom_edit_url:"https://github.com/mlflow/mlflow/edit/master/docs/docs/genai/flavors/chat-model-guide/chat-model-tool-calling.ipynb",slug:"chat-model-tool-calling"},d="Build a tool-calling model with mlflow.pyfunc.ChatModel",h={},m=[{value:"Environment setup",id:"environment-setup",level:3},{value:"Step 1: Creating the tool definition",id:"step-1-creating-the-tool-definition",level:3},{value:"Step 2: Implementing the tool",id:"step-2-implementing-the-tool",level:3},{value:"Step 3: Implementing the <code>predict</code> method",id:"step-3-implementing-the-predict-method",level:3},{value:"Step 4 (optional, but recommended): Enable tracing for the model",id:"step-4-optional-but-recommended-enable-tracing-for-the-model",level:3},{value:"Step 5: Logging the model",id:"step-5-logging-the-model",level:3},{value:"Using the model for inference",id:"using-the-model-for-inference",level:3},{value:"Serving the model",id:"serving-the-model",level:3},{value:"Conclusion",id:"conclusion",level:3}];function p(e){let t={a:"a",code:"code",h1:"h1",h3:"h3",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",...(0,s.R)(),...e.components};return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)(t.header,{children:(0,n.jsx)(t.h1,{id:"build-a-tool-calling-model-with-mlflowpyfuncchatmodel",children:"Build a tool-calling model with mlflow.pyfunc.ChatModel"})}),"\n",(0,n.jsx)(r.O,{href:"https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/chat-model-guide/chat-model-tool-calling.ipynb",children:"Download this notebook"}),"\n",(0,n.jsxs)(t.p,{children:["Welcome to the notebook tutorial on building a simple tool calling model using the ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatModel",children:"mlflow.pyfunc.ChatModel"})," wrapper. ChatModel is a subclass of MLflow's highly customizable ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel",children:"PythonModel"}),", which was specifically designed to make creating GenAI workflows easier."]}),"\n",(0,n.jsx)(t.p,{children:"Briefly, here are some of the benefits of using ChatModel:"}),"\n",(0,n.jsxs)(t.ol,{children:["\n",(0,n.jsx)(t.li,{children:"No need to define a complex signature! Chat models often accept complex inputs with many levels of nesting, and this can be cumbersome to define yourself."}),"\n",(0,n.jsx)(t.li,{children:"Support for JSON / dict inputs (no need to wrap inputs or convert to Pandas DataFrame)"}),"\n",(0,n.jsx)(t.li,{children:"Includes the use of Dataclasses for defining expected inputs / outputs for a simplified development experience"}),"\n"]}),"\n",(0,n.jsxs)(t.p,{children:["For a more in-depth exploration of ChatModel, please check out the ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/llms/chat-model-guide/index.html",children:"detailed guide"}),"."]}),"\n",(0,n.jsx)(t.p,{children:"In this tutorial, we'll be building a simple OpenAI wrapper that makes use of the tool calling support (released in MLflow 2.17.0)."}),"\n",(0,n.jsx)(t.h3,{id:"environment-setup",children:"Environment setup"}),"\n",(0,n.jsx)(t.p,{children:"First, let's set up the environment. We'll need the OpenAI Python SDK, as well as MLflow >= 2.17.0. We'll also need to set our OpenAI API key in order to use the SDK."}),"\n",(0,n.jsx)(i.d,{executionCount:" ",children:"%pip install 'mlflow[genai]>=2.17.0' 'openai>=1.0' -qq"}),"\n",(0,n.jsx)(i.d,{executionCount:1,children:`import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")`}),"\n",(0,n.jsx)(t.h3,{id:"step-1-creating-the-tool-definition",children:"Step 1: Creating the tool definition"}),"\n",(0,n.jsxs)(t.p,{children:["Let's begin to define our model! As mentioned in the introduction, we'll be subclassing ",(0,n.jsx)(t.code,{children:"mlflow.pyfunc.ChatModel"}),". For this example, we'll build a toy model that uses a tool to retrieve the weather for a given city."]}),"\n",(0,n.jsxs)(t.p,{children:["The first step is to create a tool definition that we can pass to OpenAI. We do this by using ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.FunctionToolDefinition",children:"mlflow.types.llm.FunctionToolDefinition"})," to describe the parameters that our tool accepts. The format of this dataclass is aligned with the OpenAI spec:"]}),"\n",(0,n.jsx)(i.d,{executionCount:2,children:`import mlflow
from mlflow.types.llm import (
  FunctionToolDefinition,
  ParamProperty,
  ToolParamsSchema,
)


class WeatherModel(mlflow.pyfunc.ChatModel):
  def __init__(self):
      # a sample tool definition. we use the \`FunctionToolDefinition\`
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
          # make sure to call \`to_tool_definition()\` to convert the \`FunctionToolDefinition\`
          # to a \`ToolDefinition\` object. this step is necessary to normalize the data format,
          # as multiple types of tools (besides just functions) might be available in the future.
      ).to_tool_definition()

      # OpenAI expects tools to be provided as a list of dictionaries
      self.tools = [weather_tool.to_dict()]`}),"\n",(0,n.jsx)(t.h3,{id:"step-2-implementing-the-tool",children:"Step 2: Implementing the tool"}),"\n",(0,n.jsx)(t.p,{children:"Now that we have a definition for the tool, we need to actually implement it. For the purposes of this tutorial, we're just going to mock a response, but the implementation can be arbitrary\u2014you might make an API call to an actual weather service, for example."}),"\n",(0,n.jsx)(i.d,{executionCount:3,children:`class WeatherModel(mlflow.pyfunc.ChatModel):
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
          return f"It's sunny in {city}, with a temperature of 20C"`}),"\n",(0,n.jsxs)(t.h3,{id:"step-3-implementing-the-predict-method",children:["Step 3: Implementing the ",(0,n.jsx)(t.code,{children:"predict"})," method"]}),"\n",(0,n.jsxs)(t.p,{children:["The next thing we need to do is define a ",(0,n.jsx)(t.code,{children:"predict()"})," function that accepts the following arguments:"]}),"\n",(0,n.jsxs)(t.ol,{children:["\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.code,{children:"context"}),": ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModelContext",children:"PythonModelContext"})," (not used in this tutorial)"]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.code,{children:"messages"}),": List[",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.ChatMessage",children:"ChatMessage"}),"]. This is the chat input that the model uses for generation."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.code,{children:"params"}),": ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.ChatParams",children:"ChatParams"}),". These are commonly used params used to configure the chat model, e.g. ",(0,n.jsx)(t.code,{children:"temperature"}),", ",(0,n.jsx)(t.code,{children:"max_tokens"}),", etc. This is where the tool specifications can be found."]}),"\n"]}),"\n",(0,n.jsx)(t.p,{children:"This is the function that will ultimately be called during inference."}),"\n",(0,n.jsxs)(t.p,{children:["For the implementation, we'll simply forward the user's input to OpenAI, and provide the ",(0,n.jsx)(t.code,{children:"get_weather"})," tool as an option for the LLM to use if it chooses to do so. If we receive a tool call request, we'll call the ",(0,n.jsx)(t.code,{children:"get_weather()"})," function and return the response back to OpenAI. We'll need to use what we've defined in the previous two steps in order to do this."]}),"\n",(0,n.jsx)(i.d,{executionCount:4,children:`import json

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
      if tool_calls := response.choices[0].message.tool_calls:
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
      return ChatResponse.from_dict(response.to_dict())`}),"\n",(0,n.jsx)(t.h3,{id:"step-4-optional-but-recommended-enable-tracing-for-the-model",children:"Step 4 (optional, but recommended): Enable tracing for the model"}),"\n",(0,n.jsxs)(t.p,{children:["This step is optional, but highly recommended to improve observability in your app. We'll be using ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/llms/tracing/index.html",children:"MLflow Tracing"})," to log the inputs and outputs of our model's internal functions, so we can easily debug when things go wrong. Agent-style tool calling models can make many layers of function calls during the lifespan of a single request, so tracing is invaluable in helping us understand what's going on at each step."]}),"\n",(0,n.jsxs)(t.p,{children:["Integrating tracing is easy, we simply decorate the functions we're interested in (",(0,n.jsx)(t.code,{children:"get_weather()"})," and ",(0,n.jsx)(t.code,{children:"predict()"}),") with ",(0,n.jsx)(t.code,{children:"@mlflow.trace"}),"! MLflow Tracing also has integrations with many popular GenAI frameworks, such as LangChain, OpenAI, LlamaIndex, and more. For the full list, check out this ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing",children:"documentation page"}),". In this tutorial, we're using the OpenAI SDK to make API calls, so we can enable tracing for this by calling ",(0,n.jsx)(t.code,{children:"mlflow.openai.autolog()"}),"."]}),"\n",(0,n.jsxs)(t.p,{children:["To view the traces in the UI, run ",(0,n.jsx)(t.code,{children:"mlflow server"})," in a separate terminal shell, and navigate to the ",(0,n.jsx)(t.code,{children:"Traces"})," tab after using the model for inference below."]}),"\n",(0,n.jsx)(i.d,{executionCount:5,children:`from mlflow.entities.span import (
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

      if tool_calls := response.choices[0].message.tool_calls:
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

      return ChatResponse.from_dict(response.to_dict())`}),"\n",(0,n.jsx)(t.h3,{id:"step-5-logging-the-model",children:"Step 5: Logging the model"}),"\n",(0,n.jsx)(t.p,{children:"Finally, we need to log the model. This saves the model as an artifact in MLflow Tracking, and allows us to load and serve it later on."}),"\n",(0,n.jsxs)(t.p,{children:["(Note: this is a fundamental pattern in MLflow. To learn more, check out the ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html",children:"Quickstart guide"}),"!)"]}),"\n",(0,n.jsx)(t.p,{children:"In order to do this, we need to do a few things:"}),"\n",(0,n.jsxs)(t.ol,{children:["\n",(0,n.jsx)(t.li,{children:"Define an input example to inform users about the input we expect"}),"\n",(0,n.jsx)(t.li,{children:"Instantiate the model"}),"\n",(0,n.jsxs)(t.li,{children:["Call ",(0,n.jsx)(t.code,{children:"mlflow.pyfunc.log_model()"})," with the above as arguments"]}),"\n"]}),"\n",(0,n.jsx)(t.p,{children:"Take note of the Model URI printed out at the end of the cell\u2014we'll need it when serving the model later!"}),"\n",(0,n.jsx)(i.d,{executionCount:6,children:`# messages to use as input examples
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

  print("Successfully logged the model at the following URI: ", model_info.model_uri)`}),"\n",(0,n.jsx)(a.p,{isStderr:!0,children:"2024/10/29 09:30:14 INFO mlflow.pyfunc: Predicting on input example to validate output"}),"\n",(0,n.jsx)(a.p,{children:"Received a tool call, calling the weather tool..."}),"\n",(0,n.jsx)(a.p,{children:"Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]"}),"\n",(0,n.jsx)(a.p,{children:`Received a tool call, calling the weather tool...
Successfully logged the model at the following URI:  runs:/8051850efa194a3b8b2450c4c9f4d42f/weather-model`}),"\n",(0,n.jsx)(t.h3,{id:"using-the-model-for-inference",children:"Using the model for inference"}),"\n",(0,n.jsxs)(t.p,{children:["Now that the model is logged, our work is more or less done! In order to use the model for inference, let's load it back using ",(0,n.jsx)(t.code,{children:"mlflow.pyfunc.load_model()"}),"."]}),"\n",(0,n.jsx)(i.d,{executionCount:7,children:`import mlflow

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
print(response["choices"][0]["message"]["content"])`}),"\n",(0,n.jsx)(a.p,{isStderr:!0,children:"2024/10/29 09:30:27 WARNING mlflow.tracing.processor.mlflow: Creating a trace within the default experiment with id '0'. It is strongly recommended to not use the default experiment to log traces due to ambiguous search results and probable performance issues over time due to directory table listing performance degradation with high volumes of directories within a specific path. To avoid performance and disambiguation issues, set the experiment for your environment using `mlflow.set_experiment()` API."}),"\n",(0,n.jsx)(a.p,{children:`Received a tool call, calling the weather tool...
The weather in Singapore is sunny, with a temperature of 20\xb0C.
Received a tool call, calling the weather tool...
The weather in San Francisco is sunny, with a temperature of 20\xb0C.`}),"\n",(0,n.jsx)(t.h3,{id:"serving-the-model",children:"Serving the model"}),"\n",(0,n.jsxs)(t.p,{children:["MLflow also allows you to serve models, using the ",(0,n.jsx)(t.code,{children:"mlflow models serve"})," CLI tool. In another terminal shell, run the following from the same folder as this notebook:"]}),"\n",(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-sh",children:"$ export OPENAI_API_KEY=<YOUR OPENAI API KEY>\n$ mlflow models serve -m <MODEL_URI>\n"})}),"\n",(0,n.jsxs)(t.p,{children:["This will start serving the model on ",(0,n.jsx)(t.code,{children:"http://127.0.0.1:5000"}),", and the model can be queried via POST request to the ",(0,n.jsx)(t.code,{children:"/invocations"})," route."]}),"\n",(0,n.jsx)(i.d,{executionCount:8,children:`import requests

messages = [
  system_prompt,
  {"role": "user", "content": "What's the weather in Tokyo?"},
]

response = requests.post("http://127.0.0.1:5000/invocations", json={"messages": messages})
response.raise_for_status()
response.json()`}),"\n",(0,n.jsx)(a.p,{children:`{'choices': [{'index': 0,
 'message': {'role': 'assistant',
  'content': 'The weather in Tokyo is sunny, with a temperature of 20\xb0C.'},
 'finish_reason': 'stop'}],
'usage': {'prompt_tokens': 100, 'completion_tokens': 16, 'total_tokens': 116},
'id': 'chatcmpl-ANVOhWssEiyYNFwrBPxp1gmQvZKsy',
'model': 'gpt-4o-mini-2024-07-18',
'object': 'chat.completion',
'created': 1730165599}`}),"\n",(0,n.jsx)(t.h3,{id:"conclusion",children:"Conclusion"}),"\n",(0,n.jsxs)(t.p,{children:["In this tutorial, we covered how to use MLflow's ",(0,n.jsx)(t.code,{children:"ChatModel"})," class to create a convenient OpenAI wrapper that supports tool calling. Though the use-case was simple, the concepts covered here can be easily extended to support more complex functionality."]}),"\n",(0,n.jsxs)(t.p,{children:["If you're looking to dive deeper into building quality GenAI apps, you might be also be interested in checking out ",(0,n.jsx)(t.a,{href:"https://mlflow.org/docs/latest/llms/tracing/index.html",children:"MLflow Tracing"}),", an observability tool you can use to trace the execution of arbitrary functions (such as your tool calls, for example)."]})]})}function u(e={}){let{wrapper:t}={...(0,s.R)(),...e.components};return t?(0,n.jsx)(t,{...e,children:(0,n.jsx)(p,{...e})}):p(e)}},75453(e,t,o){o.d(t,{p:()=>n});var l=o(74848);let n=({children:e,isStderr:t})=>(0,l.jsx)("pre",{style:{margin:0,borderRadius:0,background:"none",fontSize:"0.85rem",flexGrow:1,padding:"var(--padding-sm)"},children:e})},75940(e,t,o){o.d(t,{d:()=>s});var l=o(74848),n=o(37449);let s=({children:e,executionCount:t})=>(0,l.jsx)("div",{style:{flexGrow:1,minWidth:0,marginTop:"var(--padding-md)",width:"100%"},children:(0,l.jsx)(n.A,{className:"codeBlock_oJcR",language:"python",children:e})})},42676(e,t,o){o.d(t,{O:()=>i});var l=o(74848),n=o(96540);let s="3.9.1.dev0";function i({children:e,href:t}){let o=(0,n.useCallback)(async e=>{if(e.preventDefault(),window.gtag)try{window.gtag("event","notebook-download",{href:t})}catch{}s.includes("dev")||(t=t.replace(/\/master\//,`/v${s}/`));let o=await fetch(t),l=await o.blob(),n=window.URL.createObjectURL(l),i=document.createElement("a");i.style.display="none",i.href=n,i.download=t.split("/").pop(),document.body.appendChild(i),i.click(),window.URL.revokeObjectURL(n),document.body.removeChild(i)},[t]);return(0,l.jsx)("a",{className:"button button--primary",style:{marginBottom:"1rem",display:"block",width:"min-content"},href:t,download:!0,onClick:o,children:e})}},66354(e,t,o){o.d(t,{Q:()=>n});var l=o(74848);let n=({children:e})=>(0,l.jsx)("div",{style:{flexGrow:1,minWidth:0,fontSize:"0.8rem",width:"100%"},children:e})},52915(e,t,o){o.d(t,{A:()=>h});var l=o(74848);o(96540);var n=o(34164),s=o(71643),i=o(66697),a=o(92949),r=o(64560),c=o(47819);function d({language:e}){return(0,l.jsxs)("div",{className:(0,n.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,l.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,l.jsx)(c.A,{})]})}function h({className:e}){let{metadata:t}=(0,s.Ph)(),o=t.language||"text";return(0,l.jsxs)(i.A,{as:"div",className:(0,n.A)(e,t.className),children:[t.title&&(0,l.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,l.jsx)(a.A,{children:t.title})}),(0,l.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,l.jsx)(d,{language:o}),(0,l.jsx)(r.A,{})]})]})}}}]);