---
title: Using Bedrock Agent as an MLflow ChatModel with Tracing
description: A guide for using BedRock Runtime Agent with ChatModel and custom trace handling.
slug: bedrock-chat-model-part-1
authors: [jas-bali]
tags: [genai, pyfunc, bedrock, tracing]
thumbnail: /img/blog/bedrock-chatmodel.png
---

![Thumbnail](bedrock-chatmodel.png)

In this blog post, we delve into the integration of AWS Bedrock Agent as a ChatModel within MLflow, focusing on how to
leverage Bedrock's Action Groups and Knowledge Bases to build a conversational AI application. The blog will guide you
through setting up the Bedrock Agent, configuring Action Groups to enable custom actions with Lambda, and utilizing knowledge bases
for context-aware interactions. A special emphasis is placed on implementing tracing within MLflow.
By the end of this article, you'll have a good understanding of how to combine AWS Bedrock's advanced features
with MLflow's capabilities such as agent request tracing, model tracking and consistent signatures for input examples.

## What is AWS Bedrock?

Amazon Bedrock is a managed service by AWS that simplifies the development of generative AI applications.
It provides access to a variety of foundation models (FMs) from leading AI providers through a single API,
enabling developers to build and scale AI solutions securely and efficiently.

Key Components Relevant to This Integration:

**Bedrock Agent**: At a higher level, a bedrock agent is an abstraction within bedrock that consists of a foundation model,
action groups and knowledge bases.

**Action Groups**: These are customizable sets of actions that define what tasks the Bedrock Agent can perform.
Action Groups consist of an OpenAPI Schema and Lambda functions. The OpenAI Schema is used to define APIs available
for the agent to invoke and complete tasks.

**Knowledge Bases**: Amazon Bedrock supports the creation of Knowledge Bases to implement
Retrieval Augmented Generation workflows. It consists of data sources (on S3 or webpages) and a vector store.

Bedrock agent's execution process (and the following trace is grouped as such) consists of:
**Pre-processing**
This step validated, contextualizes and categorizes user input.

**Orchestration**
This step handles interpreting of user inputs, deciding when/what tasks to perform, iteratively refines responses
via observations, rationales and augmented prompts.

**Pre-processing (Optional)**
This step formats the final response before returning to the user.

**Traces**
Each step above has an execution trace, which consists of rationale, actions, queries and observations at each step
of the agent's response. This includes inputs/outputs of action groups and knowledge base queries.

We will look at these traces in detail below.

## What is a ChatModel in MLflow?

The ChatModel class is specifically designed to make it easier to implement models that are compatible with
popular large language model (LLM) chat APIs. It enables you to seamlessly bring in your own models or agents and
leverage MLflow's functionality, even if those models aren't natively supported as a flavor in MLflow. Additionally,
it automatically defines input and output signatures based on an example input.

In the following sections, we will use ChatModel to wrap the Bedrock Agent.

For more detailed information about ChatModel, you can read the MLflow documentation
[here](https://mlflow.org/docs/latest/llms/chat-model-guide/index.html) and
[here](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatModel)

## Setting up AWS Bedrock Agent with an Action group

In this section, we will deploy all components of a bedrock agent so that we can invoke it as a `ChatModel` in MLflow.

### Prerequisites

You will need to setup following items (either via the AWS console or SDKs):

- Setting up role for the agent and Lambda function. [Example](https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/bedrock-agent/scenario_get_started_with_agents.py#L148)
- Create/deploy the agent. [Example](https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/bedrock-agent/scenario_get_started_with_agents.py#L191)
  - **Important**: Save the agent ID here as we will need this below.
- Creating a Lambda function. [Example](https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/bedrock-agent/scenario_get_started_with_agents.py#L218)
- Configuring IAM permissions for agent-Lambda interaction. [Example](https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/bedrock-agent/scenario_get_started_with_agents.py#L283) and [Example](https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/bedrock-agent/scenario_get_started_with_agents.py#L297)
- Creating an action group to link the agent and Lambda. [Example](https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/bedrock-agent/scenario_get_started_with_agents.py#L312)
  - **Important**:Save the agent alias ID here as we will need this below.
- Deploy Bedrock agent with an alias. [Example](https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/bedrock-agent/scenario_get_started_with_agents.py#L342)

In our case, we are going to deploy a following example action group, which calculates the specific period of time
when it's most efficient(duration, energy efficiency etc) to launch a spacecraft from Earth to Mars.

As described above, here is the OpenAPI Schema for our action group:

```yaml
openapi: 3.0.0
info:
  title: Mars launch window API
  version: 1.0.0
  description: API to get the Mars launch window for Mission planning.
paths:
  /get-next-mars-launch-window:
    get:
      summary: Gets the next optimal launch window to Mars based on Hohmann transfer period.
      description: Gets the next optimal launch window to Mars based on the rbital period difference in days for the Hohmann transfer to Mars.
      operationId: getNextMarsLaunchWindow
      responses:
        "200":
          description: Gets the next optimal launch window to Mars.
          content:
            "application/json":
              schema:
                type: object
                properties:
                  next_launch_window:
                    type: string
                    description: date of the next optimal launch window to Mars
```

and here is the code deployment for action group's Lambda:

```python
from datetime import datetime, timedelta
import json


def lambda_handler(event, context):
    def _next_mars_launch_window():
        current_date = datetime.now()
        # Orbital period difference in days for the Hohmann transfer to Mars (~780 days)
        hohmann_transfer_period = 780
        # Last known optimal launch window to Mars (let's use a past known window for calculation)
        # For example, let's use the successful Mars launch window on July 30, 2020
        last_optimal_window = datetime(2020, 7, 30)
        # Calculate the number of days since the last optimal window
        days_since_last_window = (current_date - last_optimal_window).days
        # Calculate the days until the next launch window
        days_until_next_window = hohmann_transfer_period - (days_since_last_window % hohmann_transfer_period)
        # Calculate the date of the next optimal launch window
        next_window_date = current_date + timedelta(days=days_until_next_window)
        return next_window_date.strftime("%Y-%m-%d")

    response = {"next_launch_window": _next_mars_launch_window()}

    response_body = {"application/json": {"body": json.dumps(response)}}

    action_response = {
        "actionGroup": event["actionGroup"],
        "apiPath": event["apiPath"],
        "httpMethod": event["httpMethod"],
        "httpStatusCode": 200,
        "responseBody": response_body,
    }

    session_attributes = event["sessionAttributes"]
    prompt_session_attributes = event["promptSessionAttributes"]

    return {
        "messageVersion": "1.0",
        "response": action_response,
        "sessionAttributes": session_attributes,
        "promptSessionAttributes": prompt_session_attributes,
    }
```

Next, we are going to wrap Bedrock agent as a ChatModel so that we can register and load it for inference.

## Writing ChatModel for Bedrock agent

Here is the virtual env used for running the following example locally (click on details):

<details>

<summary>Click here to expand for the venv requirements for the following example</summary>
**Python 3.12.7**
```text
alembic==1.13.3
aniso8601==9.0.1
blinker==1.8.2
boto3==1.35.31
botocore==1.35.31
cachetools==5.5.0
certifi==2024.8.30
charset-normalizer==3.3.2
click==8.1.7
cloudpickle==3.0.0
contourpy==1.3.0
cycler==0.12.1
databricks-sdk==0.33.0
Deprecated==1.2.14
docker==7.1.0
Flask==3.0.3
fonttools==4.54.1
gitdb==4.0.11
GitPython==3.1.43
google-auth==2.35.0
graphene==3.3
graphql-core==3.2.4
graphql-relay==3.2.0
gunicorn==23.0.0
idna==3.10
importlib_metadata==8.4.0
itsdangerous==2.2.0
Jinja2==3.1.4
jmespath==1.0.1
joblib==1.4.2
kiwisolver==1.4.7
Mako==1.3.5
Markdown==3.7
MarkupSafe==2.1.5
matplotlib==3.9.2
mlflow==2.16.2
mlflow-skinny==2.16.2
numpy==2.1.1
opentelemetry-api==1.27.0
opentelemetry-sdk==1.27.0
opentelemetry-semantic-conventions==0.48b0
packaging==24.1
pandas==2.2.3
pillow==10.4.0
protobuf==5.28.2
pyarrow==17.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.1
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2024.2
PyYAML==6.0.2
requests==2.32.3
rsa==4.9
s3transfer==0.10.2
scikit-learn==1.5.2
scipy==1.14.1
six==1.16.0
smmap==5.0.1
SQLAlchemy==2.0.35
sqlparse==0.5.1
threadpoolctl==3.5.0
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
Werkzeug==3.0.4
wrapt==1.16.0
zipp==3.20.2
```
</details>

### Implementing Bedrock Agent as an MLflow ChatModel with Tracing

```python
import copy
import os
import uuid
from typing import List, Optional

import boto3
import mlflow
from botocore.config import Config
from mlflow.entities import SpanType
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatResponse, ChatMessage, ChatParams, ChatChoice


class BedrockModel(ChatModel):
    def __init__(self):
        """
        Initializes the BedrockModel instance with placeholder values.

        Note:
            The `load_context` method cannot create new instance variables; it can only modify existing ones.
            Therefore, all instance variables should be defined in the `__init__` method with placeholder values.
        """
        self.brt = None
        self._main_bedrock_agent = None
        self._bedrock_agent_id = None
        self._bedrock_agent_alias_id = None
        self._inference_configuration = None
        self._agent_instruction = None
        self._model = None
        self._aws_region = None

    def __getstate__(self):
        """
        Prepares the instance state for pickling.

        This method is needed because the `boto3` client (`self.brt`) cannot be pickled.
        By excluding `self.brt` from the state, we ensure that the model can be serialized and deserialized properly.
        """
        # Create a dictionary of the instance's state, excluding the boto3 client
        state = self.__dict__.copy()
        del state["brt"]
        return state

    def __setstate__(self, state):
        """
        Restores the instance state during unpickling.

        This method is needed to reinitialize the `boto3` client (`self.brt`) after the instance is unpickled,
        because the client was excluded during pickling.
        """
        self.__dict__.update(state)
        self.brt = None

    def load_context(self, context):
        """
        Initializes the Bedrock client with AWS credentials.

        Args:
            context: The MLflow context containing model configuration.

        Note:
            Dependent secret variables must be in the execution environment prior to loading the model;
            else they will not be available during model initialization.
        """
        self._main_bedrock_agent = context.model_config.get("agents", {}).get(
            "main", {}
        )
        self._bedrock_agent_id = self._main_bedrock_agent.get("bedrock_agent_id")
        self._bedrock_agent_alias_id = self._main_bedrock_agent.get(
            "bedrock_agent_alias_id"
        )
        self._inference_configuration = self._main_bedrock_agent.get(
            "inference_configuration"
        )
        self._agent_instruction = self._main_bedrock_agent.get("instruction")
        self._model = self._main_bedrock_agent.get("model")
        self._aws_region = self._main_bedrock_agent.get("aws_region")

        # Initialize the Bedrock client
        self.brt = boto3.client(
            service_name="bedrock-agent-runtime",
            config=Config(region_name=self._aws_region),
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["AWS_SESSION_TOKEN"],
            region_name=self._aws_region,
        )

    @staticmethod
    def _extract_trace_groups(events):
        """
        Extracts trace groups from a list of events based on their trace IDs.

        Args:
            events (list): A list of event dictionaries.

        Returns:
            dict: A dictionary where keys are trace IDs and values are lists of trace items.
        """
        from collections import defaultdict

        trace_groups = defaultdict(list)

        def find_trace_ids(obj, depth=0, parent_key=None):
            if depth > 5:
                return  # Stop recursion after 5 levels if no traceId has been found
            if isinstance(obj, dict):
                trace_id = obj.get("traceId")
                if trace_id:
                    # Include the parent key as the 'type'
                    item = {"type": parent_key, "data": obj}
                    trace_groups[trace_id].append(item)
                else:
                    for key, value in obj.items():
                        find_trace_ids(value, depth=depth + 1, parent_key=key)
            elif isinstance(obj, list):
                for item in obj:
                    find_trace_ids(item, depth=depth + 1, parent_key=parent_key)

        find_trace_ids(events)
        return dict(trace_groups)

    @staticmethod
    def _get_final_response_with_trace(trace_id_groups: dict[str, list[dict]]):
        """
        Processes trace groups to extract the final response and create relevant MLflow spans.

        Args:
            trace_id_groups (dict): A dictionary of trace groups keyed by trace IDs.

        Returns:
            str: The final response text extracted from the trace groups.
        """
        trace_id_groups_copy = copy.deepcopy(trace_id_groups)
        model_invocation_input_key = "modelInvocationInput"

        def _create_trace_by_type(trace_name, _trace_id, context_input):
            @mlflow.trace(
                name=trace_name,
                attributes={"trace_attributes": trace_id_groups_copy[_trace_id]},
            )
            def _trace_agent_pre_context(inner_input_trace):
                return str(trace_id_groups_copy[_trace_id])

            trace_id_groups_copy[_trace_id].remove(context_input)
            _trace_agent_pre_context(context_input.get("data", {}).get("text"))

        def _extract_action_group_trace(
            _trace_id, trace_group, action_group_invocation_input: dict
        ):
            @mlflow.trace(
                name="action-group-invocation",
                attributes={"trace_attributes": trace_id_groups_copy[_trace_id]},
            )
            def _action_group_trace(inner_trace_group):
                for _trace in trace_group:
                    action_group_invocation_output = _trace.get("data", {}).get(
                        "actionGroupInvocationOutput"
                    )
                    if action_group_invocation_output is not None:
                        return str(
                            {
                                "action_group_name": action_group_invocation_input.get(
                                    "actionGroupName"
                                ),
                                "api_path": action_group_invocation_input.get(
                                    "apiPath"
                                ),
                                "execution_type": action_group_invocation_input.get(
                                    "executionType"
                                ),
                                "execution_output": action_group_invocation_output.get(
                                    "text"
                                ),
                            }
                        )

            _action_group_trace(str(action_group_invocation_input))

        def _extract_knowledge_base_trace(
            _trace_id, trace_group, knowledge_base_lookup_input
        ):
            @mlflow.trace(
                name="knowledge-base-lookup",
                attributes={"trace_attributes": trace_id_groups_copy[_trace_id]},
            )
            def _knowledge_base_trace(inner_trace_group):
                for _trace in trace_group:
                    knowledge_base_lookup_output = _trace.get("data", {}).get(
                        "knowledgeBaseLookupOutput"
                    )
                    if knowledge_base_lookup_output is not None:
                        return str(
                            {
                                "knowledge_base_id": knowledge_base_lookup_input.get(
                                    "knowledgeBaseId"
                                ),
                                "text": knowledge_base_lookup_input.get("text"),
                                "retrieved_references": knowledge_base_lookup_output.get(
                                    "retrievedReferences"
                                ),
                            }
                        )

            _knowledge_base_trace(str(trace_group))

        def _find_trace_group_type(_trace_id, trace_group):
            trace_name = "observation"
            pre_processing_trace_id_suffix = "-pre"
            if pre_processing_trace_id_suffix in _trace_id:
                trace_name = "agent-initial-context"
            else:
                for _trace in trace_group:
                    action_group_invocation_input = _trace.get("data", {}).get(
                        "actionGroupInvocationInput"
                    )
                    if action_group_invocation_input is not None:
                        action_group_name = action_group_invocation_input.get(
                            "actionGroupName"
                        )
                        trace_name = f"ACTION-GROUP-{action_group_name}"
                        _extract_action_group_trace(
                            _trace_id, trace_group, action_group_invocation_input
                        )
                        break
                    knowledge_base_lookup_input = _trace.get("data", {}).get(
                        "knowledgeBaseLookupInput"
                    )
                    if knowledge_base_lookup_input is not None:
                        knowledge_base_id = knowledge_base_lookup_input.get(
                            "knowledgeBaseId"
                        )
                        trace_name = f"KNOWLEDGE_BASE_{knowledge_base_id}"
                        _extract_knowledge_base_trace(
                            _trace_id, trace_group, knowledge_base_lookup_input
                        )
            return trace_name

        final_response = ""
        for _trace_id, _trace_group in trace_id_groups_copy.items():
            for _trace in _trace_group:
                if model_invocation_input_key == _trace.get("type", ""):
                    trace_name = _find_trace_group_type(_trace_id, _trace_group)
                    _create_trace_by_type(trace_name, _trace_id, _trace)
                final_response = (
                    _trace.get("data", {}).get("finalResponse", {}).get("text", "")
                )
        return final_response

    @mlflow.trace(name="Bedrock Input Prompt")
    def _get_agent_prompt(self, raw_input_question):
        """
        Constructs the agent prompt by combining the input question and the agent instruction.

        Args:
            raw_input_question (str): The user's input question.

        Returns:
            str: The formatted agent prompt.
        """
        return f"""
        Answer the following question and pay strong attention to the prompt:
        <question>
        {raw_input_question}
        </question>
        <instruction>
        {self._agent_instruction}
        </instruction>
        """

    @mlflow.trace(name="bedrock-agent", span_type=SpanType.CHAT_MODEL)
    def predict(
        self, context, messages: List[ChatMessage], params: Optional[ChatParams]
    ) -> ChatResponse:
        """
        Makes a prediction using the Bedrock agent and processes the response.

        Args:
            context: The MLflow context.
            messages (List[ChatMessage]): A list of chat messages.
            params (Optional[ChatParams]): Optional parameters for the chat.

        Returns:
            ChatResponse: The response from the Bedrock agent.
        """
        formatted_input = messages[-1].content
        session_id = uuid.uuid4().hex

        response = self.brt.invoke_agent(
            agentId=self._bedrock_agent_id,
            agentAliasId=self._bedrock_agent_alias_id,
            inputText=self._get_agent_prompt(formatted_input),
            enableTrace=True,
            sessionId=session_id,
            endSession=False,
        )

        # Since this provider's output doesn't match the OpenAI specification,
        # we need to go through the returned trace data and map it appropriately
        # to create the MLflow span object.
        events = list(response.get("completion", []))
        trace_id_groups = self._extract_trace_groups(events)
        final_response = self._get_final_response_with_trace(trace_id_groups)
        with mlflow.start_span(
            name="retrieved-response", span_type=SpanType.AGENT
        ) as span:
            span.set_inputs(messages)
            span.set_attributes({})

            output = ChatResponse(
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="user", content=final_response),
                    )
                ],
                usage={},
                model=self._model,
            )

            span.set_outputs(output)

        return output
```

Here are some important remarks about this `BedrockModel` implementation:

- AWS access key ID, secret key and the session token are externalized here. These need to be present in the environment before we can run inference.
  You will need to generate it for your IAM user and set them as environment variables.

```bash
aws sts get-session-token --duration-seconds 3600
```

And then set the following:

```python
import os

os.environ['AWS_ACCESS_KEY'] = "<AccessKeyId>"
os.environ['AWS_SECRET_ACCESS_KEY'] = "<SecretAccessKey>"
os.environ['AWS_SESSION_TOKEN'] = "<SessionToken>"

```

As noticed in the code above, these do not get logged with the model and are only set inside `load_context`.
This method is called when ChatModel is constructed. Further details are [here](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel.load_context)

- Bedrock agent ID and agent alias ID are passed via `model_config` that we will use below.

- boto3 module has been excluded from getting pickled. This is done via `__getstate__` and `__setstate__` where we exclude it and reset it respectively

### Log and load the BedrockModel

```python
import mlflow
import os
import logging
from mlflow.models import infer_signature

input_example = [{
    "messages": [
        {
            "role": "user",
            "content": "what is the next launch window for Mars?",
        }
    ]
}]

output_example = {
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "test content"},
        }
    ]
}
signature = infer_signature(example, output_example)

with mlflow.start_run():

    model_config = {
        "agents": {
            "main": {
                "model": "anthropic.claude-v2",
                "aws_region": "us-east-1",
                "bedrock_agent_id": "LQDMKZPELG",
                "bedrock_agent_alias_id": "3A6N13GCMY",
                "instruction": (
                    "You have functions available at your disposal to use when anwering any questions about orbital mechanics."
                    "if you can't find a function to answer a question about orbital mechanics, simply reply "
                    "'I do not know'"
                ),
                "inference_configuration": {
                    "temperature": 0.5,
                    "maximumLength": 2000,
                }
            },
        },
    }

    logged_chain_info = mlflow.pyfunc.log_model(
        python_model=BedrockModel(),
        model_config=model_config,
        artifact_path="chain",  # This string is used as the path inside the MLflow model where artifacts are stored
        input_example=input_example,  # Must be a valid input to your chain
    )

loaded = mlflow.pyfunc.load_model(logged_chain_info.model_uri)

response = loaded.predict(
    {
        "messages": [
            {
                "role": "user",
                "content": "what is the next launch window for Mars?",
            }
        ]
    }
)
```

```text

```

### Mapping Bedrock Agent Trace Data to MLflow Span Objects

In this step, we need to iterate over the data that is returned within the bedrock agent's response trace
to provide relevant mappings to create the MLflow span object.
AWS Bedrock agent's response is a flat list with trace events connected by `traceId`.
Here is the raw trace sent in the bedrock agent's response:

<details>
<summary>Expand to see AWS Bedrock agent's raw trace</summary>
```text
[
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'preProcessingTrace': {
          'modelInvocationInput': {
            'inferenceConfiguration': {
              ...
            },
            'text': '\n\nHuman: You are a classifying agent that filters user inputs into categories. Your job is to sort these inputs before they...<thinking> XML tags before providing only the category letter to sort the input into within <category> XML tags.\n\nAssistant:',
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-pre-0',
            'type': 'PRE_PROCESSING'
          }
        }
      }
    }
  },
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'preProcessingTrace': {
          'modelInvocationOutput': {
            'parsedResponse': {
              ...
            },
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-pre-0'
          }
        }
      }
    }
  },
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'orchestrationTrace': {
          'modelInvocationInput': {
            'inferenceConfiguration': {
              ...
            },
            'text': '\n\nHuman:\nYou are a research assistant AI that has been equipped with one or more functions to help you answer a <question>...\n\nAssistant: <scratchpad> I understand I cannot use functions that have not been provided to me to answer this question.\n\n',
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0',
            'type': 'ORCHESTRATION'
          }
        }
      }
    }
  },
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'orchestrationTrace': {
          'modelInvocationOutput': {
            'metadata': {
              ...
            },
            'rawResponse': {
              ...
            },
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0'
          }
        }
      }
    }
  },
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'orchestrationTrace': {
          'rationale': {
            'text': 'To answer this question, I will:\n\n1. Call the GET::current_date_and_time::getNextMarsLaunchWindow function to get the next ...unch window to Mars.\n\nI have checked that I have access to the GET::current_date_and_time::getNextMarsLaunchWindow function.',
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0'
          }
        }
      }
    }
  },
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'orchestrationTrace': {
          'invocationInput': {
            'actionGroupInvocationInput': {
              ...
            },
            'invocationType': 'ACTION_GROUP',
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0'
          }
        }
      }
    }
  },
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'orchestrationTrace': {
          'observation': {
            'actionGroupInvocationOutput': {
              ...
            },
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0',
            'type': 'ACTION_GROUP'
          }
        }
      }
    }
  },
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'orchestrationTrace': {
          'modelInvocationInput': {
            'inferenceConfiguration': {
              ...
            },
            'text': '\n\nHuman:\nYou are a research assistant AI that has been equipped with one or more functions to help you answer a <question>...and_time::getNextMarsLaunchWindow()</function_call>\n<function_result>{"next_launch_window": "2026-12-26"}</function_result>\n',
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-1',
            'type': 'ORCHESTRATION'
          }
        }
      }
    }
  },
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'orchestrationTrace': {
          'modelInvocationOutput': {
            'metadata': {
              ...
            },
            'rawResponse': {
              ...
            },
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-1'
          }
        }
      }
    }
  },
  {
    'trace': {
      'agentAliasId': '3A6N13GCMY',
      'agentId': 'LQDMKZPELG',
      'agentVersion': '1',
      'sessionId': '8a888330158d432f9bf90633c378e095',
      'trace': {
        'orchestrationTrace': {
          'observation': {
            'finalResponse': {
              ...
            },
            'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-1',
            'type': 'FINISH'
          }
        }
      }
    }
  },
  {
    'chunk': {
      'bytes': b
      'The next optimal launch window to Mars is 2026-12-26 UTC.'
    }
  }
]
```
</details>

To fit this structure into MLflow's span, we first need to go through the raw response trace and group events by their `traceId`.
After grouping the trace events by _`traceId`_, the structure looks like this:

<details>
<summary>Expand to see trace grouped by _`traceId`_</summary>
```text
{
  '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0': [
    {
      'data': {
        'inferenceConfiguration': {
          'maximumLength': 2048,
          'stopSequences': [
            '</function_call>',
            '</answer>',
            '</error>'
          ],
          'temperature': 0.0,
          'topK': 250,
          'topP': 1.0
        },
        'text': '\n\nHuman:\nYou are a research assistant AI that has been equipped with one or more functions to help you answer a <question>...\n\nAssistant: <scratchpad> I understand I cannot use functions that have not been provided to me to answer this question.\n\n',
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0',
        'type': 'ORCHESTRATION'
      },
      'type': 'modelInvocationInput'
    },
    {
      'data': {
        'metadata': {
          'usage': {
            'inputTokens': 5040,
            'outputTokens': 101
          }
        },
        'rawResponse': {
          'content': 'To answer this question, I will:\n\n1. Call the GET::current_date_and_time::getNextMarsLaunchWindow function to get the next ...::getNextMarsLaunchWindow function.\n\n</scratchpad>\n\n<function_call>\nGET::current_date_and_time::getNextMarsLaunchWindow()'
        },
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0'
      },
      'type': 'modelInvocationOutput'
    },
    {
      'data': {
        'text': 'To answer this question, I will:\n\n1. Call the GET::current_date_and_time::getNextMarsLaunchWindow function to get the next ...unch window to Mars.\n\nI have checked that I have access to the GET::current_date_and_time::getNextMarsLaunchWindow function.',
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0'
      },
      'type': 'rationale'
    },
    {
      'data': {
        'actionGroupInvocationInput': {
          'actionGroupName': 'current_date_and_time',
          'apiPath': '/get-next-mars-launch-window',
          'executionType': 'LAMBDA',
          'verb': 'get'
        },
        'invocationType': 'ACTION_GROUP',
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0'
      },
      'type': 'invocationInput'
    },
    {
      'data': {
        'actionGroupInvocationOutput': {
          'text': '{"next_launch_window": "2026-12-26"}'
        },
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-0',
        'type': 'ACTION_GROUP'
      },
      'type': 'observation'
    }
  ],
  '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-1': [
    {
      'data': {
        'inferenceConfiguration': {
          'maximumLength': 2048,
          'stopSequences': [
            '</function_call>',
            '</answer>',
            '</error>'
          ],
          'temperature': 0.0,
          'topK': 250,
          'topP': 1.0
        },
        'text': '\n\nHuman:\nYou are a research assistant AI that has been equipped with one or more functions to help you answer a <question>...and_time::getNextMarsLaunchWindow()</function_call>\n<function_result>{"next_launch_window": "2026-12-26"}</function_result>\n',
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-1',
        'type': 'ORCHESTRATION'
      },
      'type': 'modelInvocationInput'
    },
    {
      'data': {
        'metadata': {
          'usage': {
            'inputTokens': 5164,
            'outputTokens': 25
          }
        },
        'rawResponse': {
          'content': '<answer>\nThe next optimal launch window to Mars is 2026-12-26 UTC.'
        },
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-1'
      },
      'type': 'modelInvocationOutput'
    },
    {
      'data': {
        'finalResponse': {
          'text': 'The next optimal launch window to Mars is 2026-12-26 UTC.'
        },
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-1',
        'type': 'FINISH'
      },
      'type': 'observation'
    }
  ],
  '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-pre-0': [
    {
      'data': {
        'inferenceConfiguration': {
          'maximumLength': 2048,
          'stopSequences': [
            '\n\nHuman:'
          ],
          'temperature': 0.0,
          'topK': 250,
          'topP': 1.0
        },
        'text': '\n\nHuman: You are a classifying agent that filters user inputs into categories. Your job is to sort these inputs before they...<thinking> XML tags before providing only the category letter to sort the input into within <category> XML tags.\n\nAssistant:',
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-pre-0',
        'type': 'PRE_PROCESSING'
      },
      'type': 'modelInvocationInput'
    },
    {
      'data': {
        'parsedResponse': {
          'isValid': True,
          'rationale': "The user's input is asking what the next launch window for Mars is. Based on the provided functions, there is a function call...ut falls into Category D, as it is a question that can be answered by the function calling agent using the provided functions."
        },
        'traceId': '8471abc8-88fd-4880-b21e-7e0cc4cf73ea-pre-0'
      },
      'type': 'modelInvocationOutput'
    }
  ]
}
```
</details>

Each group of events with the same _`traceId`_ will contain at least two events: one of type _`modelInvocationInput`_ and
one of type _`modelInvocationOutput`_. Groups that involve action group traces will also include events of type
_`actionGroupInvocationInput`_ and _`actionGroupInvocationOutput`_. Similarly, groups that use knowledge bases will have
additional events of type _`knowledgeBaseLookupInput`_ and _`knowledgeBaseLookupOutput`_.
In the BedrockModel mentioned above, it implements an approach to parse these event groups into trace nodes.
This method allows the trace to display the reasoning behind selecting action groups/knowledge bases to answer queries and invoking
the corresponding Lambda function calls, as defined in out example OpenAPI spec above.
This structure helps to clearly show the flow of information and decision-making process that bedrock agent follows.

## Conclusion

In this blog, we explored how to integrate the AWS Bedrock Agent as an MLflow ChatModel, focusing on Action Groups,
Knowledge Bases, and Tracing. We demonstrated how to easily build a custom ChatModel using MLflow's flexible and
powerful APIs. This approach enables you to leverage MLflow's tracing and logging capabilities, even for models or
flavors that are not natively supported by MLflow.

Key Takeaways from This Blog:

- Deploying a Bedrock Agent with Action Groups as AWS Lambda Functions:
  - We covered how to set up a Bedrock Agent and implement custom actions using AWS Lambda functions within Action Groups.
- Mapping the AWS Bedrock Agent's Custom Tracing to MLflow span/trace objects:
  - We demonstrated how to convert the agent's custom tracing data into MLflow span objects for better observability.
- Logging and Loading the Bedrock Agent as an MLflow ChatModel:
  - We showed how to log the Bedrock Agent into MLflow as a _`ChatModel`_ and how to load it for future use.
- Externalizing AWS Client and Bedrock Configurations:
  - We explained how to externalize AWS client and Bedrock configurations to safeguard secrets and make it easy to adjust model settings without the need to re-log the model.

## Further Reading and References

- [How Amazon Bedrock Agents work](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-how.html)
- [Amazon Bedrock Tracing](https://docs.aws.amazon.com/bedrock/latest/userguide/trace-events.html)
- [Creating a Custom GenAI chat agent](https://mlflow.org/docs/latest/llms/chat-model-guide/index.html)
- [AWS Code Examples Repository](https://github.com/awsdocs/aws-doc-sdk-examples)
