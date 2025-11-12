# Evaluating Agents

AI Agents are an emerging pattern of GenAI applications that can use tools, make decisions, and execute multi-step workflows. However, evaluating the performance of those complex agents is challenging. MLflow provides a powerful toolkit to systematically evaluate the agent behavior precisely using traces and scorers.

![Agent Evaluation](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/agent-evaluation-hero.png)

## Workflow[​](#workflow "Direct link to Workflow")

#### Build your agent

Create an AI agent with tools, instructions, and capabilities for your specific use case.

#### Create evaluation dataset

Design test cases with inputs and expectations for both outputs and agent behaviors like tool usage.

#### Define agent-specific scorers

Create scorers that evaluate multi-step agent behaviors using traces.

#### Run evaluation

Execute the evaluation and analyze both final outputs and intermediate agent behaviors in MLflow UI.

## Example: Evaluating a Tool-Calling Agent[​](#example-evaluating-a-tool-calling-agent "Direct link to Example: Evaluating a Tool-Calling Agent")

### Prerequisites[​](#prerequisites "Direct link to Prerequisites")

First, install the required packages by running the following command:

bash

```bash
pip install --upgrade mlflow>=3.3 openai

```

MLflow stores evaluation results in a tracking server. Connect your local environment to the tracking server by one of the following methods.

* Local (pip)
* Local (docker)
* Remote MLflow Server
* Databricks

For the fastest setup, you can install the [mlflow](https://pypi.org/project/mlflow/) Python package and run MLflow locally:

bash

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

```

This will start the server at port 5000 on your local machine. Connect your notebook/IDE to the server by setting the tracking URI. You can also access to the MLflow UI at <http://localhost:5000>.

python

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

```

You can also brows the MLflow UI at <http://localhost:5000>.

MLflow provides a Docker Compose file to start a local MLflow server with a postgres database and a minio server.

bash

```bash
git clone https://github.com/mlflow/mlflow.git
cd docker-compose
cp .env.dev.example .env
docker compose up -d

```

This will start the server at port 5000 on your local machine. Connect your notebook/IDE to the server by setting the tracking URI. You can also access to the MLflow UI at <http://localhost:5000>.

python

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

```

Refer to the [instruction](https://github.com/mlflow/mlflow/tree/master/docker-compose/README.md) for more details, e.g., overriding the default environment variables.

If you have a remote MLflow tracking server, configure the connection:

python

```python
import os
import mlflow

# Set your MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://your-mlflow-server:5000"
# Or directly in code
mlflow.set_tracking_uri("http://your-mlflow-server:5000")

```

If you have a Databricks account, configure the connection:

python

```python
import mlflow

mlflow.login()

```

This will prompt you for your configuration details (Databricks Host url and a PAT).

tip

If you are unsure about how to set up an MLflow tracking server, you can start with the cloud-based MLflow powered by Databricks: [Sign up for free →](https://login.databricks.com/?destination_url=%2Fml%2Fexperiments-signup%3Fsource%3DTRY_MLFLOW\&dbx_source=TRY_MLFLOW\&signup_experience_step=EXPRESS\&provider=MLFLOW\&utm_source=mlflow_org\&tuuid=a9534f33-78bf-4b81-becc-4334e993251d\&rl_aid=e6685d78-9f85-4fed-b64f-08e247f53547\&intent=SIGN_UP)

### Step 1: Build an agent[​](#step-1-build-an-agent "Direct link to Step 1: Build an agent")

Create a math agent that can use tools to answer questions. We use [OpenAI Agents](/mlflow-website/docs/latest/genai/tracing/integrations/listing/openai-agent.md) to build the tool-calling agent in a few lines of code.

python

```python
from agents import Agent, Runner, function_tool


@function_tool
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


@function_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@function_tool
def modular(a: int, b: int) -> int:
    """Modular arithmetic"""
    return a % b


agent = Agent(
    name="Math Agent",
    instructions=(
        "You will be given a math question. Calculate the answer using the given calculator tools. "
        "Return the final number only as an integer."
    ),
    tools=[add, multiply, modular],
)

```

Make sure you can run the agent locally.

python

```python
from agents import Runner

result = await Runner.run(agent, "What is 15% of 240?")
print(result.final_output)
# 36

```

Lastly, let's wrap it in a function that MLflow can call. Note that MLflow runs each prediction in a threadpool, so using a synchronous function does not slow down the evaluation.

python

```python
from openai import OpenAI

# If you are using Jupyter Notebook, you need to apply nest_asyncio.
# import nest_asyncio
# nest_asyncio.apply()


def predict_fn(question: str) -> str:
    return Runner.run_sync(agent, question).final_output

```

### Step 2: Create evaluation dataset[​](#step-2-create-evaluation-dataset "Direct link to Step 2: Create evaluation dataset")

Design test cases as a list of dictionaries, each with an `inputs`, `expectations`, and an optional `tags` field. We would like to evaluate the correctness of the output, but also the tool calls used by the agent.

python

```python
eval_dataset = [
    {
        "inputs": {"task": "What is 15% of 240?"},
        "expectations": {"answer": 36, "tool_calls": ["multiply"]},
        "tags": {"topic": "math"},
    },
    {
        "inputs": {
            "task": "I have 8 cookies and 3 friends. How many more cookies should I buy to share equally?"
        },
        "expectations": {"answer": 1, "tool_calls": ["modular", "add"]},
        "tags": {"topic": "math"},
    },
    {
        "inputs": {
            "task": "I bought 2 shares of stock at $100 each. It's now worth $150. How much profit did I make?"
        },
        "expectations": {"answer": 100, "tool_calls": ["add", "multiply"]},
        "tags": {"topic": "math"},
    },
]

```

### Step 3: Define agent-specific scorers[​](#step-3-define-agent-specific-scorers "Direct link to Step 3: Define agent-specific scorers")

Create scorers that evaluate agent-specific behaviors.

tip

MLflow's scorer can take the **Trace** from the agent execution. Trace is a powerful way to evaluate the agent's behavior precisely, not only the final output. For example, here we use the [`Trace.search_spans`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Trace.search_spans) method to extract the order of tool calls and compare it with the expected tool calls.

For more details, see the [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md) guide.

python

```python
from mlflow.entities import Feedback, SpanType, Trace
from mlflow.genai import scorer


@scorer
def exact_match(outputs, expectations) -> bool:
    return int(outputs) == expectations["answer"]


@scorer
def uses_correct_tools(trace: Trace, expectations: dict) -> Feedback:
    """Evaluate if agent used tools appropriately"""
    expected_tools = expectations["tool_calls"]

    # Parse the trace to get the actual tool calls
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    tool_names = [span.name for span in tool_spans]

    score = "yes" if tool_names == expected_tools else "no"
    rationale = (
        "The agent used the correct tools."
        if tool_names == expected_tools
        else f"The agent used the incorrect tools: {tool_names}"
    )
    # Return a Feedback object with the score and rationale
    return Feedback(value=score, rationale=rationale)

```

### Step 4: Run the evaluation[​](#step-4-run-the-evaluation "Direct link to Step 4: Run the evaluation")

Now we are ready to run the evaluation!

python

```python
results = mlflow.genai.evaluate(
    data=eval_dataset, predict_fn=predict_fn, scorers=[exact_match, uses_correct_tools]
)

```

Once the evaluation is done, open the MLflow UI in your browser and navigate to the experiment page. You should see MLflow creates a new Run and logs the evaluation results.

![Agent Evaluation](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/agent-evaluation-result.png)

It seems the agent does not call tools in the correct order for the second test case. Let's click on the row to **open the trace and inspect what happened under the hood**.

![Agent Evaluation](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/agent-evaluation-trace.png)

By looking at the trace, we can figure out the agent computes the answer in three steps (1) compute 100 \_ 2 (2) compute 150 \_ 2 (3) subtract the two results. However, the more effective way is (1) subtract 100 from 150 (2) multiply the result by 2. In the next version, we can update the system instruction to use tools in a more effective way.

## Configure parallelization[​](#configure-parallelization "Direct link to Configure parallelization")

Running a complex agent can take a long time. MLflow by default uses background threadpool to speed up the evaluation process. You can configure the number of workers to use by setting the `MLFLOW_GENAI_EVAL_MAX_WORKERS` environment variable.

bash

```bash
export MLFLOW_GENAI_EVAL_MAX_WORKERS=10

```

## Evaluating MLflow Models[​](#evaluating-mlflow-models "Direct link to Evaluating MLflow Models")

In MLflow 2.x, you can pass the model URI directly to the `model` argument of the legacy [`mlflow.evaluate()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.evaluate) API (deprecated). The new GenAI evaluation API in MLflow **3.x** still support evaluating MLflow Models, but the workflow is slightly different.

python

```python
import mlflow

# Load the model **outside** the prediction function.
model = mlflow.pyfunc.load_model("models:/math_agent/1")


# Wrap the model in a function that MLflow can call.
def predict_fn(question: str) -> str:
    return model.predict(question)


# Run the evaluation as usual.
mlflow.genai.evaluate(
    data=eval_dataset, predict_fn=predict_fn, scorers=[exact_match, uses_correct_tools]
)

```

## Next steps[​](#next-steps "Direct link to Next steps")

### [Customize Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md)

[Build advanced evaluation criteria and metrics specifically designed for agent behaviors and tool usage patterns.](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md)

[Create custom scorers →](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md)

### [Evaluate Production Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Analyze real agent executions in production environments to understand performance and identify improvement opportunities.](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Analyze traces →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

### [Collect User Feedback](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Gather human feedback on agent performance to create training data and improve evaluation accuracy.](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Start collecting →](/mlflow-website/docs/latest/genai/assessments/feedback.md)
