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
pip install --upgrade 'mlflow[genai]>=3.3' openai

```

MLflow stores evaluation results in a tracking server. Connect your local environment to the tracking server by one of the following methods.

* Local (uv)
* Local (pip)
* Local (docker)

Install the Python package manager [uv](https://docs.astral.sh/uv/getting-started/installation/) (that will also install [`uvx` command](https://docs.astral.sh/uv/guides/tools/) to invoke Python tools without installing them).

Start a MLflow server locally.

shell

```shell
uvx mlflow server

```

**Python Environment**: Python 3.10+

Install the `mlflow` Python package via `pip` and start a MLflow server locally.

shell

```shell
pip install --upgrade 'mlflow[genai]'
mlflow server

```

MLflow provides a Docker Compose file to start a local MLflow server with a PostgreSQL database and a MinIO server.

shell

```shell
git clone --depth 1 --filter=blob:none --sparse https://github.com/mlflow/mlflow.git
cd mlflow
git sparse-checkout set docker-compose
cd docker-compose
cp .env.dev.example .env
docker compose up -d

```

Refer to the [instruction](https://github.com/mlflow/mlflow/tree/master/docker-compose/README.md) for more details (e.g., overriding the default environment variables).

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

tip

**Async Functions are Supported!**

If your agent library provides an async API, you can use it directly without converting to sync. MLflow automatically detects and handles async functions:

python

```python
async def predict_fn(question: str) -> str:
    result = await Runner.run(agent, question)
    return result.final_output


mlflow.genai.evaluate(
    data=eval_dataset, predict_fn=predict_fn, scorers=[exact_match, uses_correct_tools]
)

```

By default, async functions have a 5-minute timeout. Configure this using the `MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT` environment variable:

bash

```bash
export MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT=600  # 10 minutes

```

### Step 2: Create evaluation dataset[​](#step-2-create-evaluation-dataset "Direct link to Step 2: Create evaluation dataset")

Design test cases as a list of dictionaries, each with an `inputs`, `expectations`, and an optional `tags` field.

python

```python
eval_dataset = [
    {
        "inputs": {"task": "What is 15% of 240?"},
        "expectations": {"answer": 36},
        "tags": {"topic": "math"},
    },
    {
        "inputs": {
            "task": "I have 8 cookies and 3 friends. How many more cookies should I buy to share equally?"
        },
        "expectations": {"answer": 1},
        "tags": {"topic": "math"},
    },
    {
        "inputs": {
            "task": "I bought 2 shares of stock at $100 each. It's now worth $150. How much profit did I make?"
        },
        "expectations": {"answer": 100},
        "tags": {"topic": "math"},
    },
]

```

### Step 3: Define scorers[​](#step-3-define-scorers "Direct link to Step 3: Define scorers")

We'll define a custom scorer to evaluate the correctness of the final output.

python

```python
from mlflow.genai import scorer


@scorer
def exact_match(outputs, expectations) -> bool:
    return int(outputs) == expectations["answer"]

```

tip

MLflow provides built-in scorers for evaluating agent tool usage:

* [ToolCallCorrectness](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.ToolCallCorrectness): Evaluates if tool calls and arguments are correct for the user query
* [ToolCallEfficiency](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.ToolCallEfficiency): Evaluates if tool calls are efficient without redundancy

These scorers automatically analyze traces to assess tool usage patterns. See the [Built-in Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md) guide for more details.

For custom evaluation patterns, you can create your own scorers that parse traces. See the [Custom Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md) guide.

### Step 4: Run the evaluation[​](#step-4-run-the-evaluation "Direct link to Step 4: Run the evaluation")

Now we are ready to run the evaluation!

python

```python
from mlflow.genai.scorers import ToolCallCorrectness, ToolCallEfficiency

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[
        exact_match,
        ToolCallCorrectness(),
        ToolCallEfficiency(),
    ],
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
