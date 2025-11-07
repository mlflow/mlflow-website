# GenAI Evaluation Quickstart

This quickstart guide will walk you through evaluating your GenAI applications with MLflow's comprehensive evaluation framework. In less than 5 minutes, you'll learn how to evaluate LLM outputs, use built-in and custom evaluation criteria, and analyze results in the MLflow UI.

![Simple Evaluation Results](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/quickstart-eval-hero.png)

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

Install the required packages by running the following command:

bash

```
pip install --upgrade mlflow>=3.3 openai
```

info

The code examples in this guide use the OpenAI SDK; however, MLflow's evaluation framework works with any LLM provider, including Anthropic, Google, Bedrock, and more.

## Step 1: Set up your environment[​](#step-1-set-up-your-environment "Direct link to Step 1: Set up your environment")

### Connect to MLflow[​](#connect-to-mlflow "Direct link to Connect to MLflow")

MLflow stores evaluation results in a tracking server. Connect your local environment to the tracking server by one of the following methods.

* Local (pip)
* Local (docker)
* Remote MLflow Server
* Databricks

For the fastest setup, you can install the [mlflow](https://pypi.org/project/mlflow/) Python package and run MLflow locally:

bash

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

This will start the server at port 5000 on your local machine. Connect your notebook/IDE to the server by setting the tracking URI. You can also access to the MLflow UI at <http://localhost:5000>.

python

```
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
```

You can also brows the MLflow UI at <http://localhost:5000>.

MLflow provides a Docker Compose file to start a local MLflow server with a postgres database and a minio server.

bash

```
git clone https://github.com/mlflow/mlflow.git
cd docker-compose
cp .env.dev.example .env
docker compose up -d
```

This will start the server at port 5000 on your local machine. Connect your notebook/IDE to the server by setting the tracking URI. You can also access to the MLflow UI at <http://localhost:5000>.

python

```
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
```

Refer to the [instruction](https://github.com/mlflow/mlflow/tree/master/docker-compose/README.md) for more details, e.g., overriding the default environment variables.

If you have a remote MLflow tracking server, configure the connection:

python

```
import os
import mlflow

# Set your MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://your-mlflow-server:5000"
# Or directly in code
mlflow.set_tracking_uri("http://your-mlflow-server:5000")
```

If you have a Databricks account, configure the connection:

python

```
import mlflow

mlflow.login()
```

This will prompt you for your configuration details (Databricks Host url and a PAT).

tip

If you are unsure about how to set up an MLflow tracking server, you can start with the cloud-based MLflow powered by Databricks: [Sign up for free →](https://login.databricks.com/?destination_url=%2Fml%2Fexperiments-signup%3Fsource%3DTRY_MLFLOW\&dbx_source=TRY_MLFLOW\&signup_experience_step=EXPRESS\&provider=MLFLOW\&utm_source=mlflow_org\&tuuid=a9534f33-78bf-4b81-becc-4334e993251d\&rl_aid=e6685d78-9f85-4fed-b64f-08e247f53547\&intent=SIGN_UP)

### Create a new MLflow Experiment[​](#create-a-new-mlflow-experiment "Direct link to Create a new MLflow Experiment")

python

```
import mlflow

# This will create a new experiment called "GenAI Evaluation Quickstart" and set it as active
mlflow.set_experiment("GenAI Evaluation Quickstart")
```

### Configure OpenAI API Key (or other LLM providers)[​](#configure-openai-api-key-or-other-llm-providers "Direct link to Configure OpenAI API Key (or other LLM providers)")

python

```
import os

# Use different env variable when using a different LLM provider
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your actual API key
```

## Step 2: Create a simple QA function[​](#step-2-create-a-simple-qa-function "Direct link to Step 2: Create a simple QA function")

First, we need to create a prediction function that takes a question and returns an answer. Here we use OpenAI's `gpt-4o-mini` model to generate the answer, but you can use any other LLM provider if you prefer.

python

```
from openai import OpenAI

client = OpenAI()


def qa_predict_fn(question: str) -> str:
    """Simple Q&A prediction function using OpenAI"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
```

## Step 3: Prepare an evaluation dataset[​](#step-3-prepare-an-evaluation-dataset "Direct link to Step 3: Prepare an evaluation dataset")

The evaluation dataset is a list of samples, each with an `inputs` and `expectations` field.

* `inputs`: The input to the `predict_fn` function above. **The key(s) must match the parameter name of the `predict_fn` function**.
* `expectations`: The expected output from the `predict_fn` function, namely, ground truth for the answer.

The dataset can be a list of dictionaries, a pandas DataFrame, a spark DataFrame. Here we use a list of dictionaries for simplicity.

python

```
# Define a simple Q&A dataset with questions and expected answers
eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "expectations": {"expected_response": "Wright Brothers"},
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"},
    },
]
```

## Step 4: Define evaluation criteria using Scorers[​](#step-4-define-evaluation-criteria-using-scorers "Direct link to Step 4: Define evaluation criteria using Scorers")

**Scorer** is a function that computes a score for a given input-output pair against various evaluation criteria. You can use built-in scorers provided by MLflow for common evaluation criteria, as well as create your own custom scorers.

python

```
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines


@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if the answer is concise (less than 5 words)"""
    return len(outputs.split()) <= 5


scorers = [
    Correctness(),
    Guidelines(name="is_english", guidelines="The answer must be in English"),
    is_concise,
]
```

Here we use three scorers:

* [Correctness](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Correctness): Evaluates if the answer is factually correct, using the "expected\_response" field in the dataset.
* [Guidelines](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Guidelines): Evaluates if the answer meets the given guidelines.
* `is_concise`: A custom scorer defined using the [scorer](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.scorer) decorator to judge if the answer is concise (less than 5 words).

The first two scorers use LLMs to evaluate the response, so-called **LLM-as-a-Judge**. This is a powerful technique to assess the quality of the response, because it provides a human-like evaluation for complex language tasks while being more scalable and cost-effective than human evaluation.

The Scorer interface allows you to define various types of quality metrics for your application in a simple way. From a simple natural language guideline to a code function with the full control of the evaluation logic.

tip

The default model used for LLM-as-a-Judge scorers such as Correctness and Guidelines is OpenAI `gpt-4o-mini`. MLflow supports all major LLM providers, such as Anthropic, Bedrock, Google, xAI, and more, through the built-in adopters and LiteLLM.

Example of using different model providers for the judge model

python

```
# Anthropic
Correctness(model="anthropic:/claude-sonnet-4-20250514")

# Bedrock
Correctness(model="bedrock:/anthropic.claude-sonnet-4-20250514")

# Google
# Run `pip install litellm` to use Google as the judge model
Correctness(model="gemini/gemini-2.5-flash")

# xAI
# Run `pip install litellm` to use xAI as the judge model
Correctness(model="xai/grok-2-latest")
```

## Step 5: Run the evaluation[​](#step-5-run-the-evaluation "Direct link to Step 5: Run the evaluation")

Now we have all three components of the evaluation: dataset, prediction function, and scorers. Let's run the evaluation!

python

```
import mlflow

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=qa_predict_fn,
    scorers=scorers,
)
```

After running the code above, go to the MLflow UI and navigate to your experiment. You'll see the evaluation results with detailed metrics for each scorer.

![Detailed Evaluation Results](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/quickstart-eval-result.png)

By clicking on the each row in the table, you can see the detailed rationale behind the score and the trace of the prediction.

![Detailed Evaluation Results](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/quickstart-eval-trace.png)

## Summary[​](#summary "Direct link to Summary")

Congratulations! You've successfully:

* ✅ Set up MLflow GenAI Evaluation for your applications
* ✅ Evaluated a Q\&A application with built-in scorers
* ✅ Created custom evaluation guidelines
* ✅ Learned to analyze results in the MLflow UI

MLflow's evaluation framework provides comprehensive tools for assessing GenAI application quality, helping you build more reliable and effective AI systems.
