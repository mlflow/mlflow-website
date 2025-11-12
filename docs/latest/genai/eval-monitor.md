# Evaluating LLMs/Agents with MLflow

Modern GenAI Evaluation

This documentation covers MLflow's **GenAI evaluation system** which uses:

* `mlflow.genai.evaluate()` for evaluation
* `Scorer` objects for metrics
* Built-in and custom LLM judges

**Note**: This system is separate from the [classic ML evaluation](/mlflow-website/docs/latest/ml/evaluation.md) system that uses `mlflow.evaluate()` and `EvaluationMetric`. The two systems serve different purposes and are not interoperable.

MLflow's evaluation and monitoring capabilities help you systematically measure, improve, and maintain the quality of your GenAI applications throughout their lifecycle from development through production.

![Prompt Evaluation](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/evaluation-result-video.gif)

A core tenet of MLflow's evaluation capabilities is **Evaluation-Driven Development**. This is an emerging practice to tackle the challenge of building high-quality LLM/Agentic applications. MLflow is an **end-to-end** platform that is designed to support this practice and help you deploy AI applications with confidence.

![Evaluation Driven Development](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/evaluation-driven-development.png)

## Key Capabilities[​](#key-capabilities "Direct link to Key Capabilities")

* Dataset Management
* Human Feedback
* LLM-as-a-Judge
* Systematic Evaluation
* Production Monitoring

#### Create and maintain a High-Quality Dataset[​](#create-and-maintain-a-high-quality-dataset "Direct link to Create and maintain a High-Quality Dataset")

Before you can evaluate your GenAI application, you need test data. **Evaluation Datasets** provide a centralized repository for managing test cases, ground truth expectations, and evaluation data at scale.

Think of Evaluation Datasets as your "test database" - a single source of truth for all the data needed to evaluate your AI systems. They transform ad-hoc testing into systematic quality assurance.

[Learn more →](/mlflow-website/docs/latest/genai/datasets.md)

![Trace Dataset](/mlflow-website/docs/latest/assets/images/genai-trace-dataset-0db517dfd5b8e13ae6732b0a1b0b098f.png)

#### Track Annotation and Human Feedbacks[​](#track-annotation-and-human-feedbacks "Direct link to Track Annotation and Human Feedbacks")

Human feedback is essential for building high-quality GenAI applications that meet user expectations. MLflow supports collecting, managing, and utilizing feedback from end-users and domain experts.

Feedbacks are attached to traces and recorded with metadata, including user, timestamp, revisions, etc.

[Learn more →](/mlflow-website/docs/latest/genai/tracing/collect-user-feedback.md)

![Trace Feedback](/mlflow-website/docs/latest/assets/images/genai-human-feedback-9a8ea2ba10a5f7c7bb192aea22345b19.png)

#### Scale Quality Assessment with Automation[​](#scale-quality-assessment-with-automation "Direct link to Scale Quality Assessment with Automation")

Quality assessment is a critical part of building high-quality GenAI applications, however, it is often time-consuming and requires human expertise. LLMs are powerful tools to automate quality assessment.

MLflow offers various built-in LLM-as-a-Judge scorers to help automate the process, as well as a flexible toolset to build your own LLM judges with ease.

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor.md)

![Trace Evaluation](/mlflow-website/docs/latest/assets/images/genai-trace-evaluation-5b5e6ba86f0f0f06ee27db356e4e59e4.png)

#### Evaluate and Enhance quality[​](#evaluate-and-enhance-quality "Direct link to Evaluate and Enhance quality")

Systematically assessing and improving the quality of GenAI applications is a challenge. MLflow provides a comprehensive set of tools to help you evaluate and enhance the quality of your applications.

Being the industry's most-trusted experiment tracking platform, MLflow provides a strong foundation for tracking your evaluation results and effectively collaborating with your team.

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)

![Trace Evaluation](/mlflow-website/docs/latest/assets/images/genai-evaluation-compare-e16ede1d0aa60604dd3eb43ecda3d631.png)

#### Monitor Applications in Production[​](#monitor-applications-in-production "Direct link to Monitor Applications in Production")

Understanding and optimizing GenAI application performance is crucial for efficient operations. MLflow Tracing captures key metrics like latency and token usage at each step, as well as various quality metrics, helping you identify bottlenecks, monitor efficiency, and find optimization opportunities.

[Learn more →](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)

![Monitoring](/mlflow-website/docs/latest/assets/images/genai-monitoring-8ebda32e5cc07cb9cc97cb0297e583c3.png)

## Running an Evaluation[​](#running-an-evaluation "Direct link to Running an Evaluation")

Each evaluation is defined by three components:

| Component                                                                                | Example                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**<br />Inputs & expectations (and optionally pre-generated outputs and traces) | ```
[
  {"inputs": {"question": "2+2"}, "expectations": {"answer": "4"}},
  {"inputs": {"question": "2+3"}, "expectations": {"answer": "5"}}
]
```                                                                                    |
| **Scorer**<br />Evaluation criteria                                                      | ```
@scorer
def exact_match(expectations, outputs):
    return expectations == outputs
```                                                                                                                                           |
| **Predict Function**<br />Generates outputs for the dataset                              | ```
def predict_fn(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content
``` |

The following example shows a simple evaluation of a dataset of questions and expected answers.

python

```python
import os
import openai
import mlflow
from mlflow.genai.scorers import Correctness, Guidelines

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define a simple QA dataset
dataset = [
    {
        "inputs": {"question": "Can MLflow manage prompts?"},
        "expectations": {"expected_response": "Yes!"},
    },
    {
        "inputs": {"question": "Can MLflow create a taco for my lunch?"},
        "expectations": {
            "expected_response": "No, unfortunately, MLflow is not a taco maker."
        },
    },
]


# 2. Define a prediction function to generate responses
def predict_fn(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


# 3.Run the evaluation
results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[
        # Built-in LLM judge
        Correctness(),
        # Custom criteria using LLM judge
        Guidelines(name="is_english", guidelines="The answer must be in English"),
    ],
)

```

## Review the results[​](#review-the-results "Direct link to Review the results")

Open the MLflow UI to review the evaluation results. If you are using OSS MLflow, you can use the following command to start the UI:

bash

```bash
mlflow ui --port 5000

```

If you are using cloud-based MLflow, open the experiment page in the platform. You should see a new evaluation run is created under the "Runs" tab. Click on the run name to view the evaluation results.

![Evaluation Results](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/quickstart-eval-hero.png)

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Quickstart](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)

[Learn MLflow's evaluation workflow in action.](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)

[Start evaluating →](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)

### [Evaluate Agents](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Evaluate AI agents with specialized techniques and custom scorers.](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Evaluate agents →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

### [Building Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md)

[Get started with MLflow's powerful scorers for evaluating qualities.](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md)

[Learn about scorers →](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md)
