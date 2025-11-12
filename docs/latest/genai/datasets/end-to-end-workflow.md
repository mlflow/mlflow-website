# End-to-End Workflow: Evaluation-Driven Development

This guide demonstrates the complete workflow for building and evaluating GenAI applications using MLflow's evaluation-driven development approach.

note

**Databricks Users**: To use Evaluation Datasets with Databricks Unity Catalog, MLflow requires the additional installation of the `databricks-agents` package. This package uses Unity Catalog to store datasets. Install it with: `pip install databricks-agents`

## Workflow Overview[​](#workflow-overview "Direct link to Workflow Overview")

### Evaluation-Driven Development

Build & Trace

Capture Traces

Add Expectations

Create Dataset

Evaluate

Analyze Results

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

bash

```bash
pip install --upgrade mlflow>=3.4 openai

```

## Step 1: Build & Trace Your Application[​](#step-1-build--trace-your-application "Direct link to Step 1: Build & Trace Your Application")

Start with a traced GenAI application. This example shows a customer support bot, but the pattern applies to any LLM application. You can use the [mlflow.trace decorator](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.trace) for manual instrumentation or [enable automatic tracing for OpenAI](/mlflow-website/docs/latest/api_reference/python_api/mlflow.openai.html#mlflow.openai.autolog) as shown below.

python

```python
import mlflow
import openai
import os

# Configure environment
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
mlflow.set_experiment("Customer Support Bot")

# Enable automatic tracing for OpenAI
mlflow.openai.autolog()


class CustomerSupportBot:
    def __init__(self):
        self.client = openai.OpenAI()
        self.knowledge_base = {
            "refund": "Full refunds within 30 days with receipt.",
            "shipping": "Standard: 5-7 days. Express available.",
            "warranty": "1-year manufacturer warranty included.",
        }

    @mlflow.trace
    def answer(self, question: str) -> str:
        # Retrieve relevant context
        context = self._get_context(question)

        # Generate response
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful support assistant."},
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def _get_context(self, question: str) -> str:
        # Simple keyword matching for demo
        for key, value in self.knowledge_base.items():
            if key in question.lower():
                return value
        return "General customer support information."


bot = CustomerSupportBot()

```

## Step 2: Capture Production Traces[​](#step-2-capture-production-traces "Direct link to Step 2: Capture Production Traces")

Run your application with real or test scenarios to capture traces. Later, you'll use [mlflow.search\_traces()](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.search_traces) to retrieve these traces for annotation and dataset creation.

python

```python
# Test scenarios
test_questions = [
    "What is your refund policy?",
    "How long does shipping take?",
    "Is my product under warranty?",
    "Can I get express shipping?",
]

# Capture traces - automatically logged to the active experiment
for question in test_questions:
    response = bot.answer(question)

```

## Step 3: Add Ground Truth Expectations[​](#step-3-add-ground-truth-expectations "Direct link to Step 3: Add Ground Truth Expectations")

Add expectations to your traces to define what constitutes correct behavior. Use [mlflow.log\_expectation()](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_expectation) to annotate traces with ground truth values that will serve as your evaluation baseline.

python

```python
# Search for recent traces (uses current active experiment by default)
traces = mlflow.search_traces(
    max_results=10, return_type="list"  # Return list of Trace objects for iteration
)

# Add expectations to specific traces
for trace in traces:
    # Get the question from the root span inputs
    root_span = trace.data._get_root_span()
    question = (
        root_span.inputs.get("question", "") if root_span and root_span.inputs else ""
    )

    if "refund" in question.lower():
        mlflow.log_expectation(
            trace_id=trace.info.trace_id,
            name="key_information",
            value={"must_mention": ["30 days", "receipt"], "tone": "helpful"},
        )
    elif "shipping" in question.lower():
        mlflow.log_expectation(
            trace_id=trace.info.trace_id,
            name="key_information",
            value={"must_mention": ["5-7 days"], "offers_express": True},
        )

```

## Step 4: Create an Evaluation Dataset[​](#step-4-create-an-evaluation-dataset "Direct link to Step 4: Create an Evaluation Dataset")

Transform your annotated traces into a reusable evaluation dataset. Use [create\_dataset()](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.datasets.create_dataset) to initialize your dataset and [merge\_records()](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.EvaluationDataset.merge_records) to add test cases from multiple sources.

python

```python
from mlflow.genai.datasets import create_dataset

# Create dataset from current experiment
dataset = create_dataset(
    name="customer_support_qa_v1",
    experiment_id=mlflow.get_experiment_by_name("Customer Support Bot").experiment_id,
    tags={"stage": "validation", "domain": "customer_support"},
)

# Re-fetch traces to get the attached expectations
# The expectations are now part of the trace data
annotated_traces = mlflow.search_traces(
    max_results=100,
    return_type="list",  # Need list for merge_records
)

# Add traces to dataset
dataset.merge_records(annotated_traces)

# Optionally add manual test cases
manual_tests = [
    {
        "inputs": {"question": "Can I return an item after 45 days?"},
        "expectations": {"should_clarify": "30-day policy", "tone": "apologetic"},
    },
    {
        "inputs": {"question": "Do you ship internationally?"},
        "expectations": {"provides_alternatives": True},
    },
]
dataset.merge_records(manual_tests)

```

## Step 5: Run Systematic Evaluation[​](#step-5-run-systematic-evaluation "Direct link to Step 5: Run Systematic Evaluation")

Evaluate your application against the dataset using built-in and custom scorers. Use [mlflow.genai.evaluate()](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate) to run comprehensive evaluations with scorers like [Correctness](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Correctness) for factual accuracy assessment. You can also create custom scorers using the [@scorer decorator](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.scorer) to evaluate domain-specific criteria.

python

```python
from mlflow.genai import evaluate
from mlflow.genai.scorers import Correctness, Guidelines, scorer


# Define custom scorer for your specific needs
@scorer
def contains_required_info(outputs: str, expectations: dict) -> float:
    """Check if response contains required information."""
    if "must_mention" not in expectations:
        return 1.0

    output_lower = outputs.lower()
    mentioned = [term for term in expectations["must_mention"] if term in output_lower]
    return len(mentioned) / len(expectations["must_mention"])


# Configure evaluation
scorers = [
    Correctness(name="factual_accuracy"),
    Guidelines(
        name="support_quality",
        guidelines="Response must be helpful, accurate, and professional",
    ),
    contains_required_info,
]

# Run evaluation
results = evaluate(
    data=dataset,
    predict_fn=bot.answer,
    scorers=scorers,
    model_id="customer-support-bot-v1",
)

# Access results
metrics = results.metrics
detailed_results = results.tables["eval_results_table"]

```

## Step 6: Iterate and Improve[​](#step-6-iterate-and-improve "Direct link to Step 6: Iterate and Improve")

Use evaluation results to improve your application, then re-evaluate using the same dataset.

python

```python
# Analyze results
low_scores = detailed_results[detailed_results["factual_accuracy/score"] < 0.8]
if not low_scores.empty:
    # Identify patterns in failures
    failed_questions = low_scores["inputs.question"].tolist()

    # Example improvements based on failure analysis
    bot.knowledge_base[
        "refund"
    ] = "Full refunds available within 30 days with original receipt. Store credit offered after 30 days."
    bot.client.temperature = 0.2  # Reduce temperature for more consistent responses

    # Re-evaluate with same dataset for comparison
    improved_results = evaluate(
        data=dataset,
        predict_fn=bot.answer,  # Updated bot
        scorers=scorers,
        model_id="customer-support-bot-v2",
    )

    # Compare versions
    improvement = (
        improved_results.metrics["factual_accuracy/score"]
        - metrics["factual_accuracy/score"]
    )

```

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Custom Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md)

[Build sophisticated scorers for complex evaluation criteria](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md)

### [SDK Reference](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

[Deep dive into dataset management APIs](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

[View guide →](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

### [Production Monitoring](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)

[Set up continuous evaluation for production](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)

[Learn more →](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)
