# Evaluation Datasets

## Transform Your GenAI Testing with Structured Evaluation Data[​](#transform-your-genai-testing-with-structured-evaluation-data "Direct link to Transform Your GenAI Testing with Structured Evaluation Data")

Evaluation datasets are the foundation of systematic GenAI application testing. They provide a centralized way to manage test data, ground truth expectations, and evaluation results—enabling you to measure and improve the quality of your AI applications with confidence.

SQL Backend Required

Evaluation Datasets require an MLflow Tracking Server with a **[SQL backend](/mlflow-website/docs/latest/self-hosting/architecture/backend-store.md#types-of-backend-stores)** (PostgreSQL, MySQL, SQLite, or MSSQL). This feature is **not available** in FileStore (local file system-based tracking). If you need a simple local configuration for MLflow, use the sqlite option when starting MLflow.

## Quickstart: Build Your First Evaluation Dataset[​](#quickstart-build-your-first-evaluation-dataset "Direct link to Quickstart: Build Your First Evaluation Dataset")

There are several ways to create evaluation datasets, each suited to different stages of your GenAI development process.

The simplest way to create one is through MLflow's UI. Navigate to an Experiment that you want the evaluation dataset to be associated with and you can directly create a new one by supplying a unique name. After adding records to it, you can view the dataset's entries in the UI.

![Evaluation Datasets Video](/mlflow-website/docs/latest/images/eval-datasets.gif)

At its core, evaluation datasets are comprised of **inputs** and **expectations**. **Outputs** are an optional addition that can be added to an evaluation dataset for post-hoc evaluation with scorers. Adding these elements can be done either directly from traces, dictionaries, or via a Pandas DataFrame.

* Build from Traces
* From Dictionaries
* From DataFrame

python

```python
import mlflow
from mlflow.genai.datasets import create_dataset, set_dataset_tags

# Create your evaluation dataset
dataset = create_dataset(
    name="production_validation_set",
    experiment_id=["0"],  # "0" is the default experiment
    tags={"team": "ml-platform", "stage": "validation"},
)

# Optionally, add additional tags to your dataset.
# Tags can be used to search for datasets with search_datasets API
set_dataset_tags(
    dataset_id=dataset.dataset_id,
    tags={"environment": "dev", "validation_version": "1.3"},
)

# First, retrieve traces that will become the basis of the dataset
traces = mlflow.search_traces(
    experiment_ids=["0"],
    max_results=20,
    filter_string="attributes.name = 'chat_completion'",
    return_type="list",  # Returns list[Trace]
)

# Add expectations to the traces
for trace in traces:
    mlflow.log_expectation(
        trace_id=trace.info.trace_id,
        name="expected_answer",
        value=(
            "The correct answer should include step-by-step instructions "
            "for password reset with email verification"
        ),
    )

# Retrieve the traces with added expectations
annotated_traces = mlflow.search_traces(
    experiment_ids=["0"],
    max_results=20,
    return_type="list",
)

# Merge the list of Trace objects directly into your dataset
dataset.merge_records(annotated_traces)

```

python

```python
import mlflow
from mlflow.genai.datasets import create_dataset

# Create dataset with manual test cases
dataset = create_dataset(
    name="regression_test_suite",
    experiment_id=["0", "1"],  # Multiple experiments
    tags={"type": "regression", "priority": "critical"},
)

# Define test cases with expected outputs (ground truth)
test_cases = [
    {
        "inputs": {
            "question": "How do I reset my password?",
            "context": "user_support",
        },
        "expectations": {
            "expected_answer": (
                "To reset your password, click 'Forgot Password' on the login page, "
                "enter your email, and follow the link sent to your inbox"
            ),
            "must_contain_steps": True,
            "expected_tone": "helpful",
        },
    },
    {
        "inputs": {
            "question": "What are your refund policies?",
            "context": "customer_service",
        },
        "expectations": {
            "expected_answer": (
                "We offer full refunds within 30 days of purchase. "
                "Refunds after 30 days are subject to approval."
            ),
            "must_include_timeframe": True,
            "must_mention_exceptions": True,
        },
    },
]

dataset.merge_records(test_cases)

```

python

```python
import pandas as pd
from mlflow.genai.datasets import create_dataset

# Create dataset
dataset = create_dataset(
    name="benchmark_dataset",
    experiment_id=["0"],
    tags={"source": "benchmark", "version": "2024.1"},
)

# Create DataFrame with inputs and expectations (ground truth)
df = pd.DataFrame(
    [
        {
            "inputs": {
                "question": "What is MLflow?",
                "domain": "general",
            },
            "expectations": {
                "expected_answer": "MLflow is an open-source platform for ML",
                "must_mention": ["tracking", "experiments", "models"],
            },
        },
        {
            "inputs": {
                "question": "How do I track experiments?",
                "domain": "technical",
            },
            "expectations": {
                "expected_answer": "Use mlflow.start_run() and mlflow.log_params()",
                "must_mention": ["log_params", "log_metrics"],
            },
        },
        {
            "inputs": {
                "question": "Explain model versioning",
                "domain": "technical",
            },
            "expectations": {
                "expected_answer": "Model Registry provides versioning",
                "must_mention": ["Model Registry", "versions"],
            },
        },
    ]
)

# Add records from DataFrame
dataset.merge_records(df)

```

## Understanding Source Types[​](#understanding-source-types "Direct link to Understanding Source Types")

Every record in an evaluation dataset has a **source type** that tracks its provenance. This enables you to analyze model performance by data origin and understand which types of test data are most valuable.

#### TRACE

Records from production traces - automatically assigned when adding traces via mlflow\.search\_traces()

#### HUMAN

Subject matter expert annotations - automatically inferred for records with expectations (ground truth)

#### CODE

Programmatically generated test cases - automatically inferred for records without expectations

#### DOCUMENT

Test cases extracted from documentation or specs - must be explicitly specified with source metadata

Source types are automatically inferred based on record characteristics but can be explicitly overridden when needed. See the [SDK Guide](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md#source-type-inference) for detailed inference rules and examples.

## Why Evaluation Datasets?[​](#why-evaluation-datasets "Direct link to Why Evaluation Datasets?")

#### Centralized Test Management

Store all your test cases, expected outputs, and evaluation criteria in one place. No more scattered CSV files or hardcoded test data.

#### Consistent Evaluation Source

Maintain a concrete representation of test data that can be used repeatedly as your project evolves. Eliminate manual testing and avoid repeatedly assembling evaluation data for each iteration.

#### Systematic Testing

Move beyond ad-hoc testing to systematic evaluation. Define clear expectations and measure performance consistently across deployments.

#### Collaborative Improvement

Enable your entire team to contribute test cases and expectations. Share evaluation datasets across projects and teams.

## The Evaluation Loop[​](#the-evaluation-loop "Direct link to The Evaluation Loop")

Evaluation datasets bridge the critical gap between trace generation and evaluation execution in the GenAI development lifecycle. As you test your application and capture traces with expectations, **evaluation datasets transform these individual test cases into a materialized, reusable evaluation suite**. This creates a consistent and evolving collection of evaluation records that grows with your application—each iteration adds new test cases while preserving the historical test coverage. Rather than losing valuable test scenarios after each development cycle, you build a comprehensive evaluation asset that can immediately assess the quality of changes and improvements to your implementation.

### The Evaluation Loop

Iterate on Code

Test App

Collect Traces

Add Expectations

Create Dataset

Run Evaluation

Analyze Results

## Key Features[​](#key-features "Direct link to Key Features")

#### Ground Truth Management

Define and maintain expected outputs for your test cases. Capture expert knowledge about what constitutes correct behavior for your AI system.

#### Schema Evolution

Automatically track the structure of your test data as it evolves. Add new fields and test dimensions without breaking existing evaluations.

#### Incremental Updates

Continuously improve your test suite by adding new cases from production. Update expectations as your understanding of correct behavior evolves.

#### Flexible Tagging

Organize datasets with tags for easy discovery and filtering. Track metadata like data sources, annotation guidelines, and quality levels.

#### Performance Tracking

Monitor how your application performs against the same test data over time. Identify regressions and improvements across deployments.

#### Experiment Integration

Link datasets to MLflow experiments for complete traceability. Understand which test data was used for each model evaluation.

## Next Steps[​](#next-steps "Direct link to Next Steps")

Ready to improve your GenAI testing? Start with these resources:

### [Dataset Structure](/mlflow-website/docs/latest/genai/concepts/evaluation-datasets.md)

[Understand how evaluation datasets organize test inputs, expectations, and metadata](/mlflow-website/docs/latest/genai/concepts/evaluation-datasets.md)

[Learn the concepts →](/mlflow-website/docs/latest/genai/concepts/evaluation-datasets.md)

### [SDK Guide](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

[Complete guide to creating and managing evaluation datasets programmatically](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

[View SDK guide →](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

### [Setting Expectations](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Learn how to define ground truth and expected outputs for your AI system](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Define expectations →](/mlflow-website/docs/latest/genai/assessments/expectations.md)

### [Tracing Guide](/mlflow-website/docs/latest/genai/tracing.md)

[Capture detailed execution data from your GenAI applications](/mlflow-website/docs/latest/genai/tracing.md)

[Start tracing →](/mlflow-website/docs/latest/genai/tracing.md)

### [Evaluation Framework](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Run systematic evaluations using your datasets with automated scorers](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Learn evaluation →](/mlflow-website/docs/latest/genai/eval-monitor.md)
