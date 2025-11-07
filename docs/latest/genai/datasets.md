# Evaluation Datasets

## Transform Your GenAI Testing with Structured Evaluation Data[​](#transform-your-genai-testing-with-structured-evaluation-data "Direct link to Transform Your GenAI Testing with Structured Evaluation Data")

Evaluation datasets are the foundation of systematic GenAI application testing. They provide a centralized way to manage test data, ground truth expectations, and evaluation results—enabling you to measure and improve the quality of your AI applications with confidence.

## Quickstart: Build Your First Evaluation Dataset[​](#quickstart-build-your-first-evaluation-dataset "Direct link to Quickstart: Build Your First Evaluation Dataset")

There are several ways to create evaluation datasets, each suited to different stages of your GenAI development process. **Expectations are the cornerstone of effective evaluation**—they define the ground truth against which your AI's outputs are measured, enabling systematic quality assessment across iterations.

* Build from Traces
* From Dictionaries
* From DataFrame

python

```
import mlflow
from mlflow.genai.datasets import create_dataset

# Create your evaluation dataset
dataset = create_dataset(
    name="production_validation_set",
    experiment_id=["0"],  # "0" is the default experiment
    tags={"team": "ml-platform", "stage": "validation"},
)

# First, retrieve traces that will become the basis of the dataset
# Request list format to work with individual Trace objects
traces = mlflow.search_traces(
    experiment_ids=["0"],
    max_results=50,
    filter_string="attributes.name = 'chat_completion'",
    return_type="list",  # Returns list[Trace] for direct manipulation
)

# Add expectations to the traces
for trace in traces[:20]:
    # Expectations can be structured metrics
    mlflow.log_expectation(
        trace_id=trace.info.trace_id,
        name="output_quality",
        value={"relevance": 0.95, "accuracy": 1.0, "contains_citation": True},
    )

    # They can also be specific expected text
    mlflow.log_expectation(
        trace_id=trace.info.trace_id,
        name="expected_answer",
        value="The correct answer should include step-by-step instructions for password reset with email verification",
    )

# Retrieve the traces with added expectations
annotated_traces = mlflow.search_traces(
    experiment_ids=["0"], max_results=100, return_type="list"  # Get list[Trace] objects
)

# Merge the list of Trace objects directly into your dataset
dataset.merge_records(annotated_traces)
```

python

```
import mlflow
from mlflow.genai.datasets import create_dataset

# Create dataset with manual test cases
dataset = create_dataset(
    name="regression_test_suite",
    experiment_id=["0", "1"],  # Multiple experiments
    tags={"type": "regression", "priority": "critical"},
)

# Define test cases with expected outputs
test_cases = [
    {
        "inputs": {
            "question": "How do I reset my password?",
            "context": "user_support",
        },
        "expectations": {
            "answer": "To reset your password, click 'Forgot Password' on the login page, enter your email, and follow the link sent to your inbox",
            "contains_steps": True,
            "tone": "helpful",
            "max_response_time": 2.0,
        },
    },
    {
        "inputs": {
            "question": "What are your refund policies?",
            "context": "customer_service",
        },
        "expectations": {
            "includes_timeframe": True,
            "mentions_exceptions": True,
            "accuracy": 1.0,
        },
    },
]

dataset.merge_records(test_cases)
```

python

```
import pandas as pd
import mlflow
from mlflow.genai.datasets import create_dataset

# Create dataset from existing test data
dataset = create_dataset(
    name="benchmark_dataset",
    experiment_id=["0"],  # Use your experiment ID
    tags={"source": "benchmark", "version": "2024.1"},
)

# Method 1: Use traces from search_traces (default returns DataFrame)
# search_traces returns a pandas DataFrame by default when pandas is installed
traces_df = mlflow.search_traces(
    experiment_ids=["0"],  # Search in your experiment
    max_results=100
    # No return_type specified - defaults to "pandas"
)

# The DataFrame from search_traces can be passed directly
dataset.merge_records(traces_df)

# Method 2: Create your own DataFrame with inputs and expectations
# You can also create a DataFrame with the expected structure
custom_df = pd.DataFrame(
    {
        "inputs.question": [
            "What is MLflow?",
            "How do I track experiments?",
            "Explain model versioning",
        ],
        "inputs.domain": ["general", "technical", "technical"],
        "expectations.relevance": [1.0, 0.95, 0.9],
        "expectations.technical_accuracy": [1.0, 1.0, 0.95],
        "expectations.includes_examples": [True, True, False],
        "tags.priority": ["high", "medium", "medium"],  # Optional tags
        "tags.reviewed": [True, True, False],
    }
)

# merge_records accepts DataFrames with inputs, expectations, and tags columns
dataset.merge_records(custom_df)
```

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

## Core Concepts[​](#core-concepts "Direct link to Core Concepts")

### [Dataset Structure](/mlflow-website/docs/latest/genai/concepts/evaluation-datasets.md)

[Understand how evaluation datasets organize test inputs, expectations, and metadata](/mlflow-website/docs/latest/genai/concepts/evaluation-datasets.md)

[Learn the concepts →](/mlflow-website/docs/latest/genai/concepts/evaluation-datasets.md)

### [SDK Guide](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

[Complete guide to creating and managing evaluation datasets programmatically](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

[View SDK guide →](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

### [Evaluation Integration](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Learn how to use datasets with MLflow's evaluation framework](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Explore evaluation →](/mlflow-website/docs/latest/genai/eval-monitor.md)

## Next Steps[​](#next-steps "Direct link to Next Steps")

Ready to improve your GenAI testing? Start with these resources:

### [Setting Expectations](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Learn how to define ground truth and expected outputs for your AI system](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Define expectations →](/mlflow-website/docs/latest/genai/assessments/expectations.md)

### [Tracing Guide](/mlflow-website/docs/latest/genai/tracing.md)

[Capture detailed execution data from your GenAI applications](/mlflow-website/docs/latest/genai/tracing.md)

[Start tracing →](/mlflow-website/docs/latest/genai/tracing.md)

### [Evaluation Framework](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Run systematic evaluations using your datasets with automated scorers](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Learn evaluation →](/mlflow-website/docs/latest/genai/eval-monitor.md)
