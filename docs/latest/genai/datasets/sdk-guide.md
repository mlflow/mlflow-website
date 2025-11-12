# Evaluation Datasets SDK Guide

Master the APIs for creating, evolving, and managing evaluation datasets through practical workflows and real-world patterns.

## Getting Started[​](#getting-started "Direct link to Getting Started")

MLflow provides a fluent API for working with evaluation datasets that makes common workflows simple and intuitive:

python

```python
from mlflow.genai.datasets import (
    create_dataset,
    get_dataset,
    search_datasets,
    set_dataset_tags,
    delete_dataset_tag,
)

```

## Your Dataset Journey[​](#your-dataset-journey "Direct link to Your Dataset Journey")

Follow this typical workflow to build and evolve your evaluation datasets:

### Complete Development Workflow

Create/Get Dataset

Add Test Cases

Run Evaluation

Improve Code

Test & Trace

Update Dataset

Update Tags

### Step 1: Create Your Dataset[​](#step-1-create-your-dataset "Direct link to Step 1: Create Your Dataset")

Start by creating a new evaluation dataset with meaningful metadata using the [`mlflow.genai.datasets.create_dataset()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.datasets.create_dataset) API:

python

```python
from mlflow.genai.datasets import create_dataset

# Create a new dataset with tags for organization
dataset = create_dataset(
    name="customer_support_qa_v1",
    experiment_id=["0"],  # Link to experiments ("0" is default)
    tags={
        "version": "1.0",
        "purpose": "regression_testing",
        "model": "gpt-4",
        "team": "ml-platform",
        "status": "development",
    },
)

```

### Step 2: Add Your First Test Cases[​](#step-2-add-your-first-test-cases "Direct link to Step 2: Add Your First Test Cases")

Build your dataset by adding test cases from production traces and manual curation. **Expectations are typically defined by subject matter experts (SMEs)** who understand the domain and can establish ground truth for what constitutes correct behavior.

[Learn how to define expectations →](/mlflow-website/docs/latest/genai/assessments/expectations.md) Expectations are the ground truth values that define what your AI should produce. They're added by SMEs who review outputs and establish quality standards.

* From Production Traces
* Manual Test Cases

python

```python
import mlflow

# Search for production traces to build your dataset
# Request list format to work with individual Trace objects
production_traces = mlflow.search_traces(
    experiment_ids=["0"],  # Your production experiment
    filter_string="attributes.user_feedback = 'positive'",
    max_results=100,
    return_type="list",  # Returns list[Trace] for direct manipulation
)

# Subject matter experts add expectations to define correct behavior
for trace in production_traces:
    # Subject matter experts review traces and define what the output should satisfy
    mlflow.log_expectation(
        trace_id=trace.info.trace_id,
        name="quality_assessment",
        value={
            "should_match_production": True,
            "minimum_quality": 0.8,
            "response_time_ms": 2000,
            "contains_citation": True,
        },
    )

    # Can also add textual expectations
    mlflow.log_expectation(
        trace_id=trace.info.trace_id,
        name="expected_behavior",
        value="Response should provide step-by-step instructions with security considerations",
    )

# Add annotated traces to dataset (expectations are automatically included)
dataset.merge_records(production_traces)

```

python

```python
# Test cases can be manually defined as dictionaries
# merge_records() accepts both dict and pandas.DataFrame formats for manual
# record additions
test_cases = [
    {
        "inputs": {
            "question": "How do I reset my password?",
            "user_type": "premium",
            "context": "User has been locked out after 3 failed attempts",
        },
        "expectations": {
            "answer_quality": 0.95,
            "contains_steps": True,
            "mentions_security": True,
            "response": "To reset your password, please follow these steps:\n1. Click 'Forgot Password' on the login page\n2. Enter your registered email address\n3. Check your email for the reset link\n4. Click the link and create a new password\n5. Use your new password to log in",
        },
        "tags": {
            "category": "account_management",
            "priority": "high",
            "reviewed_by": "security_team",
        },
    },
    {
        "inputs": {
            "question": "What are your business hours?",
            "user_type": "standard",
        },
        "expectations": {
            "accuracy": 1.0,
            "includes_timezone": True,
            "mentions_holidays": True,
        },
    },
]

# Add to your dataset (accepts list[dict], list[Trace] or pandas.DataFrame)
dataset.merge_records(test_cases)

```

### Step 3: Evolve Your Dataset[​](#step-3-evolve-your-dataset "Direct link to Step 3: Evolve Your Dataset")

As you discover edge cases and improve your understanding, continuously update your dataset. The [`mlflow.entities.EvaluationDataset.merge_records()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.EvaluationDataset.merge_records) method intelligently handles both new records and updates to existing ones:

python

```python
# Capture a production failure
failure_case = {
    "inputs": {"question": "'; DROP TABLE users; --", "user_type": "malicious"},
    "expectations": {
        "handles_sql_injection": True,
        "returns_safe_response": True,
        "logs_security_event": True,
    },
    "source": {
        "source_type": "HUMAN",
        "source_data": {"discovered_by": "security_team"},
    },
    "tags": {"category": "security", "severity": "critical"},
}

# Add the new edge case
dataset.merge_records([failure_case])

# Update expectations for existing records
updated_records = []
for record in dataset.records:
    if "accuracy" in record.get("expectations", {}):
        # Raise the quality bar
        record["expectations"]["accuracy"] = max(
            0.9, record["expectations"]["accuracy"]
        )
        updated_records.append(record)

# Merge updates (intelligently handles duplicates)
dataset.merge_records(updated_records)

```

### Step 4: Organize with Tags[​](#step-4-organize-with-tags "Direct link to Step 4: Organize with Tags")

Use tags to track dataset evolution and enable powerful searches. Learn more about [`mlflow.search_traces()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.search_traces) to build your datasets from production data:

python

```python
from mlflow.genai.datasets import set_dataset_tags

# Update dataset metadata
set_dataset_tags(
    dataset_id=dataset.dataset_id,
    tags={
        "status": "validated",
        "coverage": "comprehensive",
        "last_review": "2024-11-01",
    },
)

# Remove outdated tags
set_dataset_tags(
    dataset_id=dataset.dataset_id,
    tags={"development_only": None},  # Setting to None removes the tag
)

```

### Step 5: Search and Discover[​](#step-5-search-and-discover "Direct link to Step 5: Search and Discover")

Find datasets using powerful search capabilities with [`mlflow.genai.datasets.search_datasets()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.datasets.search_datasets):

python

```python
from mlflow.genai.datasets import search_datasets

# Find datasets by experiment
datasets = search_datasets(experiment_ids=["0", "1"])  # Search in multiple experiments

# Search by name pattern
regression_datasets = search_datasets(filter_string="name LIKE '%regression%'")

# Complex search with tags
production_ready = search_datasets(
    filter_string="tags.status = 'validated' AND tags.coverage = 'comprehensive'",
    order_by=["last_update_time DESC"],
    max_results=10,
)

# The PagedList automatically handles pagination when iterating

```

#### Common Filter String Examples[​](#common-filter-string-examples "Direct link to Common Filter String Examples")

Here are practical examples of filter strings to help you find the right datasets:

| Filter Expression                                 | Description             | Use Case                       |
| ------------------------------------------------- | ----------------------- | ------------------------------ |
| **`name = 'production_qa'`**                      | Exact name match        | Find a specific dataset        |
| **`name LIKE '%test%'`**                          | Pattern matching        | Find all test datasets         |
| **`tags.status = 'validated'`**                   | Tag equality            | Find production-ready datasets |
| **`tags.version = '2.0' AND tags.team = 'ml'`**   | Multiple tag conditions | Find team-specific versions    |
| **`created_by = 'alice@company.com'`**            | Creator filter          | Find datasets by author        |
| **`created_time > 1698800000000`**                | Time-based filter       | Find recent datasets           |
| **`tags.model = 'gpt-4' AND name LIKE '%eval%'`** | Combined conditions     | Model-specific evaluation sets |
| **`last_updated_by != 'bot@system'`**             | Exclusion filter        | Exclude automated updates      |

### Step 6: Manage Experiment Associations[​](#step-6-manage-experiment-associations "Direct link to Step 6: Manage Experiment Associations")

Datasets can be dynamically associated with experiments after creation using [`mlflow.genai.datasets.add_dataset_to_experiments()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.datasets.add_dataset_to_experiments) and [`mlflow.genai.datasets.remove_dataset_from_experiments()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.datasets.remove_dataset_from_experiments).

This functionality enables several important use cases:

* **Cross-team collaboration**: Share datasets across teams by adding their experiment IDs
* **Lifecycle management**: Remove outdated experiment associations as projects mature
* **Project reorganization**: Dynamically reorganize datasets as your project structure evolves

python

```python
from mlflow.genai.datasets import (
    add_dataset_to_experiments,
    remove_dataset_from_experiments,
)

# Add dataset to additional experiments
dataset = add_dataset_to_experiments(
    dataset_id="d-1a2b3c4d5e6f7890abcdef1234567890", experiment_ids=["3", "4", "5"]
)
print(f"Dataset now linked to experiments: {dataset.experiment_ids}")

# Remove dataset from specific experiments
dataset = remove_dataset_from_experiments(
    dataset_id="d-1a2b3c4d5e6f7890abcdef1234567890", experiment_ids=["3"]
)
print(f"Updated experiment associations: {dataset.experiment_ids}")

```

## The Active Record Pattern[​](#the-active-record-pattern "Direct link to The Active Record Pattern")

The `EvaluationDataset` object follows an active record pattern—it's both a data container and provides methods to interact with the backend:

python

```python
# Get a dataset
dataset = get_dataset(dataset_id="d-1a2b3c4d5e6f7890abcdef1234567890")

# The dataset object is "live" - it can fetch and update data
current_record_count = len(dataset.records)  # Lazy loads if needed

# Add new records directly on the object
new_records = [
    {
        "inputs": {"question": "What are your business hours?"},
        "expectations": {"mentions_hours": True, "includes_timezone": True},
    }
]
dataset.merge_records(new_records)  # Updates backend immediately

# Convert to DataFrame for analysis
df = dataset.to_df()
# Access auto-computed properties
schema = dataset.schema  # Field structure
profile = dataset.profile  # Dataset statistics

```

## How Record Merging Works[​](#how-record-merging-works "Direct link to How Record Merging Works")

The `merge_records()` method intelligently handles both new records and updates to existing ones. **Records are matched based on a hash of their inputs** - if a record with identical inputs already exists, its expectations and tags will be updated rather than creating a duplicate record.

* Adding New Records
* Updating Existing Records
* Bulk Updates from Traces
* Input Uniqueness

When you add records for the first time, they're stored with their inputs, expectations, and metadata:

python

```python
# Initial record
record_v1 = {
    "inputs": {"question": "What is MLflow?", "context": "ML platform overview"},
    "expectations": {"accuracy": 0.8, "mentions_tracking": True},
}

dataset.merge_records([record_v1])
# Creates a new record in the dataset

```

When you merge a record with identical inputs, the existing record is updated by **merging** the new expectations and tags with the existing ones:

python

```python
# Updated version with same inputs but enhanced expectations
record_v2 = {
    "inputs": {
        "question": "What is MLflow?",  # Same question
        "context": "ML platform overview",  # Same context
    },
    "expectations": {
        "accuracy": 0.95,  # Updates existing value
        "mentions_models": True,  # Adds new expectation
        "clarity": 0.9  # Adds new metric
        # Note: "mentions_tracking": True is preserved from record_v1
    },
    "tags": {"reviewed": "true", "reviewer": "ml_team"},
}

dataset.merge_records([record_v2])
# The record is updated, not duplicated
# Final record has all expectations from both v1 and v2 merged together

```

This update behavior is particularly useful when adding expectations to production traces:

python

```python
# First pass: Add traces without expectations
traces = mlflow.search_traces(experiment_ids=["0"], max_results=100, return_type="list")
dataset.merge_records(traces)

# Later: Subject matter experts review and add expectations
for trace in traces[:20]:  # Review subset
    mlflow.log_expectation(
        trace_id=trace.info.trace_id,
        name="quality_check",
        value={"approved": True, "quality_score": 0.9},
    )

# IMPORTANT: Re-fetch traces to get the attached expectations
updated_traces = mlflow.search_traces(
    experiment_ids=["0"], max_results=100, return_type="list"
)

# Re-merge the updated traces - existing records are updated with expectations
dataset.merge_records(updated_traces[:20])

```

Records are considered unique based on their **entire inputs dictionary**. Even small differences create separate records:

python

```python
# These are treated as different records due to different inputs
record_a = {
    "inputs": {"question": "What is MLflow?", "temperature": 0.7},
    "expectations": {"accuracy": 0.9},
}

record_b = {
    "inputs": {
        "question": "What is MLflow?",
        "temperature": 0.8,
    },  # Different temperature
    "expectations": {"accuracy": 0.9},
}

dataset.merge_records([record_a, record_b])
# Results in 2 separate records due to different temperature values

```

## Understanding Source Types[​](#understanding-source-types "Direct link to Understanding Source Types")

MLflow tracks the provenance of each record in your evaluation dataset through source types. This helps you understand where your test data came from and analyze performance by data source.

### Source Type Behavior

#### Automatic Inference

MLflow automatically infers source types based on record characteristics when no explicit source is provided.

#### Manual Override

You can always specify explicit source information to override automatic inference.

#### Provenance Tracking

Source types enable filtering and analysis of performance by data origin.

### Automatic Source Assignment[​](#automatic-source-assignment "Direct link to Automatic Source Assignment")

MLflow automatically assigns source types based on the characteristics of your records:

* TRACE Source
* HUMAN Source
* CODE Source

Records created from MLflow traces are automatically assigned the `TRACE` source type:

python

```python
# When adding traces directly (automatic TRACE source)
traces = mlflow.search_traces(experiment_ids=["0"], return_type="list")
dataset.merge_records(traces)  # All records get TRACE source type

# Or when using DataFrame from search_traces
traces_df = mlflow.search_traces(experiment_ids=["0"])  # Returns DataFrame
dataset.merge_records(
    traces_df
)  # Automatically detects traces and assigns TRACE source

```

Records with expectations are inferred as `HUMAN` source (subject matter expert annotations):

python

```python
# Records with expectations indicate human review/annotation
human_curated = [
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {"answer": "MLflow is an ML platform", "quality": 0.9}
        # Automatically inferred as HUMAN source due to expectations
    }
]
dataset.merge_records(human_curated)

```

Records with only inputs (no expectations) are inferred as `CODE` source (programmatically generated):

python

```python
# Records without expectations indicate programmatic generation
generated_tests = [
    {"inputs": {"question": f"Test question {i}"}}
    for i in range(100)
    # Automatically inferred as CODE source (no expectations field)
]
dataset.merge_records(generated_tests)

```

### Manual Source Specification[​](#manual-source-specification "Direct link to Manual Source Specification")

You can explicitly specify the source type and metadata for any record. When no explicit source is provided, MLflow automatically infers the source type before sending records to the backend using these rules:

* **Records with expectations** → Inferred as `HUMAN` source (indicates manual annotation or ground truth)
* **Records with only inputs** (no expectations) → Inferred as `CODE` source (indicates programmatic generation)
* **Records from traces** → Always marked as `TRACE` source (regardless of expectations)

This inference happens client-side in the `merge_records()` method before records are sent to the tracking backend. You can override this automatic inference by providing explicit source information:

python

```python
# Specify HUMAN source for manually curated test cases
human_curated = {
    "inputs": {"question": "What are your business hours?"},
    "expectations": {"accuracy": 1.0, "includes_timezone": True},
    "source": {
        "source_type": "HUMAN",
        "source_data": {"curator": "support_team", "date": "2024-11-01"},
    },
}

# Specify DOCUMENT source for data from documentation
from_docs = {
    "inputs": {"question": "How to install MLflow?"},
    "expectations": {"mentions_pip": True, "mentions_conda": True},
    "source": {
        "source_type": "DOCUMENT",
        "source_data": {"document_id": "install_guide", "page": 1},
    },
}

# Specify CODE source for programmatically generated data
generated = {
    "inputs": {"question": f"Test question {i}" for i in range(100)},
    "source": {
        "source_type": "CODE",
        "source_data": {"generator": "test_suite_v2", "seed": 42},
    },
}

dataset.merge_records([human_curated, from_docs, generated])

```

### Available Source Types[​](#available-source-types "Direct link to Available Source Types")

Source types enable powerful filtering and analysis of your evaluation results. You can analyze performance by data origin to understand if your model performs differently on human-curated vs. generated test cases, or production traces vs. documentation examples.

#### TRACE

Production data captured via MLflow tracing - automatically assigned when adding traces

#### HUMAN

Subject matter expert annotations - inferred for records with expectations

#### CODE

Programmatically generated tests - inferred for records without expectations

#### DOCUMENT

Test cases from documentation or specs - must be explicitly specified

#### UNSPECIFIED

Source unknown or not provided - for legacy or imported data

### Analyzing Data by Source[​](#analyzing-data-by-source "Direct link to Analyzing Data by Source")

* Source Distribution
* Filter by Source
* Source Metadata

python

```python
# Convert dataset to DataFrame for analysis
df = dataset.to_df()

# Check source type distribution
source_distribution = df["source_type"].value_counts()
print("Data sources in dataset:")
for source_type, count in source_distribution.items():
    print(f"  {source_type}: {count} records")

```

python

```python
# Analyze expectations by source
human_records = df[df["source_type"] == "HUMAN"]
trace_records = df[df["source_type"] == "TRACE"]
code_records = df[df["source_type"] == "CODE"]

print(f"Human-curated records: {len(human_records)}")
print(f"Production trace records: {len(trace_records)}")
print(f"Generated test records: {len(code_records)}")

# Filter high-value test cases for critical evaluation
high_value_test_cases = df[
    (df["source_type"] == "HUMAN") | (df["source_type"] == "DOCUMENT")
]

```

The `source_data` field stores rich metadata about record origins:

python

```python
# Example with detailed source metadata
detailed_source = {
    "inputs": {"question": "Complex integration test"},
    "expectations": {"passes_validation": True},
    "source": {
        "source_type": "TRACE",
        "source_data": {
            "trace_id": "tr-abc123",
            "environment": "production",
            "user_segment": "enterprise",
            "timestamp": "2024-11-01T10:30:00Z",
            "session_id": "sess-xyz789",
            "feedback_score": 0.95,
        },
    },
}

# Access metadata after merging
dataset.merge_records([detailed_source])
df = dataset.to_df()
# source_data preserved for analysis

```

## Search Filter Reference[​](#search-filter-reference "Direct link to Search Filter Reference")

Use these fields in your filter strings. **Note:** The fluent API returns a `PagedList` that can be iterated directly - pagination is handled automatically when you iterate over the results.

| Field              | Type      | Example                               |
| ------------------ | --------- | ------------------------------------- |
| `name`             | string    | `name = 'production_tests'`           |
| `tags.<key>`       | string    | `tags.status = 'validated'`           |
| `created_by`       | string    | `created_by = 'alice@company.com'`    |
| `last_updated_by`  | string    | `last_updated_by = 'bob@company.com'` |
| `created_time`     | timestamp | `created_time > 1698800000000`        |
| `last_update_time` | timestamp | `last_update_time > 1698800000000`    |

### Filter Operators[​](#filter-operators "Direct link to Filter Operators")

* `=`, `!=`: Exact match
* `LIKE`, `ILIKE`: Pattern matching with `%` wildcard (ILIKE is case-insensitive)
* `>`, `<`, `>=`, `<=`: Numeric/timestamp comparison
* `AND`: Combine conditions (OR is not currently supported for evaluation datasets)

python

```python
# Complex filter example
datasets = search_datasets(
    filter_string="""
        tags.status = 'production'
        AND name LIKE '%customer%'
        AND created_time > 1698800000000
    """,
    order_by=["last_update_time DESC"],
)

```

## Using the Client API[​](#using-the-client-api "Direct link to Using the Client API")

For applications and advanced use cases, you can also use the `MlflowClient` API which provides the same functionality with an object-oriented interface:

* Create Dataset
* Get Dataset
* Search Datasets
* Manage Tags
* Delete Dataset

python

```python
from mlflow import MlflowClient

client = MlflowClient()

# Create a dataset
dataset = client.create_dataset(
    name="customer_support_qa",
    experiment_id=["0"],
    tags={"version": "1.0", "team": "ml-platform"},
)

```

python

```python
# Get a dataset by ID
dataset = client.get_dataset(dataset_id="d-7f2e3a9b8c1d4e5f6a7b8c9d0e1f2a3b")

# Access properties
print(f"Dataset: {dataset.name}")
print(f"Records: {len(dataset.records)}")

```

python

```python
# Search for datasets
datasets = client.search_datasets(
    experiment_ids=["0"],
    filter_string="tags.status = 'validated'",
    order_by=["created_time DESC"],
    max_results=50,
)

for dataset in datasets:
    print(f"{dataset.name}: {dataset.dataset_id}")

```

python

```python
# Set tags
client.set_dataset_tags(
    dataset_id=dataset.dataset_id, tags={"status": "production", "validated": "true"}
)

# Delete a tag
client.delete_dataset_tag(dataset_id=dataset.dataset_id, key="deprecated")

```

python

```python
# Delete a dataset
client.delete_dataset(dataset_id=dataset.dataset_id)

```

The client API provides the same capabilities as the fluent API but is better suited for:

* Production applications that need explicit client management
* Scenarios requiring custom tracking URIs or authentication
* Integration with existing MLflow client-based workflows

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [End-to-End Workflow](/mlflow-website/docs/latest/genai/datasets/end-to-end-workflow.md)

[Learn the complete evaluation-driven development workflow from app building to production](/mlflow-website/docs/latest/genai/datasets/end-to-end-workflow.md)

[View complete workflow →](/mlflow-website/docs/latest/genai/datasets/end-to-end-workflow.md)

### [Run Evaluations](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Use your datasets to systematically evaluate and improve your GenAI applications](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Start evaluating →](/mlflow-website/docs/latest/genai/eval-monitor.md)

### [Define Expectations](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Learn how to add ground truth expectations to your test data for quality validation](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Set expectations →](/mlflow-website/docs/latest/genai/assessments/expectations.md)

### [Capture Traces](/mlflow-website/docs/latest/genai/tracing.md)

[Instrument your applications to capture production data for building datasets](/mlflow-website/docs/latest/genai/tracing.md)

[Enable tracing →](/mlflow-website/docs/latest/genai/tracing.md)
