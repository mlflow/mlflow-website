# Evaluation Dataset Concepts

SQL Backend Required

Evaluation Datasets require an MLflow Tracking Server with a **[SQL backend](/mlflow-website/docs/latest/self-hosting/architecture/backend-store.md#types-of-backend-stores)** (PostgreSQL, MySQL, SQLite, or MSSQL). This feature is **not available** in FileStore (local mode) due to the relational data requirements for managing dataset records, associations, and schema evolution.

## What are Evaluation Datasets?[​](#what-are-evaluation-datasets "Direct link to What are Evaluation Datasets?")

**Evaluation Datasets** in MLflow provide a structured way to organize and manage test data for GenAI applications. They serve as centralized repositories for test inputs, optional test outputs, expected outputs (expectations), and evaluation results, enabling systematic quality assessment across your AI development lifecycle.

Unlike static test files, evaluation datasets are **living validation collections** designed to grow and evolve with your application. Records can be continuously added from production traces, manual curation, or programmatic generation.

They can be viewed directly within the MLflow UI.

![Evaluation Datasets Video](/mlflow-website/docs/latest/images/eval-datasets.gif)

## Core Components[​](#core-components "Direct link to Core Components")

Evaluation datasets are composed of several key elements that work together to provide comprehensive test management:

#### Dataset Records

Individual test cases containing inputs (what goes into your model), expectations (what should come out), optional outputs (what your application returned), and metadata about the source and tags for organization.

#### Schema & Profile

Automatically computed structure and statistics of your dataset. Schema tracks field names and types across records, while profile provides statistical summaries.

#### Expectations

Ground truth values and quality criteria that define correct behavior. These are the set of standards against which your model outputs are evaluated.

#### Experiment Association

Links to MLflow experiments enable tracking which datasets were used for which model evaluations, providing full lineage and organizational control.

## Dataset Object Schema[​](#dataset-object-schema "Direct link to Dataset Object Schema")

The [`mlflow.entities.EvaluationDataset()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.EvaluationDataset) object contains the following fields:

| Field              | Type                  | Description                                                              |
| ------------------ | --------------------- | ------------------------------------------------------------------------ |
| `dataset_id`       | `str`                 | Unique identifier for the dataset (format: `d-{32 hex chars}`)           |
| `name`             | `str`                 | Human-readable name for the dataset                                      |
| `digest`           | `str`                 | Content hash for data integrity verification                             |
| `records`          | `list[DatasetRecord]` | The actual test data records containing inputs and expectations          |
| `schema`           | `Optional[str]`       | JSON string describing the structure of records (automatically computed) |
| `profile`          | `Optional[str]`       | JSON string containing statistical information about the dataset         |
| `tags`             | `dict[str, str]`      | Key-value pairs for organizing and categorizing datasets                 |
| `experiment_ids`   | `list[str]`           | List of MLflow experiment IDs this dataset is associated with            |
| `created_time`     | `int`                 | Timestamp when the dataset was created (milliseconds)                    |
| `last_update_time` | `int`                 | Timestamp of the last modification (milliseconds)                        |
| `created_by`       | `Optional[str]`       | User who created the dataset (auto-detected from tags)                   |
| `last_updated_by`  | `Optional[str]`       | User who last modified the dataset                                       |

## Record Structure[​](#record-structure "Direct link to Record Structure")

Each record in an evaluation dataset represents a single test case with the following structure:

json

```json
{
    "inputs": {
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe",
        "temperature": 0.7
    },
    "outputs": {
        "answer": "The capital of France is Paris."
    },
    "expectations": {
        "name": "expected_answer",
        "value": "Paris",
    },
    "source": {
        "source_type": "HUMAN",
        "source_data": {
            "annotator": "geography_expert@company.com",
            "annotation_date": "2024-08-07"
        }
    },
    "tags": {
        "category": "geography",
        "difficulty": "easy",
        "validated": "true"
    }
}

```

### Record Fields[​](#record-fields "Direct link to Record Fields")

* **inputs** (required): The test input data that will be passed to your model or application
* **outputs** (optional): The actual outputs generated by your model (typically used for post-hoc evaluation)
* **expectations** (optional): The expected outputs or quality criteria that define correct behavior
* **source** (optional): Provenance information about how this record was created (automatically inferred if not provided)
* **tags** (optional): Metadata specific to this individual record for organization and filtering

### Record Identity and Deduplication[​](#record-identity-and-deduplication "Direct link to Record Identity and Deduplication")

Records are uniquely identified by a **hash of their inputs**. When merging records with [`mlflow.entities.EvaluationDataset.merge_records()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.EvaluationDataset.merge_records), if a record with identical inputs already exists, its expectations and tags are merged rather than creating a duplicate. This enables iterative refinement of test cases without data duplication.

## Schema Evolution[​](#schema-evolution "Direct link to Schema Evolution")

Dataset schemas automatically evolve as you add records with new fields. The `schema` property tracks all field names and types encountered across records, while `profile` maintains statistical summaries. This automatic adaptation means you can start with simple test cases and progressively add complexity without manual schema migrations.

When new fields are introduced in subsequent records, they're automatically incorporated into the schema. Existing records without those fields are handled gracefully during evaluation and analysis.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [SDK Guide](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

[Complete reference for creating and managing evaluation datasets via the MLflow SDK](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

[View SDK guide →](/mlflow-website/docs/latest/genai/datasets/sdk-guide.md)

### [End-to-End Workflow](/mlflow-website/docs/latest/genai/datasets/end-to-end-workflow.md)

[Learn the complete evaluation-driven development workflow from building to production](/mlflow-website/docs/latest/genai/datasets/end-to-end-workflow.md)

[See workflow →](/mlflow-website/docs/latest/genai/datasets/end-to-end-workflow.md)

### [Expectations](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Learn how to define ground truth and quality criteria for your test cases](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Understand expectations →](/mlflow-website/docs/latest/genai/assessments/expectations.md)
