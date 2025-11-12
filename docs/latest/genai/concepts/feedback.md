# Feedback Concepts

## What is Feedback?[​](#what-is-feedback "Direct link to What is Feedback?")

**Feedback** in MLflow represents the result of any quality assessment performed on your GenAI application outputs. It provides a standardized way to capture evaluations, whether they come from automated systems, LLM judges, or human reviewers.

Feedback serves as the bridge between running your application and understanding its quality, enabling you to systematically track performance across different dimensions like correctness, relevance, safety, and adherence to guidelines.

![Feedbacks attached to traces](/mlflow-website/docs/latest/images/llms/tracing/genai-human-feedback.png)

Feedbacks attached to traces

## Use Cases[​](#use-cases "Direct link to Use Cases")

#### Manual Quality Checks

Manual quality checks are important for ensuring the quality of your GenAI application. For example, you can attach a feedback to indicate the hallucination in the response and compare quality between different models.

#### End-User Feedbacks

Feedbacks from end-users are precious for improving the quality of your GenAI application. By storing feedbacks on your traces, you can easily monitor the user satisfaction of your application over time.

#### LLM Judge Evaluation

LLM judges are powerful tools for systematically running quality checks at scale. When using MLflow's GenAI Evaluation, Feedbacks from LLM judges are attached to the traces, enabling you to track evaluation results in the unified way as manual quality checks.

#### Collaborative Annotation

Quality checks are often performed by multiple annotators to ensure the robustness of the output. MLflow tracks metadata and revision history of the feedbacks and supports aggregation of feedbacks from multiple annotators.

## Core Structure[​](#core-structure "Direct link to Core Structure")

![Feedback Architecture](/mlflow-website/docs/latest/assets/images/feedback_architecture-4294fe94ea1d82f20b5b16edddeab840.png)

Feedback is often created by different sources, such as human annotators, LLM judges, or real user's feedback in the application. The [`Feedback`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Feedback) object in MLflow is a standard container for storing these signals along with metadata to track how they are created. Feedbacks are associated with a Trace, or a particular Span in the Trace.

## Feedback Object Schema[​](#feedback-object-schema "Direct link to Feedback Object Schema")

| Field                 | Type                        | Description                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                | `str`                       | A string identifying the specific quality aspect being assessed                                                                                                                                                                                                                                                                                                                       |
| `value`               | `Any`                       | The actual feedback value, which can be<br /><br />- Numeric scores (e.g., `0.0` to `1.0`, `1` to `5`)<br />- Boolean values (`True`/`False`)<br />- Categorical labels (e.g., `"PASS"`, `"FAIL"`, `"EXCELLENT"`)<br />- Structured data (e.g., `{"score": 0.8, "confidence": 0.9}`)                                                                                                  |
| `rationale`           | `str`                       | A string explaining why the feedback is given to the trace.                                                                                                                                                                                                                                                                                                                           |
| `source`              | `AssessmentSource`          | The source of the feedback, composed of the type of the source and ID.<br /><br />- `HUMAN`: Represents a human review. ID can be the identifier for the annotator, such as name, account, email, etc.<br />- `LLM_JUDGE`: Represents an LLM-based evaluation. ID may be the name of the judge model e.g., "openai:/4o-mini".<br />- `CODE`: Represents any other programmatic check. |
| `error`               | `Optional[AssessmentError]` | An optional error associated with the feedback. This is used to indicate that the feedback was not processed successfully, for example, an exception from the LLM judge execution.                                                                                                                                                                                                    |
| `metadata`            | `Optional[dict[str, str]]`  | Optional key-value pairs associated with the feedback.                                                                                                                                                                                                                                                                                                                                |
| `create_time_ms`      | `int`                       | The timestamp of when the feedback is created, in milliseconds.                                                                                                                                                                                                                                                                                                                       |
| `last_update_time_ms` | `int`                       | The timestamp of when the feedback is updated, in milliseconds.                                                                                                                                                                                                                                                                                                                       |
| `trace_id`            | `str`                       | The ID of the trace that the feedback is attached to.                                                                                                                                                                                                                                                                                                                                 |
| `span_id`             | `Optional[str]`             | The ID of the span that the feedback is attached to, if it is associated with a particular span in the trace. For example, you can give a feedback to the specific retriever output in the RAG application.                                                                                                                                                                           |

## Feedback Examples[​](#feedback-examples "Direct link to Feedback Examples")

**Human Feedback for Hallucination in the Response**

json

```json
{
    "name": "hallucination",
    "value": false,
    "rationale": "The response is factual and does not contain any hallucinations.",
    "source": {
        "source_type": "HUMAN",
        "source_id": "john@example.com"
    }
}

```

**LLM Judge Feedback for Factual Accuracy**

json

```json
{
    "name": "factual_accuracy",
    "value": 0.85,
    "rationale": "The response correctly identifies 3 out of 4 key facts about MLflow, but incorrectly states the founding year.",
    "source": {
        "source_type": "LLM_JUDGE",
        "source_id": "openai:/4o-mini"
    },
    "metadata": {
        # Store link to the prompt used for the judge, registered in MLflow Prompt Registry
        "judge_prompt": "prompts:factual_accuracy_judge/1"
    }
}

```

**Error Feedback from LLM Judge (Rate Limit Exceeded)**

json

```json
{
    "name": "safety",
    "error": {
        "error_code": "RATE_LIMIT_EXCEEDED",
        "error_message": "Rate limit for the judge exceeded.",
        "stack_trace": "..."
    },
    "source": {
        "source_type": "LLM_JUDGE",
        "source_id": "openai:/4o-mini"
    }
}

```

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Feedback Guide](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Complete guide for using mlflow.log\_feedback with practical examples and code samples](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[View the feedback guide →](/mlflow-website/docs/latest/genai/assessments/feedback.md)

### [Expectations Concepts](/mlflow-website/docs/latest/genai/concepts/expectations.md)

[Learn how to define ground truth expectations for comprehensive evaluation](/mlflow-website/docs/latest/genai/concepts/expectations.md)

[Learn about expectations →](/mlflow-website/docs/latest/genai/concepts/expectations.md)

### [Ground Truth Expectations](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Understand how to define expected outputs for comprehensive evaluation](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Learn about expectations →](/mlflow-website/docs/latest/genai/assessments/expectations.md)
