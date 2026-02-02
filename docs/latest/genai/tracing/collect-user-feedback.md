# Collect User Feedback

Capturing user feedback is critical for understanding the real-world quality of your GenAI application. MLflow's Feedback API provides a structured, standardized approach to collecting, storing, and analyzing user feedback directly within your traces.

## Adding Feedback with MLflow UI[​](#adding-feedback-with-mlflow-ui "Direct link to Adding Feedback with MLflow UI")

[](/mlflow-website/docs/latest/images/llms/tracing/logging-feedback.mp4)

## Adding Feedback with API[​](#adding-feedback-with-api "Direct link to Adding Feedback with API")

To annotate traces with feedback programmatically, use the [`mlflow.log_feedback()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_feedback) API.

python

```python
import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

mlflow.log_feedback(
    trace_id="<your trace id>",
    name="user_satisfaction",
    value=True,
    rationale="User indicated response was helpful",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="user_123"
    ),
)

```

If you have a `Feedback` object already (e.g., a response from LLM-as-a-Judge), you can log it directly using the [`mlflow.log_assessment()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_assessment) API. This is equivalent to using the

[`mlflow.log_feedback()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_feedback) API with unpacked fields.

python

```python
import mlflow
from mlflow.genai.judges import make_judge
from typing import Literal

coherence_judge = make_judge(
    name="coherence",
    instructions=(
        "Evaluate if the response is coherent, maintaining a constant tone "
        "and following a clear flow of thoughts/concepts"
        "Trace: {{ trace }}\n"
    ),
    feedback_value_type=Literal["coherent", "somewhat coherent", "incoherent"],
    model="anthropic:/claude-opus-4-1-20250805",
)

trace = mlflow.get_trace("<your trace id>")
feedback = coherence_judge(trace=trace)

mlflow.log_assessment(trace_id="<your trace id>", assessment=feedback)
# Equivalent to log_feedback(trace_id="<trace_id>", name=feedback.name, value=feedback.value, ...)"

```

## Supported Value Types[​](#supported-value-types "Direct link to Supported Value Types")

MLflow feedback supports various formats to match your application's needs:

| Feedback Type | Description                    | Example Use Cases                   |
| ------------- | ------------------------------ | ----------------------------------- |
| **Boolean**   | Simple `True`/`False` feedback | Thumbs up/down, correct/incorrect   |
| **Numeric**   | Integer or float ratings       | 1-5 star ratings, confidence scores |
| **Text**      | Free-form text feedback        | Detailed quality breakdowns         |

## Supported Feedback Sources[​](#supported-feedback-sources "Direct link to Supported Feedback Sources")

The `source` field of the feedback provides information about where the feedback came from.

| Source Type    | Description           | Example Use Cases                        |
| -------------- | --------------------- | ---------------------------------------- |
| **HUMAN**      | Human feedback        | User thumbs up/down, correct/incorrect   |
| **LLM\_JUDGE** | LLM-based feedback    | Score traces with an LLM-based judge     |
| **CODE**       | Programmatic feedback | Score traces with a programmatic checker |

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Feedback Concepts](/mlflow-website/docs/latest/genai/concepts/feedback.md)

[Deep dive into feedback architecture, schema, and best practices](/mlflow-website/docs/latest/genai/concepts/feedback.md)

[Learn concepts →](/mlflow-website/docs/latest/genai/concepts/feedback.md)

### [Search and Analyze Traces](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

[Query traces with feedback data and analyze patterns for quality insights](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

[Start analyzing →](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Learn how to evaluate traces with feedback data and analyze patterns for quality insights](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Start evaluating →](/mlflow-website/docs/latest/genai/eval-monitor.md)
