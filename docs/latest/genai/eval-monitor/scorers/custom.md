# Custom Code-based Scorers

Custom scorers offer the ultimate flexibility to define precisely how your GenAI application's quality is measured. They provide the flexibility to define evaluation metrics tailored to your specific business use case, whether based on simple heuristics, advanced logic, or programmatic evaluations.

## Example Usage[​](#example-usage "Direct link to Example Usage")

To define a custom scorer, you can define a function that takes in the [input arguments](#input-format) and add the [@scorer](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.scorer) decorator to the function.

python

```
from mlflow.genai import scorer


@scorer
def exact_match(outputs: dict, expectations: dict) -> bool:
    return outputs == expectations["expected_response"]
```

To return richer information beyond primitive values, you can return a [Feedback](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Feedback) object.

python

```
from mlflow.entities import Feedback


@scorer
def is_short(outputs: dict) -> Feedback:
    score = len(outputs.split()) <= 5
    rationale = (
        "The response is short enough."
        if score
        else f"The response is not short enough because it has ({len(outputs.split())} words)."
    )
    return Feedback(value=score, rationale=rationale)
```

Then you can pass the functions directly to the [mlflow.genai.evaluate](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate) function, just like other predefined or LLM-based scorers.

python

```
import mlflow

eval_dataset = [
    {
        "inputs": {"question": "How many countries are there in the world?"},
        "outputs": "195",
        "expectations": {"expected_response": "195"},
    },
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": "The capital of France is Paris.",
        "expectations": {"expected_response": "Paris"},
    },
]

mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[exact_match, is_short],
)
```

![Code-based Scorers](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/scorers/code-scorers-results.png)

## Input Format[​](#input-format "Direct link to Input Format")

As input, custom scorers have access to:

* The `inputs` dictionary, derived from either the input dataset or MLflow post-processing from your trace.
* The `outputs` value, derived from either the input dataset or trace. If `predict_fn` is provided, the `outputs` value will be the return value of the `predict_fn`.
* The `expectations` dictionary, derived from the `expectations` field in the input dataset, or associated with the trace.
* The complete [MLflow trace](/mlflow-website/docs/latest/genai/concepts/trace.md), including spans, attributes, and outputs.

python

```
@scorer
def my_scorer(
    *,
    inputs: dict[str, Any],
    outputs: Any,
    expectations: dict[str, Any],
    trace: Trace,
) -> float | bool | str | Feedback | list[Feedback]:
    # Your evaluation logic here
    ...
```

All parameters are **optional**; declare only what your scorer needs:

text

```
# ✔️ All of these signatures are valid for scorers
def my_scorer(inputs, outputs, expectations, trace) -> bool:
def my_scorer(inputs, outputs) -> str:
def my_scorer(outputs, expectations) -> Feedback:
def my_scorer(trace) -> list[Feedback]:

# 🔴 Additional parameters are not allowed
def my_scorer(inputs, outputs, expectations, trace, additional_param) -> float
```

Where do these values come from?

When running `mlflow.genai.evaluate()`, the inputs, outputs, and expectations parameters can be specified in the data argument, or parsed from the trace. See [How Scorers Work](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#how-scorers-work) for more details.

## Return Types[​](#return-types "Direct link to Return Types")

Scorers can return different types depending on your evaluation needs:

### Simple values[​](#simple-values "Direct link to Simple values")

Return primitive values for straightforward pass/fail or numeric assessments.

* Pass/fail strings: `"yes"` or `"no"` render as

  Pass

  or

  Fail

  in the UI

* Boolean values: `True` or `False` for binary evaluations

* Numeric values: Integers or floats for scores, counts, or measurements

### Rich feedback[​](#rich-feedback "Direct link to Rich feedback")

Return [Feedback](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Feedback) objects for detailed assessments with additional metadata such as explanation, source info, and error summary.

python

```
from mlflow.entities import Feedback, AssessmentSource


@scorer
def content_quality(outputs):
    return Feedback(
        value=0.85,  # Can be numeric, boolean, or string
        rationale="Clear and accurate, minor grammar issues",
        # Optional: source of the assessment. Several source types are supported,
        # such as "HUMAN", "CODE", "LLM_JUDGE".
        source=AssessmentSource(source_type="CODE", source_id="grammar_checker_v1"),
        # Optional: additional metadata about the assessment.
        metadata={
            "annotator": "me@example.com",
        },
    )
```

Multiple feedback objects can be returned as a list. Each feedback object will be displayed as a separate metric in the evaluation results.

text

```
@scorer
def comprehensive_check(inputs, outputs):
    return [
        Feedback(name="relevance", value=True, rationale="Directly addresses query"),
        Feedback(name="tone", value="professional", rationale="Appropriate for audience"),
        Feedback(name="length", value=150, rationale="Word count within limits")
    ]
```

## Parsing Traces for Scoring[​](#parsing-traces-for-scoring "Direct link to Parsing Traces for Scoring")

Important: Agent-as-a-Judge Scorers Require Active Traces

Scorers that accept a `trace` parameter **cannot be used with pandas DataFrames**. They require actual execution traces from your application.

If you need to evaluate static data (e.g., a CSV file with pre-generated responses), use field-based scorers that work with `inputs`, `outputs`, and `expectations` parameters only.

Scorers have access to the complete MLflow traces, including spans, attributes, and outputs, allowing you to evaluate the agent's behavior precisely, not just the final output. The [`Trace.search_spans`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Trace.search_spans) API is a powerful way to retrieve such intermediate information from the trace.

Open the tabs below to see examples of custom scorers that evaluate the detailed behavior of agents by parsing the trace.

* Retrieved Document Recall
* Tool Call Trajectory
* Sub-Agents Routing

### Example 1: Evaluating Retrieved Documents Recall[​](#example-1-evaluating-retrieved-documents-recall "Direct link to Example 1: Evaluating Retrieved Documents Recall")

python

```
from mlflow.entities import SpanType, Trace
from mlflow.genai import scorer


@scorer
def retrieved_document_recall(trace: Trace, expectations: dict) -> Feedback:
    # Search for retriever spans in the trace
    retriever_spans = trace.search_spans(span_type=SpanType.RETRIEVER)

    # If there are no retriever spans
    if not retriever_spans:
        return Feedback(
            value=0,
            rationale="No retriever span found in the trace.",
        )

    # Gather all retrieved document URLs from the retriever spans
    all_document_urls = []
    for span in retriever_spans:
        all_document_urls.extend([document["doc_uri"] for document in span.outputs])

    # Compute the recall
    true_positives = len(
        set(all_document_urls) & set(expectations["relevant_document_urls"])
    )
    expected_positives = len(expectations["relevant_document_urls"])
    recall = true_positives / expected_positives
    return Feedback(
        value=recall,
        rationale=f"Retrieved {true_positives} relevant documents out of {expected_positives} expected.",
    )
```

### Example 2: Evaluating Tool Call Trajectory[​](#example-2-evaluating-tool-call-trajectory "Direct link to Example 2: Evaluating Tool Call Trajectory")

python

```
from mlflow.entities import SpanType, Trace
from mlflow.genai import scorer


@scorer
def tool_call_trajectory(trace: Trace, expectations: dict) -> Feedback:
    # Search for tool call spans in the trace
    tool_call_spans = trace.search_spans(span_type=SpanType.TOOL)

    # Compare the tool trajectory with expectations
    actual_trajectory = [span.name for span in tool_call_spans]
    expected_trajectory = expectations["tool_call_trajectory"]

    if actual_trajectory == expected_trajectory:
        return Feedback(value=1, rationale="The tool call trajectory is correct.")
    else:
        return Feedback(
            value=0,
            rationale=(
                "The tool call trajectory is incorrect.\n"
                f"Expected: {expected_trajectory}.\n"
                f"Actual: {actual_trajectory}."
            ),
        )
```

### Example 3: Evaluating Sub-Agents Routing[​](#example-3-evaluating-sub-agents-routing "Direct link to Example 3: Evaluating Sub-Agents Routing")

python

```
from mlflow.entities import SpanType, Trace
from mlflow.genai import scorer


@scorer
def is_routing_correct(trace: Trace, expectations: dict) -> Feedback:
    # Search for sub-agent spans in the trace
    sub_agent_spans = trace.search_spans(span_type=SpanType.AGENT)

    invoked_agents = [span.name for span in sub_agent_spans]
    expected_agents = expectations["expected_agents"]

    if invoked_agents == expected_agents:
        return Feedback(value=True, rationale="The sub-agents routing is correct.")
    else:
        return Feedback(
            value=False,
            rationale=(
                "The sub-agents routing is incorrect.\n"
                f"Expected: {expected_agents}.\n"
                f"Actual: {invoked_agents}."
            ),
        )
```

## Error handling[​](#error-handling "Direct link to Error handling")

When a scorer encounters an error, MLflow provides two approaches:

### Let exceptions propagate (recommended)[​](#let-exceptions-propagate-recommended "Direct link to Let exceptions propagate (recommended)")

The simplest approach is to let exceptions throw naturally. MLflow automatically captures the exception and creates a Feedback object with the error details:

python

```
import json
import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


@scorer
def is_valid_response(outputs: str) -> Feedback:
    # Let json.JSONDecodeError propagate if response isn't valid JSON
    data = json.loads(outputs)

    # Let KeyError propagate if required fields are missing
    summary = data["summary"]
    confidence = data["confidence"]

    return Feedback(value=True, rationale=f"Valid JSON with confidence: {confidence}")


# Run the scorer on invalid data that triggers exceptions
invalid_data = [
    {
        # Valid JSON
        "outputs": '{"summary": "this is a summary", "confidence": 0.95}'
    },
    {
        # Invalid JSON
        "outputs": "invalid json",
    },
    {
        # Missing required fields
        "outputs": '{"summary": "this is a summary"}'
    },
]

mlflow.genai.evaluate(
    data=invalid_data,
    scorers=[is_valid_response],
)
```

When an exception occurs, MLflow creates a [Feedback](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Feedback) with:

* `value`: None
* `error`: The exception details, such as exception object, error message, and stack trace

The error information will be displayed in the evaluation results. Open the corresponding row to see the error details.

![Scorer Error](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/scorers/scorer-error.png)

### Handle exceptions explicitly[​](#handle-exceptions-explicitly "Direct link to Handle exceptions explicitly")

For custom error handling or to provide specific error messages, catch exceptions and return a [Feedback](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Feedback) with None value and error details:

python

```
import json
from mlflow.entities import AssessmentError, Feedback


@scorer
def is_valid_response(outputs):
    try:
        data = json.loads(outputs)
        required_fields = ["summary", "confidence", "sources"]
        missing = [f for f in required_fields if f not in data]

        if missing:
            # Specify the AssessmentError object explicitly
            return Feedback(
                error=AssessmentError(
                    error_code="MISSING_REQUIRED_FIELDS",
                    error_message=f"Missing required fields: {missing}",
                ),
            )

        return Feedback(value=True, rationale="Valid JSON with all required fields")

    except json.JSONDecodeError as e:
        # Can pass exception object directly to the error parameter as well
        return Feedback(error=e)
```

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Evaluate Agents](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn how to evaluate AI agents with specialized techniques and scorers](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate production traces to understand and improve your AI application's behavior](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

### [Ground Truth Expectations](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Learn how to define and manage ground truth data for accurate evaluations](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Learn more →](/mlflow-website/docs/latest/genai/assessments/expectations.md)
