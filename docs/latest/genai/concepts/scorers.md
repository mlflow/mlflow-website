# Scorer Concepts

## What are Scorers?[​](#what-are-scorers "Direct link to What are Scorers?")

**Scorers** in MLflow are evaluation functions that assess the quality of your GenAI application outputs. They provide a systematic way to measure performance across different dimensions like correctness, relevance, safety, and adherence to guidelines.

Scorers transform subjective quality assessments into measurable metrics, enabling you to track performance, compare models, and ensure your applications meet quality standards. They range from simple rule-based checks to sophisticated LLM judges that can evaluate nuanced aspects of language generation.

## Use Cases[​](#use-cases "Direct link to Use Cases")

#### Automated Quality Assessment

Replace manual review processes with automated scoring that can evaluate thousands of outputs consistently and at scale, using either deterministic rules or LLM-based evaluation.

#### Safety & Compliance Validation

Systematically check for harmful content, bias, PII leakage, and regulatory compliance. Ensure your applications meet organizational and legal standards before deployment.

#### A/B Testing & Model Comparison

Compare different models, prompts, or configurations using consistent evaluation criteria. Make data-driven decisions about which approach performs best for your use case.

#### Continuous Quality Monitoring

Track quality metrics over time in production, detect degradations early, and maintain high standards as your application evolves and scales.

## Types of Scorers[​](#types-of-scorers "Direct link to Types of Scorers")

MLflow provides several types of scorers to address different evaluation needs:

#### Agent-as-a-Judge

Autonomous agents that analyze execution traces to evaluate not just outputs, but the entire process. They can assess tool usage, reasoning chains, and error handling.

#### Human-Aligned Judges

LLM judges that have been aligned with human feedback using the built-in align() method to match your specific quality standards. These provide the consistency of automation with the nuance of human judgment.

#### LLM-based Scorers (LLM-as-a-Judge)

Use large language models to evaluate subjective qualities like helpfulness, coherence, and style. These scorers can understand context and nuance that rule-based systems miss.

#### Code-based Scorers

Custom Python functions for deterministic evaluation. Perfect for metrics that can be calculated algorithmically like ROUGE scores, exact match, or custom business logic.

## Scorer Output Structure[​](#scorer-output-structure "Direct link to Scorer Output Structure")

All scorers in MLflow produce standardized output that integrates seamlessly with the evaluation framework. Scorers return a [`mlflow.entities.Feedback()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Feedback) object containing:

| Field       | Type             | Description                                                                |
| ----------- | ---------------- | -------------------------------------------------------------------------- |
| `name`      | `str`            | Unique identifier for the scorer (e.g., "correctness", "safety")           |
| `value`     | `Any`            | The evaluation result - can be numeric, boolean, or categorical            |
| `rationale` | `Optional[str]`  | Explanation of why this score was given (especially useful for LLM judges) |
| `metadata`  | `Optional[dict]` | Additional information about the evaluation (confidence, sub-scores, etc.) |
| `error`     | `Optional[str]`  | Error message if the scorer failed to evaluate                             |

## Common Scorer Patterns[​](#common-scorer-patterns "Direct link to Common Scorer Patterns")

MLflow's scorer system is highly flexible, supporting everything from simple rule-based checks to sophisticated AI agents that analyze entire execution traces. The examples below demonstrate the breadth of evaluation capabilities available - from detecting inefficiencies in multi-step workflows to assessing text readability, measuring response latency, and ensuring output quality. Each pattern can be customized to your specific use case and combined with others for comprehensive evaluation.

* Agent-as-a-Judge (Trace Analysis)
* LLM Judge (Field-Based)
* Reading Level Assessment
* Language Perplexity Scoring
* Response Latency Tracking

python

```python
from mlflow.genai.judges import make_judge
import mlflow

# Create an Agent-as-a-Judge that analyzes execution patterns
from typing import Literal

efficiency_judge = make_judge(
    name="efficiency_analyzer",
    instructions=(
        "Analyze the {{ trace }} for inefficiencies.\n\n"
        "Check for:\n"
        "- Redundant API calls or database queries\n"
        "- Sequential operations that could be parallelized\n"
        "- Unnecessary data processing\n\n"
        "Rate as: 'efficient', 'acceptable', or 'inefficient'"
    ),
    feedback_value_type=Literal["efficient", "acceptable", "inefficient"],
    model="anthropic:/claude-opus-4-1-20250805",
)

# Example: RAG application with retrieval and generation
from mlflow.entities import SpanType
import time


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_context(query: str):
    # Simulate vector database retrieval
    time.sleep(0.5)  # Retrieval latency
    return [
        {"doc": "MLflow is an open-source platform", "score": 0.95},
        {"doc": "It manages the ML lifecycle", "score": 0.89},
        {"doc": "Includes tracking and deployment", "score": 0.87},
    ]


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_user_history(user_id: str):
    # Another retrieval that could be parallelized
    time.sleep(0.5)  # Could run parallel with above
    return {"previous_queries": ["What is MLflow?", "How to log models?"]}


@mlflow.trace(span_type=SpanType.LLM)
def generate_response(query: str, context: list, history: dict):
    # Simulate LLM generation
    return f"Based on context about '{query}': MLflow is a platform for ML lifecycle management."


@mlflow.trace(span_type=SpanType.AGENT)
def rag_agent(query: str, user_id: str):
    # Sequential operations that could be optimized
    context = retrieve_context(query)
    history = retrieve_user_history(user_id)  # Could be parallel with above
    response = generate_response(query, context, history)
    return response


# Run the RAG agent
result = rag_agent("What is MLflow?", "user123")
trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id)

# Judge analyzes the trace to identify inefficiencies
feedback = efficiency_judge(trace=trace)
print(f"Efficiency: {feedback.value}")
print(f"Analysis: {feedback.rationale}")

```

python

```python
from mlflow.genai.judges import make_judge

correctness_judge = make_judge(
    name="correctness",
    instructions=(
        "Evaluate if the response in {{ outputs }} "
        "correctly answers the question in {{ inputs }}."
    ),
    feedback_value_type=bool,
    model="anthropic:/claude-opus-4-1-20250805",
)

# Example usage
feedback = correctness_judge(
    inputs={"question": "What is MLflow?"},
    outputs={
        "response": "MLflow is an open-source platform for ML lifecycle management."
    },
)
print(f"Correctness: {feedback.value}")

```

python

```python
import textstat
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback


@scorer
def reading_level(outputs: str) -> Feedback:
    """Evaluate text complexity using Flesch Reading Ease."""
    score = textstat.flesch_reading_ease(outputs)

    if score >= 60:
        level = "easy"
        rationale = f"Reading ease score of {score:.1f} - accessible to most readers"
    elif score >= 30:
        level = "moderate"
        rationale = f"Reading ease score of {score:.1f} - college level complexity"
    else:
        level = "difficult"
        rationale = f"Reading ease score of {score:.1f} - expert level required"

    return Feedback(value=level, rationale=rationale, metadata={"score": score})

```

python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from mlflow.genai.scorers import scorer


@scorer
def perplexity_score(outputs: str) -> float:
    """Calculate perplexity to measure text quality and coherence."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    inputs = tokenizer(outputs, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    perplexity = torch.exp(outputs.loss).item()
    return perplexity  # Lower is better - indicates more natural text

```

python

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback, Trace


@scorer
def response_time(trace: Trace) -> Feedback:
    """Evaluate response time from trace spans."""
    root_span = trace.data.spans[0]
    latency_ms = (root_span.end_time - root_span.start_time) / 1e6

    if latency_ms < 100:
        value = "fast"
    elif latency_ms < 500:
        value = "acceptable"
    else:
        value = "slow"

    return Feedback(
        value=value,
        rationale=f"Response took {latency_ms:.0f}ms",
        metadata={"latency_ms": latency_ms},
    )

```

## Judge Alignment[​](#judge-alignment "Direct link to Judge Alignment")

One of the most powerful features of MLflow scorers is the ability to **align LLM judges with human preferences**. This transforms generic evaluation models into domain-specific experts that understand your unique quality standards.

### How Alignment Works[​](#how-alignment-works "Direct link to How Alignment Works")

Judge alignment uses human feedback to improve the accuracy and consistency of LLM-based scorers:

python

```python
from mlflow.genai.judges import make_judge
import mlflow

# Create an initial judge
quality_judge = make_judge(
    name="quality",
    instructions="Evaluate if {{ outputs }} meets quality standards for {{ inputs }}.",
    feedback_value_type=bool,
    model="anthropic:/claude-opus-4-1-20250805",
)

# Collect traces with both judge assessments and human feedback
traces_with_feedback = mlflow.search_traces(
    experiment_ids=[experiment_id], max_results=20  # Minimum 10 required for alignment
)

# Align the judge with human preferences (uses default DSPy-SIMBA optimizer)
aligned_judge = quality_judge.align(traces_with_feedback)

# The aligned judge now better matches your team's quality standards
feedback = aligned_judge(inputs={"query": "..."}, outputs={"response": "..."})

```

### Key Benefits of Alignment[​](#key-benefits-of-alignment "Direct link to Key Benefits of Alignment")

* **Domain Expertise**: Judges learn your specific quality criteria from expert feedback
* **Consistency**: Aligned judges apply standards uniformly across evaluations
* **Cost Efficiency**: Once aligned, smaller/cheaper models can match expert judgment
* **Continuous Improvement**: Re-align as your standards evolve

### The Plugin Architecture[​](#the-plugin-architecture "Direct link to The Plugin Architecture")

MLflow's alignment system uses a plugin architecture, allowing you to create custom optimizers by extending the [AlignmentOptimizer](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.judges.base.AlignmentOptimizer) base class:

python

```python
from mlflow.genai.judges.base import AlignmentOptimizer


class CustomOptimizer(AlignmentOptimizer):
    def align(self, judge, traces):
        # Your custom alignment logic
        return improved_judge


# Use your custom optimizer
aligned_judge = quality_judge.align(traces, CustomOptimizer())

```

## Integration with MLflow Evaluation[​](#integration-with-mlflow-evaluation "Direct link to Integration with MLflow Evaluation")

Scorers are the building blocks of MLflow's evaluation framework. They integrate seamlessly with `mlflow.genai.evaluate()`:

python

```python
import mlflow
import pandas as pd

# Your test data
test_data = pd.DataFrame(
    [
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": {
                "response": "MLflow is an open-source platform for ML lifecycle management."
            },
            "expectations": {
                "ground_truth": "MLflow is an open-source platform for managing the ML lifecycle"
            },
        },
        {
            "inputs": {"question": "How do I track experiments?"},
            "outputs": {
                "response": "Use mlflow.start_run() to track experiments in MLflow."
            },
            "expectations": {
                "ground_truth": "Use mlflow.start_run() to track experiments"
            },
        },
    ]
)


# Your application (optional if data already has outputs)
def my_app(inputs):
    # Your model logic here
    return {"response": f"Answer to: {inputs['question']}"}


# Evaluate with multiple scorers
results = mlflow.genai.evaluate(
    data=test_data,
    # predict_fn is optional if data already has outputs
    scorers=[
        correctness_judge,  # LLM judge from above
        reading_level,  # Custom scorer from above
    ],
)

# Access evaluation metrics
print(f"Correctness: {results.metrics.get('correctness/mean', 'N/A')}")
print(f"Reading Level: {results.metrics.get('reading_level/mode', 'N/A')}")

```

## Best Practices[​](#best-practices "Direct link to Best Practices")

1. **Choose the Right Scorer Type**

   * Use code-based scorers for objective, deterministic metrics
   * Use LLM judges for subjective qualities requiring understanding
   * Use Agent-as-a-Judge for evaluating complex multi-step processes

2. **Combine Multiple Scorers**

   * No single metric captures all aspects of quality
   * Use a portfolio of scorers to get comprehensive evaluation
   * Balance efficiency (fast code-based) with depth (LLM and Agent judges)

3. **Align with Human Judgment**

   * Validate that your scorers correlate with human quality assessments
   * Use human feedback to improve LLM and Agent judge instructions
   * Consider using human-aligned judges for critical evaluations

4. **Monitor Scorer Performance**

   * Track scorer execution time and costs
   * Monitor for scorer failures and handle gracefully
   * Regularly review scorer outputs for consistency

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [LLM-based Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge.md)

[Learn about using LLMs as judges for evaluation](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge.md)

[Explore LLM judges →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge.md)

### [Judge Alignment](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment.md)

[Align judges with human feedback for domain expertise](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment.md)

[Learn alignment →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment.md)

### [Code-based Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md)

[Create custom Python functions for evaluation](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md)

[Build custom scorers →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md)

### [Evaluation Guide](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)

[Learn how to run comprehensive evaluations](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)

[Start evaluating →](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)
