# Collect User Feedback

Capturing user feedback is critical for understanding the real-world quality of your GenAI application. MLflow's **Feedback API** provides a structured, standardized approach to collecting, storing, and analyzing user feedback directly within your traces.

## Why Use MLflow Feedback for User Feedback?[​](#why-use-mlflow-feedback-for-user-feedback "Direct link to Why Use MLflow Feedback for User Feedback?")

#### Direct Trace Integration

Feedback is linked directly to specific application executions, creating an immediate connection between user reactions and system performance.

#### Structured Data Model

Standardized format with clear attribution and rationale ensures consistent feedback collection across your entire application.

#### Production Ready

Available in OSS MLflow 3.2.0+ with no external dependencies, designed for high-throughput production environments.

#### Complete Audit Trail

Track every feedback change with timestamps and user attribution, enabling comprehensive quality analysis over time.

## Step-by-Step Guide: Collecting User Feedback[​](#step-by-step-guide-collecting-user-feedback "Direct link to Step-by-Step Guide: Collecting User Feedback")

### 1. Set Up Your GenAI Application with Tracing[​](#1-set-up-your-genai-application-with-tracing "Direct link to 1. Set Up Your GenAI Application with Tracing")

First, create a simple application that automatically generates traces using MLflow's OpenAI autologging:

python

```python
import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType
import openai

# Enable automatic tracing for OpenAI calls
mlflow.openai.autolog()

# Initialize your LLM client
client = openai.OpenAI()


def ask_question(question):
    """Simple Q&A application with automatic tracing."""
    # This call is automatically traced by MLflow
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions clearly and concisely.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    return answer


# Generate some traces - each call creates a trace automatically
question = "What is machine learning?"
answer = ask_question(question)
print(f"Question: {question}")
print(f"Answer: {answer}")

# You can get the trace ID from the MLflow UI or search API
# For this example, we'll show how to collect feedback programmatically

```

### 2. Collect Simple Thumbs Up/Down Feedback[​](#2-collect-simple-thumbs-updown-feedback "Direct link to 2. Collect Simple Thumbs Up/Down Feedback")

Implement basic boolean feedback collection. In a real application, you'd get the trace\_id from your tracing system:

python

```python
def collect_thumbs_feedback(trace_id, is_helpful, user_id):
    """Collect simple thumbs up/down feedback from users."""
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_satisfaction",
        value=is_helpful,
        rationale="User indicated response was helpful"
        if is_helpful
        else "User indicated response was not helpful",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id=user_id
        ),
    )
    print(f"✓ Feedback recorded: {'👍' if is_helpful else '👎'}")


# Example: Collect feedback on a trace
trace_id = mlflow.get_last_active_trace_id()
collect_thumbs_feedback(trace_id, True, "user_123")

```

### 3. View Feedback in MLflow UI[​](#3-view-feedback-in-mlflow-ui "Direct link to 3. View Feedback in MLflow UI")

After collecting feedback, you can view it in the MLflow UI:

![Feedback in MLflow UI](/mlflow-website/docs/latest/assets/images/assessments_trace_detail_ui-d460aa8741ab1feca6d3fd1ac864bcd8.png)

The trace detail page shows all feedback attached to your traces, making it easy to analyze user satisfaction and identify patterns in your application's performance.

### 4. Adding and Updating Feedback via UI[​](#4-adding-and-updating-feedback-via-ui "Direct link to 4. Adding and Updating Feedback via UI")

Users can also provide feedback directly through the MLflow UI:

**Creating New Feedback:**

![Create Feedback](/mlflow-website/docs/latest/assets/images/add_feedback_ui-d41b6141727bf92f5d87568ae5522ae1.png)

**Adding Additional Feedback:**

![Additional Feedback](/mlflow-website/docs/latest/assets/images/additional_feedback_ui-55514604c28b292d5b976cb56a30773a.png)

This collaborative approach enables both programmatic feedback collection and manual review workflows.

## Feedback Value Types[​](#feedback-value-types "Direct link to Feedback Value Types")

MLflow feedback supports various formats to match your application's needs:

| Feedback Type   | Description                              | Example Use Cases                   |
| --------------- | ---------------------------------------- | ----------------------------------- |
| **Boolean**     | Simple `True`/`False` feedback           | Thumbs up/down, correct/incorrect   |
| **Numeric**     | Integer or float ratings                 | 1-5 star ratings, confidence scores |
| **Categorical** | String classifications                   | "Helpful", "Neutral", "Unhelpful"   |
| **Structured**  | Complex objects with multiple dimensions | Detailed quality breakdowns         |

## MLflow Feedback Collection Best Practices[​](#mlflow-feedback-collection-best-practices "Direct link to MLflow Feedback Collection Best Practices")

#### Start with Boolean Feedback

Use MLflow's boolean feedback type for simple thumbs up/down collection. Once you analyze patterns with MLflow's search APIs, expand to numeric ratings or structured feedback types.

#### Link Feedback to Fresh Traces

Collect feedback immediately after trace generation when the interaction context is available. MLflow's direct trace-feedback linkage ensures you always have the full execution context.

#### Use Consistent Naming Conventions

Standardize feedback names like 'user\_satisfaction' or 'quality\_rating' across traces. This enables MLflow's search and aggregation features to provide meaningful insights across your application.

#### Use Source Attribution Properly

Set meaningful source\_id values in AssessmentSource objects for tracking feedback providers. MLflow preserves complete audit trails with timestamps and source attribution.

#### Combine Programmatic and UI Collection

Use MLflow's API for automated collection and the UI for manual review. Both methods integrate seamlessly, allowing different teams to contribute feedback through their preferred interface.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Feedback API Guide](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Comprehensive guide to all feedback APIs with advanced patterns and examples](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[View API reference →](/mlflow-website/docs/latest/genai/assessments/feedback.md)

### [Feedback Concepts](/mlflow-website/docs/latest/genai/concepts/feedback.md)

[Deep dive into feedback architecture, schema, and best practices](/mlflow-website/docs/latest/genai/concepts/feedback.md)

[Learn concepts →](/mlflow-website/docs/latest/genai/concepts/feedback.md)

### [Search and Analyze Traces](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

[Query traces with feedback data and analyze patterns for quality insights](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

[Start analyzing →](/mlflow-website/docs/latest/genai/tracing/search-traces.md)
