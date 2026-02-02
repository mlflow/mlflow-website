# Evaluate Conversations

Conversation evaluation enables you to assess entire conversation sessions rather than individual turns. This is essential for evaluating conversational AI systems where quality emerges over multiple interactions, such as user frustration patterns, conversation completeness, or overall dialogue coherence.

Experimental Feature

Multi-turn evaluation is experimental in MLflow 3.7.0. The API and behavior may change in future releases.

## Workflow[​](#workflow "Direct link to Workflow")

#### Tag traces with session IDs

Add session metadata to your traces to group related conversation turns together.

#### Search and retrieve session traces

Collect traces from your tracking server and MLflow will automatically group them by session.

#### Define conversation judges

Use built-in multi-turn judges or create custom ones to evaluate full conversations.

#### Run evaluation

Execute evaluation and analyze session-level metrics alongside individual turn metrics in MLflow UI.

## Overview[​](#overview "Direct link to Overview")

Traditional single-turn evaluation assesses each agent response independently. However, many important qualities can only be evaluated by examining the full conversation:

* **User Frustration**: Did the user become frustrated? Was it resolved?
* **Conversation Completeness**: Were all user questions answered by the end of the conversation?
* **Dialogue Coherence**: Does the conversation flow naturally?

Multi-turn evaluation addresses these needs by grouping traces into conversation sessions and applying judges that analyze the entire conversation history.

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

First, install the required packages by running the following command:

bash

```bash
pip install --upgrade 'mlflow[genai]>=3.7'

```

MLflow stores evaluation results in a tracking server. Connect your local environment to the tracking server by one of the following methods.

* Local (uv)
* Local (pip)
* Local (docker)

Install the Python package manager [uv](https://docs.astral.sh/uv/getting-started/installation/) (that will also install [`uvx` command](https://docs.astral.sh/uv/guides/tools/) to invoke Python tools without installing them).

Start a MLflow server locally.

shell

```shell
uvx mlflow server

```

**Python Environment**: Python 3.10+

Install the `mlflow` Python package via `pip` and start a MLflow server locally.

shell

```shell
pip install --upgrade 'mlflow[genai]'
mlflow server

```

MLflow provides a Docker Compose file to start a local MLflow server with a PostgreSQL database and a MinIO server.

shell

```shell
git clone --depth 1 --filter=blob:none --sparse https://github.com/mlflow/mlflow.git
cd mlflow
git sparse-checkout set docker-compose
cd docker-compose
cp .env.dev.example .env
docker compose up -d

```

Refer to the [instruction](https://github.com/mlflow/mlflow/tree/master/docker-compose/README.md) for more details (e.g., overriding the default environment variables).

## Quick Start[​](#quick-start "Direct link to Quick Start")

Multi-turn evaluation works by grouping traces into conversation sessions using the `mlflow.trace.session` metadata. When building your agent, you can set session IDs on traces to group them into conversations:

python

```python
import mlflow


@mlflow.trace
def my_chatbot(question, session_id):
    mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    return generate_response(question)

```

![Sessions View UI](/mlflow-website/docs/latest/images/genai/sessions-view-ui.png)

To evaluate conversations, [get traces from your experiment](/mlflow-website/docs/latest/genai/tracing/search-traces.md) and pass them to `mlflow.genai.evaluate`:

python

```python
from mlflow.genai.scorers import ConversationCompleteness, UserFrustration

# Get all traces
traces = mlflow.search_traces(
    experiment_ids=["<your-experiment-id>"],
    return_type="list",
)

# Evaluate all sessions - MLflow automatically groups by session ID
results = mlflow.genai.evaluate(
    data=traces,
    scorers=[
        ConversationCompleteness(),
        UserFrustration(),
    ],
)

```

**How it works:** MLflow automatically groups traces by their `mlflow.trace.session` metadata and sorts them chronologically by timestamp within each session. Multi-turn judges run once per session and analyze the complete conversation history. Multi-turn assessments are logged to the first trace (chronologically) in each session. You can use the Sessions tab to view session-level metrics for the entire conversation as well as trace-level metrics for individual turns.

## Multi-Turn Judges[​](#multi-turn-judges "Direct link to Multi-Turn Judges")

### Built-in Judges[​](#built-in-judges "Direct link to Built-in Judges")

MLflow provides built-in judges for evaluating conversations:

* **[ConversationCompleteness](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.ConversationCompleteness)**: Evaluates whether the agent addressed all user questions throughout the conversation (returns "yes" or "no")
* **[ConversationalGuidelines](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.ConversationalGuidelines)**: Evaluates whether the assistant's responses throughout the conversation comply with provided guidelines (returns "yes" or "no")
* **[KnowledgeRetention](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.KnowledgeRetention)**: Evaluates whether the assistant correctly retains information from earlier user inputs without contradiction or distortion (returns "yes" or "no")
* **[UserFrustration](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.UserFrustration)**: Detects and tracks user frustration patterns (returns "none", "resolved", or "unresolved")

See the [Built-in Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#multi-turn) page for detailed usage examples and API documentation.

### Custom Judges[​](#custom-judges "Direct link to Custom Judges")

You can create custom multi-turn judges using [make\_judge](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.judges.make_judge) with the `{{ conversation }}` template variable:

python

```python
from mlflow.genai.judges import make_judge
from typing import Literal

# Create a custom multi-turn judge
politeness_judge = make_judge(
    name="conversation_politeness",
    instructions=(
        "Analyze the {{ conversation }} and determine if the agent maintains "
        "a polite and professional tone throughout all interactions. "
        "Rate as 'consistently_polite', 'mostly_polite', or 'impolite'."
    ),
    feedback_value_type=Literal["consistently_polite", "mostly_polite", "impolite"],
    model="openai:/gpt-4o",
)

# Use in evaluation
results = mlflow.genai.evaluate(
    data=traces,
    scorers=[politeness_judge],
)

```

Conversation Template Variable

The `{{ conversation }}` variable injects the complete conversation history in a structured format.

The variable can only be used with `{{ expectations }}`, not with `{{ inputs }}`, `{{ outputs }}`, or `{{ trace }}`.

### Combining Single-Turn and Multi-Turn Judges[​](#combining-single-turn-and-multi-turn-judges "Direct link to Combining Single-Turn and Multi-Turn Judges")

You can use both single-turn and multi-turn judges in the same evaluation:

python

```python
from mlflow.genai.scorers import (
    ConversationCompleteness,
    UserFrustration,
    RelevanceToQuery,  # Single-turn scorer
)

results = mlflow.genai.evaluate(
    data=traces,
    scorers=[
        # Single-turn: evaluates each trace individually
        RelevanceToQuery(),
        # Multi-turn: evaluates entire sessions
        ConversationCompleteness(),
        UserFrustration(),
    ],
)

```

Single-turn judges run on every trace individually, while multi-turn judges run once per session and analyze the complete conversation history.

## Working with Specific Sessions[​](#working-with-specific-sessions "Direct link to Working with Specific Sessions")

If you need to evaluate specific sessions or filter traces, you can extract session IDs and retrieve traces for each:

python

```python
import mlflow

# Get all traces from your experiment
all_traces = mlflow.search_traces(
    experiment_ids=["<your-experiment-id>"],
    return_type="list",
)

# Extract unique session IDs
session_ids = set()
for trace in all_traces:
    session_id = trace.info.trace_metadata.get("mlflow.trace.session")
    if session_id:
        session_ids.add(session_id)

# Get traces for each session and combine
all_session_traces = []
for session_id in session_ids:
    session_traces = mlflow.search_traces(
        experiment_ids=["<your-experiment-id>"],
        filter_string=f"metadata.`mlflow.trace.session` = '{session_id}'",
        return_type="list",
    )
    all_session_traces.extend(session_traces)

# Evaluate all sessions
results = mlflow.genai.evaluate(
    data=all_session_traces,
    scorers=[ConversationCompleteness(), UserFrustration()],
)

```

## Limitations[​](#limitations "Direct link to Limitations")

* **No `predict_fn` support**: Multi-turn judges currently work only with pre-collected traces. You cannot use them with `predict_fn` in `mlflow.genai.evaluate`.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Session Tracing Guide](/mlflow-website/docs/latest/genai/tracing/track-users-sessions.md)

[Learn how to track users and sessions in your conversational AI applications for better evaluation.](/mlflow-website/docs/latest/genai/tracing/track-users-sessions.md)

[Learn about sessions →](/mlflow-website/docs/latest/genai/tracing/track-users-sessions.md)

### [Built-in Multi-Turn Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#multi-turn)

[Explore built-in judges for conversation completeness, user frustration, and other multi-turn metrics.](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#multi-turn)

[View judges →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#multi-turn)

### [Create Custom Multi-Turn Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md)

[Build custom LLM judges using make\_judge to evaluate conversation-specific criteria and patterns.](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md)

[Create custom judges →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md)
