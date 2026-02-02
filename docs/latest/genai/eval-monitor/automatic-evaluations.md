# Automatic Evaluation

*Automatically evaluate traces and multi-turn conversations as they're logged - no code required*

Automatic evaluation runs your LLM judges automatically on traces and multi-turn conversations as they're logged to MLflow, without requiring manual execution of code. This enables two key use cases:

* **Streamlined Quality Iteration**: Seamlessly measure quality as you iterate on your agent or LLM application in development, getting immediate feedback and quality insights without extra evaluation steps
* **Production Monitoring**: Continuously monitor for issues like hallucinations, PII leakage, or user frustration on live traffic (often referred to as online evaluation)

[](/mlflow-website/docs/latest/images/llms/tracing/automatic-evaluation-ui-setup.mp4)

## Automatic vs Offline Evaluation[​](#automatic-vs-offline-evaluation "Direct link to Automatic vs Offline Evaluation")

|                  | Automatic Evaluation                                                                 | Offline Evaluation                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **When it runs** | Automatically, as traces and conversations are logged                                | Manually, when you call [`mlflow.genai.evaluate()`](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md) |
| **Use case**     | Production quality tracking, continuous monitoring, internal QA, interactive testing | Regression testing, bug fix verification, pre-deployment testing, comparing agent versions                                       |
| **Data source**  | Live traces and conversations from your application                                  | Curated datasets or historical traces                                                                                            |

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

Before setting up automatic evaluation, ensure that:

1. **[The MLflow Server is running](/mlflow-website/docs/latest/genai/getting-started/connect-environment.md)**
2. **[MLflow Tracing is enabled](/mlflow-website/docs/latest/genai/tracing/quickstart.md)** in your agent or LLM application
   <!-- -->
   * **For [multi-turn conversation evaluation](#session-level-evaluation)**, traces must include [session IDs](/mlflow-website/docs/latest/genai/tracing/track-users-sessions.md)
3. **[An AI Gateway endpoint is configured](/mlflow-website/docs/latest/genai/governance/ai-gateway/endpoints/create-and-manage.md)** for LLM judge execution
   <!-- -->
   * LLM judges require an LLM to perform evaluations, and [AI Gateway endpoints](/mlflow-website/docs/latest/genai/governance/ai-gateway/endpoints/create-and-manage.md) provide secure, managed access to LLMs

## Setting Up Automatic Evaluation[​](#setting-up-automatic-evaluation "Direct link to Setting Up Automatic Evaluation")

These examples show how to set up LLM judges that automatically evaluate traces and multi-turn conversations as they're logged to an [MLflow Experiment](/mlflow-website/docs/latest/genai/tracing/quickstart.md#create-a-mlflow-experiment), and how to update or disable existing judges. For more details on creating LLM judges, see [LLM-as-a-Judge](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges).

note

* Automatic evaluation only supports [LLM judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges). Code-based scorers (using the [`@scorer` decorator](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md)) are not supported. Use [built-in judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md) or create custom judges with [`make_judge()`](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/create-custom-judge.md).
* When a judge is created or enabled, it evaluates traces and sessions that are **at most one hour old**. Updating a judge's configuration does not trigger re-evaluation of previously assessed traces.

- UI
- SDK

1. Navigate to your experiment and select the **Judges** tab

   ![Judges tab](/mlflow-website/docs/latest/assets/images/judges-tab-c1dea9a24e9809d6c7f8dbb4e2891767.png)

2. Click **+ New LLM judge**

   ![New LLM judge button](/mlflow-website/docs/latest/assets/images/new-llm-judge-button-ee8d6028b0f3b278dbf49cd81759de7f.png)

3. **Select scope**:

   * **Traces**: Evaluate individual traces
   * **Sessions**: Evaluate entire multi-turn conversations

4. **Configure the judge**:

   * **LLM judge**: Select a built-in judge or create a custom one
   * **Name**: A unique name for the judge
   * **Instructions**: Define evaluation criteria for the judge
   * **Output type**: Select the type of value the judge will return
   * **Model**: Select an [AI Gateway](/mlflow-website/docs/latest/genai/governance/ai-gateway/endpoints/create-and-manage.md) endpoint (LLM) to run the judge

5. **Evaluation settings**:

   * Check **"Automatically evaluate future traces using this judge"**
   * Set the **Sample rate** (percentage of traces or sessions to evaluate)
   * Optionally add a **Filter string** to target specific traces or sessions

   ![Evaluation settings](/mlflow-website/docs/latest/assets/images/evaluation-settings-015320f1e469372c39ed925295791f40.png)

6. Click **Save**

7. To edit or disable an existing judge, select it in the **Judges** tab.

   ![Edit LLM judge button](/mlflow-website/docs/latest/assets/images/edit-llm-judge-button-20eae3b628d868a4c3f014c5e478701f.png)

For more details about the APIs used in this example, see [`mlflow.genai.scorers.Scorer.start()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Scorer.start), [`mlflow.genai.scorers.Scorer.update()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Scorer.update), and [`mlflow.genai.scorers.Scorer.stop()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Scorer.stop).

**1. Specify the experiment for automatic evaluation**

python

```python
import mlflow

mlflow.set_experiment("my-experiment")

```

**2. Start automatic evaluation for a trace-level judge**

python

```python
from mlflow.genai.scorers import ToolCallCorrectness, ScorerSamplingConfig

tool_judge = ToolCallCorrectness(model="gateway:/my-llm-endpoint")
registered_tool_judge = tool_judge.register(name="tool_call_correctness")
registered_tool_judge.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5),  # Evaluate 50% of traces
)

```

**3. Start automatic evaluation for a multi-turn (session-level) judge**

python

```python
from mlflow.genai.scorers import ConversationalGuidelines, ScorerSamplingConfig

frustration_judge = ConversationalGuidelines(
    name="user_frustration",
    guidelines="The user should not express frustration, confusion, or dissatisfaction during the conversation.",
    model="gateway:/my-llm-endpoint",
)
registered_frustration_judge = frustration_judge.register(name="user_frustration")
registered_frustration_judge.start(
    sampling_config=ScorerSamplingConfig(sample_rate=1.0),  # Evaluate all conversations
)

```

**4. Update or disable automatic evaluation for an existing judge**

python

```python
from mlflow.genai.scorers import get_scorer, ScorerSamplingConfig

judge = get_scorer(name="tool_call_correctness")
judge.update(
    sampling_config=ScorerSamplingConfig(sample_rate=0.3)
)  # Change sample rate
judge.stop()  # Or, disable the judge

```

## Viewing Results[​](#viewing-results "Direct link to Viewing Results")

Assessments from automatic evaluation appear directly in the MLflow UI. For traces, assessments typically appear within a minute or two of logging. Multi-turn sessions are evaluated after 5 minutes of inactivity (no new traces have been added to the session) by default—this is [configurable](/mlflow-website/docs/latest/api_reference/python_api/mlflow.environment_variables.html#mlflow.environment_variables.MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS).

Navigate to your experiment in the MLflow UI to see results.

![Online evaluation charts showing assessment trends](/mlflow-website/docs/latest/assets/images/online-evaluation-charts-c92ee3860b1b862ab09f32df26f2bd5d.png)

Charts in the Overview tab display quality and performance trends over time

![Online evaluation results showing assessment scores for traces](/mlflow-website/docs/latest/assets/images/online-evaluation-results-ed55cb9aec590cdc084af7f3221e8db5.png)

Assessments from automatic evaluation appear as columns in the Traces tab

## Configuration Options[​](#configuration-options "Direct link to Configuration Options")

### Sampling Rate[​](#sampling-rate "Direct link to Sampling Rate")

Control what percentage of traces are evaluated (0-100%). Balance cost and coverage based on your needs:

* **Development**: Use a high sampling rate to detect as many issues as possible before production deployment
* **Production**: Consider using lower rates if necessary to control costs

### Filtering Traces[​](#filtering-traces "Direct link to Filtering Traces")

Use [trace search syntax](/mlflow-website/docs/latest/genai/tracing/search-traces.md) to target specific traces. Examples:

python

```python
# Only evaluate successful traces
filter_string = "trace.status = 'OK'"

# Only evaluate traces from production environment
filter_string = "metadata.environment = 'production'"

```

note

For session-level evaluation, filters apply to the **first trace** in the session.

### Session-Level Evaluation[​](#session-level-evaluation "Direct link to Session-Level Evaluation")

Automatic evaluation can assess entire multi-turn conversations (sessions), in addition to individual traces.

* **Session completion**: A session is considered complete (ready for automatic evaluation) after no new traces arrive for 5 minutes ([configurable](/mlflow-website/docs/latest/api_reference/python_api/mlflow.environment_variables.html#mlflow.environment_variables.MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS))
* **Re-evaluation**: If new traces are added to the session after evaluation, the session is re-evaluated and previous automatic evaluation results are replaced

For more information about session evaluation, see [Evaluate Conversations](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/multi-turn.md).

## Best Practices[​](#best-practices "Direct link to Best Practices")

* **Combine judges**: Use multiple judges for comprehensive quality coverage
* **Start with a high sampling rate, then scale down as needed**: Use a high sampling rate during development to detect as many issues as possible before production deployment, then reduce for production if necessary to control costs
* **Monitor costs**: LLM-based evaluation has associated costs—adjust sampling accordingly
* **Use filters strategically in production**: Focus evaluation on high-value or high-risk traces

## How It Works[​](#how-it-works "Direct link to How It Works")

LLM judges are periodically executed securely within the MLflow server as new traces and multi-turn conversations are received. Evaluation happens asynchronously and does not block trace logging, so your application's performance is unaffected.

The [MLflow Server](/mlflow-website/docs/latest/genai/getting-started/connect-environment.md) uses [AI Gateway](/mlflow-website/docs/latest/genai/governance/ai-gateway.md) endpoints to access LLMs for judge execution, ensuring secure and managed model access. Only the relevant trace or session data required by the judge (such as inputs, outputs, and context) is sent to the LLM.

## Troubleshooting[​](#troubleshooting "Direct link to Troubleshooting")

| Issue                                          | Solution                                                                                                                                                                  |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Missing assessments**                        | Verify that the judge is active, the filter matches your traces, the sampling rate is greater than zero, and the traces are less than one hour old                        |
| **Unexpected or unsatisfactory judge results** | Edit the judge's instructions or use the [`align()` method](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment.md) to optimize them automatically |
| **Evaluation errors**                          | Check trace/session assessments in the UI or SDK, or server logs, for details. Failed evaluations are not retried automatically                                           |

For further debugging, enable debug logging on the MLflow server by setting the [`MLFLOW_LOGGING_LEVEL=DEBUG`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.environment_variables.html#mlflow.environment_variables.MLFLOW_LOGGING_LEVEL) environment variable and checking the MLflow server logs.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [LLM-as-a-Judge](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges)

[Learn more about creating and customizing LLM judges for your specific quality criteria.](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges)

### [Evaluate Conversations](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/multi-turn.md)

[Learn more about evaluating multi-turn conversations.](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/multi-turn.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/multi-turn.md)
