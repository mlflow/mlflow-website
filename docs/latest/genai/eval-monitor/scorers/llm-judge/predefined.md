# Built-in LLM Judges

MLflow provides several pre-configured LLM judges optimized for common evaluation scenarios.

## Example Usage[​](#example-usage "Direct link to Example Usage")

* UI
* SDK

Version Requirements

The Judge Builder UI requires **MLflow >= 3.9.0**.

The MLflow UI provides a visual Judge Builder that lets you create custom LLM judges without writing code.

1. Navigate to your experiment and select the **Judges** tab, then click **New LLM judge**

2. **LLM judge**: Select a built-in judge. We're using the `RelevanceToQuery` and `Correctness` judges in this example.

![RelevanceToQuery Judge UI](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/scorers/relevance-create-judge-ui.png)

3. Click **Create judge** to save your new LLM judge

To use the built-in LLM judges, select the judge class from the [available judges](#available-judges) and pass it to the `scorers` argument of the [evaluate](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate) function.

python

```python
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines

eval_dataset = [
    {
        "inputs": {"query": "What is the most common aggregate function in SQL?"},
        "outputs": "The most common aggregate function in SQL is SUM().",
        # Correctness judge requires an "expected_facts" field.
        "expectations": {
            "expected_facts": ["Most common aggregate function in SQL is COUNT()."],
        },
    },
    {
        "inputs": {"query": "How do I use MLflow?"},
        # verbose answer
        "outputs": "Hi, I'm a chatbot that answers questions about MLflow. Thank you for asking a great question! I know MLflow well and I'm glad to help you with that. You will love it! MLflow is a Python-based platform that provides a comprehensive set of tools for logging, tracking, and visualizing machine learning models and experiments throughout their entire lifecycle. It consists of four main components: MLflow Tracking for experiment management, MLflow Projects for reproducible runs, MLflow Models for standardized model packaging, and MLflow Model Registry for centralized model lifecycle management. To get started, simply install it with 'pip install mlflow' and then use mlflow.start_run() to begin tracking your experiments with automatic logging of parameters, metrics, and artifacts. The platform creates a beautiful web UI where you can compare different runs, visualize metrics over time, and manage your entire ML workflow efficiently. MLflow integrates seamlessly with popular ML libraries like scikit-learn, TensorFlow, PyTorch, and many others, making it incredibly easy to incorporate into your existing projects!",
        "expectations": {
            "expected_facts": [
                "MLflow is a tool for managing and tracking machine learning experiments."
            ],
        },
    },
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
    ],
)

```

![Built-in judges result](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/scorers/builtin-judges-results.png)

## Available Judges[​](#available-judges "Direct link to Available Judges")

### Response Quality[​](#response-quality "Direct link to Response Quality")

| Judge                                                                                                                                                               | What does it evaluate?                                        | Requires ground-truth? | Requires traces? |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------- | ---------------- |
| [RelevanceToQuery](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/relevance.md#relevancetoquery-judge)                                        | Does the app's response directly address the user's input?    | No                     | No               |
| [Correctness](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/response-quality/correctness.md)                                                     | Are the expected facts supported by the app's response?       | Yes\*                  | No               |
| [Completeness](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Completeness)\*\*                                        | Does the agent address all questions in a single user prompt? | No                     | No               |
| [Fluency](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Fluency)                                                      | Is the response grammatically correct and naturally flowing?  | No                     | No               |
| [Safety](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/response-quality/safety.md)                                                               | Does the app's response avoid harmful or toxic content?       | No                     | No               |
| [Equivalence](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Equivalence)                                              | Is the app's response equivalent to the expected output?      | Yes                    | No               |
| [Guidelines](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md#1-built-in-guidelines-judge-global-guidelines)                          | Does the response adhere to provided guidelines?              | Yes\*                  | No               |
| [ExpectationsGuidelines](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md#2-built-in-expectationsguidelines-judge-per-row-guidelines) | Does the response meet specific expectations and guidelines?  | Yes\*                  | No               |

### RAG[​](#rag "Direct link to RAG")

| Judge                                                                                                                            | What does it evaluate?                                    | Requires ground-truth? | Requires traces?      |
| -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------- | --------------------- |
| [RetrievalRelevance](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/relevance.md#retrievalrelevance-judge) | Are retrieved documents relevant to the user's request?   | No                     | ⚠️ **Trace Required** |
| [RetrievalGroundedness](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/groundedness.md)                    | Is the app's response grounded in retrieved information?  | No                     | ⚠️ **Trace Required** |
| [RetrievalSufficiency](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/context-sufficiency.md)              | Do retrieved documents contain all necessary information? | Yes                    | ⚠️ **Trace Required** |

### Tool Call[​](#tool-call "Direct link to Tool Call")

| Judge                                                                                                                | What does it evaluate?                                       | Requires ground-truth? | Requires traces?      |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------- | --------------------- |
| [ToolCallCorrectness](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/correctness.md)\*\* | Are the tool calls and arguments correct for the user query? | No                     | ⚠️ **Trace Required** |
| [ToolCallEfficiency](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/efficiency.md)\*\*   | Are the tool calls efficient without redundancy?             | No                     | ⚠️ **Trace Required** |

\*Can extract expectations from trace assessments if available.

\*\*Indicates experimental features that may change in future releases.

### Multi-Turn[​](#multi-turn "Direct link to Multi-Turn")

Multi-turn judges evaluate entire conversation sessions rather than individual turns. They require traces with session IDs and are experimental in MLflow 3.7.0. See [Track Users and Sessions](/mlflow-website/docs/latest/genai/tracing/track-users-sessions.md)

Multi-Turn Evaluation Requirements

Multi-turn judges require:

1. **Session IDs**: Traces must have `mlflow.trace.session` metadata
2. **List or DataFrame input**: Currently only supports pre-collected traces (no `predict_fn` support yet)

| Judge                                                                                                                                                                | What does it evaluate?                                                     | Requires Session? |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------- |
| [ConversationCompleteness](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.ConversationCompleteness)\*\*                 | Does the agent address all user questions throughout the conversation?     | Yes               |
| [ConversationalGuidelines](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.ConversationalGuidelines)\*\*                 | Do the assistant's responses comply with provided guidelines?              | Yes               |
| [ConversationalRoleAdherence](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.ConversationalRoleAdherence)\*\*           | Does the assistant maintain its assigned role throughout the conversation? | Yes               |
| [ConversationalSafety](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.ConversationalSafety)\*\*                         | Are the assistant's responses safe and free of harmful content?            | Yes               |
| [ConversationalToolCallEfficiency](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.ConversationalToolCallEfficiency)\*\* | Was tool usage across the conversation efficient and appropriate?          | Yes               |
| [KnowledgeRetention](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.KnowledgeRetention)\*\*                             | Does the assistant correctly retain information from earlier user inputs?  | Yes               |
| [UserFrustration](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.UserFrustration)\*\*                                   | Is the user frustrated? Was the frustration resolved?                      | Yes               |

Availability

Safety and RetrievalRelevance judges are currently only available in [Databricks managed MLflow](https://docs.databricks.com/mlflow3/genai/eval-monitor/) and will be open-sourced soon.

tip

Typically, you can get started with evaluation using built-in judges. However, every AI application is unique and has domain-specific quality criteria. At some point, you'll need to create your own custom LLM judges.

* Your application has complex inputs/outputs that built-in judges can't parse
* You need to evaluate specific business logic or domain-specific criteria
* You want to combine multiple evaluation aspects into a single judge

See [custom LLM judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md) guide for detailed examples.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Guidelines Judge](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md)

[Learn how to use the Guidelines judge to evaluate responses against custom criteria](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md)

### [Evaluate Agents](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn how to evaluate AI agents with specialized techniques and judges](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate production traces to understand and improve your AI application's behavior](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)
