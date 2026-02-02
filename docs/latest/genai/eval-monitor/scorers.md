# LLM Judges and Scorers

Judges are a key component of the MLflow GenAI evaluation framework. They provide a unified interface to define evaluation criteria for your models, agents, and applications. Like their name suggests, judges judge how well your application did based on the evaluation criteria. This could be a pass/fail, true/false, numerical value, or a categorical value.

Choose the right type of judge depending on how much customization and control you need. Each approach builds on the previous one, adding more complexity and control.

Start with [built-in judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#available-judges) for quick evaluation. As your needs evolve, build [custom LLM judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md) for domain-specific criteria and create [custom code-based scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md) for programmatic business logic.

| Approach                                                                                                           | Level of customization | Use cases                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------ | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Built-in judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#available-judges) | Minimal                | Quickly try LLM evaluation with built-in judges such as `Correctness` and `RetrievalGroundedness`.                                                                    |
| [Guidelines judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md)                | Moderate               | A built-in judge that checks whether responses pass or fail custom natural-language rules, such as style or factuality guidelines.                                    |
| [Custom judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md)                 | Full                   | Create fully customized LLM judges with detailed evaluation criteria and feedback optimization. Capable of returning numerical scores, categories, or boolean values. |
| [Code-based scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md)                             | Full                   | Programmatic scorers that evaluate things like exact matching, format validation, and performance metrics.                                                            |

note

We'll refer to LLM judges and code-based scorers separately, but in the API, both LLM judges and code-based scorers are classified as types of scorers, such as in the functions [`list_scorers`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.list_scorers) and [`get_scorer`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.get_scorer)

## How judges work[​](#how-judges-work "Direct link to How judges work")

A judge receives a [Trace](/mlflow-website/docs/latest/genai/concepts/trace.md) from `evaluate()`. It then does the following:

1. Parses the `trace` to extract specific fields and data that are used to assess quality
2. Runs the judge to perform the quality assessment based on the extracted fields and data
3. Returns the quality assessment as [Feedback](/mlflow-website/docs/latest/genai/assessments/feedback.md) to attach to the `trace`

## LLMs as judges[​](#llms-as-judges "Direct link to LLMs as judges")

LLM judges use Large Language Models for quality assessment.

Think of a judge as an AI assistant specialized in quality assessment. It can evaluate your app's inputs, outputs, and even explore the entire execution trace to make assessments based on criteria you define. For example, when checking correctness, exact string matching would fail to recognize that `give me healthy food options` and `food to keep me fit` are semantically the same answer, but an LLM judge can understand they're both correct.

note

Judges use LLMs for evaluation. Use them directly with `mlflow.genai.evaluate()` or wrap them in [custom scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md) for advanced scoring logic.

### Built-in LLM judges[​](#built-in-llm-judges "Direct link to Built-in LLM judges")

MLflow provides research-validated judges for common use cases.

See the [complete list of built-in judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#available-judges) for details on each judge and their usage. You can further improve the judges' accuracy by [aligning them with human feedback](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment.md).

### Custom LLM judges[​](#custom-llm-judges "Direct link to Custom LLM judges")

In addition to the built-in judges, MLflow makes it easy to create your own judges with custom prompts and instructions.

Use custom LLM judges when you need to define specialized evaluation tasks, need more control over grades (not just pass/fail), or need to validate that your agent made appropriate decisions and performed operations correctly for your specific use case.

See [Custom judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md). Once you've created custom judges, you can further improve their accuracy by [aligning them with human feedback](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment.md).

### Select the LLM that powers the judge[​](#select-the-llm-that-powers-the-judge "Direct link to Select the LLM that powers the judge")

You can change the judge model by using the `model` argument in the judge definition. Specify the model in the format `<provider>:/<model-name>`. For example:

python

```python
from mlflow.genai.scorers import Correctness

Correctness(model="openai:/gpt-5-mini")

```

For a list of supported models, see [selecting judge models](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md#selecting-judge-models).

## What Judges you should use?[​](#what-judges-you-should-use "Direct link to What Judges you should use?")

MLflow provides different types of judges to address different evaluation needs:

> *I want to try evaluation quickly and get some results fast.*

 → Use [Built-in Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md) to get started.

> *I want to evaluate my application with a simple natural language criteria, such as "The response must be polite".*

 → Use [Guidelines-based Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md).

> *I want to use more advanced prompt for evaluating my application.*

 → Use [Prompt-based Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md).

> *I want to dump the entire trace to the scorer and get detailed insights from it.*

 → Use [Trace-Based Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md#trace-based-judges).

> *I want to write my own code for evaluating my application. Other scorers don't fit my advanced needs.*

 → Use [Code-based Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md) to implement your own evaluation logic with Python.

If you are still unsure about which judge to use, you can use the "Ask AI" widget in the bottom right

## Code-based scorers[​](#code-based-scorers "Direct link to Code-based scorers")

Custom code-based scorers offer the ultimate flexibility to define precisely how your GenAI application's quality is measured. You can define evaluation metrics tailored to your specific business use case, whether based on simple heuristics, advanced logic, or programmatic evaluations.

Use custom scorers for the following scenarios:

1. Defining a custom heuristic or code-based evaluation metric.
2. Customizing how the data from your app's trace is mapped to built-in LLM judges.
3. Using your own LLM for evaluation.
4. Any other use cases where you need more flexibility and control than provided by custom LLM judges.

See [Create custom code-based scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md).

## How to Write a Good Judge?[​](#how-to-write-a-good-judge "Direct link to How to Write a Good Judge?")

In practice, out-of-the-box judges such as 'Groundedness' or 'Safety' struggle to understand your domain-specific data and criteria. Successful practitioners analyze real data to uncover domain-specific failure modes and then define custom evaluation criteria from the ground up. Here is the general workflow of how to define a good judge and iterate on it with MLflow.

1

#### Generate traces or collect them from production

Start with generating [traces](/mlflow-website/docs/latest/genai/tracing.md) from a set of realistic input samples. If you already have production traces, that is even better.

2

#### Gather human feedback

Collect feedback from domain experts or users. MLflow provides [a UI and SDK](/mlflow-website/docs/latest/genai/assessments/feedback.md) for collecting feedback on traces.

3

#### Error analysis

Analyze the common failure modes (error categories) from the feedback.

<br />

To organize traces into error categories, use [Trace Tag](/mlflow-website/docs/latest/genai/tracing/attach-tags.md) to label and filter traces.

4

#### Translate failure modes into Judges

Define judges that check for the common failure modes. For example, if the answer is in an incorrect format, you may define an [LLM Judge](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges) that checks if the format is correct. We recommend starting with a simple instruction and then iteratively refine it.

5

#### Align judges with human feedback.

LLM-as-a-Judge has natural biases. Relying on biased evaluation will lead to incorrect decision making. Therefore, it is important to refine the scorer to align with human feedback. You can manually iterate on prompts or instructions, or use the [Automatic Judge Alignment](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment.md) feature of MLflow to optimize the instruction with a state-of-the-art algorithm powered by [DSPy](https://dspy.ai/).

Pro tip: Version Control Judges

As you iterate on the judge, version control becomes important. MLflow can track [Judge Versions](/mlflow-website/docs/latest/genai/eval-monitor/scorers/versioning.md) to help you maintain changes and share the improved judges with your team.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [LLM-based Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges)

[Get started with LLM judges for evaluating qualities.](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges)

### [Evaluate Agents](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Evaluate AI agents with specialized techniques and custom judges](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

### [Collect Human Feedback](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Gather and manage human feedback to improve your evaluation accuracy](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Learn more →](/mlflow-website/docs/latest/genai/assessments/feedback.md)
