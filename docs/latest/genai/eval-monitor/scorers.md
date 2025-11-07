# What are Scorers?

**Scorers** are key components of the MLflow GenAI evaluation framework. They provide a unified interface to define evaluation criteria for your models, agents, and applications.

Scorers can be considered as **metrics** in the traditional ML sense. However, they are more flexible and can return more structured quality feedback, not only the scalar values that are typically represented by metrics.

## How Scorers Work[​](#how-scorers-work "Direct link to How Scorers Work")

Scorers analyze inputs, outputs, and traces from your GenAI application and produce quality assessments. Here's the flow:

1. You provide a dataset of

   inputs

   (and optionally other columns such as

   expectations

   )

2. MLflow runs your `predict_fn` to generate

   outputs

   and

   traces

   for each row in the dataset. Alternatively, you can provide outputs and traces directly in the dataset and omit the predict function.

3. Scorers receive the

   inputs

   ,

   outputs

   ,

   expectations

   , and

   traces

   (or a subset of them) and produce scores and metadata such as explanations and source information.

4. MLflow aggregates the scorer results and saves them. You can analyze the results in the UI.

## What Scorers you should use?[​](#what-scorers-you-should-use "Direct link to What Scorers you should use?")

MLflow provides different types of scorers to address different evaluation needs:

> *I want to try evaluation quickly and get some results fast.*

 → Use [Predefined Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md) to get started.

> *I want to evaluate my application with a simple natural language criteria, such as "The response must be polite".*

 → Use [Guidelines-based Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md).

> *I want to use more advanced prompt for evaluating my application.*

 → Use [Prompt-based Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/make-judge.md).

> *I want to dump the entire trace to the scorer and get detailed insights from it.*

 → Use [Agent-as-a-Judge Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/agentic-overview.md).

> *I want to write my own code for evaluating my application. Other scorers don't fit my advanced needs.*

 → Use [Code-based Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md) to implement your own evaluation logic with Python.

If you are still not sure about which scorer to use, you can ask to the Ask AI (add image) widget in the right below.

## How to Write a Good Scorer?[​](#how-to-write-a-good-scorer "Direct link to How to Write a Good Scorer?")

The general metrics such as 'Hallucination' or 'Toxicity' rarely work in practice. Successful practitioners analyze real data to uncover domain-specific failure modes and then define custom evaluation criteria from the ground up. Here is the general workflow of how to define a good scorer and iterate on it with MLflow.

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

#### Translate failure modes into Scorers

Define scorers that check for the common failure modes.For example, if the answer is in an incorrect format, you may define an [LLM-as-a-Judge scorer](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge.md) that checks if the format is correct. We recommend starting with a simple instruction and then iteratively refine it.

5

#### Align scorers with human feedback.

LLM-as-a-Judge has natural biases. Relying on biased evaluation will lead to incorrect decision making. Therefore, it is important to refine the scorer to align with human feedback. You can manually iterate on prompts or instructions, or use the [Automatic Judge Alignment](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment.md) feature of MLflow to optimize the instruction with a state-of-the-art algorithm powered by [DSPy](https://dspy.ai/).

Pro tip: Version Control Scorers

As you iterate on the scorer, version control becomes important. MLflow can track [Scorer Versions](/mlflow-website/docs/latest/genai/eval-monitor/scorers/versioning.md) to help you maintain changes and share the improved scorers with your team.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [LLM-based Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge.md)

[Get started with LLM judges for evaluating qualities.](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge.md)

### [Evaluate Agents](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Evaluate AI agents with specialized techniques and custom scorers](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

### [Collect Human Feedback](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Gather and manage human feedback to improve your evaluation accuracy](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Learn more →](/mlflow-website/docs/latest/genai/assessments/feedback.md)
