# Optimize Prompts (Experimental)

**The simple way to continuously improve your AI agents and prompts.**

MLflow's prompt optimization lets you systematically enhance your AI applications with minimal code changes. Whether you're building with LangChain, OpenAI Agent, CrewAI, or your own custom implementation, MLflow provides a universal path from initial prototyping to steady improvement.

**Minimum rewrites, no lock-in, just better prompts.**

MLflow supports multiple optimization algorithms to improve your prompts:

* **[GEPA](https://arxiv.org/abs/2507.19457)** ([`GepaPromptOptimizer`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize.GepaPromptOptimizer)): Iteratively refines prompts using LLM-driven reflection and automated feedback, achieving systematic improvements through trial-and-error learning.
* **Metaprompting** ([`MetaPromptOptimizer`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize.MetaPromptOptimizer)): Restructures prompts to be more systematic and effective, working in both zero-shot mode (without training data) and few-shot mode (learning from your examples).

See [Choosing Your Optimizer](#choosing-your-optimizer) for guidance on which optimizer to use for your specific needs.

Why Use MLflow Prompt Optimization?

* **Zero Framework Lock-in**: Works with ANY agent framework—LangChain, OpenAI Agent, CrewAI, or custom solutions
* **Minimal Code Changes**: Add a few lines to start optimizing; no architectural rewrites needed
* **Data-Driven Improvement**: Automatically learn from your evaluation data and custom metrics
* **Multi-Prompt Optimization**: Jointly optimize multiple prompts for complex agent workflows
* **Granular Control**: Optimize single prompts or entire multi-prompt workflows—you decide what to improve
* **Production-Ready**: Built-in version control and registry for seamless deployment
* **Extensible**: Bring your own optimization algorithms with simple base class extension

Version Requirements

The `optimize_prompts` API requires **MLflow >= 3.5.0**.

## Quick Start[​](#quick-start "Direct link to Quick Start")

Here's a realistic example of optimizing a prompt for medical paper section classification:

python

```python
import mlflow
import openai
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Correctness

# Register initial prompt for classifying medical paper sections
prompt = mlflow.genai.register_prompt(
    name="medical_section_classifier",
    template="Classify this medical research paper sentence into one of these sections: CONCLUSIONS, RESULTS, METHODS, OBJECTIVE, BACKGROUND.\n\nSentence: {{sentence}}",
)


# Define your prediction function
def predict_fn(sentence: str) -> str:
    prompt = mlflow.genai.load_prompt("prompts:/medical_section_classifier/1")
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-5-nano",
        # load prompt template using PromptVersion.format()
        messages=[{"role": "user", "content": prompt.format(sentence=sentence)}],
    )
    return completion.choices[0].message.content


# Training data with medical paper sentences and ground truth labels
# fmt: off
raw_data = [
    ("The emergence of HIV as a chronic condition means that people living with HIV are required to take more responsibility for the self-management of their condition , including making physical , emotional and social adjustments .", "BACKGROUND"),
    ("This paper describes the design and evaluation of Positive Outlook , an online program aiming to enhance the self-management skills of gay men living with HIV .", "BACKGROUND"),
    ("This study is designed as a randomised controlled trial in which men living with HIV in Australia will be assigned to either an intervention group or usual care control group .", "METHODS"),
    ("The intervention group will participate in the online group program ` Positive Outlook ' .", "METHODS"),
    ("The program is based on self-efficacy theory and uses a self-management approach to enhance skills , confidence and abilities to manage the psychosocial issues associated with HIV in daily life .", "METHODS"),
    ("Participants will access the program for a minimum of 90 minutes per week over seven weeks .", "METHODS"),
    ("Primary outcomes are domain specific self-efficacy , HIV related quality of life , and outcomes of health education .", "METHODS"),
    ("Secondary outcomes include : depression , anxiety and stress ; general health and quality of life ; adjustment to HIV ; and social support .", "METHODS"),
    ("Data collection will take place at baseline , completion of the intervention ( or eight weeks post randomisation ) and at 12 week follow-up .", "METHODS"),
    ("Results of the Positive Outlook study will provide information regarding the effectiveness of online group programs improving health related outcomes for men living with HIV .", "CONCLUSIONS"),
    ("The aim of this study was to evaluate the efficacy , safety and complications of orbital steroid injection versus oral steroid therapy in the management of thyroid-related ophthalmopathy .", "OBJECTIVE"),
    ("A total of 29 patients suffering from thyroid ophthalmopathy were included in this study .", "METHODS"),
    ("Patients were randomized into two groups : group I included 15 patients treated with oral prednisolone and group II included 14 patients treated with peribulbar triamcinolone orbital injection .", "METHODS"),
    ("Both groups showed improvement in symptoms and in clinical evidence of inflammation with improvement of eye movement and proptosis in most cases .", "RESULTS"),
    ("Mean exophthalmometry value before treatment was 22.6 1.98 mm that decreased to 18.6 0.996 mm in group I , compared with 23 1.86 mm that decreased to 19.08 1.16 mm in group II .", "RESULTS"),
    ("There was no change in the best-corrected visual acuity in both groups .", "RESULTS"),
    ("There was an increase in body weight , blood sugar , blood pressure and gastritis in group I in 66.7 % , 33.3 % , 50 % and 75 % , respectively , compared with 0 % , 0 % , 8.3 % and 8.3 % in group II .", "RESULTS"),
    ("Orbital steroid injection for thyroid-related ophthalmopathy is effective and safe .", "CONCLUSIONS"),
    ("It eliminates the adverse reactions associated with oral corticosteroid use .", "CONCLUSIONS"),
    ("The aim of this prospective randomized study was to examine whether active counseling and more liberal oral fluid intake decrease postoperative pain , nausea and vomiting in pediatric ambulatory tonsillectomy .", "OBJECTIVE"),
]
# fmt: on

# Format dataset for optimization
dataset = [
    {
        "inputs": {"sentence": sentence},
        "expectations": {"expected_response": label},
    }
    for sentence, label in raw_data
]

# Optimize the prompt
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model="openai:/gpt-5", max_metric_calls=300
    ),
    scorers=[Correctness(model="openai:/gpt-5-mini")],
)

# Use the optimized prompt
optimized_prompt = result.optimized_prompts[0]
print(f"Optimized template: {optimized_prompt.template}")

```

The API will automatically improve the prompt to better classify medical paper sections by learning from the training examples.

## Choosing Your Optimizer[​](#choosing-your-optimizer "Direct link to Choosing Your Optimizer")

MLflow currently supports two optimization algorithms: **GEPA** and **Metaprompting**. Each uses different strategies to improve your prompts.

### GEPA (Genetic-Pareto)[​](#gepa-genetic-pareto "Direct link to GEPA (Genetic-Pareto)")

GEPA is a prompt optimization technique that uses natural language reflection to iteratively improve LLM performance through trial-and-error learning. It's particularly effective at extracting rich learning signals from system behavior by analyzing failures in natural language.

**Key Features:**

* **Natural Language Reflection**: Leverages interpretable language to extract learning signals from execution traces, reasoning chains, and tool interactions
* **High Efficiency**: Achieves superior results with dramatically fewer iterations (up to 35x fewer rollouts compared to traditional methods like GRPO)
* **Pareto Synthesis**: Smartly picks the past prompt to mutate and improve
* **Strong Performance**: Demonstrates reliable gains on a wide range of tasks, e.g., context compression, Q\&A agents, etc.

**Best For:**

* Tasks where you have clear evaluation metrics and a dataset of decent size (e.g., 100+ records)
* Tasks where quality is critical to your system (e.g., medical agents, financial agents, etc.), so that the optimization cost and longer prompt as produced by GEPA is worth it

Reduce the Cost of GEPA Optimization

The cost of GEPA optimization is tightly coupled with the reflection model you use and the max number of metric calls you allow. You can reduce the cost by using a cheaper reflection model or reducing the max number of metric calls.

**Learn More:** [GEPA Research Paper](https://arxiv.org/abs/2507.19457) | [GEPA GitHub Repository](https://github.com/gepa-ai/gepa)

### Metaprompting[​](#metaprompting "Direct link to Metaprompting")

Metaprompting is a prompt optimization technique that utilizes a metaprompt to call the language model to restructure your prompts to be more systematic and effective. It operates in two modes:

**Zero-Shot Mode:**

* Analyzes your initial prompt and restructures it to follow best practices
* Makes prompts more systematic without requiring training data
* Quick to run and requires no examples

**Few-Shot Mode:**

* Evaluates the initial prompt on your training data to understand task-specific patterns
* Leverages the evaluation results along with general best practices to restructure the prompt to be more systematic and effective

**Key Features:**

* **Fast Optimization**: Runs fast because it only does one evaluation round in few-shot mode and just one single call to the language model in zero-shot mode
* **Minimal Data Requirement**: Works well with zero or just a few examples (less than 10)
* **Systematic Improvement**: Restructures prompts to follow clear patterns and best practices
* **Data-Aware**: In few-shot mode, learns from your specific data to tailor improvements
* **Custom Guidelines**: You can provide custom guidelines to the optimizer to tailor the optimization to your specific needs

**Best For:** Tasks where you want quick improvements based on prompt engineering best practices, or when you have limited training data but want to leverage it for targeted improvements.

**Usage Examples:**

python

```python
from mlflow.genai.optimize import MetaPromptOptimizer

# Zero-shot mode: No training data or scorers required
# The optimizer automatically uses zero-shot mode when train_data is empty and scorers is empty
results = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=[],
    prompt_uris=[prompt.uri],
    optimizer=MetaPromptOptimizer(
        reflection_model="openai:/gpt-5",
        guidelines="This prompt is used in a finance agent to project tax situations.",
    ),
    scorers=[],
)

# Few-shot mode: Learn from training data
# The optimizer automatically uses few-shot mode when train_data and scorers are provided
results = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[prompt.uri],
    optimizer=MetaPromptOptimizer(
        reflection_model="openai:/gpt-5",
        guidelines="This prompt is used in a finance agent to project tax situations.",
    ),
    scorers=[Correctness(model="openai:/gpt-5-mini")],
)

```

### Comparison Summary[​](#comparison-summary "Direct link to Comparison Summary")

| Feature                    | GEPA                                      | Metaprompting (Zero-Shot)       | Metaprompting (Few-Shot)                |
| -------------------------- | ----------------------------------------- | ------------------------------- | --------------------------------------- |
| **Requires Training Data** | Yes                                       | No                              | Yes                                     |
| **Optimization Speed**     | Moderate                                  | Fast                            | Fast-Moderate                           |
| **Learning Approach**      | Iterative trial-and-error with reflection | Systematic restructuring        | Data-driven restructuring               |
| **Best Use Case**          | Complex tasks with clear metrics          | Quick improvements without data | Targeted improvements with limited data |

Choose the optimizer that best fits your task requirements, available data, and optimization budget.

### Example: Simple Prompt → Optimized Prompt[​](#example-simple-prompt--optimized-prompt "Direct link to Example: Simple Prompt → Optimized Prompt")

**Before Optimization:**

text

```text
Classify this medical research paper sentence
into one of these sections: CONCLUSIONS, RESULTS,
METHODS, OBJECTIVE, BACKGROUND.

Sentence: {{sentence}}

```

**After Optimization:**

text

```text
You are a single-sentence classifier for medical research abstracts. For each input sentence, decide which abstract section it belongs to and output exactly one label in UPPERCASE with no extra words, punctuation, or explanation.

Allowed labels: CONCLUSIONS, RESULTS, METHODS, OBJECTIVE, BACKGROUND

Input format:
- The prompt will be:
  "Classify this medical research paper sentence into one of these sections: CONCLUSIONS, RESULTS, METHODS, OBJECTIVE, BACKGROUND.

  Sentence: {{sentence}}"

Core rules:
- Use only the information in the single sentence.
- Classify by the sentence's function: context-setting vs aim vs procedure vs findings vs interpretation.
- Return exactly one uppercase label from the allowed set.

Decision guide and lexical cues:

1) RESULTS
- Reports observed findings/outcomes tied to data.
- Common cues: past-tense result verbs and outcome terms: "showed," "was/were associated with," "increased/decreased," "improved," "reduced," "significant," "p < …," "odds ratio," "risk ratio," "95% CI," percentages, rates, counts or numbers tied to effects/adverse events.
- If it explicitly states changes, associations, statistical significance, or quantified outcomes, choose RESULTS.

2) CONCLUSIONS
- Interpretation, implications, recommendations, or high-level takeaways.
- Common cues: "In conclusion," "These findings suggest/indicate," "We conclude," statements about practice/policy/clinical implications, benefit–risk judgments, feasibility statements.
- Sentences that forecast the significance/utility of the study's results ("Results will provide insight/information," "Findings will inform/guide practice") are CONCLUSIONS.
- Tie-break with RESULTS: If a sentence describes an outcome as a general claim without specific observed data/statistics, prefer CONCLUSIONS over RESULTS.

3) METHODS
- How the study was conducted: design, participants, interventions/programs, measurements/outcomes lists, timelines, procedures, or analyses.
- Common cues: design terms ("randomized," "double-blind," "cross-sectional," "cohort," "case-control"), "participants," "n =," inclusion/exclusion criteria, instruments/scales, dosing/protocols, schedules/timelines, statistical tests/analysis plans ("multivariate regression," "Kaplan–Meier," "ANOVA," "we will compare"), trial registration, ethics approval.
- Measurement/outcome lists are METHODS (e.g., "Secondary outcomes include: …"; "Primary outcome was …").
- Numbers specifying sample size (e.g., "n = 200") → METHODS; numbers tied to effects → RESULTS.
- Program/intervention descriptions, components, theoretical basis, and mechanisms are METHODS, even if written in present tense and even if they contain purpose phrases. Examples: "The program is based on self-efficacy theory…," "The intervention uses a self-management approach to enhance skills…," "The device is designed to…"
  - Important: An infinitive "to [verb] …" inside a program/intervention description (e.g., "uses X to improve Y") is METHODS, not OBJECTIVE, because it describes how the intervention works, not the study's aim.

4) OBJECTIVE
- The aim/purpose/hypothesis of the study.
- Common cues: "Objective(s):" "Aim/Purpose was," "We aimed/sought/intended to," "We hypothesized that …"
- Infinitive purpose phrases indicating the study's aim without procedures or results: "To determine/evaluate/assess/investigate whether …" → OBJECTIVE.
- Phrases like "The aim of this study was to evaluate the efficacy/safety of X vs Y …" → OBJECTIVE.
- If "We evaluated/assessed …" is clearly used as a purpose statement (not describing methods or results), label OBJECTIVE.

5) BACKGROUND
- Context, rationale, prior knowledge, unmet need; introduces topic without specific aims, procedures, or results.
- Common cues: burden/prevalence statements, "X is common," "X remains poorly understood," prior work summaries, general descriptions.
- If a sentence merely states that a paper describes/reports a program/design/evaluation without concrete procedures/analyses, label as BACKGROUND.

Important tie-break rules:
- RESULTS vs CONCLUSIONS: Observed data/findings → RESULTS; interpretation/generalization/recommendation → CONCLUSIONS.
- OBJECTIVE vs METHODS: Purpose/aim of the study → OBJECTIVE; concrete design/intervention details/measurements/analysis steps → METHODS.
- BACKGROUND vs OBJECTIVE: Context/motivation without an explicit study aim → BACKGROUND.
- BACKGROUND vs METHODS: General description without concrete procedures/analyses → BACKGROUND.
- The word "Results" at the start does not guarantee RESULTS; e.g., "Results will provide information …" → CONCLUSIONS.

Output constraint:
- Return exactly one uppercase label: CONCLUSIONS, RESULTS, METHODS, OBJECTIVE, or BACKGROUND. No extra text or punctuation.

```

## Components[​](#components "Direct link to Components")

The [`mlflow.genai.optimize_prompts()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize_prompts) API requires the following components:

| Component              | Description                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Target Prompt URIs** | List of prompt URIs to optimize (e.g., `["prompts:/qa/1"]`)                                                                                                                                                                                                                                                                                                                                     |
| **Predict Function**   | A callable that takes inputs as keyword arguments and returns outputs. Must load templates from MLflow prompt versions (e.g., call [`PromptVersion.format()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.model_registry.PromptVersion.format)).                                                                                                  |
| **Training Data**      | Dataset with `inputs` (dict) and `expectations` (expected results). Supports pandas DataFrame, list of dicts, or MLflow EvaluationDataset.                                                                                                                                                                                                                                                      |
| **Optimizer**          | Prompt optimizer instance (e.g., [`GepaPromptOptimizer`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize.GepaPromptOptimizer) or [`MetaPromptOptimizer`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize.MetaPromptOptimizer)). See [Choosing Your Optimizer](#choosing-your-optimizer) for guidance. |

### 1. Target Prompt URIs[​](#1-target-prompt-uris "Direct link to 1. Target Prompt URIs")

Specify which prompts to optimize using their URIs from MLflow Prompt Registry:

python

```python
prompt_uris = [
    "prompts:/qa/1",  # Specific version
    "prompts:/instruction@latest",  # Latest version
]

```

You can reference prompts by:

* **Specific version**: `"prompts:/qa/1"` - Optimize a particular version
* **Latest version**: `"prompts:/qa@latest"` - Optimize the most recent version
* **Alias**: `"prompts:/qa@champion"` - Optimize a version with a specific alias

### 2. Predict Function[​](#2-predict-function "Direct link to 2. Predict Function")

Your `predict_fn` must:

* Accept inputs as keyword arguments matching the inputs field of the dataset

* Load the template from MLflow prompt versions using one of the following methods:

  <!-- -->

  * [`PromptVersion.format()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.model_registry.PromptVersion.format)
  * [`PromptVersion.template`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.model_registry.PromptVersion.template)
  * [`PromptVersion.to_single_brace_format()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.model_registry.PromptVersion.to_single_brace_format)

* Return outputs in the same format as your training data (e.g., outputs = `{"answer": "xxx"}` if expectations = `{"expected_response": {"answer": "xxx"}}`)

python

```python
def predict_fn(question: str) -> str:
    # Load prompt from registry
    prompt = mlflow.genai.load_prompt("prompts:/qa/1")

    # Format the prompt with input variables
    formatted_prompt = prompt.format(question=question)

    # Call your LLM
    response = your_llm_call(formatted_prompt)

    return response

```

### 3. Training Data[​](#3-training-data "Direct link to 3. Training Data")

Provide a dataset with `inputs` and `expectations`. Both columns should have dictionary values. `inputs` values will be passed to the predict function as keyword arguments. Please refer to [Built-in Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md) for the expected format of each built in scorers.

python

```python
# List of dictionaries - Example: Medical paper classification
dataset = [
    {
        "inputs": {
            "sentence": "The emergence of HIV as a chronic condition means that people living with HIV are required to take more responsibility..."
        },
        "expectations": {"expected_response": "BACKGROUND"},
    },
    {
        "inputs": {
            "sentence": "This study is designed as a randomised controlled trial in which men living with HIV..."
        },
        "expectations": {"expected_response": "METHODS"},
    },
    {
        "inputs": {
            "sentence": "Both groups showed improvement in symptoms and in clinical evidence of inflammation..."
        },
        "expectations": {"expected_response": "RESULTS"},
    },
    {
        "inputs": {
            "sentence": "Orbital steroid injection for thyroid-related ophthalmopathy is effective and safe."
        },
        "expectations": {"expected_response": "CONCLUSIONS"},
    },
    {
        "inputs": {
            "sentence": "The aim of this study was to evaluate the efficacy, safety and complications..."
        },
        "expectations": {"expected_response": "OBJECTIVE"},
    },
]

# Or pandas DataFrame
import pandas as pd

dataset = pd.DataFrame(
    {
        "inputs": [
            {"sentence": "The emergence of HIV as a chronic condition..."},
            {"sentence": "This study is designed as a randomised controlled trial..."},
            {"sentence": "Both groups showed improvement in symptoms..."},
        ],
        "expectations": [
            {"expected_response": "BACKGROUND"},
            {"expected_response": "METHODS"},
            {"expected_response": "RESULTS"},
        ],
    }
)

```

### 4. Optimizer[​](#4-optimizer "Direct link to 4. Optimizer")

Create an optimizer instance for the optimization algorithm. MLflow supports [`GepaPromptOptimizer`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize.GepaPromptOptimizer) and [`MetaPromptOptimizer`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize.MetaPromptOptimizer). See [Choosing Your Optimizer](#choosing-your-optimizer) for detailed guidance on which optimizer to use.

python

```python
from mlflow.genai.optimize import GepaPromptOptimizer, MetaPromptOptimizer

# Option 1: GEPA optimizer
optimizer = GepaPromptOptimizer(
    reflection_model="openai:/gpt-5",  # Powerful model for optimization
    max_metric_calls=100,
    display_progress_bar=False,
)

# Option 2: Metaprompting optimizer
# Note: Zero-shot vs few-shot is determined by whether you provide
# scorers and train_data to optimize_prompts()
optimizer = MetaPromptOptimizer(
    reflection_model="openai:/gpt-5",
    guidelines="Optional custom guidelines for optimization",
)

```

## Advanced Usage[​](#advanced-usage "Direct link to Advanced Usage")

### Works with Any Agent Framework[​](#works-with-any-agent-framework "Direct link to Works with Any Agent Framework")

MLflow's optimization is **framework-agnostic**—it works seamlessly with LangChain, LangGraph, OpenAI Agent, Pydantic AI, CrewAI, AutoGen, or any custom framework. No need to rewrite your existing agents or switch frameworks.

See these framework-specific guides for detailed examples:

[![LangChain Logo](/mlflow-website/docs/latest/assets/images/langchain-logo-39d51f94cc9aebac2c191cca0e8189de.png)](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts/langchain-optimization.md)

[![LangGraph Logo](data:image/png;base64,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)](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts/langgraph-optimization.md)

[![OpenAI Agent Logo](/mlflow-website/docs/latest/assets/images/openai-agent-logo-8afddf736341f2b2dfe5cbb9caeaff45.png)](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts/openai-agent-optimization.md)

[![Pydantic AI Logo](/mlflow-website/docs/latest/assets/images/pydanticai-logo-e225ccce7e8699654c01ff1aeeed6b10.png)](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts/pydantic-ai-optimization.md)

### Using Custom Scorers[​](#using-custom-scorers "Direct link to Using Custom Scorers")

Define custom evaluation metrics to guide optimization:

python

```python
from typing import Any
from mlflow.genai.scorers import scorer


@scorer
def accuracy_scorer(outputs: Any, expectations: dict[str, Any]):
    """Check if output matches expected value."""
    return 1.0 if outputs.lower() == expectations.lower() else 0.0


@scorer
def brevity_scorer(outputs: Any):
    """Prefer shorter outputs (max 50 chars)."""
    return min(1.0, 50 / max(len(outputs), 1))


# Combine scorers with a weighted objective
def weighted_objective(scores: dict[str, Any]):
    return 0.7 * scores["accuracy_scorer"] + 0.3 * scores["brevity_scorer"]


# Use custom scorers
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-5"),
    scorers=[accuracy_scorer, brevity_scorer],
    aggregation=weighted_objective,
)

```

### Custom Optimization Algorithm[​](#custom-optimization-algorithm "Direct link to Custom Optimization Algorithm")

Implement your own optimizer by extending [`BasePromptOptimizer`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize.BasePromptOptimizer):

python

```python
from mlflow.genai.optimize import BasePromptOptimizer, PromptOptimizerOutput
from mlflow.genai.scorers import Correctness


class MyCustomOptimizer(BasePromptOptimizer):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def optimize(self, eval_fn, train_data, target_prompts, enable_tracking):
        # Your custom optimization logic
        optimized_prompts = {}
        for prompt_name, prompt_template in target_prompts.items():
            # Implement your algorithm
            optimized_prompts[prompt_name] = your_optimization_algorithm(
                prompt_template, train_data, self.model_name
            )

        return PromptOptimizerOutput(optimized_prompts=optimized_prompts)


# Use custom optimizer
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[prompt.uri],
    optimizer=MyCustomOptimizer(model_name="openai:/gpt-5"),
    scorers=[Correctness(model="openai:/gpt-5")],
)

```

### Multi-Prompt Optimization[​](#multi-prompt-optimization "Direct link to Multi-Prompt Optimization")

Optimize multiple prompts together:

python

```python
import mlflow
from mlflow.genai.scorers import Correctness

# Register multiple prompts
plan_prompt = mlflow.genai.register_prompt(
    name="plan",
    template="Make a plan to answer {{question}}.",
)
answer_prompt = mlflow.genai.register_prompt(
    name="answer",
    template="Answer {{question}} following the plan: {{plan}}",
)


def predict_fn(question: str) -> str:
    plan_prompt = mlflow.genai.load_prompt("prompts:/plan/1")
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-5",  # strong model
        messages=[{"role": "user", "content": plan_prompt.format(question=question)}],
    )
    plan = completion.choices[0].message.content

    answer_prompt = mlflow.genai.load_prompt("prompts:/answer/1")
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-5-mini",  # cost efficient model
        messages=[
            {
                "role": "user",
                "content": answer_prompt.format(question=question, plan=plan),
            }
        ],
    )
    return completion.choices[0].message.content


# Optimize both
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[plan_prompt.uri, answer_prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-5"),
    scorers=[Correctness(model="openai:/gpt-5")],
)

# Access optimized prompts
optimized_plan = result.optimized_prompts[0]
optimized_answer = result.optimized_prompts[1]

```

## Result Object[​](#result-object "Direct link to Result Object")

The API returns a [`PromptOptimizationResult`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize.PromptOptimizationResult) object:

python

```python
result = mlflow.genai.optimize_prompts(...)

# Access optimized prompts
for prompt in result.optimized_prompts:
    print(f"Name: {prompt.name}")
    print(f"Version: {prompt.version}")
    print(f"Template: {prompt.template}")
    print(f"URI: {prompt.uri}")

# Check optimizer used
print(f"Optimizer: {result.optimizer_name}")

# View evaluation scores (if available)
print(f"Initial score: {result.initial_eval_score}")
print(f"Final score: {result.final_eval_score}")

```

## Common Use Cases[​](#common-use-cases "Direct link to Common Use Cases")

### Improving Accuracy[​](#improving-accuracy "Direct link to Improving Accuracy")

Optimize prompts to produce more accurate outputs:

python

```python
from mlflow.genai.scorers import Correctness


result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-5"),
    scorers=[Correctness(model="openai:/gpt-5")],
)

```

### Optimizing for Safeness[​](#optimizing-for-safeness "Direct link to Optimizing for Safeness")

Ensure outputs are safe:

python

```python
from mlflow.genai.scorers import Safety


result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-5"),
    scorers=[Safety(model="openai:/gpt-5")],
)

```

### Model Switching and Migration[​](#model-switching-and-migration "Direct link to Model Switching and Migration")

When switching between different language models (e.g., migrating from `gpt-5` to `gpt-5-mini` for cost reduction), you may need to rewrite your prompts to maintain output quality with the new model. The [`mlflow.genai.optimize_prompts()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize_prompts) API can help adapt prompts automatically using your existing application outputs as training data.

See the [Auto-rewrite Prompts for New Models](/mlflow-website/docs/latest/genai/prompt-registry/rewrite-prompts.md) guide for a complete model migration workflow.

## Troubleshooting[​](#troubleshooting "Direct link to Troubleshooting")

### Issue: Optimization Takes Too Long[​](#issue-optimization-takes-too-long "Direct link to Issue: Optimization Takes Too Long")

**Solution**: Reduce dataset size or reduce the optimizer budget:

python

```python
# Use fewer examples
small_dataset = dataset[:20]

# Use faster model for optimization
optimizer = GepaPromptOptimizer(
    reflection_model="openai:/gpt-5-mini", max_metric_calls=100
)

```

### Issue: No Improvement Observed[​](#issue-no-improvement-observed "Direct link to Issue: No Improvement Observed")

**Solution**: Check your evaluation metrics and increase dataset diversity:

* Ensure scorers accurately measure what you care about
* Increase training data size and diversity
* Try to modify optimizer configurations
* Verify outputs format matches expectations

### Issue: Prompts Not Being Used[​](#issue-prompts-not-being-used "Direct link to Issue: Prompts Not Being Used")

**Solution**: Ensure `predict_fn` calls [`PromptVersion.format()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.model_registry.PromptVersion.format) during execution:

python

```python
# ✅ Correct - loads from registry
def predict_fn(question: str):
    prompt = mlflow.genai.load_prompt("prompts:/qa@latest")
    return llm_call(prompt.format(question=question))


# ❌ Incorrect - hardcoded prompt
def predict_fn(question: str):
    return llm_call(f"Answer: {question}")

```

## See Also[​](#see-also "Direct link to See Also")

* [Auto-rewrite Prompts for New Models](/mlflow-website/docs/latest/genai/prompt-registry/rewrite-prompts.md): Adapt prompts when switching between language models
* [Create and Edit Prompts](/mlflow-website/docs/latest/genai/prompt-registry/create-and-edit-prompts.md): Basic Prompt Registry usage
* [Evaluate Prompts](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/prompts.md): Evaluate prompt performance
