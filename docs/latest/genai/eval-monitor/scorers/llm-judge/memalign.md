# MemAlign Optimizer (Experimental)

Experimental Feature

MemAlign is an experimental optimizer. The API may change in future releases.

**MemAlign** is an experimental optimizer that uses a dual-memory system inspired by human cognition to learn from natural language feedback. It offers significant speed and cost advantages over traditional prompt optimizers.

#### Fast Alignment

Up to 100× faster than traditional prompt optimizers like SIMBA, enabling rapid iteration on judge quality.

#### Lower Cost

Significantly lower cost per alignment cycle compared to traditional prompt optimizers.

#### Few-Shot Learning

Shows visible improvement with just a handful of examples—no need to front-load massive labeling efforts.

#### Dual-Memory System

Combines generalizable guidelines (semantic memory) with concrete examples (episodic memory) for robust alignment.

## Requirements[​](#requirements "Direct link to Requirements")

For alignment to work:

* Traces must contain human assessments (labels) with the same name as the judge
* Natural language feedback (rationale) is highly recommended for better alignment
* A mix of positive and negative labels is recommended

## How MemAlign Works[​](#how-memalign-works "Direct link to How MemAlign Works")

MemAlign maintains two types of memory:

* **Semantic Memory**: Stores distilled guidelines extracted from feedback. When an expert explains their decision, MemAlign extracts generalizable rules like "Always evaluate safety based on intent, not just language."

* **Episodic Memory**: Holds specific examples, particularly edge cases where the judge made mistakes. These serve as concrete anchors for situations that resist easy generalization.

When evaluating new inputs, MemAlign constructs a dynamic context by gathering all principles from semantic memory and retrieving the most relevant examples from episodic memory—similar to how human judges reference both a rulebook and case history.

## Installation[​](#installation "Direct link to Installation")

MemAlign requires additional dependencies:

bash

```bash
pip install mlflow[genai] dspy jinja2 tqdm

```

## Basic Usage[​](#basic-usage "Direct link to Basic Usage")

See [make\_judge documentation](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md) for details on creating judges.

python

```python
import mlflow
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer

# Create a judge
judge = make_judge(
    name="politeness",
    instructions=(
        "Given a user question, evaluate if the chatbot's response is polite and respectful. "
        "Consider the tone, language, and context of the response.\n\n"
        "Question: {{ inputs }}\n"
        "Response: {{ outputs }}"
    ),
    feedback_value_type=bool,
    model="openai:/gpt-5-mini",
)

# Create the MemAlign optimizer
optimizer = MemAlignOptimizer(
    reflection_lm="openai:/gpt-5-mini",
)

# Retrieve traces with human feedback
traces = mlflow.search_traces(return_type="list")

# Align the judge
aligned_judge = judge.align(traces=traces, optimizer=optimizer)

```

## Parameters[​](#parameters "Direct link to Parameters")

| Parameter         | Type  | Default                            | Description                                                                        |
| ----------------- | ----- | ---------------------------------- | ---------------------------------------------------------------------------------- |
| `reflection_lm`   | `str` | Required                           | Model used for extracting guidelines from feedback.                                |
| `retrieval_k`     | `int` | `5`                                | Number of relevant examples to retrieve from episodic memory during inference.     |
| `embedding_model` | `str` | `"openai:/text-embedding-3-small"` | Model for episodic memory retrieval. Must be in `<provider>:/<model-name>` format. |

Note: The number of parallel threads for LLM calls during guideline distillation can be configured via the `MLFLOW_GENAI_OPTIMIZE_MAX_WORKERS` environment variable (default: 8).

## Inspecting Learned Knowledge[​](#inspecting-learned-knowledge "Direct link to Inspecting Learned Knowledge")

After alignment, you can inspect what the judge has learned by viewing the updated instructions:

python

```python
# View the updated instructions with distilled guidelines
print(aligned_judge.instructions)
# Output includes appended guidelines like:
# "Distilled Guidelines (7):
#   - Responses must be factually accurate...
#   - Use neutral, descriptive language..."

```

## Removing Feedback (Unalignment)[​](#removing-feedback-unalignment "Direct link to Removing Feedback (Unalignment)")

If requirements change or feedback was incorrect, you can selectively remove learned knowledge:

python

```python
import mlflow

# Retrieve traces with outdated or incorrect feedback
traces_to_forget: list[mlflow.entities.Trace] = mlflow.search_traces(
    filter_string="tag.outdated = 'true'",
    return_type="list",
)

# Remove knowledge derived from those traces
updated_judge = aligned_judge.unalign(traces=traces_to_forget)

```

## Debugging[​](#debugging "Direct link to Debugging")

To debug the optimization process, enable DEBUG logging:

python

```python
import logging

logging.getLogger("mlflow.genai.judges.optimizers.memalign").setLevel(logging.DEBUG)
aligned_judge = judge.align(traces=traces, optimizer=optimizer)

```
