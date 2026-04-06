---
title: Genie Space Improvement Generator
slug: genie-space-analyzer
description: Take traces that failed evaluation, combine them with your Genie space config, and generate copy-paste-ready fixes with an LLM.
tags: [databricks, genie, evaluation, agents]
---

This cookbook takes the traces that failed evaluation in the [previous cookbook](/cookbook/genie-evaluation-judges), combines them with the current Databricks [Genie space](https://docs.databricks.com/en/genie/index.html) configuration, and passes everything to an LLM that generates copy-paste-ready fixes: text instructions, SQL expressions, example queries, and benchmarks.

<!-- truncate -->

## Prerequisites

This cookbook requires evaluation results from the [Genie Evaluation with LLM Judges](/cookbook/genie-evaluation-judges) cookbook.

## Step 1: Configure

Set your Genie space ID. The OpenAI client connects to [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html).

```python
from databricks.sdk import WorkspaceClient
from openai import OpenAI
import json
import mlflow

w = WorkspaceClient()

client = OpenAI(
    base_url=f"{w.config.host}/serving-endpoints",
    api_key=w.config.token,
)

SPACE_ID = "your-space-id"
EXPERIMENT_NAME = (
    f"/Users/{w.current_user.me().user_name}/genie_eval"
)

mlflow.set_experiment(EXPERIMENT_NAME)
```

## Step 2: Load Failed Traces from Evaluation

Pull traces from the experiment and filter to those where at least one judge flagged a problem.

```python
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
all_traces = mlflow.search_traces(
    locations=[experiment.experiment_id],
    return_type="list",
)

failed_conversations = []
for trace in all_traces:
    assessments = trace.info.assessments or []
    failures = [a for a in assessments if a.value == "no"]
    if not failures:
        continue

    root = trace.data.spans[0]
    failed_conversations.append({
        "question": root.inputs.get("question"),
        "response": root.outputs.get("response"),
        "generated_sql": root.outputs.get("generated_sql"),
        "error": root.outputs.get("error"),
        "failed_checks": [
            f"{a.name}: {a.value} - {a.rationale}"
            for a in failures
        ],
    })

print(
    f"{len(failed_conversations)} / {len(all_traces)} "
    f"traces had failures"
)
```

## Step 3: Extract the Genie Space Configuration

Pull the current space config so the LLM knows what tables, instructions, and examples are already in place.

```python
space = w.genie.get_space(
    space_id=SPACE_ID, include_serialized_space=True
)
config = (
    json.loads(space.serialized_space)
    if space.serialized_space
    else {}
)

tables = config.get("data_sources", {}).get("tables", [])
instructions = config.get("instructions", {})
text_instructions = instructions.get("text_instructions", [])
example_sqls = instructions.get("example_question_sqls", [])

print(f"Space: {space.title}")
print(
    f"Tables: {len(tables)}, "
    f"Instructions: {len(text_instructions)}"
)
```

## Step 4: Generate Fixes with an LLM

Feed the failed conversations and their failure reasons alongside the space config into the LLM via the OpenAI client and [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html).

```python
table_names = [t["identifier"] for t in tables]

system_prompt = (
    "You are an expert Databricks AI/BI Genie space consultant. "
    "You will be given conversations where Genie gave wrong or "
    "incomplete answers, along with the specific checks that failed. "
    "Generate specific, copy-paste-ready fixes: SQL expressions, "
    "text instructions, example SQL, and column descriptions. "
    "Never give vague advice. Always write the actual implementation."
)

analysis_prompt = f"""Fix the issues found in these Genie conversations.

## FAILED CONVERSATIONS
{json.dumps(failed_conversations[:20], indent=2)}

## CURRENT SPACE CONFIG
Title: {space.title}
Tables: {', '.join(table_names[:10])}
Text instructions: {len(text_instructions)}
Example SQL: {len(example_sqls)}

For each failed conversation, provide a specific fix: a new text
instruction, SQL expression, example query, or column description
that would prevent the failure. Prioritize by impact."""


@mlflow.trace
def analyze_genie_space(user_prompt, sys_prompt):
    response = client.chat.completions.create(
        model="databricks-claude-opus-4-6",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8000,
        temperature=0.2,
    )
    return response.choices[0].message.content


if not failed_conversations:
    print("No failures found - nothing to analyze!")
else:
    recommendations = analyze_genie_space(
        analysis_prompt, system_prompt
    )
    print(recommendations)
```

## Step 5: Apply Recommendations

The LLM generates text instructions, SQL expressions, and example queries you can copy directly into your Genie space settings.

![Updated Genie space with generated text instructions](/img/cookbook/databricks-genie/analyzer-genie-instructions.png)

Review the output and apply the suggested changes to your Genie space:

- Add text instructions in the Genie space settings
- Add SQL expressions and example queries
- Update [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html) table column descriptions

After applying the changes, ask your Genie space some new questions to create fresh conversations. Then re-run the [Conversation Tracing Pipeline](/cookbook/genie-tracing-pipeline) and [Evaluation](/cookbook/genie-evaluation-judges) to see if the changes improved Genie's answers.
