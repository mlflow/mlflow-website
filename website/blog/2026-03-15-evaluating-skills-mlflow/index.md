---
title: "Evaluating and Refining Claude Code Skills with MLflow"
description: How to test Claude Code skills using MLflow tracing and LLM judges, and close a self-improvement loop where Claude Code refines its own guidance documents.
slug: evaluating-skills-mlflow
authors: [mlflow-maintainers]
tags:
  [
    genai,
    evaluation,
    tracing,
    coding-agents,
    claude-code,
    skills,
    llm-judges,
  ]
thumbnail: /img/blog/evaluating-skills-mlflow-thumbnail.svg
image: /img/blog/evaluating-skills-mlflow-thumbnail.svg
date: 2026-03-15
---

## The Skill Testing Problem

You wrote a Claude Code skill — a `SKILL.md` file that tells Claude how to evaluate your AI agent, instrument code with tracing, or set up a complete MLflow experiment. You tested it manually and it looks right. But how do you *know* it reliably works?

The problem is fundamental: a skill guides LLM behavior, and LLM behavior is semantic, not syntactic. You can't assert `output == expected_output`. You need to observe *what Claude did* — which tools it called, what steps it took, whether it made the right judgment calls.

Here's the loop we built to solve this:

1. **Trace** Claude Code's own execution with MLflow while it runs the skill
2. **Judge** those traces with LLM scorers that check for correct behavior
3. **Refine** the skill based on failing judges — automatically, with Claude Code itself

---

## What Is a Claude Code Skill?

A skill is a markdown file with YAML frontmatter that Claude Code reads before acting. The `description` field tells Claude when to load it; the body provides instructions, examples, and tool guidance.

Here's the frontmatter from the `agent-evaluation` skill in the [MLflow Skills repo](https://github.com/mlflow/skills):

```yaml
---
name: agent-evaluation
description: Use this when you need to EVALUATE OR IMPROVE or OPTIMIZE an existing
  LLM agent's output quality ... Evaluates agents systematically using MLflow
  evaluation with datasets, scorers, and tracing. IMPORTANT - Always also load
  the instrumenting-with-mlflow-tracing skill before starting any work.
allowed-tools: Read, Write, Bash, Grep, Glob, WebFetch
---
```

The body is a complete walkthrough: discover the agent structure, set up tracing, register scorers, create an evaluation dataset, and run `mlflow.genai.evaluate()`. It's authoritative guidance — whatever Claude reads here shapes every decision it makes.

This is why skills are hard to test: the contract is semantic. "Did Claude discover the agent's entry point before trying to evaluate it?" cannot be checked with `assertEqual`.

---

## Tracing Claude Code with MLflow

MLflow 3.9 ships with built-in support for tracing Claude Code itself. A single command instruments every session in a project directory:

```bash
mlflow autolog claude /path/to/project \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment-id 42
```

From that point on, every tool call Claude makes — reading a file, running a shell command, calling the Claude API — becomes a span in a trace. The trace is a ground-truth record: not what Claude said it did, but what it *actually* did, in order, with full inputs and outputs.

We run two MLflow experiments side-by-side:

| Experiment | Contains |
|---|---|
| `evaluation-experiment` | Agent runs, datasets, scorer results — the work the skill is guiding |
| `claude-code-tracing-experiment` | Claude's own execution: file reads, Bash calls, reasoning spans |

This separation keeps the signal clean. The tracing experiment is purely Claude's behavior, which judges can inspect without noise from the agent under evaluation.

---

## Writing Judges as MLflow Scorers

A judge is an `@scorer` function that receives a Claude Code trace and returns `Feedback`. Two patterns cover almost every test:

**Rule-based judge** — check a side effect in MLflow directly:

```python
from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

@scorer(name="dataset-created")
def dataset_created(trace) -> Feedback:
    client = MlflowClient()
    datasets = client.search_datasets(experiment_ids=[eval_exp_id])
    if datasets:
        return Feedback(
            value="yes",
            rationale=f"Found {len(datasets)} dataset(s) in experiment {eval_exp_id}",
        )
    return Feedback(
        value="no",
        rationale=f"No datasets found in experiment {eval_exp_id}",
    )
```

This judge ignores the trace entirely — it just checks whether Claude created the artifact we expected.

**LLM judge** — use `make_judge()` to semantically analyze the trace:

```python
from mlflow.genai.judges import make_judge
from typing import Literal

agent_ran_instrumented_code = make_judge(
    name="agent-ran-instrumented-code",
    instructions=(
        "Examine the {{ trace }} and determine whether the agent ran the "
        "application or agent code after adding MLflow tracing instrumentation. "
        "Look for evidence that the agent executed the instrumented program "
        "(e.g., running a CLI command, calling an entry point, executing a script). "
        "Return 'yes' if the agent ran the code after instrumenting it, 'no' otherwise."
    ),
    feedback_value_type=Literal["yes", "no"],
)
```

This judge does what no rule can: it reads the actual span tree and reasons about whether the *sequence* of actions was correct.

The full test for `agent-evaluation` uses six judges, each checking one requirement:

- `dataset-created` — did Claude call `mlflow.genai.datasets.create_dataset()`?
- `scorer-registered` — did Claude register a scorer before evaluation?
- `evaluation-run-created` — did `mlflow.genai.evaluate()` produce a run?
- `agent-trace-logged` — did the agent under evaluation produce traces?
- `tracing-skill-invoked` — did Claude load the tracing skill as instructed?
- `agent-eval-skill-invoked` — did Claude actually read and follow the skill?

Together they define the acceptance criteria for the skill. If all six pass, the skill works.

---

## The Test Harness

A YAML config ties everything together:

```yaml
name: "agent-evaluation-test"
project_dir: mlflow-agent

setup_script: tests/scripts/setup_agent_eval.py
skills:
  - agent-evaluation
  - instrumenting-with-mlflow-tracing

prompt: "Evaluate the output quality of my agent. Do not ask for input."
timeout_seconds: 900
allowed_tools: "Bash,Read,Write,Edit,Grep,Glob,WebFetch"

judges:
  - tests/judges/dataset_created.py
  - tests/judges/scorer_registered.py
  - tests/judges/evaluation_run_created.py
  - tests/judges/agent_trace_logged.py
  - tests/judges/tracing_skill_invoked.py
  - tests/judges/agent_eval_skill_invoked.py
```

`test_skill.py` orchestrates the full sequence:

1. Start a local MLflow server and create both experiments
2. Run the setup script (clone the target agent repo, seed test data)
3. Install skills into `PROJECT_DIR/.claude/skills/`
4. Enable tracing: `mlflow autolog claude PROJECT_DIR`
5. Run `claude -p "PROMPT" --dangerously-skip-permissions` headlessly
6. Wait for traces to flush, then run all judges on traces created after step 4

```bash
python tests/test_skill.py tests/configs/agent_evaluation.yaml
```

Sample output:

```
[PASS] dataset-created on trace tr-abc123: yes
[PASS] scorer-registered on trace tr-abc123: yes
[PASS] evaluation-run-created on trace tr-abc123: yes
[FAIL] agent-ran-instrumented-code on trace tr-abc123: no
       Rationale: Claude added tracing decorators but did not execute the agent
                  afterward. No CLI invocation was found after the instrumentation step.
```

That last line is actionable. Claude didn't run the agent after instrumenting it — the skill didn't make this step explicit enough.

---

## The Automated Refinement Loop

Here's where it gets interesting. When judges fail, we don't fix the skill manually. We feed the failing trace and judge rationale back to Claude Code and ask it to fix `SKILL.md`.

The loop:

```
while judges_fail:
    run test → collect failing judge rationales
    claude -p "Judge '{name}' failed with rationale: '{rationale}'. Fix SKILL.md."
    rerun test
```

Two real examples from the `agent-evaluation` skill's history illustrate exactly how this played out.

**Example 1: Claude bypassing MLflow entirely**

Early runs saw `dataset-created` and `evaluation-run-created` both fail. Inspecting the trace revealed why: Claude had created an `evaluation/eval_dataset.py` file with a hand-rolled evaluation loop — completely bypassing MLflow's APIs. No dataset in MLflow, no run logged. The judges had nowhere to find success.

Claude Code read the trace, saw the custom file creation, and made this addition to `SKILL.md`:

```diff
+## ⛔ CRITICAL: Must Use MLflow APIs
+
+**DO NOT create custom evaluation frameworks.** You MUST use MLflow's native APIs:
+
+- **Datasets**: Use `mlflow.genai.datasets.create_dataset()` - NOT custom test case files
+- **Scorers**: Use `mlflow.genai.scorers` and `mlflow.genai.judges.make_judge()` - NOT custom scorer functions
+- **Evaluation**: Use `mlflow.genai.evaluate()` - NOT custom evaluation loops
+
+**Why?** MLflow tracks everything (datasets, scorers, traces, results) in the experiment.
+Custom frameworks bypass this and lose all observability.
+
+If you're tempted to create `evaluation/eval_dataset.py` or similar custom files,
+STOP. Use `scripts/create_dataset_template.py` instead.
```

Next run: `dataset-created` and `evaluation-run-created` both pass.

**Example 2: A missing skill dependency**

The `tracing-skill-invoked` judge kept failing: Claude was attempting evaluation without first loading the tracing skill, even though `SKILL.md` listed it as a prerequisite. The problem was in the skill's `description` field — the trigger text Claude reads *before* loading the skill body. It said nothing about the tracing skill dependency.

One line added to the description:

```diff
-description: Use this when you need to EVALUATE OR IMPROVE or OPTIMIZE an existing
- LLM agent's output quality ...
+description: Use this when you need to EVALUATE OR IMPROVE or OPTIMIZE an existing
+ LLM agent's output quality ... IMPORTANT - Always also load the
+ instrumenting-with-mlflow-tracing skill before starting any work.
```

The `tracing-skill-invoked` judge has passed on every run since.

Both fixes share the same shape: Claude reads the trace, identifies the gap between what it did and what the judge expected, and makes a targeted edit. The skill author never had to debug by hand.

MLflow is what makes the loop *grounded*. Claude isn't guessing what went wrong from a description in the prompt. It's reading the full span tree: the exact tool calls it made, in order, with timestamps. The diagnosis is direct.

---

## What We Learned

A few patterns emerged from running this system on the `agent-evaluation` skill:

**Traces reveal surprising gaps.** The `tracing-skill-invoked` judge caught cases where Claude attempted evaluation without loading the tracing skill — despite SKILL.md listing it as a prerequisite. The fix was a single sentence in the `description` field, the text Claude reads before loading the skill body. Without the trace, this failure mode would have been invisible — there's nothing in the output that signals a missing skill dependency.

**Rule-based judges are faster and more reliable.** For side effects you can check directly — datasets created, scorers registered, runs logged — rule-based judges are deterministic and cheap. Reserve LLM judges for behavioral questions like "did Claude follow the right sequence?" or "did Claude understand the agent's purpose before evaluating it?"

**Write judges before you polish the skill.** The judges are the specification. Writing them first forces you to articulate what success actually means — and often reveals that your initial skill draft was underspecified. A skill that passes all its judges on the first try probably has judges that are too weak.

**The rationale is the real value.** `Feedback.rationale` is what makes automated refinement possible. A bare yes/no from an LLM judge gives Claude Code nothing to work with. A rationale like "Claude added tracing decorators but never executed the agent afterward" gives it exactly what it needs to make a targeted, surgical fix.

---

## Get Started

```bash
git clone https://github.com/mlflow/skills.git
cd skills
python tests/test_skill.py tests/configs/agent_evaluation.yaml
```

The test spins up a local MLflow server, clones a sample agent repo, runs Claude Code headlessly, and prints judge results. Add your own judges to `tests/judges/` and reference them in a new YAML config.

The pattern works for any Claude Code skill: write the judges first, run the test, let Claude fix what's broken.
