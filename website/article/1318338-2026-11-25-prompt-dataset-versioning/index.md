---
title: "Prompt and Eval Dataset Versioning: A Practical Guide"
description: "Learn how to version prompts and evaluation datasets together so every score is reproducible, every regression is traceable, and every release is auditable."
slug: prompt-dataset-versioning
tags:
  [
    eval dataset versioning,
    evaluation dataset versioning,
    prompt dataset versioning,
  ]
date: 2026-11-25
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1795438902905_Engineer-reviewing-prompt-and-dataset-versions.jpeg
---

![Engineer reviewing prompt and dataset versions](https://media.babylovegrowth.ai/blog-images/organization-30814/1795438902905_Engineer-reviewing-prompt-and-dataset-versions.jpeg)

Prompt dataset versioning means treating the prompt and the evaluation dataset that scores it as a pair of immutable, linked artifacts. Each evaluation run records which prompt version, which dataset version, which model settings and which scorer produced the result. Without that link, a score moving from 0.82 to 0.79 tells you nothing: you cannot say whether the prompt got worse or the test set changed. MLflow is built around this loop, with a prompt registry for versioning, automated evaluation, and tracing that ties each run back to what produced it. This guide covers what to version, how to structure it, and how to wire it into releases.

## TL;DR

- Version four things together: the prompt template, the model and its inference parameters, the evaluation dataset, and the scorer. A score is only comparable when all four are recorded with it.
- Make dataset versions immutable. Corrections and additions become a new version with a changelog entry, never an in-place edit.
- Promote prompts through development, staging and production, and let a versioned evaluation set gate each step in CI/CD.
- Define a release as a snapshot: code version, prompt version, model configuration and the evaluation dataset version it was validated against.
- [MLflow's prompt registry](https://mlflow.org/prompt-registry) and [evaluation tooling](https://mlflow.org/genai/evaluations) give teams one open-source place to do all of this.

## Table of Contents

- [What prompt dataset versioning covers](#what-prompt-dataset-versioning-covers)
- [Why the pair has to move together](#why-the-pair-has-to-move-together)
- [What each version should record](#what-each-version-should-record)
- [A versioning scheme that holds up](#a-versioning-scheme-that-holds-up)
- [Wiring versions into the workflow](#wiring-versions-into-the-workflow)
- [Common pitfalls](#common-pitfalls)
- [Where MLflow fits](#where-mlflow-fits)
- [Our perspective](#our-perspective)
- [FAQ](#faq)
- [Sources](#sources)

## What prompt dataset versioning covers

The phrase bundles two kinds of artifacts that teams often version separately, or not at all.

**Prompt-side artifacts.** The prompt template, the model identifier, inference parameters such as temperature and top_p, tool definitions and retrieval settings. AWS's Well-Architected Generative AI Lens notes that [prompt behavior can change drastically when you tune hyperparameters](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/genrel04-bp01.html) like temperature and top_p, and advises keeping versioned information on those ranges and validating them against prompt versions. A prompt version without its parameters is only half a record.

**Evaluation-side artifacts.** The dataset of inputs and expected outputs (or expectations such as required facts), plus the scorers or LLM judges that grade the output. Eval dataset versioning and evaluation dataset versioning are the same practice: pinning a specific, unchangeable snapshot of the test set to every run.

The Manning liveBook chapter on versioning in AI model evaluation defines the practice as [recording distinct revisions of data, models, code, prompts, metrics and configuration](https://livebook.manning.com/learn/concepts/version-control--tool/ai-model-evaluation) so changes can be identified and reproduced. For LLM judges it adds that teams should track judge prompt revisions and timestamps, log response schemas and model version identifiers, and store prompt text and hashes in a versioned store so a result can be traced to the prompt used.

That last point is easy to miss. If you grade with an LLM judge, the judge's prompt is itself a prompt that needs a version.

![What prompt dataset versioning covers](https://media.babylovegrowth.ai/blog-images/organization-30814/1795438969042_What-prompt-dataset-versioning-covers.jpeg)

## Why the pair has to move together

AWS's experimentation guidance is blunt about the failure mode. Changing any input creates a new experiment, and [without strict versioning and tracking it is difficult to compare results](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/dev-experimenting-experimentation-loops.html) and determine whether a change led to a genuine improvement.

Here is a worked example. Your support-summarization prompt scores 0.82 on a 200-row dataset. A teammate rewrites the prompt and the score drops to 0.79. Meanwhile someone else added 30 hard cases to the dataset last week. Three explanations are now live:

1. The new prompt is worse.
2. The harder dataset lowered every score.
3. The judge prompt or model changed.

If every run stores `prompt_version`, `dataset_version` and `scorer_version`, you separate these in minutes: re-run the old prompt on the new dataset and compare. If not, you debate it in a meeting.

## What each version should record

AWS's guidance on experimentation loops lists what to capture for reproducibility: the prompt template version, the exact LLM and inference parameters, the evaluation dataset version, raw outputs, performance scores, and full execution traces, including retrieved documents, the final prompt and tool calls. Translate that into a minimal run record:

![Six fields in a reproducible evaluation run](https://media.babylovegrowth.ai/blog-images/organization-30814/1795438993662_Six-fields-in-a-reproducible-evaluation-run.jpeg)

```json
{
  "prompt": "support_summary",
  "prompt_version": 7,
  "model": "provider/model-id",
  "params": { "temperature": 0.2, "top_p": 0.9 },
  "dataset": "support_summary_eval",
  "dataset_version": "2.1.0",
  "scorer_version": "judge-v3",
  "raw_outputs_uri": "s3://bucket/runs/2026-11-23/"
}
```

For the prompt itself, AWS Prescriptive Guidance asks for [structured documentation of each version](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/dev-experimenting-prompt-lifecycle.html): its purpose, the rationale for changes, the expected output format and key performance metrics from evaluations. The same page recommends a pull-request-style workflow for prompt changes in staging and production, which creates an audit trail of who changed what, when and why.

Traces matter here too. When a regression appears, an aggregate score tells you that it happened, and the trace tells you where: a retrieval miss, a changed tool call, or a malformed final prompt. [MLflow's observability](https://mlflow.org/genai/observability) captures the reasoning steps of LLM and agent runs, so the failing rows in your dataset link to something you can inspect.

## A versioning scheme that holds up

### Version prompts by what they change

Assigning meaning to version numbers keeps reviews fast. One workable convention, adapted from semantic versioning:

| Change                          | Example                                            | Bump  | What to run                             |
| ------------------------------- | -------------------------------------------------- | ----- | --------------------------------------- |
| Output contract changes         | New JSON field, different schema                   | Major | Full regression suite, update consumers |
| Behavior changes, same contract | New reasoning instruction, added few-shot examples | Minor | Full regression suite                   |
| Wording fix                     | Typo, clearer phrasing                             | Patch | Still run the suite                     |

The last row is deliberate. With language models, no edit can be guaranteed not to alter behavior, so even a patch gets evaluated.

AWS makes the same point from the other side: a prompt version's identity is defined by its performance. A name like `customer-support-summarisation-v2.1.0` should refer to [both the template and its known metrics](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/dev-experimenting-prompt-lifecycle.html) on a specific evaluation dataset.

### Version datasets as append-only snapshots

Treat an evaluation dataset like a test suite's golden file. Rules that keep comparisons fair:

- **Immutable snapshots.** Once a version is used in a reported result, it never changes. Fix a mislabeled row by publishing a new version.
- **A changelog.** One line per version: what was added, corrected or removed, and why.
- **Separate commits or versions for data and prompt changes.** If both change at once, you cannot attribute the score movement.
- **A bridge run.** When you ship a new dataset version, evaluate the current production prompt on both the old and new versions. The gap between those two scores is the effect of the dataset change, not of your candidate.
- **Content hashes.** Store a hash of the snapshot alongside the version label so silent edits are detectable.

A simple version-to-semver mapping works for datasets as well: a major bump when the task schema or scoring rubric changes (old scores are no longer comparable), a minor bump when rows are added, a patch when a label is corrected.

### Grow the dataset from production

AWS's GenAIOps guidance describes the evaluation dataset being [expanded with real-world examples, user-reported failures and adversarial test cases](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/preprod-hardening.html), at which point it becomes the official benchmark for all application-stack changes. The same page states that any modification to it must be a versioned change so performance comparisons between application versions stay fair and consistent.

In practice: when a production trace fails, copy its input into the next dataset version with the expected behavior, and record the trace ID as the row's provenance.

## Wiring versions into the workflow

### Step 1: Put prompts in a registry, not in code strings

AWS Prescriptive Guidance recommends a [centralized prompt store or registry](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/dev-experimenting-prompt-lifecycle.html), ranging from a Git repository of YAML or JSON files to a dedicated prompt management tool, so prompt engineers and non-technical domain experts can update prompts without a full application redeployment. A registry also gives you the alias pattern: your application loads `production`, and promotion or rollback is a pointer change, not a deploy.

MLflow's [Prompt Registry](https://mlflow.org/prompt-registry) is the open-source option for this step, built for prompt management and versioning of LLM and agent prompts. A minimal registration looks like this:

```python
import mlflow

mlflow.genai.register_prompt(
    name="support_summary",
    template="Summarize this ticket in two sentences: {{ticket}}",
)
```

Each call to register under the same name creates a new version you can reference in run records.

### Step 2: Run evaluation against a pinned dataset

Load one specific dataset version, run the prompt, and score. Log the prompt version and dataset version as run metadata so the result is never an orphan. For open-ended outputs, LLM-as-a-judge scoring handles criteria like correctness or tone at scale; [MLflow's LLM-as-a-Judge evaluation](https://mlflow.org/llm-as-a-judge) is built for exactly that, and the judge configuration should carry its own version number.

Because hosted models do not always return identical outputs for identical inputs, run each configuration several times on high-stakes comparisons and compare the spread, not a single number.

### Step 3: Gate promotion in CI/CD

AWS's GenAIOps guidance describes the pipeline running the new application version against the versioned evaluation dataset with LLM-as-a-judge metrics and [failing the build if scores drop below a predefined threshold](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/preprod-hardening.html). That is how a regression is prevented rather than discovered by users.

Pair it with environment stages. AWS recommends promoting prompts from development, where iteration is fast, to staging, where they are tested rigorously against a proven evaluation set, and to production only after passing all staging tests. The proof-of-concept stage tolerates loose versioning. Preproduction does not: prompts become configuration-as-code with small, targeted changes that pass formal review and the full automated evaluation suite.

### Step 4: Define a release as a snapshot

The same AWS guidance defines a mature application version as a complete immutable snapshot of the stack: code version, prompt version, model configuration and the evaluation dataset version it was validated against, rather than just a Git commit. A release manifest can be as small as this:

```yaml
release: support-agent-2026.11.2
code_commit: 3f9a1c2
prompt: support_summary@7
model: provider/model-id
params: { temperature: 0.2, top_p: 0.9 }
eval_dataset: support_summary_eval@2.1.0
scorer: judge-v3
validated_score: 0.84
```

When production quality dips, you open the manifest, pull the traces from that release, and know exactly which dataset version the team trusted when it shipped.

### Step 5: Rehearse rollback

If production loads a prompt through an alias, rollback is repointing it to the previous version. Practice it before you need it, and confirm the previous version still passes the dataset version it was released with.

## Common pitfalls

- **Editing the dataset in place.** The most damaging mistake, because every historical score silently changes meaning.
- **Versioning prompts but not judges.** A changed judge prompt shifts every score without touching your application.
- **Mixing data and prompt changes in one release.** You lose the ability to attribute the movement.
- **Overfitting to the dataset.** If prompt authors tune against the same rows indefinitely, scores climb while real quality does not. Keep growing the set from production failures and consider a held-out slice.
- **Skipping parameters.** Temperature and top_p belong in the prompt's version record.
- **No provenance on rows.** Without knowing where a row came from, you cannot decide whether it is still representative.

## Where MLflow fits

MLflow is an open-source platform for the lifecycle management of GenAI and LLM applications, and the pieces above map onto it directly: prompt management and versioning in the registry, automated LLM-based evaluation, and deep tracing of agentic reasoning. All of its features are open source under Linux Foundation governance, with no enterprise feature paywalls, and it integrates with major AI and ML frameworks through OpenTelemetry. Teams can start with the free project, then add a gateway and deployment as they move from prototype to production.

If you want a map of the category first, [our comparison of prompt versioning platforms](https://mlflow.org/articles/top-llm-prompt-versioning-platforms-3) walks through what to look for, and the tag pages on [effective prompt versioning methods](https://mlflow.org/articles/tags/effective-prompt-versioning-methods) and [LLM prompt history tracking](https://mlflow.org/articles/tags/llm-prompt-history-tracking) collect more reading.

To put the workflow above into practice, begin with the [MLflow Prompt Registry](https://mlflow.org/prompt-registry) for versioned prompts and [Agent and LLM evaluation](https://mlflow.org/genai/evaluations) to score them against a pinned dataset. The project home is [mlflow.org](https://mlflow.org).

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

## Our perspective

> Most teams we talk to about evaluation start by asking which score is best. We would start one step earlier: which score can you still trust in three months?
>
> A score is a measurement, and a measurement only means something when you know the instrument. That instrument is the dataset, the scorer and the prompt under test. We built MLflow around a complete lifecycle for that reason: prompts versioned in a registry, evaluation that runs automatically, and traces that show why a row failed. Keeping them in one open-source platform means a run record can point to all of them without glue code.
>
> Our advice is simple. Treat the evaluation dataset as a release artifact, give every prompt an alias you can move in seconds, and never let a number travel without its versions. Nothing about this requires a license: every MLflow feature is open source.
>
> &nbsp;
>
> _Kevin_

## FAQ

### Is eval dataset versioning different from evaluation dataset versioning?

No. They are two names for the same practice: pinning an immutable snapshot of the test set to each evaluation run. The useful distinction is between versioning the dataset alone and versioning it together with the prompt, model settings and scorer, which is what makes results comparable.

### How big should an evaluation dataset be before I version it?

Version it from the first row. A small set of a few dozen representative cases is enough to catch many regressions, and it grows as production surfaces new failures. Early versioning costs almost nothing, while a missing history cannot be reconstructed later.

### Should I store datasets in Git or a registry?

For small datasets, a single JSONL file committed with the code it grades gives you history, review and diffs for free. As the set grows or includes large files, move to content-addressed storage or a dedicated dataset store, and keep the version label and hash in the run record either way.

### Do I need to re-run evaluation for a one-word prompt change?

Yes. Language models can change behavior from any edit, including a wording fix. Running the regression suite on every version, with a threshold in CI, is cheaper than finding the change in production.

## Sources

- [Creating evaluation loops for generative AI experimentation (AWS Prescriptive Guidance)](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/dev-experimenting-experimentation-loops.html)
- [Hardening the generative AI application through a GenAIOps approach (AWS Prescriptive Guidance)](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/preprod-hardening.html)
- [Managing the generative AI prompt lifecycle (AWS Prescriptive Guidance)](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-lifecycle-operational-excellence/dev-experimenting-prompt-lifecycle.html)
- [GENREL04-BP01 Implement a prompt catalog (AWS Well-Architected Generative AI Lens)](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/genrel04-bp01.html)
- [Versioning in AI Model Evaluation (Manning liveBook)](https://livebook.manning.com/learn/concepts/version-control--tool/ai-model-evaluation)
