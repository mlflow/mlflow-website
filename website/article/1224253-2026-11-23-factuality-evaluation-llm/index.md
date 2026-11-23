---
title: "Factuality Evaluation for LLMs: Metrics, Rubrics, and Eval as Code"
description: "How to evaluate LLM factuality: separate it from groundedness, build test sets, write judge rubrics, and run evals as code in CI to catch regressions."
slug: factuality-evaluation-llm
tags: [evaluation as code llm, eval rubrics llm, factuality evaluation llm]
date: 2026-11-23
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1795285870292_Engineer-reviews-an-LLM-factuality-evaluation.jpeg
---

![Engineer reviews an LLM factuality evaluation](https://media.babylovegrowth.ai/blog-images/organization-30814/1795285870292_Engineer-reviews-an-LLM-factuality-evaluation.jpeg)

Factuality evaluation checks whether the claims in a model's answer are true against an external source of truth. To do it well, you need four things: a clear definition that separates it from groundedness and hallucination, a test set that looks like your production traffic, a judge rubric validated against human reviewers, and an eval that runs as code on every change. We built MLflow around that loop of tracing, automated LLM-as-a-Judge evaluation, prompt versioning and gateway governance, so the same open-source platform covers each step below.

## TL;DR

- Factuality compares claims with external ground truth. Groundedness and faithfulness compare them with the context you retrieved. Measure the one that matches your failure mode.
- A good factuality set is scoped, production-like, diverse, decontaminated and kept alive as a living dataset.
- LLM judges disagree with each other and with humans. Validate every rubric on your own domain before you trust its numbers.
- Treat the eval as code: versioned dataset, versioned rubric, versioned prompt, run on every change, compare against a baseline.
- Track hallucination and non-response together, or your prompt will "fix" one by inflating the other.

## Table of Contents

- [What factuality evaluation actually measures](#what-factuality-evaluation-actually-measures)
- [Build a factuality test set that looks like production](#build-a-factuality-test-set-that-looks-like-production)
- [Methods and benchmarks: what each one tells you](#methods-and-benchmarks-what-each-one-tells-you)
- [Write LLM-as-a-Judge rubrics you can defend](#write-llm-as-a-judge-rubrics-you-can-defend)
- [Evaluation as code: run factuality checks on every change](#evaluation-as-code-run-factuality-checks-on-every-change)
- [Keeping factuality in view in production](#keeping-factuality-in-view-in-production)
- [Put it to work with MLflow](#put-it-to-work-with-mlflow)
- [Perspective](#perspective)
- [FAQ](#faq)
- [Sources](#sources)

## What factuality evaluation actually measures

The four terms below get used interchangeably, and they should not be. A [production guide from GMI Cloud](https://www.gmicloud.ai/en/blog/reducing-hallucination-in-production-llm-apps-grounding-citations-and-faithfulness-scoring) separates them like this:

| Metric                | Compared against                                              | What it catches                                       |
| --------------------- | ------------------------------------------------------------- | ----------------------------------------------------- |
| Factuality            | External ground truth (a gold set, a knowledge base, the web) | Claims that are simply wrong                          |
| Faithfulness          | The whole retrieved context                                   | Answers that contradict or invent beyond the evidence |
| Groundedness          | The retrieved context, claim by claim or sentence by sentence | Individual unsupported statements                     |
| Hallucination scoring | Nothing, or the final answer alone                            | A reference-free safety net                           |

The practical consequence: an answer can be perfectly faithful to a wrong document. If your RAG index contains an outdated policy, a faithfulness score will be high and the answer will still be false. Only a factuality check against ground truth exposes that.

Academic work draws the same line. The EMNLP 2024 survey [Factuality of Large Language Models](https://aclanthology.org/2024.emnlp-main.1088.pdf) treats hallucination as content not grounded in the given source, and factuality errors as a failure to learn and use factual knowledge. It also notes that a model can be factually correct and still hallucinate, for example by drifting off-topic or adding unrequested detail.

The distinction matters because the fixes differ. The KGELL benchmark catalogue by [Osborne et al.](https://kgell-cost.eu/wp-content/uploads/2026/07/KGELL_D1_1_final.pdf) notes that factfulness failures call for knowledge grounding, while hallucination failures call for input-faithfulness constraints. If you route every error to one dashboard number, you will pick the wrong fix.

### Which metric for which application

The same GMI Cloud guide gives a selection rule that holds up in practice:

- **RAG question answering:** groundedness plus faithfulness.
- **Summarization:** faithfulness.
- **Multi-step agents:** faithfulness to tool outputs, plus factuality against a gold set with human review.

Agents are the hard case. Each step can be faithful to the previous tool result while the final claim is wrong, so you need to see the intermediate reasoning, not only the last message. This is where tracing earns its place: MLflow's observability records the agent's reasoning and tool calls, so a failed factuality check can be traced back to the step that introduced the error.

If you want a quick outside view while you define your metrics, a [multi-LLM audit tool](https://babylovegrowth.ai/free-tools/multi-llm-audit) shows how several models answer the same question, which makes disagreement on facts easy to spot before you build a formal set.

## Build a factuality test set that looks like production

Most bad factuality numbers come from bad datasets, not bad models. The arXiv paper [A Practical Guide for Evaluating LLMs and LLM-Reliant Systems](https://arxiv.org/html/2506.13023v2) gives five properties, the "5 D's":

1. **Defined scope:** you know which tasks and domains the set covers.
2. **Demonstrative:** it reflects real production usage.
3. **Diverse:** it spans topics, phrasings and difficulty levels.
4. **Decontaminated:** it does not overlap with training data the model may have memorized.
5. **Dynamic:** it is a living body of work, not a one-time export.

OpenAI's [evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices) list the sources you can draw on: synthetic data, domain-specific data, purchased data, human-curated data, production data and historical data. For factuality, production logs are the richest source because they contain the questions users actually ask, including the odd ones.

### Pick prompts that can be checked

Not every prompt can be scored for truth. The [FACTORY](https://arxiv.org/html/2508.00109) prompt set for long-form factuality selects prompts that are fact-seeking, self-contained, have time-stable answers, and do not elicit unsafe responses. Borrow those four filters. A question whose correct answer changes every quarter will make your regression suite flaky, so either pin it to a date or keep it out of the gold set.

### Don't trust your reference answers blindly

Reference evidence can be wrong too. The IRB paper on [automated factuality benchmark generation](https://arxiv.org/html/2602.08070v1) does not assume that every key point and source document pair is reliable. It prompts an LLM to check whether each key point is supported by its document, and keeps only the validated ones. You can apply the same step to a synthetic set: generate candidate question-and-answer pairs, validate each against its source, then have a human review a sample before the set becomes your baseline.

Probe the edges as well. The practical guide above recommends generating questions about fictitious entities (its example is a made-up CVE number) and measuring whether the model says it does not know. Pair that with questions about real entities, where a refusal is a failure. Together they give you a hallucination rate and a non-response rate on the same dataset.

## Methods and benchmarks: what each one tells you

Public benchmarks are useful for choosing a model and for borrowing methods. They rarely substitute for a set built from your own domain. Here is what the main ones test, with figures from the papers that define them:

| Benchmark or method | What it measures                                          | Notable detail                                                                                                                                        |
| ------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| TruthfulQA          | Whether a model repeats common misconceptions             | 817 questions across 38 categories including health, law, finance and politics ([ACM survey](https://dl.acm.org/doi/full/10.1145/3742420))            |
| FActScore           | Fraction of atomic claims supported by a knowledge source | Breaks generated text into atomic facts, then verifies each ([ACM survey](https://dl.acm.org/doi/full/10.1145/3742420))                               |
| FEVER               | Claim verification                                        | 185,445 claims labeled Supported, Refuted or NotEnoughInfo ([KGELL catalogue](https://kgell-cost.eu/wp-content/uploads/2026/07/KGELL_D1_1_final.pdf)) |
| FactBench (VERIFY)  | Evidence-based classification of content units            | 1,000 prompts over 150 topics, tiered Easy, Moderate, Hard ([ACL 2025](https://aclanthology.org/2025.acl-long.1587.pdf))                              |

Two findings from FactBench are worth keeping in mind. Factual precision declines from the Easy tier to the Hard tier, so a single averaged score hides where your system breaks. And it reported that one model over-refused in more than 25% of cases, a reminder that a model can look factual by saying nothing. Report precision and refusal rate side by side.

The same EMNLP survey observes that benchmarks assess factuality either by multiple-choice discrimination (HotpotQA, StrategyQA, MMLU) or by free-form generation, and that free-form generation remains the significant challenge. Most production systems generate free text, which is why claim-level decomposition and a judge are usually required.

## Write LLM-as-a-Judge rubrics you can defend

An eval rubric is the written definition of what a score means. For factuality, a vague rubric ("is the answer accurate?") produces noisy scores. A good one names the evidence, the scale and the required reasoning.

The paper [On A Scale From 1 to 5](https://arxiv.org/abs/2410.12222) shows a workable structure. It scores faithfulness from 1 to 5 across four aspects: factual consistency, adjective regularity, knowledge congruence and style alignment. A 5 means all the information is verifiable in the reference, and a 1 means most of it cannot be validated. It also requires the judge to output its reasoning along with the score.

### A rubric template

For your own rubric, specify:

- **The reference:** which gold answer, document or evidence set the judge may use, and that it may not use its own memory.
- **The unit:** score each atomic claim, then aggregate, rather than scoring the whole paragraph at once.
- **The labels:** Supported, Unsupported, Contradicted, or Undecidable. The Undecidable label keeps the judge from guessing.
- **The anchors:** one short example per score level, including at least one borderline case.
- **The output:** the verdict plus a one- or two-sentence justification, so you can audit failures.

### Validate the judge before you trust it

This is the step teams skip. The ACL 2025 Findings paper [Verify with Caution](https://aclanthology.org/2025.findings-acl.1175.pdf) re-evaluated five state-of-the-art factuality metrics across 11 datasets covering summarization, RAG and QA. The evaluators were inconsistent with each other, often misestimated factual accuracy, and showed bias against heavily paraphrased outputs and outputs that draw on distant parts of a source document. The authors urge manual validation of metrics in each domain.

A practical validation routine:

1. Have two humans label 100 to 200 samples from your own traffic, including hard cases.
2. Run the judge on the same samples.
3. Compare per-class results, not only overall accuracy. The same paper cautions that balanced accuracy can hide trade-offs between true positive and true negative rates, and recommends reporting error-rate breakdowns.
4. Read the disagreements. If the judge misses paraphrased claims, rewrite the rubric or split the claim-extraction step.
5. Re-run this check whenever you change the judge model or the domain.

MLflow's automated evaluation uses LLM-as-a-Judge frameworks, so the rubric you validate here becomes the scorer you run everywhere else, rather than a spreadsheet exercise that lives apart from your code.

## Evaluation as code: run factuality checks on every change

"Evaluation as code" means the dataset, the rubric, the judge model, the prompt under test and the pass thresholds all live in version control and run from a command or a pipeline. Nobody opens a notebook and eyeballs ten answers.

OpenAI's guide lays out a five-step workflow that maps directly onto this:

1. Define the eval objective.
2. Collect a dataset.
3. Define metrics.
4. Run and compare evals.
5. Set up continuous evaluation on every change, monitor nondeterminism, and grow the eval set over time.

### What belongs in the repository

- **The dataset**, with a version tag and a note on how each row was validated.
- **The rubric**, as text beside the scorer code, so a change to wording shows up in a diff.
- **The prompt**, versioned. MLflow's [Prompt Registry](https://mlflow.org/prompt-registry) versions prompts so that every eval result ties to the exact prompt that produced it.
- **The thresholds**, such as minimum claim-level precision and maximum refusal rate on known-entity questions.

### Detecting regressions

A factuality regression can come from a model upgrade, a prompt edit, a retriever change or a new tool. Run the same suite against the baseline and the candidate, then compare per slice (topic, difficulty tier, question type) and not only the mean. Because model output is nondeterministic, repeat runs on borderline cases and compare distributions before you fail a build.

For deeper walkthroughs of the pipeline, see our guides on [continuous LLM evaluation for production teams](https://mlflow.org/articles/continuous-llm-evaluation) and [reproducible LLM evaluation harnesses](https://mlflow.org/articles/llm-evaluation-harness), and the posts tagged [CI/CD LLM evaluation](https://mlflow.org/articles/tags/ci-cd-llm-evaluation) and [agent regression suites](https://mlflow.org/articles/tags/agent-regression-suite).

### Balance anti-hallucination prompts

When a factuality score drops, the instinct is to add "If you don't know, say so" to the prompt. The practical guide warns that overly cautious instructions can raise undesirable non-response rates even when sufficient grounding data is provided. Make both numbers gates in your CI: hallucination rate on fictitious entities must stay low, and non-response rate on real entities must stay low as well.

## Keeping factuality in view in production

Offline suites tell you whether a change is safe to ship. Production tells you what users actually ask. A workable pattern is to sample a fraction of live traces, score them with the validated judge, and send the low scorers to human review and then into the dynamic part of your dataset. That closes the loop the 5 D's describe: the set stays dynamic because production keeps feeding it.

Cost and latency are design choices here. Score a sample rather than every request, run judges asynchronously so they never sit on the user's critical path, and use cheaper checks (such as retrieval-context overlap) as a first filter before the expensive claim-level judge. Route judge calls through one controlled endpoint so credentials, rate limits and provider changes are managed in one place. That is the role of the [MLflow AI Gateway](https://mlflow.org/ai-gateway), which provides cross-provider governance for the models your judges and applications call.

![Keeping factuality in view in production](https://media.babylovegrowth.ai/blog-images/organization-30814/1795285931686_Keeping-factuality-in-view-in-production.jpeg)

## Put it to work with MLflow

MLflow is an open-source platform for the GenAI and LLM lifecycle, with deep tracing of agent reasoning, automated LLM-as-a-Judge evaluation, prompt management and versioning, and a cross-provider AI Gateway. All of its features are open source under Linux Foundation governance, and tracing works through OpenTelemetry, so it fits the frameworks you already run. Start with [AI Observability for LLMs and Agents](https://mlflow.org/ai-observability) to capture traces, then attach your factuality scorers to them.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

If you are building from scratch, our posts on [best practices for LLM evaluation](https://mlflow.org/articles/tags/best-practices-for-llm-evaluation) and [continuous LLM evaluation](https://mlflow.org/articles/tags/continuous-llm-evaluation) are good next reads. The platform itself lives at [mlflow.org](https://mlflow.org).

## Perspective

> We think factuality is the metric teams most often measure with the least rigor. A single number from an unvalidated judge feels like evidence, but the research is clear that these judges disagree with each other and with people.
>
> Our view is that the evaluation has to be part of the application, not a side project. The rubric is code, the dataset is versioned, the prompt is versioned, and every score links back to a trace that shows what the agent actually did. When a claim fails, you should be able to click from the score to the retrieval step or tool call that caused it.
>
> That is why we built MLflow as one open-source platform covering tracing, LLM-as-a-Judge evaluation, prompt management and the AI Gateway, with no enterprise feature paywalls. Factuality work touches all of those layers, and splitting it across disconnected tools is how regressions slip through.
>
> _Kevin_

## FAQ

### What is the difference between factuality and hallucination?

Factuality asks whether a claim is true against external ground truth. Hallucination asks whether content is unsupported by, or ungrounded in, the input or source. A model can be factually correct and still hallucinate by adding off-topic or overly detailed content, as the [EMNLP 2024 survey](https://aclanthology.org/2024.emnlp-main.1088.pdf) notes. Track both because they need different fixes.

### How many test cases do I need for a factuality eval?

The research does not set a universal number. Aim for a set that is scoped to your use case, diverse across topics and difficulty, and large enough to compare slices. Start small with human-verified cases, then grow it from production traces so it stays dynamic.

### Can I use an LLM to judge factuality?

Yes, with validation. Studies show LLM-based factuality metrics can be inconsistent and biased toward or against certain output styles, so label a sample with human reviewers, compare per-class error rates, and repeat the check whenever the domain or judge model changes.

### How do I run factuality evals in CI?

Store the dataset, rubric, prompt and thresholds in version control, and run the suite on every change to the model, prompt or retriever. Compare the candidate against a baseline per slice, and gate on both hallucination rate and non-response rate. The [continuous LLM evaluation guide](https://mlflow.org/articles/continuous-llm-evaluation) walks through the pipeline.

## Sources

- [Reducing LLM Hallucinations in Production (GMI Cloud)](https://www.gmicloud.ai/en/blog/reducing-hallucination-in-production-llm-apps-grounding-citations-and-faithfulness-scoring)
- [Factuality of Large Language Models: A Survey (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1088.pdf)
- [Benchmarking Large Language Model Quality, Reliability, and More (KGELL)](https://kgell-cost.eu/wp-content/uploads/2026/07/KGELL_D1_1_final.pdf)
- [A Practical Guide for Evaluating LLMs and LLM-Reliant Systems](https://arxiv.org/html/2506.13023v2)
- [Evaluation best practices (OpenAI)](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- [FACTORY: A Challenging Human-Verified Prompt Set](https://arxiv.org/html/2508.00109)
- [IRB: Automated Generation of Robust Factuality Benchmarks](https://arxiv.org/html/2602.08070v1)
- [Survey on Factuality in Large Language Models (ACM Computing Surveys)](https://dl.acm.org/doi/full/10.1145/3742420)
- [FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality (ACL 2025)](https://aclanthology.org/2025.acl-long.1587.pdf)
- [Verify with Caution: The Pitfalls of Relying on Imperfect Factuality Metrics](https://aclanthology.org/2025.findings-acl.1175.pdf)
- [On A Scale From 1 to 5: Quantifying Hallucination in Faithfulness Evaluation](https://arxiv.org/abs/2410.12222)
