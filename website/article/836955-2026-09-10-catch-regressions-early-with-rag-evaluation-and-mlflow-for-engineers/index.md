---
title: "Catch Regressions Early With RAG Evaluation and MLflow for Engineers"
description: "For engineers: a practical RAG evaluation playbook to stop hallucinations. Run a 20–50 query CI harness, calibrate LLM judges, and trace results with MLflow."
slug: catch-regressions-early-with-rag-evaluation-and-mlflow-for-engineers
tags:
  [
    rag evaluation,
    rag evaluation metrics,
    how to conduct rag evaluation,
    rag reporting techniques,
    how to evaluate rag,
    project evaluation metrics,
    rag status assessment,
    evaluate rag systems,
    rag performance review,
    risk assessment criteria,
  ]
date: 2026-09-10
image: https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789036525659_Engineer-reviewing-RAG-evaluation-traces.jpeg
---

![Engineer reviewing RAG evaluation traces](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789036525659_Engineer-reviewing-RAG-evaluation-traces.jpeg)

RAG evaluation measures two things at once: whether your retriever finds the right context and whether your model actually uses that context instead of hallucinating around it. Faithfulness comes first, because a system that retrieves perfectly but fabricates its answer is more dangerous than one that retrieves poorly and admits it doesn't know — so you can start with a Multi-LLM Audit to check all AI models at once. The immediate next step is small and concrete: build a 20 to 50 query offline harness that scores faithfulness before you touch anything else.

---

> **TL;DR:**
>
> - Faithfulness and answer relevance are the key evaluation metrics, with faithfulness measuring if answers are supported by retrieved context and answer relevance checking if responses address the question.
> - Evaluating the retriever, reranker, and generator separately helps diagnose failure modes and improve system components individually.
> - Building a small, focused test set of 20 to 50 high-quality queries is sufficient for rapid CI checks, but larger datasets are better for periodic benchmarking.
> - Using LLM judges with calibrated prompts and structured output ensures more reliable, ground-truth-free evaluation of open-ended answers.
> - An integrated eval workflow with a version-controlled offline harness, CI gating, and production monitoring is essential for maintaining system quality over time.

---

## Table of Contents

- [What Metrics Actually Matter for RAG Evaluation?](#what-metrics-actually-matter-for-rag-evaluation)
- [Should You Evaluate the Retriever, the Reader, or the Whole System?](#should-you-evaluate-the-retriever-the-reader-or-the-whole-system)
- [How Do You Build a RAG Evaluation Dataset?](#how-do-you-build-a-rag-evaluation-dataset)
- [How Do LLM Judges Work for RAG Evaluation?](#how-do-llm-judges-work-for-rag-evaluation)
- [What Does a Production-Ready Evaluation Workflow Look Like?](#what-does-a-production-ready-evaluation-workflow-look-like)
- [How Do You Monitor a RAG System After It Ships?](#how-do-you-monitor-a-rag-system-after-it-ships)
- [How MLflow Supports the RAG Evaluation Lifecycle](#how-mlflow-supports-the-rag-evaluation-lifecycle)
- [A Practical Starting Roadmap](#a-practical-starting-roadmap)
- [Put Your RAG Evaluation Metrics on One Platform](#put-your-rag-evaluation-metrics-on-one-platform)
- [Sources](#sources)
- [FAQ](#faq)

## What Metrics Actually Matter for RAG Evaluation?

Every RAG evaluation framework eventually collapses into the same handful of questions: did the retriever find the right material, and did the model tell the truth about it? The RAG Triad framing (context relevance, groundedness, and answer relevance) has become the shorthand most teams reach for, and [Ragas](https://arxiv.org/html/2309.15217) operationalizes it with reference-free scoring that an LLM judge can compute without ground-truth answers.

**Faithfulness (groundedness)** checks whether every claim in the generated answer traces back to the retrieved context. Ragas computes this by breaking the answer into individual statements, then asking a judge model whether each statement is supported by the source passages. This metric maps directly to hallucination control: a low faithfulness score means the model is inventing details the retriever never surfaced, regardless of how fluent the answer reads.

**Answer relevance and correctness** measure something different: does the answer actually address what was asked? A response can be perfectly faithful to the retrieved context and still miss the question entirely, if the retriever pulled adjacent but irrelevant passages. For tasks with a known correct answer, reference-based metrics like exact match (EM) and F1 give you a hard number. For open-ended or generative tasks, judge-based relevance scoring works better because EM and F1 punish valid paraphrases.

**Context precision and recall** live on the retrieval side. Precision@k tells you what fraction of the top k retrieved chunks are actually relevant; recall@k tells you what fraction of all relevant chunks in the corpus made it into the top k. Both are sensitive to chunking strategy. Practitioner notebooks that vary chunk size report [large swings in retrieval quality](https://huggingface.co/learn/cookbook/en/rag_evaluation) purely from changing how documents get split, before you even touch the model.

**Hallucination rate and citation accuracy** are the production-facing cousins of faithfulness. Hallucination rate tracks the percentage of responses containing at least one unsupported claim; citation accuracy checks whether cited sources actually contain the claims attributed to them. LLM judges catch most obvious fabrications but struggle with subtle scope errors, like a citation that supports a related but slightly different claim than the one made.

A useful mental model for thresholds and sequencing:

- Faithfulness above a moderate threshold is a reasonable early bar for prompt and grounding fixes to clear before you move on.
- Context recall and precision come next, once faithfulness stabilizes, since retrieval fixes are cheaper before generation tuning.
- Answer completeness and clarity get evaluated last, after the system stops hallucinating and starts retrieving well.

**Pro Tip:** _Don't average metrics across your whole test set and call it done. A 0.85 average faithfulness score can hide a cluster of queries that score near zero. Segment your results by query type before you trust the aggregate._

Regulated domains (medical, legal, financial) need a higher bar across the board. A hallucination rate that's tolerable in a general customer-support bot is a liability in a clinical decision-support tool, and teams in those spaces should treat faithfulness thresholds as a floor, not a target.

## Should You Evaluate the Retriever, the Reader, or the Whole System?

Both, and separately. Conflating retrieval failures with generation failures is one of the fastest ways to waste an engineering sprint chasing the wrong fix. A comprehensive survey on RAG evaluation formalizes this split as internal (component-level) evaluation versus external (system-level) evaluation, and the distinction matters because each component fails in a different, diagnosable way.

**Retriever evaluation** centers on whether the right documents even entered the candidate pool.

- Recall@k: the share of ground-truth relevant documents captured within the top k results.
- Precision@k: the share of the top k results that are actually relevant.
- Context coverage: whether the retrieved set, taken together, contains enough information to answer the question at all, even if no single chunk does.
- Ranking diagnostics: checking whether relevant documents rank near the top or get buried past position 10, since most readers only attend closely to the first few chunks in the prompt window.

**Reranker evaluation** applies once you've added a cross-encoder or similar second-stage ranker on top of initial retrieval. The question is simple: does reranking measurably lift precision at the k value your reader actually consumes? Measure precision@k before and after reranking on the same query set. If the lift is marginal (a point or two), the added latency probably isn't worth it. If it's substantial, it usually shows up most on queries where the initial retriever returns several plausible but subtly wrong candidates.

**Reader (generation) evaluation** isolates the model's behavior given a fixed, known-good context. This is where faithfulness checks belong in their purest form, because you've removed retrieval noise from the equation. Also check answer completeness (did it address every part of a multi-part question) and formatting constraints (did it respect the output schema your downstream system expects).

**End-to-end evaluation** looks at the full pipeline as a black box and asks two production-relevant questions: what's the boost rate (how often does adding retrieval improve the answer versus the model's parametric knowledge alone), and what's the resilience rate (how often does the system degrade gracefully, rather than confidently wrong, when retrieval fails outright)?

Component failures leave fingerprints in these end-to-end numbers. A retriever that's degrading will show up as falling context precision alongside stable faithfulness, since the model is being faithful to bad context. A reader that's struggling will show up as falling faithfulness even when context precision holds steady. Running both layers side by side is what turns a vague "the system got worse" into an actionable ticket.

## How Do You Build a RAG Evaluation Dataset?

You don't need thousands of labeled examples to start. You need a small, high-quality set that actually exercises the failure modes your system is likely to hit, and a repeatable process for growing it over time.

1. **Select context snippets.** Pull a representative sample of chunks from your actual corpus, weighted toward document types and sections that get queried most often in production.
2. **Prompt for question generation.** Feed each snippet to an LLM with an instruction to generate a question the snippet answers. This is the synthetic QA step that the Hugging Face RAG evaluation cookbook walks through with runnable code.
3. **Run critique agents.** Pass each generated question and answer pair through a second LLM prompt that scores clarity, answerability, and whether the question is trivially guessable without the context.
4. **Filter low-quality items.** Drop anything the critique step flags: ambiguous questions, answers that leak from general knowledge rather than the retrieved snippet, and duplicates.
5. **Add human annotation where it counts.** Synthetic generation is fast but noisy. Once you need to measure actual accuracy rather than relative regression, a human needs to write or verify the ground-truth answer, particularly for domain-specific queries where an LLM's judgment is unreliable.

For CI purposes, aim for **20 to 50 high-value queries**. That's small enough to run on every pull request without burning through API budget or slowing the merge pipeline, and [practitioner guidance](https://mlflow.org/articles/tags/evaluation-metrics-for-datasets) consistently points to this range as the sweet spot for catching regressions fast. Reserve larger sets, in the hundreds, for periodic benchmarking runs rather than every commit.

Whatever size you land on, capture consistent metadata alongside each example:

- **Source document ID** and **chunk offset**, so you can trace a failure back to the exact passage that misled the system.
- **Expected answer type** (factual, list, yes/no, multi-hop reasoning), because different answer types fail differently.
- **Edge-case tags** (ambiguous phrasing, multi-document synthesis required, out-of-corpus question), so you can slice results by category instead of staring at one flat pass rate.

**Pro Tip:** _Tag a handful of your test queries as "adversarial" on purpose, questions with a plausible-sounding but wrong answer sitting right next to the correct one in the corpus. These catch faithfulness regressions that ordinary queries miss entirely._

Chunking strategy deserves its own line item in your test-set design, since notebook experiments show that chunk size changes alone can shift retrieval quality dramatically. Build a few test queries specifically designed to probe whether your current chunk boundaries split relevant information across two chunks, since that's a common and easy-to-miss failure.

![How Do You Build a RAG Evaluation Dataset? — overview diagram](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789036582147_How-Do-You-Build-a-RAG-Evaluation-Dataset-overview-diagram.jpeg)

## How Do LLM Judges Work for RAG Evaluation?

An LLM-as-a-judge scores your system's outputs using another model's reasoning instead of a fixed reference answer, and it's become the practical default for RAG evaluation precisely because ground-truth answers are expensive to collect at scale.

**Reference-free evaluators** (what Ragas primarily uses) score faithfulness, context relevance, and answer relevance without needing a human-written correct answer. They work well for open-ended or exploratory queries where "correct" isn't a single fixed string. **Reference-based evaluators**, by contrast, compare an output against a known-good answer, which is more rigorous but only works where you've invested in ground truth. The LangChain evaluation framework treats [correctness, relevance, and groundedness](https://docs.langchain.com/langsmith/evaluate-rag-tutorial) as distinct evaluator types you can mix based on which ground truth you have available for a given query.

Prompt design determines whether your judge produces stable, trustworthy scores or noise that happens to look like numbers.

- Define the metric explicitly in the prompt. Don't ask a judge to rate "quality." Ask it to rate whether every claim in the answer is supported by the provided context, on a defined scale.
- Ask for rationale before the score, not after. Requiring the judge to explain its reasoning first, then output a number, produces more stable results across repeated runs, because the model has to commit to reasoning before committing to a verdict.
- Request structured output (a JSON object with a score field and a rationale field) rather than free text, so you can parse results reliably in an automated pipeline.

**Calibration** is the step most teams skip, and it's the one that determines whether you can trust the judge at all. Run your LLM judge against a sample of human-labeled examples, at minimum 20 to 30, and compare agreement rates. Also check inter-run consistency: run the same judge prompt against the same inputs multiple times and see if the score drifts. If it does, your prompt probably needs the rationale-first structure above, or a lower temperature setting.

**Pro Tip:** _Use a different, typically more capable model as the judge than the one generating answers. Self-evaluation, where a model grades its own output, tends to inflate scores because the model's blind spots overlap with its own generation._

Known failure modes are worth watching for specifically. LLM judges struggle with domain jargon they weren't trained heavily on, so a judge scoring a legal or medical RAG system may confidently approve an answer that's subtly wrong to a specialist. They also systematically favor fluent, confident-sounding wrong answers over hedged, correct ones, since fluency and correctness are correlated in most training data but not always in the specific case being judged. Mitigate both by periodically swapping in a domain-expert human reviewer on a random sample, and by explicitly instructing the judge to penalize unsupported confidence rather than reward it.

## What Does a Production-Ready Evaluation Workflow Look Like?

An evaluation workflow that only runs when someone remembers to run it manually isn't a workflow. It's a hope. The pattern that holds up in production combines four pieces: an offline harness, CI gating, regression tests, and a checklist that governs what "ready to ship" actually means.

1. **Build the offline harness.** Store your test dataset under version control alongside your code, not in a spreadsheet somewhere. Every run should be deterministic: same queries, same retrieval index snapshot, same model version, logged outputs. This is what makes a score from last Tuesday comparable to a score from today.
2. **Set CI gate thresholds.** A typical gate might require faithfulness above 0.80 and context precision above a set floor before a pull request merges. Tie these to your test-set results, not to spot checks, and fail the build automatically when a threshold isn't met.
3. **Design regression tests for speed and impact.** Your CI suite should run in minutes, not hours. That's the entire argument for the 20 to 50 query range: small enough to run on every commit, large enough to catch the failure modes that actually recur, like a prompt change that quietly breaks citation formatting.
4. **Enforce a production readiness checklist** before anything ships: sampling plan defined, alerting thresholds set, a human review rotation assigned, and an incident runbook written for when faithfulness drops mid-deployment.

| Workflow stage      | What it catches                                      | Typical cadence                |
| ------------------- | ---------------------------------------------------- | ------------------------------ |
| Offline harness     | Regressions from prompt, model, or retrieval changes | Every commit or PR             |
| CI gate             | Sub-threshold faithfulness or precision before merge | Every PR                       |
| Production sampling | Silent drift not visible in offline tests            | Continuous, 1 to 5% of traffic |
| Periodic benchmark  | Slow degradation across a larger query set           | Weekly or monthly              |

The offline harness and the CI gate protect you from shipping a known regression. Production sampling is what protects you from the regression you didn't know to test for, which is the more dangerous category because nobody's watching for it until a user complains.

## How Do You Monitor a RAG System After It Ships?

Offline tests catch what you thought to test for. Production monitoring catches everything else, which in practice is most of what actually breaks a RAG system over time: a source document that quietly changed, an embedding model that got swapped upstream, a prompt template edited by someone who didn't run the harness first.

Minimum viable logging for a RAG system in production includes the query itself, the full set of retrieved documents (not just the ones that made the final prompt), the generated text, model and prompt version metadata, and a timestamp. Without retrieved-document logging specifically, you can't distinguish a retrieval failure from a generation failure after the fact, which puts you back in the conflation problem from earlier.

Sampling strategy determines what you actually see. A pure random sample (say, 1 to 5% of production traffic) gives you a baseline read on average quality but will miss rare, high-impact failures by design, since rare things are rare in a random sample too. Layer in edge-case triggers, like automatically flagging any query where retrieval returned unusually low similarity scores across the board, and error-driven sampling that captures every instance where a user explicitly flagged a bad response.

- Watch for faithfulness drops in your sampled evaluations, which is usually the earliest signal something upstream changed.
- Watch for a rising hallucination rate specifically on query types that were previously stable, which often points to a document update rather than a model regression.
- Watch for shifts in context precision, which frequently trace back to an index rebuild, an embedding model version bump, or new documents entering the corpus with different formatting.

When a drift signal fires, the triage playbook matters as much as the detection. Define ahead of time who gets notified (usually whoever owns the retrieval index and whoever owns the generation prompt, since the fix path differs), what data gets pulled first (the flagged query, its retrieved documents, and the last known-good version of both), and the rollback threshold at which you revert instead of patch forward. Teams that skip this step tend to spend the first two hours of an incident just figuring out who should be looking at it.

## How MLflow Supports the RAG Evaluation Lifecycle

Every pattern above needs somewhere to live: your test datasets, your judge outputs, your production samples, and the history that lets you tell whether faithfulness went up or down after last week's prompt change. MLflow's [genai](https://mlflow.org/genai) tooling and tracing capabilities are built around exactly that record-keeping problem.

- **Traceability.** The platform stores each evaluation run, including the query, retrieved context, generated answer, and the resulting scores, as a logged artifact, so you can compare today's faithfulness score against the version from previous deployments without reconstructing anything by hand.
- **LLM-judge orchestration.** Instead of scripting judge calls one-off, MLflow's LLM-as-a-judge tooling runs evaluator prompts consistently across a dataset and records the rationale and score together as structured output, which is exactly the pattern the calibration step above depends on.
- **CI and production integration.** Offline harness results can feed directly into the same tracking system that ingests production samples, so a faithfulness score computed in CI and one computed from live traffic can be compared in the same timeline.
- **Observability handoff.** For teams connecting evaluation output to broader monitoring, MLflow's [observability tooling](https://mlflow.org/ai-observability) extends the same tracing into dashboards that flag drift signals like the ones described above.

For teams building this out for the first time, MLflow's [data flow monitoring](https://mlflow.org/articles/tags/llm-data-flow-monitoring) resources and [dataset evaluation guidance](https://mlflow.org/articles/tags/how-to-evaluate-rag-datasets) walk through the practical setup with runnable examples rather than abstract description.

## A Practical Starting Roadmap

If you're starting from nothing, resist the urge to build a comprehensive evaluation suite on day one. Get a faithfulness check running on 20 to 50 queries in CI first, because that single metric catches the most damaging failure mode with the least setup cost. Only after that's stable should you introduce an LLM judge for relevance or completeness, and even then, validate its scores against a human-labeled sample before you let it gate a merge. The teams that skip calibration are the ones who eventually discover their "passing" system was passing a broken judge, not a working RAG pipeline. Keep growing your ground-truth pool slowly, and recalibrate the judge every time you change the underlying model.

> _— Kevin_

## Put Your RAG Evaluation Metrics on One Platform

Running faithfulness checks in a notebook works until you need to compare last month's scores against this month's, across three prompt versions and two embedding models, without losing track of which run produced which number. That's the gap MLflow closes: one place to log offline harness results, orchestrate LLM-judge runs, and carry those same metrics into production sampling, instead of stitching together spreadsheets and one-off scripts every time someone asks "did faithfulness actually improve?"

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

A lightweight script-based harness is fine when you're validating a single metric on a handful of queries. Once you're running CI gates on every pull request, calibrating judges against human labels, and tracking drift across production traffic, you need a system built for that lifecycle rather than a folder of disconnected scripts. MLflow's LLM-as-a-judge tooling automates the evaluator orchestration this article walks through, and its agent and LLM engineering platform ties those scores into the same tracing system your team already uses for deployment. Start by pointing your existing 20 to 50 query harness at evaluation tracking, and see whether last week's faithfulness score actually holds up against today's.

## Sources

- [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/html/2309.15217)
- [RAG Evaluation — Hugging Face Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation)
- [Evaluate a RAG application - LangChain docs](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)

## FAQ

### Is ChatGPT a RAG Model?

No. ChatGPT is a large language model that generates answers from its trained parameters; it becomes part of a RAG system only when it's paired with an external retriever that feeds it live document context before generation.

### What Does RAG Stand For?

RAG stands for Retrieval-Augmented Generation, an architecture that retrieves relevant documents from an external source and feeds them to a language model as context before it generates an answer.

### How Do You Check RAG Performance?

Run a small offline harness (20 to 50 queries) scoring faithfulness, context precision and recall, and answer relevance, using frameworks like Ragas or LLM-as-a-judge tooling such as MLflow's [evaluation orchestration](https://mlflow.org/llm-as-a-judge), then track those scores over time as you change prompts or models.

### What Is the Difference Between RAG and an LLM?

An LLM is the generation engine alone, limited to what it learned during training; RAG adds a retrieval step that grounds the LLM's answers in external, often more current, documents fetched at query time.

## Recommended

- [ML Model Evaluation](https://mlflow.org/classical-ml/model-evaluation)
- [Benchmark Your Way to Better RAG and Agents:Tuning Vector Search with MLflow](https://mlflow.org/blog/tune-and-benchmark-with-mlflow)
- [RAG Evaluation Datasets: A Developer's Reproducible Workflow](https://mlflow.org/articles/rag-evaluation-datasets)
- [Agent Trace Evaluation with TruLens Scorers in MLflow](https://mlflow.org/blog/mlflow-trulens-evaluation)
