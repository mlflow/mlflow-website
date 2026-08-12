---
title: "RAG Evaluation Datasets: A Developer's Reproducible Workflow"
description: "Explore efficient workflows for RAG evaluation datasets, utilizing top resources and metrics to enhance your model's performance and reproducibility."
slug: rag-evaluation-datasets
tags:
  [
    synthetic data for evals,
    rag data assessment,
    how to evaluate rag datasets,
    evaluation metrics for datasets,
    data analysis for rag,
    rag datasets for evaluation,
    performance evaluation datasets,
    machine learning evaluation datasets,
    rag evaluation datasets,
    rag eval dataset,
    metrics for rag evaluation,
  ]
date: 2026-08-12
image: https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1786511002918_Hands-manipulating-data-analogues-on-desk.jpeg
---

![Hands manipulating data analogues on desk](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1786511002918_Hands-manipulating-data-analogues-on-desk.jpeg)

Start with the highest-signal resources: [RAGEval](https://aclanthology.org/2025.acl-long.418.pdf) (schema-based synthetic generation + three grounding metrics), RAGAS/WikiEval (reference-free scoring), [GaRAGe](https://github.com/amazon-science/GaRAGe) (per-passage grounding annotations), MTRAG (multi-turn conversational benchmark), and TechQA-RAG-Eval (technical-support domain). The recommended evaluation approach combines grounding-aware metrics — Completeness, Hallucination, Irrelevance, Faithfulness, Answer Relevance, Context Relevance — with retrieval metrics like recall@k and MRR, scored through a mix of human grounding labels and LLM-as-a-judge for scale.

**Immediate next steps:**

- Clone the RAGEval repo (`OpenBMB/RAGEval`) and run the schema-based generation pipeline on a sample of your own documents.
- Open the Hugging Face Open-Source AI Cookbook RAG Evaluation notebook and run a small synthetic dataset through an LLM-as-a-judge scorer.
- Pull the GaRAGe dataset and inspect the `evidence_relevant` / `evidence_correct` passage-level labels to understand what grounding annotation looks like in practice.

---

## Key Takeaways

Grounding-aware metrics combined with a versioned, artifact-logged pipeline are the foundation of a trustworthy RAG evaluation — synthetic datasets accelerate early-stage testing, but human passage-level annotations are what catch silent failures before production.

| Point                            | Details                                                                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Start with established datasets  | RAGEval, RAGAS/WikiEval, GaRAGe, MTRAG, and TechQA-RAG-Eval cover the core evaluation scenarios.                              |
| Use grounding-aware metrics      | Track Completeness, Hallucination, Faithfulness, and recall@k separately to pinpoint retrieval vs. generation failures.       |
| Constrain synthetic generation   | Schema-based generation (RAGEval approach) reduces unverifiable labels; RAGEval reports a 1.67% machine-vs-human scoring gap. |
| Annotate passage-level grounding | GaRAGe's `evidence_relevant` and `evidence_correct` tags catch silent failures that answer-accuracy metrics miss.             |
| Mlflow for reproducibility       | Log corpora, embeddings, judge prompts, and raw outputs as Mlflow artifacts to reproduce any evaluation run from a single ID. |

---

## Table of Contents

- [Which public RAG evaluation datasets should you use?](#which-public-rag-evaluation-datasets-should-you-use)
- [How do you build a synthetic RAG evaluation dataset?](#how-do-you-build-a-synthetic-rag-evaluation-dataset)
- [What metrics should you track for RAG outputs?](#what-metrics-should-you-track-for-rag-outputs)
- [How do you run an end-to-end RAG benchmarking pipeline?](#how-do-you-run-an-end-to-end-rag-benchmarking-pipeline)
- [How should you design human annotation for grounding labels?](#how-should-you-design-human-annotation-for-grounding-labels)
- [What makes multi-turn RAG evaluation different?](#what-makes-multi-turn-rag-evaluation-different)
- [Running RAG evaluations with Mlflow: a workflow sketch](#running-rag-evaluations-with-mlflow-a-workflow-sketch)
- [The real trade-offs in RAG eval dataset design](#the-real-trade-offs-in-rag-eval-dataset-design)
- [Mlflow makes RAG evaluation reproducible at scale](#mlflow-makes-rag-evaluation-reproducible-at-scale)
- [Papers, repos, and notebooks worth bookmarking](#papers-repos-and-notebooks-worth-bookmarking)
- [Sources](#sources)

## Which public RAG evaluation datasets should you use?

The table below maps each resource to its primary use case, the RAG component it stresses, and the annotation type it provides. Choose based on what your system needs to prove.

![Which public RAG evaluation datasets should you use? — overview diagram](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1786511143846_Which-public-RAG-evaluation-datasets-should-you-use-overview-diagram.jpeg)

| Dataset / Resource           | Primary Use Case                                 | RAG Component Stressed              | Annotation Type                   |
| ---------------------------- | ------------------------------------------------ | ----------------------------------- | --------------------------------- |
| RAGEval (OpenBMB/RAGEval)    | Scenario-specific synthetic eval generation      | Generator + grounding               | LLM-generated, schema-constrained |
| RAGAS / WikiEval             | Reference-free automated scoring                 | Generator (faithfulness, relevance) | Human judgments + automated       |
| GaRAGe                       | Grounding-level regression testing               | Retriever + generator grounding     | Human passage-level labels        |
| MTRAG                        | Multi-turn conversational RAG                    | Retriever across turns + generator  | Human-generated conversations     |
| TechQA-RAG-Eval              | Domain-specific (technical support) benchmarking | Retriever + generator               | Human QA pairs + context passages |
| HF RAG Eval Notebook         | Hands-on synthetic dataset + LLM-judge tutorial  | End-to-end pipeline                 | Synthetic + LLM-as-judge          |
| Kaggle Single-Topic RAG Eval | Focused single-topic retrieval testing           | Retriever                           | Curated QA pairs                  |

**Key notes on scope and size:**

- **RAGEval** targets scenario-specific domains (finance, medicine, law) and generates datasets automatically from your own seed documents. The RAGEval paper reports a machine-vs-human absolute scoring difference of 1.67% and a Fleiss' Kappa of 0.7686 across annotators, which is a strong signal that LLM-based scoring is reliable for this schema.
- **RAGAS/WikiEval** is English-language and works without golden reference answers, making it the fastest path to an automated scoring loop. The RAGAS framework separates faithfulness, answer relevance, and context relevance so you can pinpoint which subsystem is failing.
- **GaRAGe** provides per-passage `evidence_relevant` and `evidence_correct` tags, enabling regression tests that catch silent failures where an answer looks correct but lacks proper grounding.
- **MTRAG** covers 110 human-generated conversations averaging 7.7 turns (842 tasks total), plus a companion synthetic set (MTRAG-S) for studying automation paths.
- **TechQA-RAG-Eval** ships with roughly 908 QA pairs, context passages, and an `is_impossible` flag — useful for testing unanswerable-question handling in knowledge-base scenarios.

---

## How do you build a synthetic RAG evaluation dataset?

When public benchmarks don't match your production domain, you build your own. Schema-constrained generation is the method that holds up under scrutiny: it ties every question and answer to a verifiable document passage, which makes debugging retrieval vs. generation failures tractable.

1. **Collect seed documents.** Pull 50–200 representative documents from your production corpus. Diversity matters more than volume at this stage — cover edge cases, short passages, and dense technical content.
2. **Extract factual keypoints.** For each document, prompt an LLM to extract discrete, verifiable facts. Each keypoint becomes a candidate answer anchor. Keep keypoints atomic; compound facts produce ambiguous labels.
3. **Define your schema.** At minimum, your schema needs: `document_id`, `question`, `answer`, `reference_passages` (list), and `keypoints_covered`. RAGEval's schema-based approach adds `completeness_target` and `hallucination_risk` fields that make scoring deterministic.
4. **Generate constrained questions.** Prompt the generator with the keypoint and the source passage as context, and instruct it to produce a question answerable _only_ from that passage. This constraint prevents the generator from producing questions that require world knowledge outside the corpus.
5. **Link answers to keypoints.** Each generated answer should map back to one or more keypoints. Store this mapping in the schema — it's what lets you compute Completeness later.
6. **Run post-checks.** Filter out questions where the answer is not grounded in the reference passage (a simple LLM judge prompt works here), remove duplicates, and verify that at least 10–15% of your dataset contains unanswerable questions to test rejection behavior.

**Pro Tip:** _Use the RAGEval GitHub repo as your schema template rather than designing from scratch. The schema enforces keypoint linkage at generation time, which eliminates the most common source of unverifiable synthetic labels before they enter your dataset._

The Hugging Face Open-Source AI Cookbook RAG Evaluation notebook walks through a similar pipeline end-to-end and is worth running on a small sample before scaling. For structured-data audits of your schema metadata, the LLM structured data audit tool can flag fields that are missing or machine-unreadable before you commit to a large generation run.

---

## What metrics should you track for RAG outputs?

Grounding-aware metrics give you a diagnostic picture that answer-accuracy alone cannot. The right metric set covers both retrieval quality and generation fidelity, and separates them clearly so you know which component to fix.

**Generation metrics:**

- **Completeness** (RAGEval): fraction of reference keypoints covered by the generated answer. Low completeness points to a retrieval gap or a generator that ignores retrieved content.
- **Hallucination** (RAGEval): presence of claims in the answer not supported by any retrieved passage. High hallucination is a generator problem, not a retrieval problem.
- **Irrelevance** (RAGEval): proportion of the answer that addresses content outside the question scope. Useful for catching verbose or topic-drifting generators.
- **Faithfulness** (RAGAS): whether every claim in the answer is attributable to the retrieved context. Computed reference-free via LLM judge.
- **Answer Relevance** (RAGAS): how well the answer addresses the question, independent of factual correctness.
- **Context Relevance** (RAGAS): fraction of the retrieved context that is actually needed to answer the question. Low scores indicate noisy retrieval.

**Retrieval metrics:**

- **Recall@k**: fraction of relevant passages appearing in the top-k retrieved results. The primary signal for retriever tuning.
- **MRR (Mean Reciprocal Rank)**: rewards systems that rank the most relevant passage highest. Useful when the first retrieved passage dominates generation quality.

| Metric            | Diagnoses                   | How to Compute                           |
| ----------------- | --------------------------- | ---------------------------------------- |
| Completeness      | Missing keypoints in answer | LLM judge vs. keypoint list              |
| Hallucination     | Unsupported claims          | LLM judge vs. retrieved passages         |
| Irrelevance       | Off-topic generation        | LLM judge on answer scope                |
| Faithfulness      | Attribution to context      | RAGAS reference-free LLM judge           |
| Answer Relevance  | Question-answer alignment   | RAGAS reference-free LLM judge           |
| Context Relevance | Retrieval noise             | RAGAS reference-free LLM judge           |
| Recall@k          | Retriever coverage          | Exact match vs. ground-truth passages    |
| MRR               | Retriever ranking quality   | Reciprocal rank of first relevant result |

Reference-free scoring (RAGAS, LLM-as-a-judge) is the right default when you don't have golden answers and need fast iteration. Reference-based scoring (ROUGE, F1, exact match) is worth adding when you have human-verified answers and want a hard accuracy floor. The RAGAS paper notes that context relevance is harder to automate reliably than faithfulness or answer relevance — report all three separately rather than averaging them, so you can see which dimension is pulling scores down.

For a broader view of LLM evaluation metrics and how they interact with human judgments, the Mlflow articles hub covers the tradeoffs in depth.

---

## How do you run an end-to-end RAG benchmarking pipeline?

A reproducible pipeline requires explicit decisions at every stage. Skipping config documentation is the most common reason teams can't reproduce a result two weeks later.

1. **Data hygiene.** Deduplicate documents, normalize whitespace, and strip boilerplate (headers, footers, navigation text). Contaminated corpora inflate retrieval recall artificially.
2. **Chunking strategy.** Fix chunk size and overlap before indexing. Record both values as pipeline parameters — changing them invalidates all prior retrieval results.
3. **Embedder selection.** Pin the embedding model name and version. Different versions of the same model produce incompatible vector spaces.
4. **Index configuration.** Record ANN parameters (number of neighbors, distance metric, ef_construction for HNSW). These directly affect recall@k.
5. **Retriever evaluation.** Run recall@k and MRR against your ground-truth passage set before touching the generator. Fix retrieval first.
6. **Generator configuration.** Log temperature, top-p, system prompt text, and model version for every experiment. A temperature change is a different experiment.
7. **Scorer configuration.** Log the judge model, judge prompt template, and scoring thresholds. Publish raw judge outputs alongside aggregated scores.
8. **Aggregate and inspect.** Compute per-metric means and distributions. Flag outliers for manual review — they often reveal systematic failure modes invisible in averages.

**Pro Tip:** _Store every pipeline configuration as code (YAML or JSON), not as notebook variables. Version the config file alongside your embeddings and index artifacts so any run can be reproduced exactly from a single commit hash._

**Logging checklist for each trial:**

- Retrieved passages (with passage IDs and scores)
- Model prompt sent to the generator
- Raw generator output
- Judge prompt and raw judge response
- Per-metric scores and aggregate summary

For [accuracy measurement in AI](https://www.babylovegrowth.ai/free-tools/structured-data-llm-audit) workflows, Mlflow's experiment tracking captures all of these artifacts in a single run record, making cross-trial comparison straightforward.

---

## How should you design human annotation for grounding labels?

Human annotation is the ground truth that validates your automated scoring. A weak annotation schema produces labels that are too coarse to diagnose retrieval vs. generation failures.

**Recommended schema fields:**

The GaRAGe benchmark uses `evidence_relevant` and `evidence_correct` at the passage level, which lets you run regression tests that catch silent failures — cases where answer accuracy holds steady but grounding quality degrades.

**Quality control steps:**

1. Write a detailed annotation guide with at least 10 worked examples covering edge cases (partial evidence, contradictory passages, unanswerable questions).
2. Run a calibration round: have all annotators label the same 20–30 items before the main annotation batch.
3. Target a Fleiss' Kappa of at least 0.60 for acceptable agreement; 0.70+ is the threshold RAGEval reports for their validation tasks.
4. Spot-check 5–10% of completed annotations per annotator per batch.
5. Adjudicate disagreements through a third annotator or a senior reviewer, not by majority vote alone.

**Tool options:** Label Studio (open-source, self-hosted), Scale AI, or a custom spreadsheet workflow for small batches under 500 items.

---

## What makes multi-turn RAG evaluation different?

Single-turn benchmarks systematically overestimate conversational RAG performance. The failure modes that matter most in production only appear across turns.

**Non-obvious failure modes to test:**

- **Later-turn retrieval degradation:** retrieval quality often drops after turn 3–4 as queries become shorter and more elliptical. MTRAG's 110 conversations averaging 7.7 turns expose this directly.
- **Non-standalone questions:** a question like "What about the second option?" is unanswerable without conversation history. The MTRAG benchmark explicitly includes these cases and shows they reveal weaknesses invisible in single-turn tests.
- **Context drift:** the generator accumulates incorrect assumptions from earlier turns and propagates them forward.
- **Stateful grounding errors:** a passage retrieved in turn 2 is incorrectly treated as relevant in turn 5 after the topic has shifted.

**Dataset design choices for multi-turn:**

1. Include at least 20% dependent-turn cases where the question cannot be answered without the prior turn's context.
2. Add unanswerable prompts at random turn positions, not just at the end.
3. Test both last-turn retrieval (only the final query triggers retrieval) and query-rewrite retrieval (the full conversation is condensed into a new query) — they stress different retrieval behaviors and often produce different failure patterns.
4. Track per-turn recall@k separately, not just aggregate recall across the conversation.

For building [multi-turn agent](https://mlflow.org/cookbook/multi-turn-agent) evaluation datasets that capture these dependencies, Mlflow's multi-turn agent cookbook provides concrete implementation patterns.

---

## Running RAG evaluations with Mlflow: a workflow sketch

Mlflow's experiment tracking and artifact management fit naturally into a RAG benchmarking pipeline. Here's a concrete workflow sketch you can adapt.

1. **Set up an experiment.** Create a named Mlflow experiment for each evaluation configuration (e.g., `rag-eval-finance-v1`). This scopes all runs and makes cross-configuration comparison clean.
2. **Log corpora and embeddings as artifacts.** Store your chunked document corpus and embedding vectors as Mlflow artifacts. Pin the embedding model name and version as run parameters.
3. **Capture retriever runs.** For each query, log the top-k retrieved passages (with passage IDs and retrieval scores) as a structured artifact. This is the data you need to compute recall@k and to audit grounding failures later.
4. **Log generations and retrieved context.** Store the full prompt sent to the generator, the raw output, and the retrieved passages together in a single run record. Never log just the final answer.
5. **Run LLM-as-a-judge scoring.** Use Mlflow's LLM-as-a-judge evaluation features to score Completeness, Hallucination, Faithfulness, and Context Relevance. Log the judge prompt template and raw judge responses as artifacts alongside the aggregated scores.
6. **Aggregate metrics and compare runs.** Use the Mlflow UI to compare metric distributions across retriever configs, chunk sizes, and generator settings in a single view.

**Pro Tip:** _Store the retrieved passages and judge prompts as first-class artifacts, not just as log lines. When a score drops between runs, you need the full context to determine whether the retriever, the generator, or the judge prompt changed — and log lines alone won't give you that._

**Integration notes:**

- Link your evaluation notebook to the Mlflow run via `mlflow.set_experiment()` and `mlflow.log_artifact()` calls.
- Save index configurations (ANN parameters, distance metric) as `mlflow.log_params()` entries so they appear in the run comparison view.
- Use Mlflow AI observability to trace agentic retrieval steps and capture per-step latency alongside quality metrics.

---

## The real trade-offs in RAG eval dataset design

The conventional wisdom says "more human labels = better evaluation." That's true at the margin, but it misses the more important question: _which_ labels, and _when_ in the project lifecycle.

At the proof-of-concept stage, a 50-item synthetic dataset built with schema-constrained generation (RAGEval's approach) gives you faster signal than 500 human-labeled items that took three weeks to collect. The synthetic set won't catch every failure mode, but it will catch the obvious ones — hallucination, low recall, irrelevant context — quickly enough to guide architecture decisions before you've committed to a retrieval stack.

Human grounding labels become worth the investment at pre-production, specifically the `evidence_relevant` and `evidence_correct` passage-level annotations from GaRAGe's schema. These are the labels that expose silent failures: cases where your aggregate accuracy metric looks fine but the generator is actually confabulating answers that happen to match the reference by coincidence. Catching that before production is worth the annotation cost.

For production monitoring, neither a full human-labeled set nor a static synthetic benchmark is the right tool. You want a live LLM-as-a-judge pipeline scoring a sample of real queries, with the judge prompts versioned and the raw outputs stored for audit. The RAGAS paper is right that you should publish judge prompts and raw outputs — not because reproducibility is a nice-to-have, but because a score you can't audit is a score you can't trust when something goes wrong in production.

One more thing teams consistently underestimate: multi-turn coverage. If your system handles conversations, a single-turn benchmark will give you an inflated performance estimate every time. Budget for at least a small multi-turn eval set before you ship.

---

![The real trade-offs in RAG eval dataset design — overview diagram](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1786511337359_The-real-trade-offs-in-RAG-eval-dataset-design-overview-diagram.jpeg)

## Mlflow makes RAG evaluation reproducible at scale

Evaluating RAG systems without a structured tracking layer means losing the context that makes scores meaningful. Mlflow's [GenAI and agent engineering platform](https://mlflow.org/genai) gives you experiment tracking, artifact versioning, and LLM-as-a-judge evaluation in a single open-source platform — so every retriever config, embedding version, and judge prompt is tied to the run that produced it.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

With Mlflow, you can log retrieved passages, generator outputs, and judge responses as structured artifacts, compare metric distributions across dozens of configurations in the UI, and reproduce any prior result from a single run ID. The LLM-as-a-judge evaluation features support Completeness, Hallucination, Faithfulness, and Context Relevance scoring out of the box, with prompt versioning through the built-in prompt registry. Start with the RAG evaluation examples in the Mlflow docs and run your first reproducible benchmark today.

---

## Papers, repos, and notebooks worth bookmarking

- **[RAGEval (ACL 2025)](https://aclanthology.org/2025.acl-long.418.pdf)** — Schema-based synthetic generation pipeline and Completeness/Hallucination/Irrelevance metrics. Bookmark for: tooling and dataset generation. Check the `OpenBMB/RAGEval` GitHub repo for runnable code.
- **RAGAS / WikiEval** — Reference-free evaluation framework with human-validated WikiEval dataset. Bookmark for: fast automated scoring loops without golden answers.
- **[GaRAGe (Amazon Science)](https://github.com/amazon-science/GaRAGe)** — Passage-level grounding annotations with `evidence_relevant` and `evidence_correct` fields. Bookmark for: regression testing and silent-failure detection. Also see the [GaRAGe ACL Findings paper](https://aclanthology.org/2025.findings-acl.875.pdf) for the full annotation methodology.
- **MTRAG (ACL TACL 2025)** — 110 human-generated multi-turn conversations, 842 tasks, plus companion MTRAG-S synthetic set. Bookmark for: multi-turn evaluation and later-turn failure analysis. See also the IBM/mt-rag-benchmark companion repo.
- **TechQA-RAG-Eval (Hugging Face)** — ~908 QA pairs from technical support forums with `is_impossible` flags. Bookmark for: domain-specific benchmarking in knowledge-base and support scenarios.
- **Hugging Face Open-Source AI Cookbook RAG Evaluation notebook** — End-to-end walkthrough of synthetic dataset construction and LLM-as-a-judge scoring. Bookmark for: hands-on pipeline prototyping. Search "RAG evaluation" in the Hugging Face cookbook index.
- **Kaggle Single-Topic RAG Evaluation Dataset** — Focused single-topic retrieval benchmark. Bookmark for: quick retrieval-only testing with a clean, scoped corpus.

**License note:** Before using any dataset in production evaluations or publications, verify its license. GaRAGe and several Hugging Face datasets use CC-BY-NC or similarly restricted licenses that prohibit commercial use. Check the dataset card or repository README for the exact terms before you build a benchmark dependency on it.

## Sources

- [RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](https://aclanthology.org/2025.acl-long.418.pdf)
- [amazon-science/GaRAGe](https://github.com/amazon-science/GaRAGe)

## Recommended

- [One post tagged with "how to use evaluation datasets" | MLflow](https://mlflow.org/articles/tags/how-to-use-evaluation-datasets)
- [One post tagged with "evaluation datasets in AI" | MLflow](https://mlflow.org/articles/tags/evaluation-datasets-in-ai)
- [MLflow](https://mlflow.org/cookbook/rag-evaluation)
- [Benchmark Your Way to Better RAG and Agents:Tuning Vector Search with MLflow | MLflow](https://mlflow.org/blog/tune-and-benchmark-with-mlflow)
