---
title: "RAG Tracing: What to Capture and How to Debug Pipelines"
description: "RAG tracing shows each retrieval, rerank and generation step in one trace. Learn what to capture, how to protect data, and which metrics to pair with traces."
slug: rag-tracing
tags: [rag agent observability, rag tracing, rag observability]
date: 2026-10-30
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1793320101941_Engineer-inspecting-nested-RAG-trace-spans.jpeg
---

![Engineer inspecting nested RAG trace spans](https://media.babylovegrowth.ai/blog-images/organization-30814/1793320101941_Engineer-inspecting-nested-RAG-trace-spans.jpeg)

RAG tracing records every step a request takes through a retrieval-augmented generation pipeline as one connected trace: the query rewrite, the vector search, the reranker, the prompt assembly, the model call and any tool calls. Each step becomes a span with its inputs, outputs, latency and token usage. When an answer is wrong, you open the trace and see whether retrieval failed, the context was ignored or the model made something up. This guide covers what to capture, how to keep sensitive data out of spans, which evaluation signals to pair with traces, and how we do it in MLflow.

## TL;DR

- A RAG trace is a tree of spans for one request. At minimum it needs a retrieval span (query, document IDs, scores), a generation span (model, prompt, tokens) and the root span that ties them together.
- Most bad RAG answers are retrieval problems, not model problems. Only a trace that stores retrieved chunk IDs and scores lets you tell the difference.
- Use OpenTelemetry GenAI attribute names so your traces work across backends and frameworks.
- Redact or reference sensitive content instead of storing it. Store document IDs, and keep raw prompt text off by default.
- Traces tell you what happened. Pair them with automated evaluation, such as LLM-as-a-judge scorers, to tell you whether it was good.

## Table of Contents

- [What RAG tracing is and why logs are not enough](#what-rag-tracing-is-and-why-logs-are-not-enough)
- [What a RAG trace should capture](#what-a-rag-trace-should-capture)
- [How RAG tracing supports debugging in production](#how-rag-tracing-supports-debugging-in-production)
- [Tracing RAG agents, tools and multi-step workflows](#tracing-rag-agents-tools-and-multi-step-workflows)
- [Protecting sensitive prompts, documents and user data](#protecting-sensitive-prompts-documents-and-user-data)
- [Metrics and evaluation signals to pair with traces](#metrics-and-evaluation-signals-to-pair-with-traces)
- [How we trace RAG with MLflow](#how-we-trace-rag-with-mlflow)
- [Start tracing your RAG pipeline with MLflow](#start-tracing-your-rag-pipeline-with-mlflow)
- [A starter checklist for RAG tracing](#a-starter-checklist-for-rag-tracing)
- [FAQ](#faq)
- [Sources](#sources)

## What RAG tracing is and why logs are not enough

A RAG application is a chain of dependent steps. A user question is rewritten, embedded, matched against an index, filtered, reranked, stuffed into a prompt and sent to a model. Sometimes the model then calls a tool and the loop repeats. A final answer that looks wrong could come from any of these steps.

Plain application logs give you timestamps and error messages. They do not show you which chunks were retrieved, in what order, with what similarity, or how much of the context window they used. A trace does, because it keeps parent-child structure: the retrieval span sits under the same root as the generation span, so you can read the whole request top to bottom.

The OpenTelemetry project makes the same argument for LLM systems generally. [Its introduction to LLM observability](https://opentelemetry.io/blog/2024/llm-observability/) explains that model calls, prompts, completions, token usage and quality-related behavior are application-specific signals that conventional service telemetry does not cover. RAG adds one more: what your knowledge base returned.

![What RAG tracing is and why logs are not enough](https://media.babylovegrowth.ai/blog-images/organization-30814/1793320125947_What-RAG-tracing-is-and-why-logs-are-not-enough.jpeg)

### The anatomy of a RAG trace

A typical single-turn RAG request produces a trace like this:

| Span                  | Type           | What it tells you                                        |
| --------------------- | -------------- | -------------------------------------------------------- |
| `rag_pipeline` (root) | Chain or agent | End-to-end latency, final answer, status, total tokens   |
| `rewrite_query`       | LLM call       | How the question was reformulated before search          |
| `embed_query`         | Embedding      | Embedding model and latency                              |
| `vector_search`       | Retriever      | Top-k, filters, returned chunk IDs and scores            |
| `rerank`              | Reranker       | How ordering changed after reranking                     |
| `build_prompt`        | Chain step     | Which chunks made it into the context, and in what order |
| `generate_answer`     | LLM call       | Model, prompt, completion, tokens, finish reason         |

If your application is an agent, the generation span may be followed by tool calls and further retrieval rounds. Each round nests under the same root, so a five-step agent run is still one readable trace.

## What a RAG trace should capture

The temptation is to log everything. The better approach is to decide which questions you will need to answer at 2 a.m. and capture the fields that answer them.

### Fields for every LLM and retrieval span

OpenTelemetry's [GenAI semantic-convention registry](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/) is the shared vocabulary here. It documents attributes for the operation name, provider name, request and response model, token usage, finish reasons, time to first chunk, retrieval query text, retrieved document identifiers and scores, tool metadata and workflow names. It also describes `gen_ai.input.messages` and `gen_ai.output.messages` for model inputs and outputs, including tool-call and tool-result message parts.

Using these names rather than inventing your own means a trace produced by one instrumentation library can be read by any backend that understands the conventions.

### Retrieval-specific fields

Retrieval is where most RAG debugging happens, so the retriever span deserves the most care. A [practitioner guide to observability for agentic RAG](https://sarangk90.github.io/awesome-agent-evals-and-guardrails/03-online-evals-and-observability/02-observability-and-tracing/) suggests capturing the rewritten query, index and snapshot version, top-k, filters and hybrid weights, the returned document or chunk IDs, scores, ranks, source URIs, ACL-filter counts and the embedding model version.

We would put three of those at the top of your list:

1. **Chunk IDs with scores and ranks.** Without them you cannot say whether the right document was retrieved but ranked too low.
2. **Index or snapshot version.** If you re-index weekly, an answer that regressed on Tuesday may simply reflect a new snapshot.
3. **Filter and ACL counts.** If permission filters removed every relevant chunk, the model had nothing to work with, and the trace should show that.

### Fields for the root span

The same guide recommends root-agent fields for trace and session identifiers, pseudonymous user and tenant context, application and prompt versions, deployment environment, redacted input, final output with citations, status, latency, time to first token, token usage, cost, and counts of LLM calls, tool calls and agent steps.

The prompt version and application version are the two we find most useful. They let you filter traces to "everything served by prompt v7" and compare it with v8.

## How RAG tracing supports debugging in production

With traces in place, the vague complaint "the bot gave a bad answer" becomes a short diagnostic path.

1. **Open the trace for the bad answer** using the session or trace ID the user or support team can supply.
2. **Read the retriever span.** Were relevant chunks returned? If not, the problem is the query rewrite, the embedding, the index or the filters.
3. **Check ordering and truncation.** If the right chunk was retrieved at rank 9 but only the top 5 reached the prompt, you have a reranking or top-k problem.
4. **Read the generation span.** If the right context was in the prompt and the answer still ignored it, the problem is the prompt or the model.
5. **Check latency and tokens.** A slow trace usually points at one fat span, such as an oversized context or a slow reranker.

This ordering matters. Teams often tune prompts first because prompts are easy to edit, then discover weeks later that retrieval was returning the wrong documents the whole time. A trace puts both in one view, so you fix the right layer.

### Latency and cost

Span timing shows where a request spends its time, and token counts per generation span show where money goes. If your retrieval returns 20 long chunks and you pass all of them to the model, the trace shows a large prompt token count on the generation span and, often, a slow time to first token. Trimming context then becomes a measurable change, not a guess.

## Tracing RAG agents, tools and multi-step workflows

A fixed pipeline runs the same steps every time. An agent decides at runtime whether to retrieve, which index to query, whether to call a tool and when to stop. That changes what you need from tracing.

- **Variable depth.** Two requests produce differently shaped traces. You need tree views and counts of agent steps, not a fixed schema.
- **Repeated retrieval.** An agent may query three times with three reformulations. Each retrieval span must carry its own query and results.
- **Tool results feeding the next step.** Tool call and tool result messages need to be captured so you can see what the agent believed when it chose its next action.
- **Framework-specific structure.** Orchestration frameworks structure their work differently.

On that last point, [OpenTelemetry's article on agent observability](https://opentelemetry.io/blog/2025/ai-agent-observability) says conventions for agent applications are established, while framework conventions are still being developed for systems such as CrewAI, AutoGen and LangGraph. In practice this means you should expect some variation between frameworks and favor tooling that builds on OpenTelemetry rather than a private format.

## Protecting sensitive prompts, documents and user data

RAG traces are risky by nature: prompts contain user questions, and retrieved chunks contain your internal documents. A trace store that holds both is a second copy of your most sensitive content. Plan for that from the first day.

**Turn content capture off by default where you can.** OpenTelemetry's [Agent Service demo documentation](https://opentelemetry.io/bn/docs/demo/services/agent/) notes that its GenAI instrumentation records prompts and completions as `gen_ai.input.messages` and `gen_ai.output.messages`, and that setting `TRACELOOP_TRACE_CONTENT=false` prevents that content from being exported. That variable applies to that instrumentation, so check how your own instrumentation library handles content capture.

**Store pointers, not text.** The agentic RAG guide above treats raw chunk text as optional and suggests an external pointer instead. That reduces span volume and privacy exposure while keeping retrieval debugging possible: you have the chunk ID and can look up the text in the source system, under its own access controls.

**Use pseudonymous identifiers.** Record a hashed user or tenant ID on the root span, not an email address.

**Redact before export.** If you must keep input text for debugging, run redaction before spans leave your process.

If your data falls under a regulation such as HIPAA or a state privacy law, have your security or legal team review what the trace store holds and who can read it. We cannot tell you which rules apply to your data.

![Protecting sensitive prompts, documents and user data](https://media.babylovegrowth.ai/blog-images/organization-30814/1793320139466_Protecting-sensitive-prompts-documents-and-user-data.jpeg)

## Metrics and evaluation signals to pair with traces

A trace is a record. It does not tell you whether an answer was good. Pair it with signals that do.

**Retrieval signals**

- _Context relevance:_ are the retrieved chunks relevant to the question?
- _Context sufficiency:_ does the retrieved context contain what is needed to answer?
- _Recall at k, when you have labeled data:_ did the known-correct document appear in the top k?

**Generation signals**

- _Groundedness (faithfulness):_ is every claim in the answer supported by the retrieved context?
- _Answer relevance:_ does the answer address the question asked?
- _Safety and tone checks:_ does it follow your policies?

**Operational signals**

- Latency (end to end and per span), time to first token, token usage and cost per request.
- Error rate and finish reasons, such as truncation by length limits.
- Number of LLM calls, tool calls and steps per request for agents.

Most teams cannot read every trace, so they use an LLM judge to score traces automatically against criteria like groundedness, then sample low scorers for human review. Keep humans in the loop: calibrate your judge against a set of traces people have labeled, and revisit it when you change the judge model.

Once scores are attached to traces, you can slice by prompt version, index snapshot or model and see which change moved the numbers. That is the loop that turns tracing from a debugging tool into a development process.

## How we trace RAG with MLflow

MLflow is an open-source platform for GenAI and LLM lifecycle management, and [tracing is built on OpenTelemetry](https://mlflow.org/articles/otel-for-llm), so traces follow the conventions described above and can work alongside the rest of your telemetry. Our [AI observability page](https://mlflow.org/ai-observability) covers the full feature set. Here is the shape of a minimal manually instrumented pipeline:

```python
import mlflow
from mlflow.entities import SpanType, Document

mlflow.set_experiment("rag-demo")

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(query: str):
    hits = vector_store.search(query, k=5)
    return [
        Document(
            page_content=h.text,
            metadata={"doc_id": h.id, "score": h.score},
        )
        for h in hits
    ]

@mlflow.trace(span_type=SpanType.LLM)
def generate(question: str, docs):
    context = "

".join(d.page_content for d in docs)
    return llm.complete(f"Context:
{context}

Question: {question}")

@mlflow.trace
def rag_pipeline(question: str):
    docs = retrieve(question)
    return generate(question, docs)
```

The decorators create the span tree: `rag_pipeline` is the root, with `retrieve` and `generate` as children. If you use a supported framework or model provider, automatic tracing can capture these spans for you, so you add manual decorators only for your own code. In this example `vector_store` and `llm` stand in for your own clients, and the retriever returns document objects so the retrieval span carries chunk IDs and scores. If your chunk text is sensitive, put only the ID in the metadata and leave the text out of what you return into the span.

From there, the workflow is:

1. **Instrument** the pipeline, automatically or with decorators like those above.
2. **Run real or replayed traffic** and inspect traces in the MLflow UI, starting with retrieval spans for your worst answers.
3. **Attach scores** using [LLM-as-a-judge evaluation](https://mlflow.org/llm-as-a-judge) for groundedness, relevance and your own custom criteria.
4. **Version prompts and the application** so each trace says which version produced it.
5. **Compare** scores across versions before you ship a change.

If your team already runs OpenTelemetry collectors, our guide to [shipping GenAI-conformant OTel traces to MLflow](https://mlflow.org/articles/otel-for-llm) walks through the setup. You can also browse more posts on [OpenTelemetry for LLM](https://mlflow.org/articles/tags/opentelemetry-for-llm) topics.

> ## Our perspective on RAG tracing
>
> We think the most common mistake in RAG engineering is treating the system as a model problem. When an answer is wrong, the instinct is to rewrite the prompt or switch models. But a RAG system is mostly a data and retrieval system with a model at the end, and if retrieval returns the wrong chunks, no prompt will rescue the answer.
>
> That is why we build tracing, evaluation, prompt management and gateway governance as one lifecycle instead of separate tools. A trace is only the starting point. It becomes valuable when scores sit next to it, prompt versions are tied to it, and the whole thing can be compared across releases. Teams that stop at tracing end up with a pile of beautiful, unread data.
>
> We also believe in open standards. MLflow is fully open source under Linux Foundation governance, builds on OpenTelemetry, and does not hold features back behind an enterprise paywall. Your traces should not be locked into one vendor's format, and your observability should not depend on a procurement cycle.
>
> If you take one thing from this guide: capture chunk IDs and scores on every retrieval span, and score groundedness automatically. Those two habits will find more bugs than any amount of prompt tuning.
>
> _Kevin_

## Start tracing your RAG pipeline with MLflow

MLflow is free and open source, so you can instrument a RAG pipeline today without a contract. It gives you tracing for LLM and agent applications, automated evaluation with LLM judges, prompt management and versioning, and an AI Gateway for governing access across model providers.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

Start with [MLflow for agent and LLM engineering](https://mlflow.org/genai) to see how the pieces fit, or go to [mlflow.org](https://mlflow.org) to get set up. Teams that need enterprise support, managed options or help with integration can reach us through the site.

## A starter checklist for RAG tracing

Use this list when you instrument a new pipeline or audit an existing one.

- [ ] Every request has a root span with trace ID, session ID, environment and app version.
- [ ] Retrieval spans record the query actually sent to the index, top-k, filters, chunk IDs, scores and ranks.
- [ ] The index or snapshot version and embedding model version are recorded.
- [ ] Generation spans record model, token usage, finish reason and latency.
- [ ] Prompt version is on every trace.
- [ ] Raw prompt and chunk text is off, redacted or replaced by pointers, and the trace store has access controls.
- [ ] Automated scorers for groundedness and relevance run on a sample of production traces.
- [ ] Low-scoring traces are reviewed by a person each week and turned into test cases.

## FAQ

### Is RAG tracing the same as RAG observability?

Tracing is the core data source of observability, not the whole of it. Tracing records what happened inside each request. Observability also includes metrics such as latency and cost, evaluation scores, and the dashboards and alerts built from them. You start with traces, then add the rest.

### What is the difference between tracing a RAG pipeline and a RAG agent?

A pipeline follows a fixed sequence, so its traces look alike. An agent chooses steps at runtime, so traces vary in depth and may include several retrievals and tool calls. For agents, capture step counts, tool calls and tool results in addition to retrieval and generation spans.

### Do I need OpenTelemetry for RAG tracing?

You do not strictly need it, but it is a good default. Shared GenAI attribute names make traces portable between tools, and [OpenTelemetry's agent observability work](https://opentelemetry.io/blog/2025/ai-agent-observability) is moving toward conventions for agent frameworks. MLflow tracing is built on OpenTelemetry, so you can adopt the standard without extra work.

### How do I keep private documents out of my traces?

Store chunk IDs and source URIs instead of chunk text, redact user input before export, and turn off prompt and completion capture where your instrumentation supports it. Restrict who can read the trace store, since it still reveals user behavior. Ask your security team to review before you trace regulated data.

### Which metrics should I watch first?

Start with groundedness, context relevance, end-to-end latency and cost per request. Those four cover whether the answer is supported, whether retrieval worked, and whether the experience and budget are sustainable. Add task-specific scorers once you know your common failure modes.

## Sources

- [Gen AI attributes registry, OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)
- [An Introduction to Observability for LLM-based applications, OpenTelemetry](https://opentelemetry.io/blog/2024/llm-observability/)
- [AI Agent Observability: Evolving Standards and Best Practices, OpenTelemetry](https://opentelemetry.io/blog/2025/ai-agent-observability)
- [Agent Service, OpenTelemetry Demo documentation](https://opentelemetry.io/bn/docs/demo/services/agent/)
- [Observability & Tracing for Agentic RAG, Agent Evals & Guardrails](https://sarangk90.github.io/awesome-agent-evals-and-guardrails/03-online-evals-and-observability/02-observability-and-tracing/)
