---
title: Your Agents Need an AI Platform
description: Getting an agent to work in a demo is easy. Getting it to work in production requires an AI engineering platform with observability, evaluation, version control, and an AI gateway.
slug: agents-need-ai-platform
authors: [mlflow-maintainers]
tags:
  [mlflow, genai, agents, tracing, evaluation, ai-gateway, version-control]
thumbnail: /img/blog/agents-need-ai-platform-thumbnail.png
image: /img/blog/agents-need-ai-platform-thumbnail.png
date: 2026-03-10
---

Getting an AI agent to work in a demo is easy. Getting it to work reliably in production is a different problem entirely.

Consider a customer support agent. It has access to three tools: look up order status, process refunds, and search your knowledge base. The system prompt is a few paragraphs. You wire it up with an LLM and a framework like LangGraph or OpenAI Agents SDK, test it on a handful of questions, and it works. Ship it.

Once it's in production, things break down quickly. A customer asks for a refund on an order that doesn't exist, and the agent hallucinates a confirmation number. Another customer's home address shows up in a response that gets logged to an analytics dashboard. A model provider updates their API, and your agent's refund success rate quietly drops from 94% to 71% over a weekend. Nobody notices for a week. You get an invoice for $12,000 in API costs because a retry loop went haywire on Thursday night.

For your agents to thrive in production, they need an **AI platform**: integrated tooling for [**observability**](/ai-observability), [**evaluation**](/llm-evaluation), [**version control**](https://mlflow.org/docs/latest/genai/prompt-registry/index.html), and [**governance**](/ai-gateway). This article walks through why each one matters and how MLflow brings them together in a unified open source platform.

## The Gap Between Demos and Production Agents

LLM outputs are nondeterministic, and that nondeterminism compounds at scale. The real world is always more complex than your demo or testing environment. **As a result, operating an agent in production requires specialized tooling**.

Building an agent typically involves choosing a framework, writing prompts, connecting tools, and testing with a few examples. In contrast, operating a production agent means understanding what it's doing across thousands of requests, measuring whether it's performing well, and controlling how it's interacting with LLMs and external services.

These operational needs map to four capabilities that every production agent requires:

1. [**Observability**](/ai-observability): full visibility into what your agent is doing, step by step
2. [**Evaluation**](/llm-evaluation): reproducible quality measurement across every dimension you care about
3. [**Version control**](https://mlflow.org/docs/latest/genai/prompt-registry/index.html): versioned prompts and configurations that can be compared, optimized, and rolled back
4. [**Governance**](/ai-gateway): centralized control over LLM calls, data access, and costs

Let's walk through each one using our customer support agent to see how an AI platform transforms a demo into a production-grade agent.

## Observability: You Can't Debug What You Can't See

Your agent processes 2,000 support tickets a day. A customer complains that the agent told them their refund was processed, but it wasn't. What happened?

Without observability, you're guessing. Maybe the tool call failed silently. Maybe the LLM hallucinated the confirmation. Maybe the retrieval step pulled the wrong knowledge base article. Maybe the agent called the right tool with the wrong parameters. You have no idea, because all you logged was the final response.

**AI platforms** provide observability through [**tracing**](/llm-tracing): a system that records every step your agent takes, so you can replay and inspect any request after the fact. As we'll see, tracing also makes the rest of the platform work: evaluation, version control, and governance all depend on having full visibility into what your agent is doing.

<div style={{display: "flex", gap: "1.5rem", alignItems: "center", marginBottom: "0"}}>
<div style={{flex: 1, minWidth: 0}}>

```python
import mlflow

# One line instruments your entire agent
mlflow.langchain.autolog()

# Or use the decorator for custom agents
@mlflow.trace(name="customer_support")
def handle_ticket(ticket):
    intent = classify_intent(ticket)
    if intent == "refund":
        order = lookup_order(ticket.order_id)
        return process_refund(order)
    return search_knowledge_base(ticket.question)
```

</div>
<div style={{flex: 1, minWidth: 0}}>

<img src="/img/GenAI_home/GenAI_trace_darkmode.png" alt="MLflow Tracing UI showing a hierarchical trace breakdown with LLM calls, tool invocations, and latency" style={{borderRadius: "8px", width: "100%"}} />

</div>
</div>
<figure style={{ marginTop: "1.5rem", marginBottom: "2.5rem" }}>
<figcaption style={{ textAlign: "center" }}>
<i>Every agent interaction produces a trace you can inspect in the MLflow UI. When that customer complains about a phantom refund, you find the trace, see that <code>lookup_order</code> returned an error the LLM ignored, and fix the problem in minutes.</i>
</figcaption>
</figure>

[MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/index.html) captures the full execution graph of every agent interaction: every LLM call, every tool invocation, every retrieval step, with inputs, outputs, token counts, and latency. MLflow Tracing is [OpenTelemetry-compatible](/blog/opentelemetry-tracing-support), so it works with any programming language, any agent framework, and any LLM provider. MLflow also offers [one-line autologging](https://mlflow.org/docs/latest/genai/tracing/#one-line-auto-tracing-integrations) for more than 30 popular frameworks and providers (LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, Pydantic AI, Google ADK, and [many more](https://mlflow.org/docs/latest/genai/tracing/integrations/)) that instruments your entire application automatically with just a single line of code.

## Evaluation: Prove Your Agent Works Before You Ship It

Your customer support agent handles refunds, answers product questions, and looks up order status. Before you deploy it, how do you know it performs well enough? And after it's in production, how do you know it's still working well? You can't manually sift through thousands of traces looking for problems. Every change you push risks breaking something that goes undetected because most users don't provide detailed feedback, they just stop using your product.

**AI platforms** solve this with an [**evaluation framework**](/llm-evaluation): a system that automatically scores your agent's outputs using defined criteria, both before deployment and continuously in production. Evaluation frameworks combine two approaches: LLM judges — models that automatically score your agent's outputs across dimensions like correctness, safety, relevance, and tool call accuracy — and human feedback, where domain experts review and label agent outputs to catch issues that automated scoring misses. You define what "good" means for your use case, and the framework measures it at scale.

<div style={{display: "flex", gap: "1.5rem", alignItems: "center", marginBottom: "0"}}>
<div style={{flex: 1, minWidth: 0}}>

```python
import mlflow
from mlflow.genai.scorers import (
    Safety,
    Correctness,
    RelevanceToQuery,
)

# Evaluate your agent against a dataset
results = mlflow.genai.evaluate(
    data=support_tickets_dataset,
    predict_fn=handle_ticket,
    scorers=[
        Safety(),
        Correctness(),
        RelevanceToQuery(),
    ],
)
```

</div>
<div style={{flex: 1, minWidth: 0}}>

<div style={{overflow: "hidden", borderRadius: "8px"}}><img src="/img/GenAI_home/GenAI_evaluation_darkmode.png" alt="MLflow Evaluation UI showing pass/fail scores for conciseness and hallucination across agent responses" style={{width: "100%", marginBottom: "-30px"}} /></div>

</div>
</div>
<figure style={{ marginTop: "1.5rem", marginBottom: "2.5rem" }}>
<figcaption style={{ textAlign: "center" }}>
<i>With MLflow, you can define judges for the dimensions you care about, run them against your agent's responses, tool calls, and reasoning steps, analyze results in a comprehensive dashboard, and monitor quality continuously in production.</i>
</figcaption>
</figure>

[MLflow's agent evaluation framework](https://mlflow.org/docs/latest/genai/eval-monitor/) ships with [70+ built-in judges](https://mlflow.org/docs/latest/genai/evaluation/judges.html) covering response quality, safety, tool call correctness, user frustration, and more. You can also [define custom judges](https://mlflow.org/docs/latest/genai/evaluation/judges.html) for your domain and [collect human feedback](https://mlflow.org/docs/latest/genai/concepts/feedback/) through a built-in labeling UI. Because MLflow evaluation is integrated with tracing, you can [run scorers against production traces continuously](https://mlflow.org/docs/latest/genai/evaluation/index.html), catching quality issues in minutes.

## Version Control: Your Agents Need a Changelog

Agents have a lot of moving parts, including system prompts, tool definitions, data retrieval configurations, and model parameters. When something changes and quality drops, you need to know what changed, compare it against the previous version, and roll back if necessary.

Most teams struggle with this because of one principal problem: **nothing ties a change to its impact on quality**. Prompts and configurations often live as hardcoded values in application code, and changes frequently blend together. One engineer rewrites the refund instructions to handle edge cases, another updates the few-shot examples to improve tone, and when refund accuracy drops, nobody can pinpoint which change caused the quality drop because no change has lineage to performance data.

**AI platforms** solve this with a [**prompt registry**](/prompt-registry): a versioned store for prompts and configurations where every version has lineage to traces and evaluation results.

<div style={{display: "flex", gap: "1.5rem", alignItems: "center", marginBottom: "0"}}>
<div style={{flex: 1, minWidth: 0}}>

```python
import mlflow

# Register a prompt in the registry
mlflow.genai.register_prompt(
    name="customer_support_system",
    template="You are a customer support agent for {{company_name}}. "
    "You can help customers with order status, refunds, and product "
    "questions. Always verify the order ID before processing refunds. "
    "Never share customer PII in your responses.",
)

# Load and use the latest version
prompt = mlflow.genai.load_prompt("customer_support_system")
```

</div>
<div style={{flex: 1, minWidth: 0}}>

<img src="/img/GenAI_home/GenAI_prompts_darkmode.png" alt="MLflow Prompt Registry UI showing versioned prompts with aliases, diff history, and commit messages" style={{borderRadius: "8px", width: "100%"}} />

</div>
</div>
<figure style={{ marginTop: "1.5rem", marginBottom: "2.5rem" }}>
<figcaption style={{ textAlign: "center" }}>
<i>MLflow's prompt registry versions every prompt and links it to the traces and evaluation results it produced, so you can pinpoint which change caused a quality drop.</i>
</figcaption>
</figure>

[MLflow's prompt registry](https://mlflow.org/docs/latest/genai/prompt-registry/index.html) links every version of your prompt directly to the traces and performance data it produced, so you can trace a quality drop back to the exact change that caused it. Additionally, MLflow offers [**prompt optimization**](/prompt-optimization), which automates prompt engineering by using an LLM to generate, test, and select improved prompt versions.

## Governance: Your Agent Has No Guardrails

Your customer support agent has API keys for OpenAI, Anthropic (as a fallback), and your internal order management system. Those keys are scattered across environment variables, config files, and `.env` files on developer laptops. One engineer hardcoded a key in a Jupyter notebook that got committed to git. The agent has no rate limits, so when a retry bug causes a loop, it burns through your API budget before anyone notices. And there's no centralized way to enforce content policies: the agent can return whatever the LLM generates, including customer PII that shouldn't be in a chat response.

**AI platforms** solve this with an [**AI gateway**](/ai-gateway): a centralized proxy between your agent and every LLM provider it calls. An AI gateway handles credential management, traffic routing, cost controls, and content guardrails in one place, so they're enforced consistently and never skipped.

<div style={{display: "flex", gap: "1.5rem", alignItems: "center", marginBottom: "0"}}>
<div style={{flex: 1, minWidth: 0}}>

```python
from openai import OpenAI

# Point your agent at the gateway instead of OpenAI directly
client = OpenAI(
    base_url="https://your-mlflow-server/gateway/v1",
)

# Your agent code doesn't change. The gateway handles:
# - API key management (keys never touch application code)
# - Automatic failover (if OpenAI is down, route to Anthropic)
# - Usage tracking (token counts, costs per endpoint)
# - Rate limiting and budget controls
# - Content guardrails and PII redaction
```

</div>
<div style={{flex: 1, minWidth: 0}}>

<img src="/img/blog/gateway-usage-tracking-dashboard.png" alt="MLflow AI Gateway usage dashboard showing requests, latency, errors, token usage, and cost breakdown" style={{borderRadius: "8px", width: "100%"}} />

</div>
</div>
<figure style={{ marginTop: "1.5rem", marginBottom: "2.5rem" }}>
<figcaption style={{ textAlign: "center" }}>
<i>MLflow's AI Gateway provides a centralized proxy for managing credentials, routing traffic, controlling costs, and enforcing guardrails across all your LLM providers.</i>
</figcaption>
</figure>

[MLflow's AI Gateway](https://mlflow.org/docs/latest/genai/ai-gateway/index.html) exposes an [OpenResponses](https://www.openresponses.org/)-compatible API, so you can switch model providers without changing your code. A/B test GPT-5 against Claude by splitting traffic 90/10. Set up usage alerts so a retry loop can't burn through your budget overnight. Enforce PII redaction with guardrails, so sensitive data never reaches users when LLMs misbehave. Because the gateway is integrated with [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/index.html), every request automatically becomes a trace with full context: model used, tokens consumed, latency, and whether any guardrails were triggered.

## Why a Unified Platform Wins

You could build your own AI platform by stitching together separate tools. Use LangSmith for tracing, DeepEval for evaluation, a git repo for prompts, and LiteLLM for gateway routing. Some teams do.

But the integration tax adds up fast. Your evaluation framework can't access your traces, so you build a pipeline to export data between them. Your gateway doesn't know about your prompt versions, so you manually track which prompt is deployed where. Your tracing tool doesn't feed into your evaluation dashboard, so quality problems go unnoticed until a customer complains.

When all four capabilities live in one platform, they **work together**:

- **Traces feed evaluations**: run judges directly against production traces to catch quality issues in real time
- **Evaluations validate prompts**: every prompt change is measured against your evaluation suite before deployment
- **Judges act as guardrails**: the same LLM judges that evaluate quality can block unsafe or low-quality responses at the gateway before they reach users
- **The gateway generates traces**: every model interaction is automatically traced with full context, no separate instrumentation needed
- **Prompt versions link to evaluation results**: see exactly how each prompt version performed across every quality dimension

That's the difference between operating your agent with a single dashboard and operating it with four disconnected tools and a Slack channel full of "is this working?" messages.

## MLflow: The Open-Source AI Platform for Agents

MLflow is the only open-source platform that provides all four of these capabilities ([tracing](/llm-tracing), [evaluation](/llm-evaluation), [version control](/prompt-optimization), and [AI gateway](/ai-gateway)) in a single, integrated platform.

Three things make this worth paying attention to:

**It's open source and vendor-neutral.** Apache 2.0 licensed, backed by the Linux Foundation, with over 20,000 GitHub stars and 30 million monthly downloads. No per-trace fees, no per-evaluation charges, no lock-in. Your data stays yours. Traces are OpenTelemetry-compatible and can be exported to any backend.

**It works with whatever you're already using.** MLflow supports LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, Pydantic AI, Google ADK, and 20+ other agent frameworks. It works with 50+ LLM providers. You pick your tools; MLflow handles the operational side.

**It's already running in production everywhere.** Thousands of organizations use MLflow today. Millions of ML practitioners already rely on it, and it now has the capabilities that production agents need.

## Getting Started

The fastest way to try this is to instrument an existing agent:

```bash
pip install 'mlflow[genai]'
mlflow server
```

```python
import mlflow

# Start tracing your agent with one line
mlflow.openai.autolog()  # or langchain, crewai, autogen, etc.
```

From there, layer on evaluation, register your prompts, and configure the gateway. Each one builds on the tracing foundation.

Your agents can do a lot. Give them the platform to do it reliably.

- [MLflow Tracing docs](https://mlflow.org/docs/latest/genai/tracing/index.html)
- [MLflow Evaluation docs](https://mlflow.org/docs/latest/genai/evaluation/index.html)
- [MLflow Prompt Registry docs](https://mlflow.org/docs/latest/genai/prompt-registry/index.html)
- [MLflow AI Gateway docs](https://mlflow.org/docs/latest/genai/ai-gateway/index.html)

If you find MLflow useful, [give us a star on GitHub](https://github.com/mlflow/mlflow).
