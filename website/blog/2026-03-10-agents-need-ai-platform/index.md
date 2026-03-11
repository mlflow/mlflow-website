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

For your agents to thrive in production, they need an AI platform: integrated tooling for observability, evaluation, version control, and governance. This post walks through why each one matters and how MLflow brings them together in a unified open source platform.

## The Gap Between Demos and Production Agents

LLM outputs are nondeterministic, and that nondeterminism compounds at scale. The real world is always more complex than your demo or testing environment. **As a result, operating an agent in production requires specialized tooling**.

Building an agent typically looks like choosing a framework, writing prompts, connecting tools, and testing with a few examples. In contrast, operating a production agent means understanding what it's doing across thousands of requests, measuring whether it's performing well, and controlling how it's interacting with LLMs and external services.

These operational needs map to four capabilities that every production agent requires:

1. **Observability**: full visibility into what your agent is doing, step by step
2. **Evaluation**: reproducible quality measurement across every dimension you care about
3. **Version control**: versioned prompts and configs that can be compared, optimized, and rolled back
4. **Governance**: centralized control over LLM calls, data access, and costs

Let's walk through each one using our customer support agent.

## Observability: You Can't Debug What You Can't See

Your agent processes 2,000 support tickets a day. A customer complains that the agent told them their refund was processed, but it wasn't. What happened?

Without observability, you're guessing. Maybe the tool call failed silently. Maybe the LLM hallucinated the confirmation. Maybe the retrieval step pulled the wrong knowledge base article. Maybe the agent called the right tool with the wrong parameters. You have no idea, because all you logged was the final response.

**MLflow Tracing** captures the full execution graph of every agent interaction: every LLM call, every tool invocation, every retrieval step, with inputs, outputs, token counts, and latency. When something goes wrong, you don't guess. You open the trace and see exactly what happened.

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

Every call to `handle_ticket` produces a trace you can inspect in the MLflow UI: a hierarchical view showing the intent classification, the tool calls, the LLM reasoning at each step, and where things went off track. When that customer complains about a phantom refund, you find the trace, see that `lookup_order` returned an error that the LLM ignored, and fix the problem in minutes instead of days.

Tracing also makes the rest of the platform work. You can't evaluate what you can't observe. You can't optimize prompts without seeing how they perform across real requests. Everything else builds on top of it.

MLflow Tracing works with [20+ agent frameworks](https://mlflow.org/docs/latest/genai/tracing/index.html) (LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, Pydantic AI, Google ADK, and others) with one-line autologging for each. It supports [50+ LLM providers](https://mlflow.org/docs/latest/genai/tracing/index.html) and is fully [OpenTelemetry-compatible](/blog/opentelemetry-tracing-support), so traces can be exported to any backend.

## Evaluation: You Don't Know if Your Agent Is Good

Your customer support agent handles refunds, answers product questions, and looks up order status. It's been running for two weeks. Is it good?

"Good" is hard to pin down for agents. It's not like traditional ML where you check accuracy against a test set. Your agent's outputs are free-form text, its reasoning involves multiple steps, and "correct" can mean different things depending on context. You can't manually review 2,000 tickets a day. And gut-checking a few examples isn't evaluation. It's wishful thinking.

**MLflow Evaluation** uses LLM judges: models that score your agent's outputs across dimensions like correctness, safety, relevance, and tool call accuracy. You define what "good" means for your use case, and MLflow measures it at scale.

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

MLflow ships with [70+ built-in judges](https://mlflow.org/docs/latest/genai/evaluation/judges.html) covering response quality, safety, groundedness, retrieval relevance, and more. For agents specifically, you can evaluate **tool call accuracy** (did it pick the right tool?), **reasoning quality** (did it follow a logical path?), and **task completion** (did it actually resolve the customer's issue?).

Because evaluation is integrated with tracing, you can also run scorers against production traces continuously, not just before deployment. When that model update quietly drops your refund success rate, you see it in your evaluation dashboard the same day, not a week later.

You can build custom scorers for your domain too:

```python
from mlflow.genai.scorers import scorer

@scorer
def refund_accuracy(request, response, trace):
    """Check if refund responses match actual refund status."""
    if "refund" in request.lower():
        actual_status = get_actual_refund_status(trace)
        claimed_status = extract_claimed_status(response)
        return actual_status == claimed_status
    return True  # Not a refund request
```

This turns agent quality from a gut feeling into something you can actually measure and track over time.

## Version Control: You Can't Improve What You Don't Version

Your agent isn't just a prompt. It's a system: a system prompt, tool definitions, retrieval configs, model parameters. When something changes and quality drops, you need to know what changed, compare it against the previous version, and roll back if necessary.

Most teams don't have this. Prompts live as hardcoded strings in application code. Changes are tracked (if at all) through git commits mixed in with unrelated logic. One engineer tweaks the refund instructions, another swaps the model from GPT-4o to GPT-4o-mini, and when the agent starts giving worse answers, nobody can pinpoint which change caused it.

**MLflow Prompt Registry** versions prompts as standalone artifacts, separate from application code. Every version has a complete diff history and is linked to evaluation results, so you can see exactly how each change affected quality.

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

Because prompts are versioned and evaluation is built in, you can also run **automated prompt optimization**. MLflow integrates with algorithms like [GEPA](https://github.com/gepa-ai/gepa) that generate improved prompt variants, evaluate them against your dataset, and select the best performer. Every optimization run is tracked, so comparing versions and rolling back is straightforward.

```python
from mlflow.genai import optimize_prompts
from mlflow.genai.optimize import GepaPromptOptimizer

result = optimize_prompts(
    target_llm_params={"model": "gpt-4o-mini"},
    prompts=["customer_support_system"],
    train_data=training_examples,
    scorers=[Correctness(), Safety()],
    optimizer=GepaPromptOptimizer(),
)
```

## Governance: Your Agent Has No Guardrails

Your customer support agent has API keys for OpenAI, Anthropic (as a fallback), and your internal order management system. Those keys are scattered across environment variables, config files, and `.env` files on developer laptops. One engineer hardcoded a key in a Jupyter notebook that got committed to git. The agent has no rate limits, so when a retry bug causes a loop, it burns through your API budget before anyone notices. And there's no centralized way to enforce content policies: the agent can return whatever the LLM generates, including customer PII that shouldn't be in a chat response.

**MLflow AI Gateway** is a centralized proxy between your agent and every LLM provider it talks to. It handles credential management, traffic routing, cost controls, and guardrails, all without changing your agent's code.

```python
from openai import OpenAI

# Point your agent at the gateway instead of OpenAI directly
client = OpenAI(
    base_url="https://your-mlflow-server/gateway/v1",
    api_key=""  # Gateway handles authentication
)

# Your agent code doesn't change. The gateway handles:
# - API key management (keys never touch application code)
# - Automatic failover (if OpenAI is down, route to Anthropic)
# - Usage tracking (token counts, costs per endpoint)
# - Rate limiting and budget controls
# - Content guardrails and PII redaction
```

The gateway exposes an OpenAI-compatible API, so switching providers is a config change, not a code change. A/B test GPT-4o against Claude by splitting traffic 90/10. Set budget alerts so a retry loop can't burn $12,000 overnight. Enforce PII redaction at the gateway level so sensitive data never reaches the response, regardless of what the LLM generates.

Because the gateway is integrated with MLflow Tracing, every request that passes through it automatically becomes a trace with full context: model used, tokens consumed, latency, and whether any guardrails were triggered.

## Why You Need All Four Together

You could stitch together separate tools for each of these. Use LangSmith for tracing, DeepEval for evaluation, a git repo for prompts, and LiteLLM for gateway routing. Some teams do.

But the integration tax adds up fast. Your evaluation framework can't access your traces, so you build a pipeline to export data between them. Your gateway doesn't know about your prompt versions, so you manually track which prompt is deployed where. Your tracing tool doesn't feed into your evaluation dashboard, so quality problems go unnoticed until a customer complains.

When all four capabilities live in one platform, they **work together**:

- **Traces feed evaluations**: run scorers directly against production traces to catch quality issues in real time
- **Evaluations validate prompts**: every prompt change is measured against your evaluation suite before deployment
- **The gateway generates traces**: every model interaction is automatically traced with full context, no separate instrumentation needed
- **Prompt versions link to evaluation results**: see exactly how each prompt version performed across every quality dimension

That's the difference between operating your agent with a single dashboard and operating it with four disconnected tools and a Slack channel full of "is this working?" messages.

## MLflow: The Open-Source AI Platform for Agents

MLflow is the only open-source platform that provides all four of these capabilities ([tracing](/llm-tracing), [evaluation](/llm-evaluation), [version control](/prompt-optimization), and [AI gateway](/ai-gateway)) in a single, integrated platform.

Three things make this worth paying attention to:

**It's open source and vendor-neutral.** Apache 2.0 licensed, backed by the Linux Foundation, with over 20,000 GitHub stars and 30 million monthly downloads. No per-trace fees, no per-evaluation charges, no lock-in. Your data stays yours. Traces are OpenTelemetry-compatible and can be exported to any backend.

**It works with whatever you're already using.** MLflow supports LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, Pydantic AI, Google ADK, and 20+ other agent frameworks. It works with 50+ LLM providers. You pick your tools; MLflow handles the operational side.

**It's already running in production everywhere.** Thousands of organizations use MLflow today. It's not a new project looking for users. It's the platform that millions of ML practitioners already rely on, extended with the capabilities that production agents need.

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
