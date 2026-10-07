---
title: "Evaluating and Improving Agent Skills with MLflow"
description: How to build measurable, testable, and continuously improving AI agent capabilities.
slug: evaluating-improving-agent-skills
authors: [dandresavid]
tags: [genai, evaluation, tracing, skills, agents, llm-judges]
thumbnail: /img/blog/evaluating-improving-agent-skills-thumbnail.png
image: /img/blog/evaluating-improving-agent-skills-thumbnail.png
date: 2026-08-02
---

How to build measurable, testable, and continuously improving AI agent capabilities.

Agents are becoming increasingly capable of solving complex tasks by combining reasoning, tools, memory, and structured workflows. Across frameworks such as LangGraph, OpenAI Agents SDK, Claude Code, CrewAI, Cortex, OpenClaw, and Hermes, a common pattern has emerged: agents are built from reusable skills. These skills encapsulate specific capabilities, such as retrieving information, generating SQL, validating invoices, planning multi-step tasks, or interacting with APIs, making agents easier to develop and maintain. But this modular approach introduces an important question: how do you know whether a skill is actually improving?

<!-- truncate -->

Many teams still evaluate skills manually by running a few prompts and checking whether the responses "look good," but this approach is difficult to reproduce, doesn't scale, and can miss subtle regressions. Instead, skills should be treated like software components: versioned, tested, measured, and continuously improved. In this blog, we'll explore how MLflow enables evaluation-driven development for agent skills using traces, datasets, custom evaluators, and experiment tracking.

## What Is an Agent Skill?

A [skill](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) is a reusable capability that an agent can invoke to complete part of a task, such as retrieving documents from a vector database, generating SQL queries, calling external APIs, validating receipts, planning execution steps, or reviewing generated code. Rather than embedding all instructions inside a monolithic system prompt, developers can create focused skills that evolve independently. For example, instead of building a customer support agent around one large prompt containing every policy, we might define a reusable refund skill.

```yaml
---
name: refund-evaluation
description: Use this skill to evaluate whether a customer is eligible for a refund according to the applicable refund policy.
---
## Responsibilities

- Validate purchase date
- Verify order status
- Apply refund policy
- Escalate edge cases
- Explain decisions clearly
```

This skill can now be reused across multiple agents while remaining independently testable.

## The Problem: Skills Drift Over Time

Skills rarely stay static, and as production feedback arrives, developers continuously modify them.

Version 1:

```yaml
---
name: refund-evaluation
description: Use this skill to evaluate whether a customer is eligible for a refund according to the applicable refund policy.
---
Validate receipts for refund.
```

Version 2:

```yaml
---
name: refund-evaluation
description: Use this skill to evaluate whether a customer is eligible for a refund according to the applicable refund policy.
---
- Ignore duplicate uploads
- Accept PDF receipts
- Reject blurry images
- Handle foreign currencies
```

The updated instructions appear better, but appearances can be misleading. Duplicate detection may have improved while blurry receipt detection became less reliable, or latency may have increased because the skill now performs additional reasoning. Without structured evaluation, these regressions often go unnoticed until users report them.

## Why Evaluating Final Answers Isn't Enough

Traditional LLM evaluation focuses on whether the final answer is correct, but for agent skills, the process also matters. A refund agent may give the correct refund while skipping identity verification, or a retrieval skill may return the right answer while making unnecessary tool calls that increase latency and cost.

Skill evaluation should measure behaviors, not just outputs. This is where traces become essential. Traces capture each step the agent takes, making it possible to verify that the skill selected the correct tools, invoked APIs in the expected order, followed business policies, generated an efficient execution plan, completed required validation steps, and cited retrieved evidence where appropriate.

These behavioral metrics provide much richer insight into skill quality than answer accuracy alone.

![Answer-only evaluation checks the final response, while behavioral evaluation inspects each step in the trace](./behavioral-metrics.png)

## Building an Evaluation Dataset

Every skill should have a dedicated evaluation dataset representing realistic scenarios.

For our refund skill:

| Customer Request        | Expected Outcome      |
| ----------------------- | --------------------- |
| Refund after 5 days     | Approve               |
| Refund after 45 days    | Reject                |
| Wrong product delivered | Escalate              |
| Digital purchase        | Follow digital policy |
| Missing receipt         | Request documentation |

Unlike benchmark datasets, evaluation datasets evolve with production, so whenever users discover failure cases, add them to the dataset to prevent future regressions.

![Evaluation datasets grow over time as new failure cases are added](./evaluation-dataset.png)

## Scorers

**1. Correctness: Output-based**

Compares the skill's final response against Expected Outcome (Using Expectations / Ground Truth). Doesn't look inside the trace just the answer.

```python
from mlflow.entities import Feedback
from mlflow.genai import scorer


@scorer
def correctness(outputs, expectations) -> Feedback:
    """Final decision matches the dataset's expected decision (ground truth)."""
    decision = outputs.get("decision")
    expected = expectations.get("expected_decision")
    return Feedback(value=decision == expected,
                    rationale=f"decision={decision} vs expected={expected}")
```

**2. Policy_compliance: rule-based (partly trace-aware)**

Checks whether the skill honored the business rule for that scenario. It reads the Expected Outcome to know which rule applies, and often the trace to confirm the rule was followed.

- Digital purchase → response must be FOLLOW_DIGITAL_POLICY
- Missing receipt → response must be REQUEST_DOCUMENTATION (don't refund blind)

**3. Correct_tool_selection: Trace-based**

Reads the execution trace to confirm the expected tools ran, in the right order, before the response. This scorer does not evaluate the final answer but rather the intermediate steps that agent went through. Because the trace records every span, the scorer can require that identity verification happened before the refund decision:

```python
from mlflow.entities import Feedback, Trace
from mlflow.genai import scorer


@scorer
def correct_tool_selection(trace: Trace) -> Feedback:
    """verify_identity must run before decide_and_respond."""
    verify = trace.search_spans(name="verify_identity")
    decide = trace.search_spans(name="decide_and_respond")

    ok = bool(verify and decide and verify[0].start_time_ns < decide[0].start_time_ns)
    return Feedback(value=ok, rationale="verified before the decision" if ok else "verification missing")
```

Every evaluation run now checks that the workflow was followed, not just that the answer was right. The same pattern extends to other behaviors: API sequencing, citation completeness, planning quality, cost efficiency, and retry behavior after failures.

## Running Skill Evaluations with MLflow

The predict_fn can wrap any agent implementation, regardless of the framework. Its job is simply to execute the skill for a single evaluation example and return the result in a structured format. For example, if your refund skill is implemented as a Python function:

```python
def run_agent(customer_message, order_id=None):
    result = refund_agent.run(
        message=customer_message,
        order_id=order_id,
    )
    return {
        "response": result.response,
        "decision": result.decision,
        "tool_calls": result.tool_calls,
    }
```

Once the dataset exists, evaluating a skill becomes straightforward, and instead of manually inspecting dozens of conversations, MLflow automatically computes evaluation metrics across the entire dataset, so improvements become measurable.

```python
import mlflow

results = mlflow.genai.evaluate(
    data=refund_dataset,
    predict_fn=run_agent,
    scorers=[
        correctness,
        policy_compliance,
        correct_tool_selection
    ]
)
```

## Using Traces to Understand Failures

Evaluation tells you that something failed, while tracing tells you why.

![Evaluation report showing Correct Tool Selection at 43% before the fix](./eval-report-before.png)

Opening the trace reveals:

User Request → Search Orders → Generate Response

Notice anything missing? The workflow skipped Verify Customer Identity, meaning the skill instructions never explicitly required identity verification before accessing order history. After updating the skill to require identity verification before accessing order data, the trace correctly includes the missing verification step.

Re-running evaluation yields:

A small change to the skill instructions resulted in a substantial improvement in agent behavior, with the Correct Tool Selection score increasing from 43% to 98%.

![Evaluation report showing Correct Tool Selection improving from 43% to 98% after the fix](./eval-report-after.png)

## Why MLflow for Skill Evaluation

Evaluating skills requires more than a single benchmark metric. It demands an end-to-end workflow that captures execution, measures behavior, tracks experiments, and supports continuous improvement.

MLflow brings these capabilities together in one platform:

- Tracing captures how a skill executes.
- Evaluation measures outputs and behaviors using built-in and custom scorers.
- Experiment Tracking records every run for reproducibility and comparison.
- Datasets enable regression testing with representative scenarios.

Together, these capabilities enable an evaluation-driven development process where every skill change is measurable, reproducible, and backed by data. As reusable skills become fundamental building blocks of modern AI agents, systematically evaluating and improving them becomes increasingly important. Rather than asking whether an agent simply "seems to work," teams can identify which skills fail, understand why, compare versions, detect regressions, and validate improvements with objective data. By combining tracing, datasets, custom evaluators, and experiment tracking, MLflow helps teams treat skills as measurable, versioned, and continuously improving components, resulting in more reliable, maintainable, and trustworthy AI agents.

If this is useful, give us a ⭐ on [GitHub](https://github.com/mlflow/mlflow).

### Related reading

- [Testing and Refining Claude Code Skills with MLflow](/blog/evaluating-skills-mlflow)
- [Ship LLM Agents Faster with Coding Assistants and MLflow Skills](/blog/self-improving-agent-loop)
- [Structuring AI Evaluation and Observability with MLflow: From Development to Production](/blog/structured-ai-eval)
