---
title: "Make LLM Red Teaming Run on Every Release for Engineers"
description: "A practical, engineering-focused guide to plan, run, and operationalize LLM red teaming into continuous evaluation pipelines with MLflow and automated..."
slug: red-teaming-llm
tags:
  [
    safety evaluation llm,
    red teaming llm,
    red team vs blue team,
    penetration testing LLM,
    LLM security testing,
    AI red teaming,
    red teaming tools,
    red teaming framework,
    LLM threat assessment,
    red team methodology,
    red team prompts,
    red team exercises,
    llm content moderation,
    conducting red teaming,
    adversarial testing llm,
  ]
date: 2026-09-08
image: https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1788856743332_Engineer-conducting-adversarial-AI-testing.jpeg
---

![Engineer conducting adversarial AI testing](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1788856743332_Engineer-conducting-adversarial-AI-testing.jpeg)

LLM red teaming is a controlled adversarial testing practice that uncovers safety, privacy, and jailbreak risks before your model reaches production. The first move isn't buying a tool or writing an automation script. It's running a short manual red-team pass to map your system's actual risk surface, a sequence that both [Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/red-teaming) and [NIST](https://csrc.nist.gov/glossary/term/red_team_exercise) back, and one MLflow can help you operationalize once you have findings worth tracking.

---

> **TL;DR:**
>
> - Manual red teaming should be the first step to map the actual risk surface before automation or scripting begins.
> - Testing efforts must distinguish between model vulnerabilities like unsafe content and application-level risks such as data leakage and tool misuse.
> - A clear scope, defined rules, and a small, well-roles team are essential to avoid legal issues and noise during red-team engagements.
> - Combining manual creativity with automated attack tools improves coverage and identifies varied attack vectors, especially when using structured metrics.
> - Logged adversarial findings need to be systematically stored and integrated into continuous evaluation pipelines to track progress and ensure safety over time.

---

## Table of Contents

- [Why LLM Red Teaming Matters More Than Standard QA](#why-llm-red-teaming-matters-more-than-standard-qa)
- [Where LLMs Actually Break: Model vs. Application Layer](#where-llms-actually-break-model-vs-application-layer)
- [Planning Your Red-Team Engagement Before You Start Testing](#planning-your-red-team-engagement-before-you-start-testing)
- [Choosing Your Testing Method: Manual, Automated, or Both](#choosing-your-testing-method-manual-automated-or-both)
- [Building Repeatable Test Scenarios and Measuring What Happened](#building-repeatable-test-scenarios-and-measuring-what-happened)
- [Frameworks and Tools That Do the Heavy Lifting](#frameworks-and-tools-that-do-the-heavy-lifting)
- [Turning Red-Team Findings Into a Working Evaluation Pipeline](#turning-red-team-findings-into-a-working-evaluation-pipeline)
- [Prioritizing Fixes and Setting a Sustainable Testing Cadence](#prioritizing-fixes-and-setting-a-sustainable-testing-cadence)
- [Author Perspective: Resource Allocation and When to Bring in Outside Help](#author-perspective-resource-allocation-and-when-to-bring-in-outside-help)
- [Sources](#sources)

## Why LLM Red Teaming Matters More Than Standard QA

Traditional QA checks whether your model does what it's supposed to do. Red teaming checks what happens when someone tries to make it do something else entirely. That distinction shapes everything about how you should approach it.

Penetration testing looks for known vulnerability classes in code and infrastructure. Red teaming is broader: NIST describes a red team exercise as a simulated adversarial attempt to assess an organization's security capability, not just its code's flaw count. For LLMs, that means testing behaviors nobody explicitly coded, because generative models produce emergent responses that static unit tests can't anticipate.

Here's where red teaming earns its place in your pipeline:

- It surfaces harms that only appear under adversarial pressure, like a model that refuses a direct harmful request but complies when the same request is wrapped in a fictional scenario.
- It tests detection and response, not just prevention. Does your logging catch the attempt? Does your monitoring flag it?
- It generates the raw material for automated regression tests, since Microsoft Foundry's guidance treats manual findings as the seed data for later systematic measurement.

Skip this step and you're measuring your model against a checklist you wrote yourself. That checklist has blind spots by definition.

## Where LLMs Actually Break: Model vs. Application Layer

Vulnerabilities split cleanly into two layers, and conflating them wastes testing effort. The model layer concerns the LLM's own behavior in isolation. The application layer concerns everything you built around it, retrieval systems, tool integrations, memory stores, that create new attack paths the base model never had.

1. **Unsafe content generation.** The model produces disallowed material when prompted with the right framing, roleplay, or incremental escalation.
2. **Policy evasion through obfuscation.** Encoding requests in code, foreign languages, or fictional narratives to slip past content filters.
3. **Hallucination-driven misinformation.** The model states false facts with confident, authoritative phrasing, a risk that compounds when outputs feed downstream decisions.
4. **Retrieval-augmented generation (RAG) data leakage.** Poisoned or overly permissive retrieval indexes expose private documents through crafted queries.
5. **Agent tool misuse.** An agent with API or file-system access gets manipulated into calling tools in unintended sequences.
6. **Exfiltration via tool calls.** Attackers use a legitimate tool integration (email, search, code execution) as a covert channel to move data out.
7. **Prompt injection.** Malicious instructions embedded in retrieved content or user input override the system prompt's intended behavior.
8. **Jailbreaks.** Multi-turn or single-turn techniques that erode a model's safety training through persistence or clever framing.

The [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) catalogs most of these patterns with more granularity, and it's worth treating as your baseline checklist before you invent your own taxonomy from scratch.

## Planning Your Red-Team Engagement Before You Start Testing

A red-team run without defined boundaries produces noise, or worse, a legal problem. Before anyone types an adversarial prompt, nail down scope and authorization.

Start by defining exactly what's in bounds: the model API directly, the production UI, the retrieval index, the agent's toolset, or some combination. NIST's guidance on red team exercises recommends aligning scope with your actual organizational learning goals rather than testing everything indiscriminately, because a test designed to find jailbreaks won't surface the same issues as one designed to find data leakage.

Assemble a small team with clear roles:

- Adversarial testers who generate and execute attack attempts.
- A product owner who understands what "success" and "failure" mean for the system under test.
- A small deconfliction group aware the test is happening, so nobody mistakes a red-team probe for a real incident.

Rules of engagement need to cover authorization in writing, boundaries on what data testers can access, and a safe-handling protocol for any genuinely harmful output the team generates, storage, redaction, and who gets to see it.

**Pro Tip:** _Kroll's guidance on red-team readiness notes that organizations without basic logging and monitoring maturity get less value from full-scope engagements. If you don't have solid observability yet, run a narrower, cheaper test first and build monitoring before scaling up._

## Choosing Your Testing Method: Manual, Automated, or Both

Manual and automated testing produce fundamentally different kinds of evidence, and most mature programs eventually run both in sequence rather than picking one.

Manual testing means a human creatively probes the system, chaining ideas, improvising on partial successes, and following intuition down paths a script wouldn't try. This is where you find the genuinely novel jailbreak, the one nobody anticipated because it depends on cultural context or timing a rule set can't encode. Microsoft Foundry's planning guidance treats this creative mapping phase as the necessary precursor to anything systematic.

Automated testing trades some of that creativity for scale. Research on automated red-teaming agents, including the GPT-Red approach trained through self-play, shows automated attackers can discover more working attacks than human testers on certain benchmarks simply by running thousands of variations humans wouldn't have patience for. The tradeoff: automated agents can overfit to narrow failure modes if their training environment isn't designed carefully.

Your adversary model matters just as much as your method:

- **White-box** testers see model weights, system prompts, and architecture, producing the most thorough but least realistic findings.
- **Black-box** testers only see what an external attacker would see, closest to real-world risk but slower to find deep issues.
- **Gray-box** testers get partial visibility, a common and practical middle ground for internal teams.

## Building Repeatable Test Scenarios and Measuring What Happened

A red-team finding that can't be reproduced is a rumor, not a data point. Structure matters as much as creativity here.

1. **Build a harms catalog first.** Define categories (unsafe content, data leakage, policy evasion) with concrete examples, so every finding gets classified consistently instead of described in ad hoc prose.
2. **Write scenario templates.** Each template should specify the adversary goal, the starting prompt or conversation state, and the success condition, so a tester six months from now can rerun it exactly.
3. **Log the full interaction, not just the outcome.** Capture the complete conversation, model version, temperature, and any system prompt in effect at test time.
4. **Score severity and reproducibility separately.** A finding that works once out of twenty attempts is a different risk than one that works reliably every time.

Three metrics do most of the heavy lifting in any red-team report: attack success rate (the percentage of attempts that achieve the adversary's goal), severity banding (how bad is the outcome if it reaches a real user), and reproducibility rate (does the same input reliably trigger the same failure). Automated red-teaming research, including work on [scaled adversarial-prompt generation](https://arxiv.org/html/2512.20677v1), depends entirely on this kind of structured, comparable data to be useful at all. Without it, you're comparing findings that were never measured the same way.

## Frameworks and Tools That Do the Heavy Lifting

You don't need to build your attack-generation pipeline from raw string manipulation. Several open-source and research-driven options already handle the tedious parts.

- **DeepTeam** is an open-source framework built specifically to automate LLM red teaming: it generates adversarial attacks, runs them against your target system, and manages the test harness so you're not scripting HTTP calls and retry logic by hand.
- **GPT-Red-style automated agents** represent the research frontier, systems trained through self-play to generate genuinely novel attacks rather than replaying a fixed library. The [underlying paper](https://arxiv.org/html/2607.26115v1) shows these agents complement human red-teamers well, though the paper's authors note humans still catch scenario types the trained agent misses.
- **Academic automation frameworks** for [generating and evaluating adversarial prompts at scale](https://arxiv.org/html/2512.20677v1) offer architectural patterns worth studying even if you don't adopt the exact codebase.

Decision rule of thumb: adopt a framework like DeepTeam when you need breadth across many attack categories fast, and build custom scripts when you're targeting a narrow, unusual attack surface specific to your application, like a proprietary tool-calling sequence no off-the-shelf framework anticipates. Teams building internal engineering documentation around this process may also find general [developer onboarding resources](https://ammarai-creative-hub.lovable.app/ai-for-developers) useful for getting new team members up to speed on the tooling quickly.

## Turning Red-Team Findings Into a Working Evaluation Pipeline

A red-team report that lives in a slide deck dies in a slide deck. The findings only compound in value once they're wired into something your team runs repeatedly, and that's where [MLflow's evaluation and tracing capabilities](https://mlflow.org/genai/evaluations) earn their place in the workflow.

![Adversarial findings entering evaluation pipeline](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1788856744534_Adversarial-findings-entering-evaluation-pipeline.jpeg)

Every adversarial example your manual pass or automated framework surfaces should get logged as an artifact, complete with the prompt, the model's response, the severity label, and whether it reproduced. Tracing can capture this kind of agentic reasoning trail in detail, which matters when an attack succeeded through a multi-step tool call rather than a single bad response. That same [observability layer](https://mlflow.org/genai/observability) lets you replay exactly what the model saw and did at each step, instead of reconstructing it from memory after the fact.

Once you've accumulated enough labeled adversarial examples, they become your regression suite. MLflow's LLM-as-a-Judge framework can score new model versions against that suite automatically, converting a one-time red-team engagement into a continuous check that runs every time you change a prompt, swap a model, or update your retrieval index.

- Store adversarial examples and their severity labels as versioned artifacts, not scattered spreadsheets.
- Run automated evaluations against your harms catalog every time you ship a prompt or model change.
- Use an AI Gateway to standardize prompt versions across environments, so the version that passed red-team review is provably the version running in production.

**Pro Tip:** _Teams that skip prompt version control often discover, after an incident, that the "fixed" prompt in staging was never actually the one deployed. [Standardized prompt management](https://mlflow.org/genai) closes that gap and gives you an audit trail for free._

## Prioritizing Fixes and Setting a Sustainable Testing Cadence

Not every finding deserves an emergency patch. Rank them on three axes: how severe the outcome is if it reaches a real user, how easy the exploit is to reproduce, and how much business exposure it creates if it leaks publicly. A jailbreak that requires eleven specific conversation turns and produces mildly inappropriate text sits in a different bucket than a one-shot prompt that leaks another customer's data.

Common mitigation patterns map fairly predictably to the vulnerability class:

- Prompt-level filters and classifiers for unsafe content categories.
- Retrieval sanitization to strip untrusted instructions from documents before they reach the context window.
- API rate limits and restricted tool permissions to cap what an agent can do even if manipulated.
- Role-based access control (RBAC) so a compromised session can't reach data or systems beyond its intended scope.

| Program stage | Primary activity                        | Typical output                       |
| ------------- | --------------------------------------- | ------------------------------------ |
| Discovery     | Manual creative probing                 | Harms catalog, novel attack examples |
| Measurement   | Structured scoring of known attacks     | Severity bands, success rates        |
| Regression    | Automated replay against fixed test set | Pass/fail tracking per model version |
| Monitoring    | Continuous production observability     | Drift alerts, new-pattern detection  |

That progression, from manual discovery to measurement to automated regression to continuous monitoring, mirrors what Microsoft Foundry recommends and gives you a natural cadence: deep manual passes regularly throughout the year, automated regression on every release, and lightweight monitoring running always.

## Author Perspective: Resource Allocation and When to Bring in Outside Help

Most teams over-invest in one long, exhaustive red-team engagement and under-invest in the boring follow-up work. A single two-week deep-dive generates a great report and then nothing happens for six months because nobody built the regression harness to keep testing what was found. I'd rather see a team run a shorter initial pass and spend the saved budget on making findings replayable.

![Author Perspective: Resource Allocation and When to Bring in Outside Help — overview diagram](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1788856806596_Author-Perspective-Resource-Allocation-and-When-to-Bring-in-Outside-Help-overview-diagram.jpeg)

Building internal red-team capability pays off if you're shipping model or prompt changes weekly. If you ship quarterly, contracting specialists for a periodic deep engagement probably beats maintaining a dedicated internal team that goes idle between cycles.

The tradeoff nobody talks about enough: test realism versus safe failure modes. The most realistic adversarial test is one your team doesn't fully control, but that's also the one most likely to produce a genuinely harmful output you now have to handle responsibly. Decide your risk tolerance for that tradeoff before you start, not mid-engagement.

> _— Kevin_

## Sources

- [Planning red teaming for large language models (LLMs) and their applications - Microsoft Foundry | Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/red-teaming)
- [NIST glossary: red team exercise](https://csrc.nist.gov/glossary/term/red_team_exercise)
- [OWASP Top 10 for large language model applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [GPT-Red (automated red-teaming agent) — research paper (arXiv)](https://arxiv.org/html/2607.26115v1)

## Recommended

- [Automatically find the bad LLM responses in your LLM Evals with Cleanlab](https://mlflow.org/blog/tlm-tracing)
- [LLM Application Architecture: A 2026 Engineer's Guide](https://mlflow.org/articles/llm-application-architecture-a-2026-engineers-guide)
