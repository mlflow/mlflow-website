---
title: "SOC 2 for LLM Apps: 9 Artifacts Auditors Sample First"
description: "Practical SOC 2 playbook for LLM apps. See the nine audit grade artifacts auditors sample, plus Type II timing, model registries, redaction, and DPAs."
slug: soc-2-for-llm-apps
tags:
  [
    SOC 2 compliance for AI,
    soc 2 for llm apps,
    LLM application security,
    SOC 2 requirements for software,
    LLM app compliance standards,
    how to achieve SOC 2 certification,
    best practices for SOC 2,
  ]
date: 2026-08-28
image: https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1787895660856_Hands-arranging-audit-materials-on-tech-desk.jpeg
---

![Hands arranging audit materials on tech desk](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1787895660856_Hands-arranging-audit-materials-on-tech-desk.jpeg)

SOC 2 applies to LLM applications, and auditors are not waiting for a special "AI attestation" to catch up before they start testing. They map existing Trust Services Criteria onto AI-specific risks and want AI-specific evidence: a model registry with version history, redacted prompt and inference logs, signed DPAs with your LLM vendors, drift monitoring, and change-control artifacts. If you're starting from zero, scope the AI features with your auditor now and start collecting six months of operating evidence toward a Type II report.

---

> **TL;DR:**
>
> - Building a model registry with version history, per-call logging, and tamper-evident retention is essential for SOC 2 compliance in AI systems.
> - Auditors will expect evidence such as drift monitoring, vendor risk assessments, and signed data processing agreements with third-party providers.
> - Focus on controlling prompt injection, data leakage, and model outage risks, with proof such as redaction logs, SLA documents, and rollback tests.
> - Prepare for a multi-month observation window, with regular evidence collection including daily logs and monthly risk reviews.
> - Prioritize simple, high-impact controls like model approval gates, prompt redaction, and drift alerts to avoid costly fixes during audits.

---

## Table of Contents

- [How SOC 2 Maps to LLM Risks](#how-soc-2-maps-to-llm-risks)
- [The Evidence Checklist Auditors Actually Sample](#the-evidence-checklist-auditors-actually-sample)
- [Scoping the Audit: Timeline and Sampling](#scoping-the-audit-timeline-and-sampling)
- [Mapping Controls to Criteria with Real Artifacts](#mapping-controls-to-criteria-with-real-artifacts)
- [An Implementation Playbook You Can Actually Run](#an-implementation-playbook-you-can-actually-run)
- [How Observability Tooling Produces Audit-Grade Artifacts](#how-observability-tooling-produces-audit-grade-artifacts)
- [Author's Perspective: What Actually Gets Teams in Trouble](#authors-perspective-what-actually-gets-teams-in-trouble)
- [Sources](#sources)

## How SOC 2 Maps to LLM Risks

SOC 2 doesn't get a new set of criteria for generative AI. Auditors take the five Trust Services Criteria and stretch them over failure modes that didn't exist in traditional software audits.

**Security** covers prompt injection and excessive agent permissions, the two vulnerabilities the [OWASP GenAI LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) ranks highest for generative systems. **Confidentiality** covers data leakage through model outputs, including cases where a model regurgitates training data or a user's prompt gets logged in plaintext. **Availability** now has to account for third-party LLM provider outages, not just your own infrastructure. **Privacy** governs how personal data flows into prompts and whether it ends up in a vendor's training pipeline.

**Processing Integrity** is where LLM audits get genuinely hard. Traditional software either processes a transaction correctly or it doesn't. An LLM can produce a plausible, well-formatted, completely wrong answer, and there's no single "correct output" to check against. Auditors handle this by asking you to define acceptable output boundaries in advance: confidence thresholds, guardrail rules, human-review triggers for high-stakes outputs, and documented rates of hallucination or refusal against a golden dataset.

What auditors will accept as evidence, by criterion:

- **Security:** access logs, secrets rotation records, penetration test results scoped to prompt injection
- **Processing Integrity:** regression test results, golden dataset scoring, documented output boundaries
- **Confidentiality:** redaction logs, data classification policy, DLP scan output
- **Availability:** provider SLA documentation, failover test results
- **Privacy:** training opt-out clauses, data subject request handling for AI-processed data

The OWASP Top 10 for Large Language Model Applications is worth keeping open while you build this map, since it names the threats in the same vocabulary auditors are starting to use.

## The Evidence Checklist Auditors Actually Sample

Auditors reviewing AI systems for SOC 2 have converged on a fairly consistent list of things they ask for first: model lineage, redacted prompt and inference logs, drift monitoring output, and vendor risk assessments, according to [Soc2auditors](https://soc2auditors.org/insights/soc-2-for-ai-companies/). Build toward this checklist rather than guessing at what fieldwork will look like.

1. **Model registry with immutable history.** Every model version needs a tag, a timestamp, an approver, and a deployment record. If you can't reproduce which model version served a given request six weeks ago, that's a finding.
2. **Per-call logging.** Model version, timestamp, caller identity, prompt hash (or redacted prompt text), output or a hashed reference to it, and token counts. This is the single most-requested artifact in AI-inclusive SOC 2 engagements, and field reports on 2026 audits list [per-call logging alongside immutable registries and rollback proof](https://www.knowlee.ai/blog/soc-2-type-2-for-ai-companies-2026) as the top items examiners sample.
3. **Tamper-evident retention.** Append-only logs, WORM storage, or cryptographic hashes proving logs weren't altered after the fact. A retention policy without tamper-evidence controls is a paper policy.
4. **Vendor DPAs with training opt-out or zero-data-retention clauses.** If you route prompts to a third-party model provider, you need a signed agreement stating your data isn't used to train their models, plus their own SOC 2 or ISO 27001 report.
5. **Drift detection and regression testing.** Scheduled comparisons against a baseline dataset, with alerting when output quality degrades. Auditors want to see the alert history, not just the dashboard.
6. **Bias and fairness testing reports.** Especially for any LLM feature touching hiring, lending, healthcare, or other regulated decisions.
7. **Rollback proof.** Evidence that you can revert to a prior model version and that you've actually tested the rollback, not just documented it.
8. **Secrets management for provider API keys.** Rotation schedules, access scoping, and audit trails showing who pulled a key and when.
9. **Tool-level permissioning for agents.** If an LLM agent calls external tools or APIs, each tool needs independent authorization so the agent can't exceed its intended privileges during autonomous execution, a control [Trend Micro's OWASP commentary](https://owasp.org/www-project-top-10-for-large-language-model-applications/) flags as a common gap in agentic deployments.

**Pro Tip:** _Don't wait for the auditor to ask for training-data provenance. Poisoned or mislabeled training data is a documented attack surface, and [CyLab research on dataset poisoning](https://www.cylab.cmu.edu/news/2025/06/11-poisoned-datasets-put-ai-models-at-risk-for-attack.html) is exactly the kind of citation an auditor will expect you to have already considered in your risk assessment._

The [OWASP LLMSVS v2.0 standard](https://owasp.org/www-project-llm-verification-standard/LLMSVS-v2.0-en.html) gives this checklist a leveled structure. Level 2 assurance is the reasonable target for most production LLM apps handling sensitive data, and mapping your controls to its numbered requirements (V1 through V8) gives auditors a framework they can cross-reference instead of taking your word for completeness.

## Scoping the Audit: Timeline and Sampling

Start with a scope workshop, not a control build. Sit down with your auditor and decide explicitly which LLM features are in scope (customer-facing chat, internal copilots, agentic workflows) and which are excluded (a one-off internal prototype, for instance). Getting this wrong in either direction wastes months.

Plan for a Type II observation window spanning several months for any system with LLM features. Type I only proves controls exist on a given day; it says nothing about whether prompt logging or drift monitoring actually ran continuously. Because [evidence collection timing determines finding strength](https://devopsvibe.io/en/blog/soc2-ai-controls-addendum), a control implemented three months before the audit window closes produces a materially weaker result than one running for the full period.

Evidence collection cadence matters more than most teams expect:

- **Daily:** inference logs, secrets access logs
- **Weekly:** drift monitoring reports, model performance dashboards review
- **Monthly:** vendor risk reassessment, access reviews, regression test runs
- **Per deployment:** model registry update, approval sign-off, rollback test

Auditors typically sample a subset of logs and change records rather than reviewing every entry. Expect sampling of multiple records per control area from different weeks to check consistency, not just volume.

Run a gap assessment before you commit to an observation start date. Budget **8 to 16 weeks** for remediation between the gap assessment and the day the clock starts on your Type II window. Rushing into observation with half-built controls guarantees exceptions in the final report.

## Mapping Controls to Criteria with Real Artifacts

Abstract control descriptions don't survive fieldwork. Auditors want the actual file, export, or log line. Here's how specific artifacts map to specific criteria.

| Trust Services Criterion | Artifact auditors sample                                                                        | What it proves                                                    |
| ------------------------ | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Security                 | Secrets rotation logs, IAM access reviews, MFA enforcement records for model-promote operations | Only authorized people can push a new model to production         |
| Processing Integrity     | Model lineage exports, regression test CSVs, golden dataset scoring results                     | Output quality is measured and tracked release over release       |
| Confidentiality          | Prompt redaction logs, data retention policy, DLP scan output                                   | Sensitive data doesn't leak into logs or model context unredacted |
| Availability             | LLM provider SLA documents, multi-provider failover configuration, failover test results        | The app survives a single provider's outage                       |

Each row corresponds to a fieldwork test an auditor actually runs, not a policy statement they read. A model lineage export, for example, needs to show the version that served a real production request, tied back to a training or fine-tuning event, with a timestamp an auditor can independently verify against your deployment logs. [MLflow's model registry](https://mlflow.org/articles/tags/ai-model-governance) is designed to produce exactly this kind of exportable lineage record, and regression results scored against a [golden dataset](https://mlflow.org/articles/tags/understanding-llm-benchmarks) turn a vague "we test our models" claim into a file an auditor can open.

Multi-provider failover deserves particular attention. If your LLM app depends on a single vendor's API and that vendor has an outage, your availability commitment breaks with it. Auditors increasingly ask for evidence you've tested failover to a secondary provider, not just configured one.

![Hands swapping connectors for LLM failover test](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1787895656858_Hands-swapping-connectors-for-LLM-failover-test.jpeg)

## An Implementation Playbook You Can Actually Run

Treat this as a three-phase project, each with a defined output, not an open-ended compliance initiative.

1. **Phase 1: Assessment (weeks 1 to 4).** Deliverables: a scoping memo naming which LLM features are in audit scope, a prioritized control backlog ranked by risk, and a risk register that explicitly includes model poisoning, prompt injection, and vendor concentration risk.
2. **Phase 2: Build (weeks 5 to 16).** Deliverables: a working model registry, prompt-scanning middleware that redacts sensitive fields before logging, drift dashboards with alerting thresholds, and signed DPAs from every LLM vendor in scope.
3. **Phase 3: Operate and observe (months 4 to 12+).** Deliverables: logs retained in WORM or equivalent tamper-evident storage, a full change-ticket history for every model promotion, incident response runbooks specific to LLM failures, and at least one completed tabletop exercise with documented results.

Assign owners by role, not by department name. Engineering owns the registry and logging pipeline. Security owns secrets management and access reviews. Legal owns vendor DPAs and data processing terms. Product owns defining acceptable output boundaries for Processing Integrity, since engineering alone shouldn't decide what counts as an acceptable hallucination rate for a customer-facing feature.

**Pro Tip:** _Keep the toolchain minimal on purpose. A model registry, a secrets manager, and an observability stack cover the majority of the checklist above. Adding five overlapping tools rarely produces better evidence; it produces five places an auditor now has to check for consistency._

![An Implementation Playbook You Can Actually Run — overview diagram](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1787895714528_An-Implementation-Playbook-You-Can-Actually-Run-overview-diagram.jpeg)

## How Observability Tooling Produces Audit-Grade Artifacts

The gap between "we have a policy" and "here is the file" is where most SOC 2 findings happen, and it's exactly the gap that a proper model registry and evaluation harness close.

A [model registry](https://mlflow.org/genai) that enforces version tags and approval gates produces the lineage export an auditor asks for in Phase 1 of fieldwork, without anyone manually reconstructing deployment history from Slack messages. [MLflow](https://mlflow.org) generates this kind of versioned record as a byproduct of normal deployment, not as a separate compliance exercise.

- LLM-as-a-Judge [evaluation runs](https://mlflow.org/llm-as-a-judge) create a dated, repeatable log of output quality scoring, which is the concrete evidence Processing Integrity testing asks for.
- An [AI gateway or prompt manager](https://mlflow.org/cookbook/prompt-engineering) that enforces redaction rules and retention TTLs at the point of logging turns a written policy into an automatically enforced control.
- Vendor routing records from a gateway double as the audit trail showing which provider handled which request, supporting your vendor-risk evidence.

## Author's Perspective: What Actually Gets Teams in Trouble

The highest-leverage fixes are the cheapest ones: a model registry, prompt redaction at the logging layer, signed DPAs, and drift alerts that actually page someone. Teams that skip these four spend far more fixing findings later than they would have spent building them first.

The failures I see repeatedly aren't exotic. Teams scope the audit too late and discover mid-observation that a feature they forgot was in scope. Teams log raw prompts with customer PII intact because redaction was an afterthought. Teams can point to a rollback procedure document but have never actually run one. None of these require sophisticated tooling to fix, just sequencing. Talk to your auditor before you build controls, not after, so you're not building evidence for a scope they were never going to accept.

> _— Kevin_

## Sources

- [OWASP Large Language Model Security Verification Standard (LLMSVS) v2.0](https://owasp.org/www-project-llm-verification-standard/LLMSVS-v2.0-en.html)
- [Soc2auditors](https://soc2auditors.org/insights/soc-2-for-ai-companies/)

## Recommended

- [LLM Application Architecture: A 2026 Engineer's Guide](https://mlflow.org/articles/llm-application-architecture-a-2026-engineers-guide)
