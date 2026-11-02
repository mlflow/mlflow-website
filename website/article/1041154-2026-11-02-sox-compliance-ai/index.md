---
title: "SOX Compliance and AI: Controls, Evidence and Audit Readiness"
description: "How SOX applies to AI in financial reporting: what auditors ask for, the risks to control, and how tracing and evaluation produce audit-ready evidence."
slug: sox-compliance-ai
tags: [sox compliance ai]
date: 2026-11-02
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1793419735851_Finance-analyst-reviews-AI-audit-evidence.jpeg
---

![Finance analyst reviews AI audit evidence](https://media.babylovegrowth.ai/blog-images/organization-30814/1793419735851_Finance-analyst-reviews-AI-audit-evidence.jpeg)

SOX does not have a separate rulebook for AI. If an AI system touches journal entries, estimates, reconciliations or close processes, it becomes part of your internal control over financial reporting (ICFR), and you need to show how it works, who reviews it and what it did on a given day. This guide explains what that means in practice and how engineering teams can produce the evidence finance and audit will ask for.

## TL;DR

- AI used in financial reporting falls under existing ICFR scoping. Treat its failure modes (wrong outputs, drift, over-reliance) as risks of material misstatement.
- Start with an inventory of every AI use in finance, then update risk and control matrices, narratives and flowcharts.
- Auditors will look for reproducibility: logged inputs, prompts, model versions and reviewer identity.
- Management stays responsible for the financial statements. AI supports people and does not replace them.
- Tracing, automated evaluation and a governed gateway give engineers a practical way to generate that evidence.

## Table of Contents

- [What SOX means once AI is in the process](#what-sox-means-once-ai-is-in-the-process)
- [Where AI shows up in SOX processes](#where-ai-shows-up-in-sox-processes)
- [The risks your controls need to address](#the-risks-your-controls-need-to-address)
- [What to document and test](#what-to-document-and-test)
- [Governance that holds up in an audit](#governance-that-holds-up-in-an-audit)
- [Generating audit evidence from your AI stack](#generating-audit-evidence-from-your-ai-stack)
- [Put the evidence layer in place](#put-the-evidence-layer-in-place)
- [Can AI replace management or the auditor?](#can-ai-replace-management-or-the-auditor)
- [A 30-day starting plan](#a-30-day-starting-plan)
- [FAQ](#faq)
- [Sources](#sources)

## What SOX means once AI is in the process

SOX Section 404 puts the burden on management. Under the SEC's interpretive guidance, management must [maintain ICFR, assess its effectiveness every year and keep the evidential matter that supports the assessment](https://www.sec.gov/files/rules/interp/2007/33-8810.pdf). The auditor, in turn, must obtain sufficient appropriate evidence to support an opinion on ICFR, as set out in [PCAOB AS 2201](https://pcaobus.org/oversight/standards/auditing-standards/details/AS2201).

Nothing in that structure changes when a model enters the workflow. What changes is the kind of evidence. A spreadsheet formula is deterministic and can be inspected. A language model call depends on a prompt, a model version, retrieved context and sampling behavior, and any of those can change without anyone editing a ledger.

Regulators and auditors are already paying attention. In June 2024, an SEC Investor Advisory Committee panel presentation advised auditors to [discuss the company's AI strategy and policies, including GenAI, and consider the impact on financial reporting and underlying controls](https://www.sec.gov/files/sec-iac-panel-ai-final-materials-brian-croteau.pdf). Deloitte frames the same point from the company side: [GenAI can streamline risk assessment and control testing, but it needs rigorous professional oversight](https://www.deloitte.com/us/en/services/audit-assurance/blogs/accounting-finance/ai-finance-reporting-automation-public-companies.html).

![What SOX means once AI is in the process](https://media.babylovegrowth.ai/blog-images/organization-30814/1793419781339_What-SOX-means-once-AI-is-in-the-process.jpeg)

## Where AI shows up in SOX processes

There are two different situations, and they carry different obligations.

**AI inside the controlled process.** A model drafts journal entry descriptions, proposes accrual estimates, matches reconciliation items or classifies transactions. Its output can affect the financial statements, so it is in scope. [COSO notes that organizations already use AI-enabled tools to automate reconciliations, accelerate analysis and support decisions](https://www.coso.org/generative-ai). PwC recommends that public companies [clearly flag AI used in journal entry automation, estimates and close processes](https://www.pwc.com/us/en/tech-effect/ai-analytics/trust-ai-audits.html) within their SOX programs.

**AI used to run the SOX program itself.** Agents that collect evidence, monitor controls or help test them. KPMG describes how [agentic AI can collect and organize evidence, monitor controls continuously and perform testing more consistently](https://kpmg.com/kpmg-us/content/dam/kpmg/pdf/2025/1a-thought-leadership-v3.pdf). That is useful, but the testing output still has to be reviewable by a person and defensible to the external auditor.

Both cases need the same discipline: you must be able to explain what the AI did and prove it.

## The risks your controls need to address

EY advises companies to [translate AI-specific characteristics, such as inaccurate outputs, limited explainability and over-reliance, into risks of material misstatement](https://www.ey.com/en_us/insights/assurance/ai-audit-readiness-for-financial-reporting) so they fit into existing ICFR scoping rather than sitting in a separate AI risk register. That is a good working rule.

COSO's guidance on internal control over generative AI lists the failure modes engineers should recognize: [hallucination, bias, prompt injection, sensitive data exposure, drift, inconsistent results, and authorization or excessive-agency problems](https://auditoresinternos.es/wp-content/uploads/2026/02/Achieving-Effective-Internal-Control-Over-Generative-AI_COSO_compressed.pdf). The SOX translation looks like this:

| AI failure mode                   | SOX concern                            | Control question                                             |
| --------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| Hallucinated or inaccurate output | Completeness and accuracy of data      | Who reviews the output, and against what source?             |
| Model or prompt drift             | Control no longer operates as designed | Are prompt, threshold and model changes approved and logged? |
| Agent with broad permissions      | Segregation of duties                  | Can the same system configure, execute and approve?          |
| Prompt injection or data leakage  | Access and data integrity              | Are inputs filtered and credentials centralized?             |
| Over-reliance on automation       | Weak management review                 | Is review tiered by risk and evidenced?                      |

BDO adds that [failing to inventory AI in financial reporting creates risks around accuracy, bias and over-reliance](https://www.bdo.com/getContentAsset/3df991ce-5af6-4c9b-a178-c7aed52a014c/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/ASSR-TRA-AI-in-Financial-Reporting-insight-V-1.pdf?language=en), which is why the inventory comes first.

## What to document and test

PwC's guidance says companies should [inventory AI use and update risk and control matrices (RCMs), narratives and flowcharts where applicable](https://www.pwc.com/us/en/tech-effect/ai-analytics/trust-ai-audits.html), and be able to demonstrate validation, review, control testing and how the AI tools were developed, deployed and tested.

A workable documentation set for each AI use case:

1. **Inventory entry.** Business purpose, owner, the financial statement area it touches, the model and provider, and whether it is inside the process or supports the SOX program.
2. **Process narrative and flowchart.** Show where the AI step sits, what feeds it and who consumes the output.
3. **Control design.** The human review, the approval of configuration changes, access restrictions and monitoring.
4. **Validation record.** How you tested accuracy before go-live and how you retest after changes.
5. **Operating evidence.** What the system actually did, retained for the period your auditor needs.

On operating evidence, EY suggests designing for [point-in-time explainability, with durable logging of inputs, prompts, model versions and reviewer identity](https://www.ey.com/en_us/insights/assurance/ai-audit-readiness-for-financial-reporting). It adds that when AI outputs serve as audit evidence, auditors will examine how accuracy is supported, whether the process is reproducible and the nature of management review.

The PCAOB is moving the same way on the audit side. Its Technology Innovation Alliance work [recommends standardizing audit documentation to make AI and data-analytics analysis easier](https://pcaobus.org/news-events/speeches/speech-detail/ai-and-the-pursuit-of-audit-quality--a-regulatory-perspective), which means clean, structured evidence will matter more over time.

## Governance that holds up in an audit

Three practices recur across the professional guidance.

![Three AI governance practices for audit readiness](https://media.babylovegrowth.ai/blog-images/organization-30814/1793419810188_Three-AI-governance-practices-for-audit-readiness.jpeg)

**Separate configuring from approving.** COSO recommends [separating the ability to configure AI from the authority to approve or review its outputs](https://auditoresinternos.es/wp-content/uploads/2026/02/Achieving-Effective-Internal-Control-Over-Generative-AI_COSO_compressed.pdf), documenting approvals for prompt, threshold and retrieval-corpus changes, and monitoring failures, drift, model updates and overrides.

**Extend IT general controls to AI components.** EY points to [prompt libraries as an example of what ITGC scope should now cover](https://www.ey.com/en_us/insights/assurance/ai-audit-readiness-for-financial-reporting). Crowe likewise recommends that IT teams [vet AI-related system changes, especially those touching financially relevant data](https://www.crowe.com/insights/ai-in-financial-reporting-balancing-innovation-and-risk).

**Tier the human review.** Crowe suggests a tiered validation approach that starts with a [human-in-the-loop control where a person double-checks AI output](https://www.crowe.com/insights/ai-in-financial-reporting-balancing-innovation-and-risk). Low-risk, high-volume outputs can then move to sampled review and automated checks, while estimates and manual-entry equivalents get full review.

EY also stresses that governance should involve [clear accountability across finance, IT, risk and compliance rather than sitting with data science alone](https://www.ey.com/en_us/insights/assurance/ai-audit-readiness-for-financial-reporting). Engineers supply the evidence, finance owns the control.

## Generating audit evidence from your AI stack

This is where engineering work turns SOX requirements into something concrete. Map each ask from the auditor to a technical capability.

| What the auditor asks                   | Engineering capability                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| What did the AI do on this transaction? | Traces capturing inputs, intermediate reasoning steps, tool calls and outputs                           |
| Is accuracy supported?                  | Repeatable evaluations against a reference set, including LLM-as-a-judge scoring with human spot checks |
| Which prompt and model were live?       | Versioned prompts and recorded model identifiers per request                                            |
| Who can change the system?              | Centralized credentials and provider access through a gateway, with change history                      |
| Can you reproduce it?                   | Stored traces plus versioned configuration                                                              |

With MLflow, tracing can be switched on in a few lines. A minimal example for an OpenAI-based close-support assistant:

```python
import mlflow

mlflow.set_experiment("close-reconciliation-assistant")
mlflow.openai.autolog()  # captures each call as a trace

@mlflow.trace
def suggest_match(item: dict) -> str:
    # your retrieval and model call here
    ...
```

Each run then leaves a record of inputs, outputs and intermediate steps that a reviewer can open later. For agents, that includes the reasoning path and tool calls, which is the part auditors find hardest to reconstruct after the fact. For more on the underlying idea, see our material on [audit trails for machine learning](https://mlflow.org/articles/tags/audit-trails-for-machine-learning) and [why to audit AI decision making](https://mlflow.org/articles/why-audit-ai-decision-making-a-2026-guide).

A practical pattern for a SOX-relevant AI use case:

1. Trace every production call and retain traces for your control's retention period.
2. Build an evaluation set from past, reviewed transactions. Run it before every prompt or model change and keep the results as validation evidence.
3. Route model access through a governed gateway so keys, provider choice and usage policies sit in one place instead of in application code.
4. Record the reviewer's decision next to the trace, so the evidence shows both what the AI proposed and who accepted or overrode it.
5. Alert on drift in your evaluation scores, and treat a failed threshold as a control exception to investigate.

None of this makes a control effective on its own. Whether the control design is adequate remains a judgment for management, internal audit and the external auditor. What good tooling does is make that judgment cheaper and better informed.

> ## Our view on AI controls
>
> We see SOX teams make the same mistake with AI that engineering teams once made with logging: they treat evidence as something to assemble at quarter end. By then the prompt has changed twice, the model has been updated and nobody remembers which version approved which entry.
>
> Our view is that evidence should be a by-product of running the system. When tracing, evaluation, prompt versioning and provider access are built on one open-source platform, the audit trail exists before anyone asks for it. That is the idea behind MLflow: the whole lifecycle, from observability to LLM-as-a-judge evaluation to gateway governance, available without enterprise feature paywalls and under Linux Foundation governance.
>
> We would also argue for openness for a reason auditors will appreciate. Traces built on OpenTelemetry-compatible standards stay readable in your own infrastructure, so your evidence does not depend on one vendor's retention policy.
>
> _Kevin_

## Put the evidence layer in place

If you are scoping AI into your SOX program, start with the tooling that records what your models and agents do. MLflow is open source and covers [agent and LLM engineering](https://mlflow.org/genai), [automated evaluation](https://mlflow.org/genai/evaluations) and a [cross-provider AI Gateway](https://mlflow.org/genai/ai-gateway) in one platform, so tracing, validation and access governance share a single record.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

If your evaluation approach relies on model-based scoring, our page on [LLM-as-a-judge evaluation](https://mlflow.org/llm-as-a-judge) shows how to build repeatable checks you can run before each change and retain as validation evidence. You can explore the platform at [mlflow.org](https://mlflow.org).

## Can AI replace management or the auditor?

No. AS 2201 requires the auditor to obtain sufficient appropriate evidence for an ICFR opinion while exercising [due professional care and professional skepticism](https://pcaobus.org/oversight/standards/auditing-standards/details/AS2201). Those are human duties.

The SEC has said the same about management. On August 6, 2026, Chief Accountant Kurt Hohl stated that [management remains ultimately responsible for financial statements and needs to understand AI-related risks](https://www.sec.gov/newsroom/podcasts/material-matters-sec-chairman-paul-atkins/chief-accountant-kurt-hohl-building-trust-our-capital-markets), and that the SEC was assessing whether auditors use AI responsibly.

The design implication is simple. Build AI as a preparer and a tester that a qualified person can challenge. COSO's guidance says qualified personnel should [review outputs, resolve contradictions and retain traceable evidence of their judgments](https://auditoresinternos.es/wp-content/uploads/2026/02/Achieving-Effective-Internal-Control-Over-Generative-AI_COSO_compressed.pdf). Your system should make that review easy to perform and easy to prove.

## A 30-day starting plan

- **Week 1:** Inventory every AI use in finance and accounting, including shadow tools and vendor features that call models.
- **Week 2:** With finance and internal audit, decide which uses are in ICFR scope and map them to existing risks, using the table above.
- **Week 3:** Turn on tracing for in-scope systems, version the prompts and define the evaluation set for each.
- **Week 4:** Draft the narrative, control description and review procedure, then run a walkthrough with your auditor's questions in hand.

For further reading on audit practice, see our posts on [best practices for AI auditing](https://mlflow.org/articles/tags/best-practices-for-ai-auditing) and [challenges in auditing AI](https://mlflow.org/articles/tags/challenges-in-auditing-ai).

## FAQ

### Does SOX apply to AI tools directly?

SOX applies to internal control over financial reporting at public companies, not to a technology as such. An AI tool is in scope when it affects the preparation of financial statements or the operation of a control. In that case, it is documented, tested and monitored like any other part of ICFR.

### What evidence will auditors want for an AI-assisted control?

Expect questions on accuracy, reproducibility and management review. EY points to durable logs of inputs, prompts, model versions and reviewer identity as the basis for explaining an output at a point in time. Validation results and change approvals complete the picture.

### Should AI be in our SOX risk and control matrix?

Yes, where it touches financial reporting. PwC recommends flagging AI used in journal entries, estimates and close processes, and updating RCMs, narratives and flowcharts to reflect it. An inventory is the first step.

### Can we use AI agents to test our own SOX controls?

Yes, with oversight. Agents can help gather evidence and run tests, but a qualified reviewer should be able to inspect, override and sign off on the results. The external auditor still forms an independent opinion.

## Sources

- [Modernized SOX compliance with GenAI, Deloitte US](https://www.deloitte.com/us/en/services/audit-assurance/blogs/accounting-finance/ai-finance-reporting-automation-public-companies.html)
- [AI Materials, SEC Investor Advisory Committee panel](https://www.sec.gov/files/sec-iac-panel-ai-final-materials-brian-croteau.pdf)
- [Generative AI, COSO](https://www.coso.org/generative-ai)
- [Achieving Effective Internal Control Over Generative AI, COSO](https://auditoresinternos.es/wp-content/uploads/2026/02/Achieving-Effective-Internal-Control-Over-Generative-AI_COSO_compressed.pdf)
- [AI audit readiness for financial reporting, EY US](https://www.ey.com/en_us/insights/assurance/ai-audit-readiness-for-financial-reporting)
- [Trust AI and audits, PwC](https://www.pwc.com/us/en/tech-effect/ai-analytics/trust-ai-audits.html)
- [Seize the future, KPMG](https://kpmg.com/kpmg-us/content/dam/kpmg/pdf/2025/1a-thought-leadership-v3.pdf)
- [AI in Financial Reporting: Key Governance and Control Considerations, BDO](https://www.bdo.com/getContentAsset/3df991ce-5af6-4c9b-a178-c7aed52a014c/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/ASSR-TRA-AI-in-Financial-Reporting-insight-V-1.pdf?language=en)
- [AI in Financial Reporting: Balancing Innovation and Risk, Crowe](https://www.crowe.com/insights/ai-in-financial-reporting-balancing-innovation-and-risk)
- [AI and the Pursuit of Audit Quality, PCAOB](https://pcaobus.org/news-events/speeches/speech-detail/ai-and-the-pursuit-of-audit-quality--a-regulatory-perspective)
- [SEC Interpretive Release: Commission Guidance Regarding Management's Report on ICFR](https://www.sec.gov/files/rules/interp/2007/33-8810.pdf)
- [AS 2201, PCAOB](https://pcaobus.org/oversight/standards/auditing-standards/details/AS2201)
- [Chief Accountant Kurt Hohl: Building Trust in Our Capital Markets, SEC](https://www.sec.gov/newsroom/podcasts/material-matters-sec-chairman-paul-atkins/chief-accountant-kurt-hohl-building-trust-our-capital-markets)

## Recommended

- [Enterprise AI Compliance Documentation: A Practical Guide](https://mlflow.org/articles/enterprise-ai-compliance-documentation)
- [Why Audit AI Decision Making: A 2026 Guide](https://mlflow.org/articles/why-audit-ai-decision-making-a-2026-guide)
