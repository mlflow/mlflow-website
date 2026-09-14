---
title: "Developers: Stop 80% of PII Before Any LLM Call"
description: "Developer focused PII redaction: run regex and NER in a gateway, use reversible placeholders, and keep raw identifiers out of LLM calls."
slug: pii-redaction-llm
tags:
  [
    pii redaction prompts,
    pii in prompts,
    pii masking traces,
    how to redact PII using LLMs,
    LLM privacy compliance,
    machine learning data redaction,
    PII data protection,
    automated PII removal,
    pii redaction llm,
    pii masking llm,
  ]
date: 2026-09-14
image: https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789361184859_Developer-reviewing-sanitized-API-request.jpeg
---

![Developer reviewing sanitized API request](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789361184859_Developer-reviewing-sanitized-API-request.jpeg)

Redact PII before any model call, using a layered detection pipeline and reversible placeholders. That single operational rule beats every alternative because it keeps raw identifiers out of provider logs, training pipelines, and third-party subprocessors entirely. Layer regex and checksum validation first, add named-entity recognition for names and locations, and reserve a local LLM for edge cases you can't resolve deterministically. Use reversible placeholders, not permanent masking, whenever the workflow needs the original value restored later.

---

> **TL;DR:**
>
> - Layered detection combining regex, checksum validation, and NER is essential to catch structured and unstructured PII before it reaches the LLM.
> - Reversible placeholders preserve context and enable accurate rehydration, making them preferable to permanent masking in production workflows.
> - Deployment at the gateway or proxy level ensures centralized policy enforcement and minimizes risks of inconsistent client-side implementation.
> - Regular testing against adversarial inputs like homoglyphs, leetspeak, and token splits is critical to maintain detection accuracy and avoid evasions.
> - Maintaining detailed, versioned audit logs and integrating telemetry with lifecycle management platforms like MLflow enhances compliance, monitoring, and ongoing redaction performance.

---

## Table of Contents

- [What Is PII Redaction LLM Pipeline Design?](#what-is-pii-redaction-llm-pipeline-design)
- [Detection Methods: Regex, NER, Classifiers, and LLM Review](#detection-methods-regex-ner-classifiers-and-llm-review)
- [Detect, Anonymize, Rehydrate: The Core Pipeline Pattern](#detect-anonymize-rehydrate-the-core-pipeline-pattern)
- [Client, Gateway, or Server: Where Should Redaction Run?](#client-gateway-or-server-where-should-redaction-run)
- [Building the Redaction Layer: Checklist and API Flow](#building-the-redaction-layer-checklist-and-api-flow)
- [How Do You Measure PII Detection Accuracy?](#how-do-you-measure-pii-detection-accuracy)
- [Compliance and Audit: What Regulators Actually Expect](#compliance-and-audit-what-regulators-actually-expect)
- [Operationalizing Redaction at Platform Scale](#operationalizing-redaction-at-platform-scale)
- [Multilingual and Unstructured Data Challenges](#multilingual-and-unstructured-data-challenges)
- [Redaction and Data Lifecycle Management](#redaction-and-data-lifecycle-management)
- [Where LLM-Based Redaction Breaks Down](#where-llm-based-redaction-breaks-down)
- [Author Perspective: A Rolling Plan, Not a Perfect One](#author-perspective-a-rolling-plan-not-a-perfect-one)
- [Bring Redaction Telemetry Into Your LLM Lifecycle](#bring-redaction-telemetry-into-your-llm-lifecycle)
- [Sources](#sources)
- [FAQ](#faq)

## What Is PII Redaction LLM Pipeline Design?

A PII redaction LLM pipeline is the set of detection and substitution steps that run between a user's input and the model call, designed so no raw personally identifiable information ever reaches the LLM provider. This differs from generic content filtering: the goal isn't blocking bad prompts, it's swapping sensitive spans (a Social Security number, an email, a patient name) for safe placeholders before the API request fires, then optionally restoring them in the response.

Most teams get this wrong by treating it as a single-model problem. They ask an LLM to "find and remove PII" in one shot and call it done. The [PRvL research on arXiv](https://arxiv.org/html/2508.05545v1) shows why that's fragile: model architecture and training choices materially affect redaction accuracy, and even well-tuned models miss context-dependent identifiers or over-redact benign text. The fix isn't a better prompt. It's a layered pipeline where deterministic tools catch the easy [80%](https://www.nvidia.com/en-us/glossary/frontier-models/), and a model only handles the ambiguous remainder, ideally running locally rather than as an unvetted third-party call.

This matters most for three architectures: chat applications where users paste raw customer data into a prompt, retrieval-augmented generation (RAG) systems that pull PII-laden documents into context, and multi-hop agents that pass data between tool calls where nobody's watching the middle steps. Each needs redaction enforced at a different point, which is why "PII masking traces" and "PII in prompts" have become distinct engineering concerns rather than one generic compliance checkbox.

## Detection Methods: Regex, NER, Classifiers, and LLM Review

Structured identifiers should never reach a neural network for detection. Credit card numbers, Social Security numbers, and IBANs follow fixed formats with checksum algorithms (Luhn for card numbers, mod-97 for IBANs), so regex plus checksum validation catches them with near-zero false positives at microsecond latency. This is your first pass, and it should run on every single request without exception.

Free-text identifiers need a different tool. Names, addresses, and locations don't follow a pattern, so this is where named-entity recognition (NER) earns its place. NER models trained on datasets like CoNLL or OntoNotes fill the gap regex leaves open, catching "meet me at 44 Birch Lane" or "forward this to Priya Chandrasekaran" that no regular expression could anticipate. The trade-off: NER trades some precision for recall. It will flag business names as people occasionally, and it can miss names it wasn't trained to recognize, particularly non-Western naming conventions.

Context-sensitive PII, the kind that's only sensitive because of surrounding text (a room number that's fine alone but identifying next to a patient name), needs a classifier trained or fine-tuned on your domain's actual documents. Generic models won't know that "Room 214B" matters in a hospital intake form but not in a hotel booking confirmation.

Distil-PII demonstrates where LLM-assisted review actually earns its place: small, specialized models in the 1B to 3B parameter range scored approximately 0.81 accuracy on redaction tasks, close to the accuracy scored by much larger models exceeding 600 billion parameters.

Evasions remain the hardest open problem in this space:

- **Homoglyphs** swap Latin characters for visually identical Unicode lookalikes, defeating naive regex.
- **Leetspeak and spacing tricks** ("S-S-N: 123 45 6789") break tokenization boundaries that detectors rely on.
- **Tokenization splits** can separate a phone number across multiple tokens, hiding it from pattern matchers that only scan whole tokens.

Mitigate these with normalization passes (Unicode NFKC normalization, whitespace collapsing) before detection runs, not after.

**Pro Tip:** _Run your detection stack against a deliberately adversarial test set with homoglyphs and leetspeak baked in before you trust any recall number your team reports internally._

## Detect, Anonymize, Rehydrate: The Core Pipeline Pattern

The pattern that actually ships in production follows three steps: detect PII in the outbound payload, replace it with a placeholder token, send the sanitized text to the LLM, then restore the original values in the response before it reaches whatever system displays it. This is what [Preserve implements](https://pypi.org/project/preserve-pii/) with local-first layered detection and a reversible placeholder map, paired with an OpenAI-compatible proxy that scrubs prompts before they ever leave the user's machine.

Reversible placeholders beat permanent masking whenever downstream logic needs the real value. A support agent summarizing a customer ticket still needs to know which customer they're talking about after the LLM responds; a hardcoded `[REDACTED]` breaks that. A session-scoped vault mapping `[PERSON_1]` back to "Maria Gonzalez" solves it without the raw name ever touching the model.

Three deployment patterns cover most production cases:

1. **Direct pre-send redaction.** The application layer detects and swaps PII immediately before constructing the API request. Simplest to reason about, but every calling service needs the same logic implemented correctly.
2. **Gateway/proxy interception.** A self-hosted proxy sits between your application and the LLM provider, redacting every outbound call regardless of which internal team wrote the code. This is the pragmatic default for most organizations because it enforces policy at one choke point instead of trusting every developer to remember.
3. **Per-hop agent enforcement.** In multi-step agent workflows, redact not just the initial user input but every retrieved RAG chunk, every tool output, and every intermediate message passed between agent steps. A pattern catalog for agent PII redaction recommends two-pass inspection, checking both inputs and outputs, precisely because agents leak PII at hops nobody reviews manually.

Vault design deserves its own attention. Keep placeholder maps ephemeral and scoped to a single session or conversation, never a global lookup table that persists indefinitely. Encrypt the vault at rest, expire entries aggressively, and treat access to the vault itself as a privileged operation subject to the same audit logging as the redaction decisions it supports.

**Pro Tip:** _If your agent calls three tools before generating a final answer, redact after every single tool call, not just once at the start. PII reintroduced by a tool response at hop two will sail straight into hop three if you only scrubbed the original prompt._

## Client, Gateway, or Server: Where Should Redaction Run?

Client-side redaction gives the strongest privacy guarantee because sensitive data never leaves the user's device unredacted, but it demands every client implementation, web, mobile, CLI, get the detection logic right and keep it updated. Miss one client and you've got a silent gap in coverage.

Gateway or proxy-based redaction is the default most engineering teams should reach for first. A single enforcement point means you patch detection logic in one place, apply one policy version across every calling service, and get centralized logging for free. [Philterd's guidance](https://philterd.ai/blog/redact-pii-before-sending-to-an-llm/) backs this pattern specifically: keep PII inside your network boundary by redacting before the request crosses it, and do it with deterministic, self-hosted tooling rather than trusting an external API.

![Client, Gateway, or Server: Where Should Redaction Run? — overview diagram](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1789361246054_Client-Gateway-or-Server-Where-Should-Redaction-Run-overview-diagram.jpeg)

Server-side or post-receive redaction, scrubbing the payload after your backend already received the raw PII, does not shrink your compliance scope. The moment raw PII touches your server's memory, logs, or request queue, you've already incurred the handling obligation that redaction was supposed to prevent. Post-receive scrubbing before the LLM call is fine as an implementation detail; treating it as equivalent to never receiving the data in the first place is a compliance misunderstanding that shows up in audits.

Latency is the real constraint that decides architecture in practice:

- Regex and checksum checks add sub-millisecond overhead and should never be skipped regardless of load.
- NER inference adds tens of milliseconds depending on model size and hardware.
- Occludra's staged design keeps inline text redaction under 50 milliseconds by short-circuiting on high-confidence regex matches and only escalating to heavier NER or OCR stages when the cheap pass is inconclusive.
- Reserve any LLM-based review for asynchronous or non-blocking paths; never gate a real-time chat response on a second model call if you can avoid it.

## Building the Redaction Layer: Checklist and API Flow

Start with a versioned policy document, not code. Define every entity type you'll detect (names, emails, phone numbers, medical record numbers, financial account numbers), the masking rule for each (full redaction, partial masking, reversible placeholder), and any allow-lists for terms that look like PII but aren't (a product named "Jordan," a street called "Liberty").

The API flow that most production systems converge on looks like this:

1. **Detect** entities in the outbound payload using your layered pipeline (regex, then NER, then classifier where needed).
2. **Scrub** each match, replacing it with a placeholder token following a consistent naming convention (`[EMAIL_1]`, `[PERSON_2]`) so multi-turn conversations stay coherent across turns.
3. **Call the LLM** with the sanitized payload only, never the original.
4. **Scan the output** for any PII the model may have hallucinated or echoed back in an unexpected form.
5. **Restore or refuse.** If the response references a placeholder, rehydrate it from the session vault. If the output contains PII that wasn't in your placeholder map, refuse to return it and log the incident.

Placeholder naming conventions matter more than teams expect. Using a stable, incrementing scheme within a session (`[PERSON_1]` always maps to the same individual across ten conversation turns) keeps the LLM's reasoning coherent. Random or per-call placeholder generation breaks context and can produce confused, inconsistent responses when a user references "that person I mentioned earlier."

Ship this in three modes before you ever enforce it: **detect-only** (log what would be caught, change nothing), **warn-only** (surface findings to developers without blocking), and **enforce** (actually redact and refuse). Add CI gates and pre-commit scanners so PII patterns can't sneak into test fixtures or example code either.

**Pro Tip:** _Keep policy versioning explicit in your logs, every redaction decision should record which policy version made the call, so a false negative discovered next quarter can be traced back to the exact rule set that missed it._

Operational controls round this out: feature flags to toggle detection stages independently, an emergency rollback path if a new NER model starts over-redacting legitimate content, and monitoring dashboards tracking redaction volume by entity type over time.

## How Do You Measure PII Detection Accuracy?

Precision and recall pull against each other, and picking the wrong balance either lets PII through or breaks your product with false positives. Precision measures how many flagged spans were actually PII; recall measures how much real PII you caught. Track both per entity type, since your NER model might hit 95% recall on names but only 70% on less common identifiers like passport numbers, and a single blended F1 score hides that gap.

| Metric              | What it measures                             | Why it matters for redaction                                                      |
| ------------------- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| Precision           | Share of flagged items that are true PII     | Low precision means excessive false-positive masking, breaking legitimate content |
| Recall              | Share of actual PII successfully caught      | Low recall means real PII slips through to the model                              |
| F1 score            | Harmonic mean of precision and recall        | Single-number tracking for regression testing across releases                     |
| Per-entity hit rate | Precision/recall broken out by entity type   | Surfaces weak spots a blended score would hide                                    |
| Latency per token   | Processing time added by the detection stage | Determines whether staged short-circuiting is needed                              |

Build your test corpus from three sources: synthetic data generated specifically to cover entity types you rarely see in production, a real-sample slice reviewed and scrubbed by hand for ground truth, and an adversarial edge-case set targeting the evasions covered earlier. The PRvL paper specifically recommends stress-testing with homoglyphs, partial tokens, and mixed-language input before trusting any reported accuracy figure.

Roll new detection thresholds out in warn-only mode first, measure the delta against your existing pipeline through telemetry, then graduate to canary enforcement on a small traffic percentage before full rollout. Your audit harness should log the entity category detected, the action taken, and the policy version applied, never the raw matched value itself.

## Compliance and Audit: What Regulators Actually Expect

Redaction, anonymization, and de-identification are not interchangeable terms, and conflating them creates audit risk. Redaction removes or masks specific data before storage or transmission. Anonymization aims to make re-identification statistically infeasible. De-identification, the term [HHS uses for HIPAA](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html), refers to a formal process with two accepted methods (Safe Harbor or Expert Determination) that carries specific regulatory weight healthcare organizations must document.

HHS guidance emphasizes controlled handling and documented de-identification methodology rather than trusting an opaque tool to guarantee anonymity on its own. That means your redaction pipeline is a component of a compliance story, not the whole story, and you need to document which method you're relying on and why it satisfies the applicable standard for your data flow.

Audit logs need a specific shape to be useful during a review:

- Record the detection category (e.g., "SSN," "patient name") and the policy version that fired.
- Record the action taken (redacted, allowed through an allow-list exception, flagged for human review).
- Never write the raw matched value into the log itself, that defeats the entire purpose of redaction.
- Set explicit retention windows for both the audit logs and any session-scoped placeholder vaults.
- Define an escalation path for cases flagged for human review, with access controls limiting who can view unredacted content.

## Operationalizing Redaction at Platform Scale

Redaction decisions become far more useful when they're traceable, not just logged in isolation. Attaching detection metadata to prompt traces, which entity types fired, which policy version applied, whether a human reviewer overrode a decision, turns a compliance checkbox into a debugging tool your team actually uses.

[MLflow's tracing and observability capabilities](https://mlflow.org/genai/observability) apply directly here. Rather than bolting redaction logging onto a separate system, teams can:

- Capture redaction events as spans within the same trace that captures the LLM call itself, so debugging a bad response and auditing a redaction decision happen in one place.
- Version detection policies alongside model and prompt versions, gating rollout of a new NER threshold the same way you'd gate a new model version.
- Feed detection metrics (per-entity precision and recall) into an LLM evaluation harness and alert automatically when accuracy drifts between releases.
- Centralize policy governance across teams so a redaction rule change in one service doesn't silently diverge from another.

This is the difference between redaction as a one-off script and redaction as a governed, monitored part of your LLM lifecycle.

## Multilingual and Unstructured Data Challenges

English-trained NER models degrade fast on other languages, and PII detection is no exception. A model tuned on English name patterns will miss compound surnames common in Spanish or Portuguese, patronymic naming conventions in Russian or Icelandic, and non-Latin scripts entirely unless it was explicitly trained on multilingual corpora.

The practical fix is language-specific model routing rather than one universal detector. Detect the input language first (a fast, cheap classification step), then route to a language-appropriate NER model or a multilingual model specifically validated against your target languages. Don't assume a single English-centric pipeline will generalize.

Unstructured data compounds the problem. PDFs, scanned documents, and images embed PII in formats that text-based regex and NER simply can't see. This is where OCR enters the pipeline as a distinct, heavier stage, run asynchronously rather than inline, since optical character recognition adds latency that real-time chat can't absorb. Occludra's staged architecture treats OCR as an out-of-band step precisely for this reason, keeping the fast text path under its latency budget while image content gets processed separately.

Mixed-language input within a single conversation, common in global products, breaks detectors that assume one language per request. A user switching between English and Tagalog mid-message needs a detection pipeline that can segment and route each portion independently, or a multilingual model robust enough to handle code-switching without a language-detection pre-pass at all. Test your pipeline specifically against this case; it's one of the more common blind spots teams discover only after a production incident.

## Redaction and Data Lifecycle Management

Redacted placeholder vaults are still sensitive data, and treating them as an afterthought outside your retention policy creates a second compliance gap you didn't intend to open. If a vault maps `[PERSON_3]` back to a real customer indefinitely, you've effectively recreated the PII exposure the redaction was meant to prevent, just one layer removed.

Tie vault lifecycle explicitly to your existing data retention schedule. If your organization deletes customer support tickets after 90 days, the placeholder vault entries tied to those tickets should expire on the same schedule, not persist separately with no defined end date. This requires the vault to carry metadata linking each entry back to its source record, so automated retention jobs can find and purge it.

Redaction also needs to integrate with data subject access and deletion requests. If a user exercises a deletion right under an applicable privacy law, your process needs to purge not just the original record but any placeholder vault entries and audit log references tied to that individual, wherever they live across your systems. Building this connective tissue after the fact is significantly harder than designing the vault schema with lifecycle hooks from day one.

Session-scoped vaults with aggressive, automatic expiration solve most of this by default: if a vault entry only lives for the duration of an active conversation and expires within hours, you've sidestepped most long-term retention risk without needing to build custom deletion tooling. Reserve longer-lived vaults for the narrow set of workflows that genuinely need persistent rehydration across sessions, and apply extra scrutiny and shorter retention windows to those specifically.

## Where LLM-Based Redaction Breaks Down

False negatives are the risk that gets the most attention, PII slipping through undetected, but false positives carry real cost too. Over-aggressive redaction that masks a product name, a common word that happens to match a name pattern, or a legitimate business address breaks the user experience and erodes trust in the system fast. Both failure modes need active monitoring, not just recall optimization.

Model drift is the quieter problem. A NER model tuned against last year's data can degrade as naming conventions, product terminology, or user demographics shift, and nothing alerts you unless you're actively tracking per-entity accuracy over time. This is exactly why warn-only telemetry and continuous monitoring, not a one-time accuracy benchmark at launch, need to be part of the operational plan from day one.

Using an LLM itself as the primary redaction engine introduces a distinct risk category the PRvL research documents directly: model architecture and training choices affect redaction reliability in ways that aren't always predictable from a model's general capability. A model that's excellent at reasoning tasks isn't automatically excellent at exhaustively finding every identifier in a document, and treating "smart" as synonymous with "reliable redactor" is a category error that shows up as inconsistent production behavior.

Adversarial input is the risk category most teams underweight. Someone deliberately trying to smuggle PII past your filters, whether testing your system's boundaries or attempting to exfiltrate data through a chatbot, will use homoglyphs, unusual formatting, and split tokens specifically because they know regex-only detection will miss them. Treat your redaction pipeline as a security boundary that gets tested adversarially, not just a data-quality filter validated against clean synthetic examples.

## Author Perspective: A Rolling Plan, Not a Perfect One

Chasing perfect recall on day one is the wrong first move. Start with deterministic detection (regex, checksums) behind a gateway proxy, and run NER in warn-only mode long enough to see what it actually catches against your real traffic before you let it block anything. Reserve local LLM review for your genuinely highest-sensitivity flows, healthcare intake, financial account data, not as a blanket first-pass filter. Reversible placeholders and solid audit logging matter more than squeezing out the last few points of recall; a system you can explain and roll back beats one that's marginally more accurate and opaque. Let warn-only telemetry shape your policy before you flip enforcement on.

> _— Kevin_

## Bring Redaction Telemetry Into Your LLM Lifecycle

Building a layered redaction pipeline solves detection and masking, but the harder long-term problem is keeping that pipeline observable as your models, policies, and traffic evolve. That's where an integrated lifecycle platform earns its place alongside your redaction tooling rather than replacing it.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

Mlflow gives teams a home for the parts of this workflow that tend to sprawl across disconnected scripts and spreadsheets: tracing that captures redaction decisions alongside the LLM calls they protect, evaluation harnesses that catch detection drift before it becomes an incident, and gateway hooks for centralizing policy across every team calling a model. If you're already running a layered detector and a proxy, [MLflow's AI platform](https://mlflow.org/ai-platform) gives you a place to version those policies, gate their rollout, and monitor accuracy over time instead of tracking it in ad hoc dashboards. Explore [MLflow's observability tooling](https://mlflow.org/ai-observability) to see how trace-level visibility fits into a redaction workflow you're already running, and start by connecting your existing pipeline's telemetry to a single tracked experiment.

## Sources

The technical claims in this guide draw on peer-reviewed and practitioner sources worth reading directly. The PRvL paper on arXiv covers LLM redaction capabilities and risks in depth. Preserve's project documentation details the reversible placeholder and proxy pattern referenced throughout. Philterd's engineering blog lays out the deterministic, layered detection approach. Occludra's architecture post documents the sub-50ms staged pipeline design. HHS provides the official HIPAA de-identification guidance referenced in the compliance section. The Distil-PII repository documents small-model redaction benchmarks, and the [Agent Patterns Catalog](https://github.com/agentpatternscatalog/patterns/blob/main/patterns/pii-redaction.md) outlines agent-specific redaction practices. For broader context on data ownership when building on hosted LLMs, see this [Gainable engineering post](https://gainable.dev/blog/if-the-llm-is-your-engine-you-dont-own-your-product).

- [PRvL: Quantifying the Capabilities and Risks of Large Language Models for PII Redaction (arXiv)](https://arxiv.org/html/2508.05545v1)
- [preserve-pii (PyPI / project docs)](https://pypi.org/project/preserve-pii/)
- [Philterd blog — Redact PII Before Sending to an LLM](https://philterd.ai/blog/redact-pii-before-sending-to-an-llm/)
- [HHS — De-identification guidance (HIPAA)](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)

## FAQ

### How do you handle PII data without exposing it to an LLM?

Detect PII with a layered pipeline (regex, NER, and classifiers), replace matches with reversible placeholders, send only the sanitized text to the model, then restore the original values in the response using a session-scoped vault.

### What is PII redaction?

PII redaction is the process of detecting and removing or masking personally identifiable information, like names, Social Security numbers, or emails, from text before it's stored, shared, or sent to a third-party system such as an LLM provider.

### Which approach works best for detecting sensitive data before an LLM call?

A layered pipeline combining regex and checksum validation for structured identifiers, NER for free-text names and locations, and an optional local model like those in the Distil-PII family for edge cases outperforms relying on any single method, including a general-purpose LLM alone.

### What are the two main types of PII?

Direct identifiers uniquely identify a person on their own (a Social Security number, a passport number), while indirect or quasi-identifiers only become identifying in combination with other data (a birth date plus a ZIP code plus a job title).

### Does server-side redaction reduce compliance scope?

No. Once raw PII reaches your server's memory or logs, you've already incurred the handling obligation; redacting after receipt doesn't retroactively remove that exposure, so pre-send redaction remains the stronger control.

## Recommended

- [One post tagged with "how LLM works"](https://mlflow.org/articles/tags/how-llm-works)
- [LLM Application Architecture: A 2026 Engineer's Guide](https://mlflow.org/articles/llm-application-architecture-a-2026-engineers-guide)
- [LLM Tracing & AI Tracing for Agents](https://mlflow.org/llm-tracing)
- [One post tagged with "LLM application components"](https://mlflow.org/articles/tags/llm-application-components)
