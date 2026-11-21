---
title: "Chat History Retention for AI Teams: What to Keep and When"
description: "How AI teams set chat history retention: separate platform history from business records, classify prompts and traces, apply purge rules, legal holds and audits."
slug: chat-history-retention
tags: [chat history retention]
date: 2026-11-21
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1795122080247_Engineer-setting-an-AI-chat-retention-policy.jpeg
---

![Engineer setting an AI chat retention policy](https://media.babylovegrowth.ai/blog-images/organization-30814/1795122080247_Engineer-setting-an-AI-chat-retention-policy.jpeg)

Chat history retention is the rule that decides how long prompts, model outputs and agent traces stay stored, who can read them, and what deletes them. For an AI team the working answer is to stop treating the chat window as the system of record. Classify every conversation and trace by purpose, give each class its own retention clock, and make purge and legal hold automatic. Teams that run agents through an observable, open-source layer such as [MLflow](https://mlflow.org/) can apply that policy to the one place where the full reasoning of an agent is captured: the trace.

## TL;DR

- A vendor's chat sidebar is not a retention policy. Its settings, plans and deletion windows can change without your review.
- Set retention per artifact (chat, trace, evaluation set, audit log), tied to purpose, risk and jurisdiction, not one global number.
- Purge has to reach copies: indexes, embeddings, caches, exports and logs. Legal hold must override ordinary purge.
- In the U.S., content and function decide whether a conversation is a record. Public bodies and regulated teams have the strictest rules.
- Verify the policy on a schedule. A written rule nobody tests is a liability.

## Table of Contents

- [Platform history is not a retention policy](#platform-history-is-not-a-retention-policy)
- [What U.S. rules actually push on](#what-us-rules-actually-push-on)
- [Classify first, then set the clock](#classify-first-then-set-the-clock)
- [Agent traces are the store people forget](#agent-traces-are-the-store-people-forget)
- [Purge, exports and legal hold](#purge-exports-and-legal-hold)
- [Controlling who can call what](#controlling-who-can-call-what)
- [Verify that the policy actually runs](#verify-that-the-policy-actually-runs)
- [Putting it together: a one-week rollout](#putting-it-together-a-one-week-rollout)
- [Where MLflow fits](#where-mlflow-fits)
- [Perspective](#perspective)
- [FAQ](#faq)
- [Sources](#sources)

## Platform history is not a retention policy

Most teams first meet retention as a product behavior. In ChatGPT, for example, [regular and archived chats stay on the account until the user deletes them or a workspace retention policy removes them](https://help.openai.com/en/articles/8983778-chat-and-file-retention-in-chatgpt). Deleting a chat removes it from the account view at once, and permanent deletion is scheduled within 30 days unless the data was already de-identified or must be kept for security or legal reasons.

That is a description of one vendor's defaults. It tells you nothing about what your organization must keep. Three problems follow from relying on it:

1. **The defaults move.** Plans change, admins change settings, vendors revise deletion rules. Your obligations do not move with them.
2. **History is not authority.** A transcript records what was said, including dropped options, stale numbers and confident mistakes. The approved report, ticket or case record is what your business relies on. The chat is a working file.
3. **Your own stack stores more than the chat.** Proxy logs, traces, evaluation datasets, vector indexes and backups each hold fragments of the same conversation, with their own clocks.

The practical move is to decide, per kind of artifact, what is the authoritative copy and what is just working material. Keep the authoritative copy in a system you control, in a format that opens without the original AI interface.

## What U.S. rules actually push on

There is no single U.S. statute that says "keep AI chats for X days." What exists is a set of records, discovery and privacy obligations that attach to the content and its use. Four are worth knowing.

### Public-sector records

The National Archives and Records Administration says [AI materials become federal records depending on circumstances such as official use or reliance in decision-making](https://www.archives.gov/files/records-mgmt/policy/nara-fra-ai-guidance.pdf), and disposal requires an approved schedule. Transitory records needed for less than 180 days and not needed for financial or legal obligations can be destroyed when no longer needed. Intermediary records used to create a later record can be destroyed once that record exists.

Local government follows the same logic. King County, Washington, states that [generative AI prompts and outputs are potentially public records, and that retention depends on content and function rather than a single period](https://cdn.kingcounty.gov/-/media/king-county/depts/executive-services/records-management/rm/rm-guidance-generative-ai.pdf). Its approved Copilot Chat enterprise storage setting is 30 days, after which content cannot be retrieved unless it was saved earlier.

The Peace Corps goes further operationally: its guidance requires users of public commercial AI services to [export federal-record inputs and outputs to a government recordkeeping system within 20 days](https://files.peacecorps.gov/documents/MS-549-Attachment-B.pdf), and warns that exported records may then be subject to FOIA and litigation discovery.

If you build for public-sector customers, expect these export-and-schedule requirements to land in your contracts.

![Public-sector records](https://media.babylovegrowth.ai/blog-images/organization-30814/1795122110378_Public-sector-records.jpeg)

### Litigation and legal hold

When litigation is reasonably anticipated, ordinary deletion has to stop for relevant material. A 2025 case showed how far that can reach: a U.S. federal court [ordered OpenAI to preserve user-generated ChatGPT and API content, including deleted chats](https://cyberinsider.com/us-court-orders-openai-to-preserve-all-chatgpt-logs-indefinitely/), with enterprise, education and zero-data-retention API customers exempt. OpenAI later stated that [its obligations under that order ended September 26, 2025](https://openai.com/index/response-to-nyt-data-demands/), and that it returned to standard retention, with deleted conversations and API data removed within 30 days.

The lesson for your team is not about one vendor. A deletion promise from a third party can be suspended by a court. Retention you control, on infrastructure you control, is the part you can plan around.

### Vendor terms for API and enterprise data

OpenAI's November 2024 court filing describes its U.S. practice: [workspace admins control retention for ChatGPT Enterprise, and API business data is kept up to 30 days, with zero-day retention available to eligible customers](https://storage.courtlistener.com/recap/gov.uscourts.nysd.612697/gov.uscourts.nysd.612697.379.13.pdf). Read your own provider's current terms for the same facts, and record them in your retention schedule as vendor deletion windows.

### Privacy and regulated work

If prompts contain personal data, state privacy laws such as the CCPA bring access and deletion rights, so "keep everything forever" becomes a privacy exposure as well as a cost. In regulated work such as GxP environments, an AI conversation that shapes an electronic record may need to meet the same audit-trail standard as that record. Have counsel or your compliance lead confirm which regimes apply to each use case before you fix the numbers.

## Classify first, then set the clock

The governance guidance from IF4IT is direct: don't use one blanket rule. [Define retention by use category, risk tier, data sensitivity, jurisdiction, regulatory obligations, business-record status, agent authority and evidence requirements](https://if4it.org/best-practices/enterprise-ai-governance-best-practices/govern-ai-interaction-output-and-evidence-retention/). Choose the least retentive option that still meets legal, audit, operational and evidence needs. Metadata-only logging can suit low-risk or privacy-sensitive uses, while full transcripts may be needed for high-risk, regulated, customer-facing, agentic or incident-related work.

For an engineering team, four classes cover most of what you store:

| Class                  | Examples                                                                             | Typical purpose       | Retention approach                                 |
| ---------------------- | ------------------------------------------------------------------------------------ | --------------------- | -------------------------------------------------- |
| Transient diagnostic   | Raw prompts and outputs in dev, debug traces                                         | Fixing bugs           | Days; mask personal fields at write time           |
| Operational monitoring | Latency, token counts, error rates, tool-call metadata                               | Performance and cost  | Longer, metadata only                              |
| Decision evidence      | Full agent traces for customer-facing or high-impact actions, with model and version | Audit, dispute review | Tied to the workflow and its regulator or contract |
| Business record        | The approved report, ticket, case note                                               | Operating record      | Your normal records schedule                       |

The numbers are yours to set. What matters is that each row has a stated purpose, a start event, an owner and an expiry action.

### The schedule fields

For each class, write down:

- the record class and its purpose;
- the start event (ticket closed, session ended, release shipped);
- the period or end event;
- the review owner;
- the expiry action (delete, anonymize, export to records system);
- exceptions, especially legal hold;
- derived-data behavior (summaries, embeddings, evaluation copies);
- the vendor's deletion window;
- how you will evidence the deletion.

That list is an engineering-friendly template. Put it in the repository next to the service it governs so it gets reviewed when the service changes.

## Agent traces are the store people forget

Chat transcripts are the visible store. For agents, the larger one is the trace: every model call, tool call, retrieved document and intermediate reasoning step. A trace often holds more sensitive content than the final conversation, because it includes what the agent fetched and what it considered, not only what the user saw.

That makes traces both your most valuable operational evidence and your biggest retention risk. Keep them too briefly and an incident cannot be reconstructed. Keep them indefinitely and every customer detail pasted into a prompt sits in your log store, in discovery scope and in breach scope.

IF4IT makes the same trade-off explicit: [permanent retention of every prompt and response raises privacy, discovery and breach exposure plus storage cost, while premature deletion can destroy records needed for audit, litigation or incident investigation](https://if4it.org/best-practices/enterprise-ai-governance-best-practices/govern-ai-interaction-output-and-evidence-retention/).

A workable pattern for traces:

1. Split raw content from metadata at write time, so you can keep timing, token and error data long after raw text expires.
2. Mask personal fields before storage, not after.
3. Attach identifiers (model, version, prompt version, session) so a trace can be linked to the decision it supported.
4. Give raw traces the shortest clock that supports debugging and incident review, and promote only specific traces to decision evidence.

The [MLflow LLM tracing](https://mlflow.org/llm-tracing) capability records agentic reasoning in depth and is built on OpenTelemetry, which means trace data fits the same pipelines and storage controls you already apply to other telemetry. Because MLflow is fully open source with no enterprise feature paywalls, you apply the retention classes above to traces without waiting on a vendor tier. Our notes on [LLM audit trails](https://mlflow.org/articles/tags/llm-audit-trails) and [LLM compliance logging](https://mlflow.org/articles/tags/llm-compliance-logging) go deeper on what a defensible log looks like, and our piece on [data residency for LLM logs](https://mlflow.org/articles/data-residency-llm-logs) covers where prompts and traces live.

## Purge, exports and legal hold

A retention rule without a purge rule is a wish. IF4IT's guidance says [AI interactions, transcripts, outputs and logs should be purged automatically when the approved period ends](https://if4it.org/best-practices/enterprise-ai-governance-best-practices/govern-ai-interaction-output-and-evidence-retention/), unless a legal, audit, incident, regulatory or business-record hold applies, and legal hold must override ordinary purge. It adds that purge should be reliable, documented and auditable, and should cover primary records, copies, indexes, embeddings, caches, exports, vendor-held records and downstream repositories where feasible.

The embeddings point catches teams most often. If you delete a transcript but leave the vector index built from it, the content is still retrievable in effect. Build deletion so that removing a conversation also removes the summaries, memories and search derivatives made from it.

A purge checklist you can hand to an engineer:

- **Inventory every store** that can hold conversation content: application database, trace store, proxy logs, evaluation datasets, vector indexes, caches, backups, exports.
- **Set the longest-lived copy first.** The shortest rule is irrelevant if one copy has no expiry.
- **Automate expiry** with jobs that log what they deleted and when.
- **Wire legal hold as a flag** that suspends expiry for matching records, and test that it does.
- **Redact evaluation sets.** Keep only redacted or synthetic copies of production conversations for regression testing.

That last point matters for teams using LLM-as-a-Judge evaluation: your evaluation data is a copy of production data with a different owner and, usually, no clock.

## Controlling who can call what

Retention starts at the point of entry. If every service calls model providers directly with its own keys, you have no single place to apply logging rules, redaction or provider-level settings such as zero-data-retention endpoints. A central gateway gives you one place to decide what is logged and which provider settings apply. The [MLflow AI Gateway](https://mlflow.org/ai-gateway) provides cross-provider governance and prompt management in one layer, so the retention decision is made once rather than per application.

## Verify that the policy actually runs

Auditors, customers and your own incident reviews will ask the same question: how do you know it works? The ARCS standard, a custody framework for AI records, lays out five verification expectations that translate well to any team:

1. Maintain [current documentation of lifecycle and custody controls](https://arcsstandard.org/full-standard/) (VER-01).
2. Run periodic internal checks that controls operate as documented (VER-02).
3. Obtain vendor confirmation of retention, deletion and preservation behavior (VER-03).
4. Confirm that deletion is suspended during holds (VER-04).
5. Keep documentation available for audit, regulatory or legal review (VER-05).

Make it concrete with a quarterly routine: sample a handful of expired conversations and confirm they are gone from every store, place a test hold and confirm nothing in scope expires, and review access logs for unexpected reads of raw traces. Record the results. The record of the check is itself evidence.

## Putting it together: a one-week rollout

- **Day 1:** list every AI use case and the stores it writes to.
- **Day 2:** assign each use case a class and a risk tier with legal or compliance input.
- **Day 3:** write the schedule fields for each class and name an owner.
- **Day 4:** implement masking and the raw/metadata split in your tracing and logging path.
- **Day 5:** automate expiry and the legal-hold flag, then run the first verification sample.

Start narrow. Cover customer-facing agents first, because they carry the highest evidence and privacy stakes, then extend to internal tools.

## Where MLflow fits

MLflow is an open-source platform for GenAI and LLM lifecycle management: observability through deep tracing of agent reasoning, automated evaluation with LLM-as-a-Judge, prompt management and versioning, and a cross-provider AI Gateway. For retention work, the useful part is that these pieces live in one place, so the traces, evaluation data and prompt versions you need to classify and govern are not scattered across tools. Start with the [Agent and LLM engineering overview](https://mlflow.org/genai) or the broader [AI platform guide](https://mlflow.org/ai-platform).

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

If you want to see how your agents' requests are captured end to end, begin with our walkthrough of [how to trace LLM requests](https://mlflow.org/articles/tags/how-to-trace-llm-requests) and the notes on [LLM logging best practices](https://mlflow.org/articles/tags/llm-logging-best-practices).

## Perspective

> Retention debates usually start with storage cost and end with an incident nobody can reconstruct. We think the order should be reversed: start from what you would need to prove, then keep exactly that, for exactly as long as it is needed.
>
> For agents, the proof lives in the trace. A final answer tells you what the user saw. The trace tells you which prompt version ran, which tools were called, what was retrieved and where the reasoning went wrong. That is why we built MLflow around tracing, evaluation and gateway governance as one lifecycle rather than three separate tools, and why we keep every feature open source under Linux Foundation governance, with no enterprise paywalls. Your retention controls should not depend on a pricing tier.
>
> Our advice to teams is simple. Treat the chat history as a working file, the trace as evidence, and the approved output as the record. Give each its own clock, automate the purge, and test the legal hold before you need it.
>
> _Kevin_

## FAQ

### How long should we keep AI chat history?

There is no universal period. Set it by use category, risk and legal obligation, then choose the least retentive option that still meets audit and operational needs. Public bodies may be bound by approved records schedules, such as the 180-day transitory category in NARA's guidance, while debugging data usually warrants a much shorter clock than decision evidence.

### Does deleting a chat delete it everywhere?

Not automatically. Deleting a conversation in an app removes it from the account view, and a vendor then schedules permanent deletion on its own timeline, for ChatGPT within 30 days in the normal case. Copies in your logs, traces, evaluation sets, embeddings, caches and backups need their own deletion rules.

### When does a legal hold override our retention schedule?

When litigation, an investigation or a regulatory inquiry is reasonably anticipated, relevant material has to be preserved and ordinary purge suspended. Build the hold as a flag that stops expiry for matching records, and have your legal counsel decide what is in scope.

### Are AI chats public records?

For U.S. government bodies, they can be. Guidance from agencies such as NARA and King County says the status depends on the content and function of the conversation, such as official use or reliance in decisions, rather than the tool that produced it. Private companies should check with counsel whether discovery, sector regulation or contracts treat their AI conversations as records.

## Sources

- [Chat and file retention in ChatGPT, OpenAI Help Center](https://help.openai.com/en/articles/8983778-chat-and-file-retention-in-chatgpt)
- [NARA guidance on applying the Federal Records Act to AI](https://www.archives.gov/files/records-mgmt/policy/nara-fra-ai-guidance.pdf)
- [King County records management guidance on generative AI](https://cdn.kingcounty.gov/-/media/king-county/depts/executive-services/records-management/rm/rm-guidance-generative-ai.pdf)
- [Peace Corps MS 549 Attachment B, AI records management considerations](https://files.peacecorps.gov/documents/MS-549-Attachment-B.pdf)
- [US court orders OpenAI to preserve ChatGPT logs, CyberInsider](https://cyberinsider.com/us-court-orders-openai-to-preserve-all-chatgpt-logs-indefinitely/)
- [OpenAI response to The New York Times data demands](https://openai.com/index/response-to-nyt-data-demands/)
- [OpenAI court filing exhibit on retention practice](https://storage.courtlistener.com/recap/gov.uscourts.nysd.612697/gov.uscourts.nysd.612697.379.13.pdf)
- [IF4IT: Govern AI interaction, output and evidence retention](https://if4it.org/best-practices/enterprise-ai-governance-best-practices/govern-ai-interaction-output-and-evidence-retention/)
- [ARCS, Automated Record Custody Standard](https://arcsstandard.org/full-standard/)

## Recommended

- [Context Window Management: Why Retrieval Beats Bigger Context](https://mlflow.org/articles/context-window-management)
