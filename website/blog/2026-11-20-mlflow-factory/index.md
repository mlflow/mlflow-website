---
title: "From Bug Report to Pull Request in 10 Minutes: How We Built and Monitor MLflow's Factory"
description: "We walk you through how MLflow turns a bug report into a reviewed pull request in about ten minutes: the agents, the sandbox that keeps them from writing anything, and how we monitor every run with MLflow Tracing."
slug: mlflow-factory
authors: [mlflow-maintainers]
tags: [mlflow, agents, genai, open-source, observability, databricks]
thumbnail: /img/blog/mlflow-factory-thumbnail.png
image: /img/blog/mlflow-factory-thumbnail.png
---

Before a maintainer can fix a bug, someone has to reproduce it, and on a busy open source project that is where reports wait. Until this October, only 22% of MLflow bug reports were closed within two days.

Today, when you open a bug report on [mlflow/mlflow](https://github.com/mlflow/mlflow), an agent reproduces it, finds the root cause and posts a report, usually within ten minutes. If the fix is clear, a pull request with a regression test follows. Other agents review that pull request, answer review comments and repair CI, and a maintainer merges it. 66% of bug reports now close within two days.

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src={require("./images/triage-report.png").default} alt="The triage agent's report on a bug report: a summary that says the bug was reproduced, and a before and after diagram." style={{ width: "100%", maxWidth: 832 }} />
</div>

Together, these agents form MLflow's factory. In this post we walk you through:

- **How it is built:** five small agents on GitHub Actions that hand work to each other through issues and pull requests.
- **How we keep it safe:** an agent with a shell reads text written by strangers, and it still cannot write anything.
- **How we monitor it:** every run exports a trace to [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/), which tells us what the agent did and what it cost.
- **What the numbers look like** after six weeks, and where they flatter us.

Everything is public, so you can copy any part of it. All numbers come from the GitHub API as of November 20, 2026.

## What happens when you open an issue

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src={require("./images/flow.png").default} alt="A new issue is screened, then a sandboxed triage agent produces a report and a patch. A publish job posts the report on the issue and opens a pull request. Three agents review, respond and repair on the pull request, and a maintainer merges." style={{ width: "100%", maxWidth: 832 }} />
</div>

1. **Within two minutes**, a comment tells you an agent has picked up the issue. The median is 1.4 minutes.
2. **About nine minutes in**, that comment becomes the report: a verdict, expected and actual behavior, steps to reproduce, the root cause with links to the code, and a suggested fix.
3. **If the bug reproduced and the fix is clear**, the fix arrives as a pull request with a regression test. A fix that needs a design decision stays in the report as a suggestion.
4. **On the pull request**, one agent reviews the diff. When a maintainer leaves comments, a second agent addresses each one and replies. A third watches CI and fixes failures.
5. **A maintainer merges it.** The median is 8.5 hours after the pull request opened.

Here is the pull request that followed the report above:

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src={require("./images/triage-pr.png").default} alt="A merged pull request opened by the triage agent. The description links the issue it closes and explains the root cause and the fix." style={{ width: "100%", maxWidth: 832 }} />
</div>

## How it is built: five agents that hand off through GitHub

The factory is five small agents, each with one job. Each agent is a separate GitHub Actions workflow with a short instruction file and a schema for its output. Most of them are [Claude Code](https://claude.com/product/claude-code) sessions running non-interactively, and the ones that read untrusted input run inside Anthropic's [sandbox runtime](https://github.com/anthropics/sandbox-runtime).

The agents never talk to each other directly. They hand off through GitHub: Triage opens a pull request and requests a review, which starts Review. A maintainer's comments start Resolve. Failing CI brings in Repair. Every hand-off is an issue, a pull request or a comment, so anyone can read it and a maintainer can step in at any point.

| Agent   | Runs when                                | Agent does                                            | Workflow publishes                                  |
| ------- | ---------------------------------------- | ----------------------------------------------------- | --------------------------------------------------- |
| Triage  | An issue is opened, or a maintainer asks | Reproduces, finds root cause, optionally writes a fix | Label, comment, optional pull request               |
| Review  | A review is requested on a pull request  | Reviews the diff                                      | Review comments                                     |
| Resolve | A maintainer reviews an agent PR         | Addresses each comment, writes a patch                | Commit, replies, resolved threads                   |
| Repair  | CI fails on a PR that opted in           | Diagnoses the failure                                 | Rerun, merge base, or fix commit                    |
| Nightly | Daily                                    | Groups CI failures by root cause, checks upstream     | One issue per new cause, which Triage then picks up |

The last two columns are separate on purpose. The agent only produces a result, and the workflow around it decides whether to publish that result. The next section explains why.

Every model call goes through a gateway, and every run exports a trace. We use Databricks for both, with [Unity Gateway](https://www.databricks.com/product/artificial-intelligence/unity-gateway) and managed MLflow. Open source MLflow does the same job with its [AI Gateway](https://mlflow.org/docs/latest/genai/governance/ai-gateway/) and [Tracing](https://mlflow.org/docs/latest/genai/tracing/).

## Security: the agent cannot write anything

Anyone can write an issue, and the agent that reads it has a shell. Prompt injection is the obvious risk. We assume the agent can be talked into anything, and make sure it has nothing useful to do with that.

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src={require("./images/trust-boundary.png").default} alt="Issue text is screened and handed to a triage agent in a sandbox that holds a read-only token and an allowlisted network. The agent's only output is a report and a patch as files. A separate publish job validates the files and is the only place that holds the write token. The agent cannot push, comment or call other hosts." style={{ width: "100%", maxWidth: 832 }} />
</div>

The agent that reads the issue runs in a sandbox and can only read. Its only output is two files: a report and an optional patch. A separate job, which never runs a model, validates those files, and it is the only job that can comment or push.

Here is what someone who controls the issue text might try, and what stops each attempt:

| An injected issue tries to                                     | What stops it                                                                  |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Hide instructions in the issue                                 | Hidden text is stripped, and a separate model screens the issue before any run |
| Push code or post comments                                     | The agent's GitHub token is read-only                                          |
| Send code or secrets to another server                         | The sandbox only allows network access to a short list of hosts                |
| Steal the model credential                                     | A proxy outside the sandbox holds it, and the agent only has a placeholder     |
| Change CI, dependencies or the agent's own instructions        | The publish job rejects any patch that touches more than source, tests or docs |
| Get a harmful fix merged                                       | A maintainer reviews and merges every pull request                             |
| Flood the factory with issues, or run up the bill with one run | Each author is rate limited, and each run has a cost cap and a time limit      |

There is also a kill switch, and a dry-run mode for testing changes to the factory itself.

## Monitoring: watching the agents with MLflow Tracing

An unattended agent that comments on public issues needs the same observability as any production service. We get it from [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/). Every run exports a trace, so we can open one run and read it step by step, or query all of them with SQL.

### Collecting traces from Claude Code

Claude Code can export OpenTelemetry traces, and MLflow accepts OpenTelemetry on its `/v1/traces` endpoint. Collecting traces takes a few environment variables on the step that runs the agent, and no code changes:

```yaml
- name: Run the agent
  env:
    CLAUDE_CODE_ENABLE_TELEMETRY: "1"
    # Export spans, not only logs and metrics
    CLAUDE_CODE_ENHANCED_TELEMETRY_BETA: "1"
    OTEL_TRACES_EXPORTER: otlp
    OTEL_EXPORTER_OTLP_PROTOCOL: http/protobuf
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: https://<your-mlflow-server>/v1/traces
    OTEL_EXPORTER_OTLP_TRACES_HEADERS: x-mlflow-experiment-id=<experiment-id>
    # Attach the issue and the workflow run to every span
    OTEL_RESOURCE_ATTRIBUTES: github.issue_url=${{ github.event.issue.html_url }},github.run_url=${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
  run: claude -p "$PROMPT"
```

In our workflows the endpoint is a small local proxy that adds the credential, for the same reason as the model proxy: nothing inside the sandbox should be worth stealing.

If you use Claude Code on your own machine, `mlflow autolog claude` sets up tracing for a project in one command. See [Tracing Claude Code](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/claude_code/) for that, and [Collect OpenTelemetry Traces into MLflow](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/ingest/) for any other agent that speaks OpenTelemetry.

### What we do with the traces

Each trace links back to its issue and workflow run, and is tagged with the verdict, the model, the effort and whether a pull request was proposed.

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src={require("./images/monitoring.png").default} alt="Each agent run exports traces to MLflow Tracing. The traces are used to debug a bad run, compare models, find waste and track cost per agent." style={{ width: "100%", maxWidth: 832 }} />
</div>

- **Debug a bad triage.** Open the trace from the issue and read the tool calls in order.
- **Compare models and settings.** Filter traces by verdict, model or effort.
- **Track cost.** A dashboard built on the traces shows spend per agent, per model and per token type.
- **Find waste.** Traces show spend that the agent's own cost estimate leaves out, such as the permission classifier in the chart below.

The dashboard needs nothing beyond the exported traces. This is the view for the triage agent:

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src={require("./images/cost-dashboard.png").default} alt="A cost dashboard named MLflow Factory with one tab per agent. The triage tab shows tiles for runs, total cost and average cost per run over 30 days, and a stacked column chart of cost by caller and token type." style={{ width: "100%", maxWidth: 832 }} />
</div>

Reading traces one at a time does not scale. For patterns across many runs, MLflow has [automatic issue detection](https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues/): it analyzes the traces in an experiment and groups recurring problems, such as an agent that repeats the same search, into issues you can review.

## Results

Three numbers first:

- **231 bug reports triaged** since October 6, 216 of them reproduced.
- **131 of 143 agent pull requests merged** since October 15, with 6 closed and 6 still open.
- **Bugs closed within 2 days: 22% before, 66% after.**

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src={require("./images/weekly-prs.png").default} alt="Stacked columns of agent pull requests opened per week from mid October to mid November 2026. Weekly totals are 9, 17, 31, 30, 26 and 30, and most are merged." style={{ width: "100%", maxWidth: 832 }} />
</div>

The pull requests are small: a median of 49 changed lines across 2 files. Most merge quickly.

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src={require("./images/time-to-merge.png").default} alt="Columns of time from pull request opened to merged for 131 merged agent PRs: 37 under 2 hours, 26 in 2 to 8 hours, 43 in 8 to 24 hours, 20 in 1 to 2 days, and 5 over 2 days." style={{ width: "100%", maxWidth: 832 }} />
</div>

Bug reports close faster than they did before triage. The median time to close went from 4.4 days to 0.8 days.

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src={require("./images/close-rate.png").default} alt="Share of bug reports closed within 2 days rose from 22 percent to 66 percent, and within 7 days from 57 percent to 80 percent." style={{ width: "100%", maxWidth: 832 }} />
</div>

The triage agent now writes a real share of MLflow's fixes: 131 of the 755 pull requests merged since October 15, or 17%.

## Caveats

These numbers do not come from a controlled experiment. If you build something similar, these are the gaps to watch for:

- **The before and after groups are different.** Bug reports went from 67 in September to 170 in October. Filing an issue became the cheapest way to get a fix, so maintainers and regular contributors file more of them: 146 of the 231 triaged reports came from members, collaborators or prior contributors.
- **We have not audited the verdicts.** The agent reproduced 94% of reports and asked for more information only once. Either the reports are good or the agent is too agreeable, and we do not know which yet.
- **Human review is light.** 112 of the 131 merged pull requests had no human review comment or change request. They are small, but that is a lot of trust.
- **The backlog is untouched.** Triage only sees new issues.

## What's next

- **Merging without a human**, for the pull requests that qualify. Today a maintainer merges every change, and that merge is one of the safeguards above, so this needs a clear rule for which pull requests qualify and what replaces the human check.
- **A backlog pass**, modelled on the [Next.js close queue](https://nextjs.org/blog/how-we-closed-1500-github-issues).
- **Measuring quality:** revert and reopen rates, and sampled audits of verdicts.

## You can clone what we do

Everything is in the [public repository](https://github.com/mlflow/mlflow): the workflows, the agent instructions and the sandbox configuration. The building blocks are open source too:

- [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/) and its [Claude Code integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/claude_code/), to see what your agents do.
- [OpenTelemetry ingestion](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/ingest/), for agents that already export OpenTelemetry.
- [Token usage and cost tracking](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/), to see what they cost.
- [MLflow AI Gateway](https://mlflow.org/docs/latest/genai/governance/ai-gateway/), to keep model keys and budgets in one place.
- [Automatic issue detection](https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues/), to find recurring problems across traces.

If you open an issue on [mlflow/mlflow](https://github.com/mlflow/mlflow/issues/new/choose), you will meet the factory within a couple of minutes.
