---
title: "How to Trace AI Agents and MCP Tool Calls End to End"
description: "Trace an AI agent end to end with MLflow: capture LLM calls, MCP tool calls, sessions, and cross-service hops as one OpenTelemetry-compatible trace."
slug: trace-ai-agents-and-mcp-tool-calls-end-to-end
tags:
  [
    AI agent tracing,
    MCP tracing,
    trace MCP tool calls,
    agent observability,
    end-to-end tracing,
    OpenTelemetry,
    MLflow Tracing,
    debug AI agents,
  ]
date: 2026-11-05
---

To trace an AI agent end to end, record every step of one run as a span inside a single trace: the user request, each LLM call, each tool or MCP call, and the final answer. Then group the traces by session. With MLflow Tracing, that takes a tracking server, one autolog call for your framework, and one decorator around each MCP tool call.

This guide is for engineers whose agent works in a demo and does something odd in production, and who need to see what the agent actually did. We build MLflow, so the examples use MLflow Tracing. The steps apply to any OpenTelemetry-compatible tracer.

**Disclosure:** the MLflow team wrote this guide.

## TL;DR

An agent trace is a tree of spans, one per step, and you add the MCP step yourself.

- An agent run is a tree: one root `AGENT` span, with `LLM` and `TOOL` spans below it. A trace is that whole tree.
- Frameworks such as the OpenAI Agents SDK, LangGraph, and the Claude Agent SDK can be traced with one `autolog()` call. Hand-written agent loops need the `@mlflow.trace` decorator.
- An MCP tool call is a JSON-RPC `tools/call` request. Plain MCP client code does not record it as a span, so wrap the call in a `TOOL` span (or check that your framework does it for you, as PydanticAI does).
- Use a `session_id` to join the traces of a multi-turn conversation. Use W3C trace-context headers over HTTP to join traces across services.
- Everything here is OpenTelemetry-compatible, so you can send the same traces to another backend.

## What does "end to end" mean for an AI agent?

End to end means that one trace shows the full path from the user's request to the final answer, including every model call and every tool call in between. If a step is missing from the trace, you cannot say why the agent chose a wrong tool or returned a wrong answer.

A normal web request is mostly deterministic. An agent is not. The model decides at run time which tool to call, with which arguments, and how many times. Two runs of the same prompt can take different paths. That is why logs of the final answer do not help much. You need the path.

MLflow's [span model](https://mlflow.org/docs/latest/genai/concepts/span/) is built for this. A span is one step. Spans nest in a tree, and the tree is the trace. MLflow defines span types such as `AGENT` (an autonomous agent operation), `LLM`, `RETRIEVER`, and `TOOL` (a tool execution, typically by an agent). The span object is compatible with the OpenTelemetry span spec and is exported as standard OTLP.

Here is the shape of one traced run in this guide (an illustrative example):

```text
trace: "Where is order A-1001?"
└── run_agent            (AGENT)   whole run
    ├── chat completion  (LLM)     model picks get_order_status
    ├── call_mcp_tool    (TOOL)    tools/call get_order_status
    └── chat completion  (LLM)     model writes the answer
```

The LLM spans in this diagram come from autologging a provider SDK. The `AGENT` and `TOOL` spans are the two you add yourself for a hand-written loop. The code below produces the `AGENT` and `TOOL` pair; we ran it and checked the span tree (see Step 3).

## Who this guide is for

This guide is for you if you build an agent that calls tools and want to answer three questions from a trace: which tool did the agent call, what did it return, and where did the time go.

It works for framework agents and hand-written loops. It also covers MCP servers, because the Model Context Protocol (MCP) is an open standard for connecting agents to tools. If your agent has no tools, a plain LLM trace is enough, and the [guide to open-source LLM tracing tools](https://mlflow.org/articles/best-open-source-llm-tracing-tools-2026/) is a better start.

## Why MLflow for agent and MCP tracing

MLflow is a good fit for this job if you want tracing, evaluation, and storage in one Apache-2.0 server, and you want your traces in a vendor-neutral format. MLflow is [backed by the Linux Foundation](https://mlflow.org/). MLflow Tracing is OpenTelemetry-native: the tracking server accepts traces over OTLP at `/v1/traces` from any language, and traces made with the MLflow SDK can be exported to other OpenTelemetry backends.

That is not the same as "best for every setup". If you only need instrumentation and already run an observability backend, a lighter library may be enough. If your agent is not written in Python or TypeScript, you can still send traces over OTLP, but you lose the one-line autologging. The [MLflow tracing overview](https://mlflow.org/llm-tracing/) lists what is supported.

## How do you trace an AI agent end to end? Six steps

Follow these steps in order. Each step adds one kind of visibility and works on its own.

### Step 1: Run a tracking server

The server stores traces and shows them in the trace UI. For local work, start one with:

```bash
pip install mlflow
mlflow server --port 5000
```

In your agent code, point MLflow at it and name an experiment. Set the experiment before you turn on tracing, so every trace lands in the same place:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-agent")
```

For a quick test without a server, a local SQLite URI such as `sqlite:///mlflow.db` also works. We ran the example in Step 3 against a SQLite URI; swap it in for the server URL there to run without a server.

### Step 2: Turn on autologging for your agent framework

If you use a supported framework, one call traces it. Each framework has its own autolog function:

| Framework                 | Call                           | Note                                                                                                                                                                                               |
| ------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OpenAI Agents SDK         | `mlflow.openai.autolog()`      | Captures agent handoffs and the calls to the OpenAI API ([docs](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai-agent/))                                                  |
| LangGraph                 | `mlflow.langchain.autolog()`   | LangGraph tracing extends the LangChain integration ([docs](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph/))                                                         |
| Claude Agent SDK (Python) | `mlflow.anthropic.autolog()`   | Needs MLflow 3.5 or newer. Traces `ClaudeSDKClient`; a direct call to `query()` is not traced ([docs](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/claude_agent_sdk_python/)) |
| PydanticAI                | `mlflow.pydantic_ai.autolog()` | Also captures MCP `call_tool` and `list_tools` ([docs](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/pydantic_ai/))                                                            |

Call the autolog function once, before you create the agent or client. After that, each agent run produces a trace with its LLM calls and tool calls.

PydanticAI deserves a note. Its integration records MCP server interactions, including `call_tool` and `list_tools`, as individual spans in the trace UI. If you use PydanticAI with MCP, you may not need Step 3 at all. Check one trace first.

The full list of integrations is in the [MLflow tracing docs](https://mlflow.org/docs/latest/genai/tracing/integrations/). If your framework is not listed, go to Step 3.

### Step 3: Trace MCP tool calls

An MCP client calls a tool by sending a `tools/call` request to the server, as the [MCP specification](https://modelcontextprotocol.io/specification/latest/server/tools) describes. The spec defines two standard transports, `stdio` and Streamable HTTP, and the call looks the same to your code on both. If your client library or framework does not trace that call, the trace shows a gap: the model asked for a tool, and then the answer appears, with nothing in between.

Close the gap by wrapping the call in a `TOOL` span. Wrap the whole run in an `AGENT` span. This example uses the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) and a tiny local server.

First, the server:

```python title="server.py"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("orders")


@mcp.tool()
def get_order_status(order_id: str) -> str:
    """Look up the status of an order."""
    return f"Order {order_id} shipped"


if __name__ == "__main__":
    mcp.run()
```

Then the traced client:

```python title="client.py"
import asyncio
import sys
from pathlib import Path

import mlflow
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mlflow.entities import SpanType

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-agent")


@mlflow.trace(span_type=SpanType.TOOL)
async def call_mcp_tool(session: ClientSession, name: str, arguments: dict):
    result = await session.call_tool(name, arguments)
    return result.content[0].text


@mlflow.trace(span_type=SpanType.AGENT)
async def run_agent(question: str) -> str:
    params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name("server.py"))])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # A real agent lets the LLM choose the tool. It is hard-coded here.
            return await call_mcp_tool(
                session, "get_order_status", {"order_id": "A-1001"}
            )


print(asyncio.run(run_agent("Where is order A-1001?")))
```

We ran this with MLflow 3.18.0 and `mcp` 1.30.0. The output was `Order A-1001 shipped`, and the trace held a root `run_agent` span of type `AGENT` with a child `call_mcp_tool` span of type `TOOL`. Because `@mlflow.trace` records a function's inputs and outputs, the tool span shows the tool name, the arguments, and the text the server returned.

Two notes on this code:

- Pin `mcp<2` if you copy it. The `mcp` package moved to a 2.x line in which `FastMCP` was renamed `MCPServer`, so this server code needs 1.x.
- In a real agent, add your LLM provider's autolog call (Step 2). The LLM spans then appear as siblings of the tool span inside the same `AGENT` span, and you can read the whole decision loop in order.

Name the span after the tool, not after your wrapper, when you have many tools. A trace full of `call_mcp_tool` rows is hard to scan. The `@mlflow.trace(name=...)` argument sets the name, and `mlflow.start_span(name=..., span_type=SpanType.TOOL)` does the same when you need a span inside a block instead of a function.

### Step 4: Group traces into sessions

A chat agent produces one trace per turn. To debug a conversation, you need the turns together. MLflow lets you attach a session ID and a user ID to every trace created in a block of code:

```python
import mlflow

mlflow.openai.autolog()

with mlflow.tracing.context(session_id="session-123", user="user-456"):
    agent_turn("What is the status of order A-1001?")  # your agent's entry point
```

The `session_id` and `user` parameters need MLflow 3.11.0 or newer. On older versions, set the `mlflow.trace.session` or `mlflow.trace.user` key in the trace metadata. See [Track Users & Sessions](https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/).

In a web app, wrap the request handler with this context manager. Every trace from that request then carries the user and session without any change to your agent code.

### Step 5: Connect traces across services

Often the MCP server, or a sub-agent, runs in a different process or on a different host. Without extra work, each process makes its own trace, and you must match them by hand.

MLflow's distributed tracing fixes this for HTTP calls. It passes the active trace context in headers that follow the W3C TraceContext standard. The caller, inside an active span, uses `get_tracing_context_headers_for_http_request` to get the headers. The callee uses `set_tracing_context_from_http_request_headers` to read them, and its spans join the same trace. Both services must log to the same tracking server and the same experiment, or the spans end up in separate traces. The [distributed tracing guide](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/distributed-tracing/) has a runnable two-service example.

There is a limit you should know about. That guide shows HTTP services. A `stdio` MCP server has no HTTP headers, so this method does not apply to it. We have not tested header propagation through the MCP Streamable HTTP transport, so we do not claim it works. For a `stdio` server, trace the tool function inside the server with `@mlflow.trace`, log to the same experiment, and tag both traces with the same session ID so you can find them together.

### Step 6: Keep your traces portable with OpenTelemetry

MLflow ingests and exports traces in the OpenTelemetry format, including the OpenTelemetry [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Two things follow:

- An agent written in Java, Go, or Rust can send spans to the MLflow server. Set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to `<tracking-uri>/v1/traces`, and set `OTEL_EXPORTER_OTLP_TRACES_HEADERS` to `x-mlflow-experiment-id=<experiment-id>` so the server knows which experiment gets the spans.
- Traces from the MLflow SDK can go to other OpenTelemetry backends, so you are not locked in.

Read the [OpenTelemetry integration page](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/) for setup, and the [OpenTelemetry tracing announcement](https://mlflow.org/blog/opentelemetry-tracing-support) for background.

## What should you look for in an agent trace?

Open a trace and read the tree from top to bottom. These five patterns cover the agent failures we see most often in traces:

| Pattern in the trace                                        | What it usually means                                                                       |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| A `TOOL` span with an error, followed by a retry loop       | The tool failed and the agent keeps trying. Check the tool's error text in the span output. |
| The same `TOOL` span many times with near-equal inputs      | The agent is stuck. Add a step limit and look at the prompt before the loop.                |
| A long gap with no child span                               | Work happens outside any traced code. Add a span around it.                                 |
| A `TOOL` span with correct output, but a wrong final answer | The model misused the result. Look at the next `LLM` span's input.                          |
| The wrong tool chosen at the first `LLM` span               | The tool descriptions are unclear. Read the tool list the model saw.                        |

Two habits help. First, read the failing run and one passing run of the same input side by side. The first different span is usually the cause. Second, when you fix a bug, keep the failing trace and turn it into a test case. MLflow's evaluation tools can score traces later, which closes the loop between debugging and regression tests.

## What are the common mistakes when tracing agents?

These five mistakes are common.

- **Enabling autolog after the client is created.** Call autolog first, then build the agent or client.
- **Splitting one run across experiments.** Distributed traces need the same tracking server and experiment. Otherwise they do not merge.
- **Tracing only the LLM.** An LLM-only trace shows the model's tool request but not the tool's result. The tool span is where most failures show up.
- **Logging secrets.** Tool inputs and outputs are recorded. If a tool handles passwords, tokens, or personal data, mask them before they reach the trace. MLflow has a [masking guide](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/masking/).
- **Assuming a framework traces MCP.** Some integrations do, as PydanticAI does. Others do not. Run one request and look for the tool span.

## What about other tracing tools?

You can do this with other tools. Langfuse, Arize Phoenix, Opik, OpenLIT, and OpenLLMetry all trace LLM applications, and they differ in license, hosting, and OpenTelemetry support. We compare them on one page: [Best Open-Source LLM Tracing Tools in 2026](https://mlflow.org/articles/best-open-source-llm-tracing-tools-2026/). The idea in the six steps above (one trace per run, one span per step, one session ID per conversation) is not specific to MLflow, but the setup and API names differ by tool, so check each tool's docs.

For tracing multi-agent systems in particular, see our post on [observability for multi-agent systems](https://mlflow.org/blog/observability-multi-agent-part-1).

## Can an agent read its own traces through MCP?

Yes, in a different direction. MLflow also ships an MCP server that exposes trace operations to AI assistants. It can search and retrieve traces, analyze trace performance, log feedback, and manage tags. It is marked experimental and needs MLflow 3.5.1 or newer, installed with `pip install 'mlflow[mcp]>=3.5.1'`. See the [MLflow MCP server docs](https://mlflow.org/docs/latest/genai/mcp/).

This is separate from tracing your own MCP calls. It lets a coding assistant such as Claude Code look at the traces you already collected.

## FAQ

### How do I trace AI agent activity end to end?

Start an `mlflow server`, call your framework's autolog function (for example `mlflow.openai.autolog()` or `mlflow.langchain.autolog()`) before you create the client, and run one request. If the tree has a long gap with no child span, some work runs outside traced code. Wrap that code in `@mlflow.trace` and run the request again.

### How do I trace MCP tool calls?

First, run one request and look for a `TOOL` span under the `LLM` span that asked for the tool. If there is none, your framework does not trace MCP. Then put the client's `call_tool` call inside `mlflow.start_span(name=..., span_type=SpanType.TOOL)`, so the span records the tool name, its arguments, and its result.

### Can I trace an MCP server that runs in another process?

For an HTTP server, yes. The client sends headers from `get_tracing_context_headers_for_http_request`, and the server reads them with `set_tracing_context_from_http_request_headers`. Both sides must log to the same tracking server and experiment, or the spans do not merge. A `stdio` server has no HTTP headers, so trace inside the server and use the same session ID to find its traces.

### Do I need OpenTelemetry to trace an agent?

No. You only need it when spans come from services that do not use the MLflow SDK. Those services set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to `<tracking-uri>/v1/traces` and pass `x-mlflow-experiment-id=<experiment-id>` in `OTEL_EXPORTER_OTLP_TRACES_HEADERS`.

### How do I make agent behavior observable in production?

Attach a user and a session to each trace (the `session_id` and `user` parameters, or the `mlflow.trace.session` and `mlflow.trace.user` metadata keys before MLflow 3.11.0), so you can filter by customer and conversation. Mask passwords, tokens, and personal data before they reach a span. When you fix a bug, keep the failing trace as a test case, so the same failure is caught next time.

## Conclusion

Before tracing, an agent bug is a guess. After tracing, it is a span you can open. Start small: run `mlflow server`, add the autolog call for your framework, and wrap each MCP call in a `TOOL` span. Run one request and read the tree.

Next step: copy the example from Step 3 into a file, run it against a local MLflow server, and open the trace in the UI. The [MLflow Tracing overview](https://mlflow.org/llm-tracing/) shows what to add next.

_Updated 2026-11-05._
