# Tracing LangGraph🦜🕸️

[](/mlflow-website/docs/latest/images/llms/tracing/langgraph-tracing.mp4)

[LangGraph](https://www.langchain.com/langgraph) is an open-source library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows.

[MLflow Tracing](/mlflow-website/docs/latest/genai/tracing.md) provides automatic tracing capability for LangGraph, as a extension of its LangChain integration. By enabling auto-tracing for LangChain by calling the [`mlflow.langchain.autolog()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.langchain.html#mlflow.langchain.autolog) function, MLflow will automatically capture the graph execution into a trace and log it to the active MLflow Experiment. In TypeScript, you can pass the MLflow LangChain callback to the `callbacks` option.

* Python
* JS / TS

python

```python
import mlflow

mlflow.langchain.autolog()

```

LangGraph.js tracing is supported via the OpenTelemetry ingestion. See the [Getting Started section](#getting-started) below for the full setup.

## Getting Started[​](#getting-started "Direct link to Getting Started")

MLflow support tracing for LangGraph in both Python and TypeScript/JavaScript. Please select the appropriate tab below to get started.

* Python
* JS / TS

### 1. Start MLflow[​](#1-start-mlflow "Direct link to 1. Start MLflow")

Start the MLflow server following the [Self-Hosting Guide](/mlflow-website/docs/latest/self-hosting.md), if you don't have one already.

### 2. Install dependencies[​](#2-install-dependencies "Direct link to 2. Install dependencies")

bash

```bash
pip install langgraph langchain-openai 'mlflow[genai]'

```

### 3. Enable tracing[​](#3-enable-tracing "Direct link to 3. Enable tracing")

python

```python
import mlflow

# Calling autolog for LangChain will enable trace logging.
mlflow.langchain.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_experiment("LangChain")
mlflow.set_tracking_uri("http://localhost:5000")

```

### 4. Define the LangGraph agent and invoke it[​](#4-define-the-langgraph-agent-and-invoke-it "Direct link to 4. Define the LangGraph agent and invoke it")

python

```python
from typing import Literal

import mlflow

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Enabling tracing for LangGraph (LangChain)
mlflow.langchain.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LangGraph")


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"


llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_weather]
graph = create_react_agent(llm, tools)

# Invoke the graph
result = graph.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf?"}]}
)

```

### 5. View the trace in the MLflow UI[​](#5-view-the-trace-in-the-mlflow-ui "Direct link to 5. View the trace in the MLflow UI")

Visit `http://localhost:5000` (or your custom MLflow tracking server URL) to view the trace in the MLflow UI.

### 1. Start MLflow[​](#1-start-mlflow-1 "Direct link to 1. Start MLflow")

Start the MLflow server following the [Self-Hosting Guide](/mlflow-website/docs/latest/self-hosting.md), if you don't have one already.

### 2. Install the required dependencies:[​](#2-install-the-required-dependencies "Direct link to 2. Install the required dependencies:")

bash

```bash
npm i @langchain/langgraph @langchain/core @langchain/openai @arizeai/openinference-instrumentation-langchain

```

### 3. Enable OpenTelemetry[​](#3-enable-opentelemetry "Direct link to 3. Enable OpenTelemetry")

Enable OpenTelemetry instrumentation for LangChain in your application:

typescript

```typescript
import { NodeTracerProvider, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { LangChainInstrumentation } from "@arizeai/openinference-instrumentation-langchain";
import * as CallbackManagerModule from "@langchain/core/callbacks/manager";

// Set up the OpenTelemetry
const provider = new NodeTracerProvider(
  {
    spanProcessors: [new SimpleSpanProcessor(new OTLPTraceExporter({
      // Set MLflow tracking server URL with `/v1/traces` path. You can also use the OTEL_EXPORTER_OTLP_TRACES_ENDPOINT environment variable instead.
      url: "http://localhost:5000/v1/traces",
      // Set the experiment ID in the header. You can also use the OTEL_EXPORTER_OTLP_TRACES_HEADERS environment variable instead.
      headers: {
        "x-mlflow-experiment-id": "123",
      },
    }))],
  }
);
provider.register();

// Enable LangChain instrumentation
const lcInstrumentation = new LangChainInstrumentation();
lcInstrumentation.manuallyInstrument(CallbackManagerModule);

```

### 4. Define the LangGraph agent and invoke it[​](#4-define-the-langgraph-agent-and-invoke-it-1 "Direct link to 4. Define the LangGraph agent and invoke it")

Define the LangGraph agent following the [LangGraph example](https://docs.langchain.com/oss/javascript/langgraph/quickstart#full-code-example) and invoke it.

### 5. View the trace in the MLflow UI[​](#5-view-the-trace-in-the-mlflow-ui-1 "Direct link to 5. View the trace in the MLflow UI")

Visit `http://localhost:5000` (or your custom MLflow tracking server URL) to view the trace in the MLflow UI.

## Token Usage Tracking[​](#token-usage-tracking "Direct link to Token Usage Tracking")

MLflow >= 3.1.0 supports token usage tracking for LangGraph. The token usage for each LLM call during a graph invocation will be logged in the `mlflow.chat.tokenUsage` span attribute, and the total usage in the entire trace will be logged in the `mlflow.trace.tokenUsage` metadata field.

python

```python
import json
import mlflow

mlflow.langchain.autolog()

# Execute the agent graph defined in the previous example
graph.invoke({"messages": [{"role": "user", "content": "what is the weather in sf?"}]})

# Get the trace object just created
last_trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id=last_trace_id)

# Print the token usage
total_usage = trace.info.token_usage
print("== Total token usage: ==")
print(f"  Input tokens: {total_usage['input_tokens']}")
print(f"  Output tokens: {total_usage['output_tokens']}")
print(f"  Total tokens: {total_usage['total_tokens']}")

# Print the token usage for each LLM call
print("\n== Token usage for each LLM call: ==")
for span in trace.data.spans:
    if usage := span.get_attribute("mlflow.chat.tokenUsage"):
        print(f"{span.name}:")
        print(f"  Input tokens: {usage['input_tokens']}")
        print(f"  Output tokens: {usage['output_tokens']}")
        print(f"  Total tokens: {usage['total_tokens']}")

```

bash

```bash
== Total token usage: ==
  Input tokens: 149
  Output tokens: 135
  Total tokens: 284

== Token usage for each LLM call: ==
ChatOpenAI_1:
  Input tokens: 58
  Output tokens: 87
  Total tokens: 145
ChatOpenAI_2:
  Input tokens: 91
  Output tokens: 48
  Total tokens: 139

```

## Adding spans within a node or a tool[​](#adding-spans-within-a-node-or-a-tool "Direct link to Adding spans within a node or a tool")

By combining auto-tracing with the [manual tracing APIs](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md), you can add child spans inside a node or tool, to get more detailed insights for the step.

Let's take LangGraph's [Code Assistant](https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/#graph) tutorial for example. The `check_code` node actually consists of two different validations for the generated code. You may want to add span for each validation to see which validation were executed. To do so, simply create manual spans inside the node function.

python

```python
def code_check(state: GraphState):
    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        # Create a child span manually with mlflow.start_span() API
        with mlflow.start_span(name="import_check", span_type=SpanType.TOOL) as span:
            span.set_inputs(imports)
            exec(imports)
            span.set_outputs("ok")
    except Exception as e:
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check execution
    try:
        code = imports + "\n" + code
        with mlflow.start_span(name="execution_check", span_type=SpanType.TOOL) as span:
            span.set_inputs(code)
            exec(code)
            span.set_outputs("ok")
    except Exception as e:
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # No errors
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }

```

This way, the span for the `check_code` node will have child spans, which record whether the each validation fails or not, with their exception details.

![LangGraph Child Span](/mlflow-website/docs/latest/assets/images/langgraph-child-span-076b0cb599aeabce965b36602d5fda82.png)

Async Context Propagation

When using async methods like `ainvoke()` with manual `@mlflow.trace` decorators inside LangGraph nodes or tools, enable inline tracer execution to ensure proper context propagation:

python

```python
mlflow.langchain.autolog(run_tracer_inline=True)

```

This ensures that manually traced spans are properly nested within the autolog trace hierarchy. Without this setting, manual spans may appear as separate traces in async scenarios.

warning

When `run_tracer_inline=True` is enabled, avoid calling multiple graph invocations sequentially within the same async function, as this may cause traces to merge unexpectedly. If you need to make multiple sequential invocations, either:

* Wrap each invocation in a separate async task
* Use the default `run_tracer_inline=False` if you don't need manual tracing integration

## Thread ID Tracking[​](#thread-id-tracking "Direct link to Thread ID Tracking")

Since MLflow 3.6, MLflow will automatically record the thread (session) ID for the trace and let you view a group of traces as a session in the UI. To enable this feature, you need to pass the `thread_id` in the config when invoking the graph.

python

```python
graph.invoke(inputs, {"configurable": {"thread_id": "1"}})

```

The thread ID will be recorded in the trace metadata and displayed in the MLflow Trace UI.

![LangGraph Thread ID](/mlflow-website/docs/latest/assets/images/langgraph-thread-id-f180f62c969e783d2a312a5d69ccd2fd.png)

By navigating to the Session tab on the side bar, you can view all the traces in the session.

![LangGraph Session Page](/mlflow-website/docs/latest/assets/images/langgraph-session-page-c27d68787ae5bbefa08c8bb485273697.png)

## Disable auto-tracing[​](#disable-auto-tracing "Direct link to Disable auto-tracing")

Auto tracing for LangGraph can be disabled globally by calling `mlflow.langchain.autolog(disable=True)` or `mlflow.autolog(disable=True)`.
