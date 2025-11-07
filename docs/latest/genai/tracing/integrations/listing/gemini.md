# Tracing Gemini

![OpenAI Tracing via autolog](/mlflow-website/docs/latest/assets/images/gemini-tracing-fd066260b1e75fa44bfdd89dd000f76d.png)

[MLflow Tracing](/mlflow-website/docs/latest/genai/tracing/integrations.md) provides automatic tracing capability for Google Gemini. By enabling auto tracing for Gemini by calling the [`mlflow.gemini.autolog()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.gemini.html#mlflow.gemini.autolog) function, MLflow will capture nested traces and log them to the active MLflow Experiment upon invocation of Gemini Python SDK. In Typescript, you can instead use the `tracedGemini` function to wrap the Gemini client.

* Python
* JS / TS

python

```
import mlflow

mlflow.gemini.autolog()
```

typescript

```
import { GoogleGenAI } from "@google/genai";
import { tracedGemini } from "mlflow-gemini";

const client = tracedGemini(new GoogleGenAI());
```

note

Current MLflow tracing integration supports both new [Google GenAI SDK](https://github.com/googleapis/python-genai) and legacy [Google AI Python SDK](https://github.com/google-gemini/generative-ai-python). However, it may drop support for the legacy package without notice, and it is highly recommended to migrate your use cases to the new Google GenAI SDK.

MLflow trace automatically captures the following information about Gemini calls:

* Prompts and completion responses
* Latencies
* Model name
* Additional metadata such as `temperature`, `max_tokens`, if specified.
* Token usage (input, output, and total tokens)
* Function calling if returned in the response
* Any exception if raised

## Supported APIs[​](#supported-apis "Direct link to Supported APIs")

MLflow supports automatic tracing for the following Gemini APIs:

### Python[​](#python "Direct link to Python")

| Text Generation | Chat | Function Calling | Streaming | Async    | Image | Video |
| --------------- | ---- | ---------------- | --------- | -------- | ----- | ----- |
| ✅              | ✅   | ✅               | -         | ✅ (\*1) | -     | -     |

(\*1) Async support was added in MLflow 3.2.0.

### TypeScript / JavaScript[​](#typescript--javascript "Direct link to TypeScript / JavaScript")

| Content Generation | Chat | Function Calling | Streaming | Async |
| ------------------ | ---- | ---------------- | --------- | ----- |
| ✅                 | -    | ✅ (\*2)         | -         | ✅    |

(\*2) Only `models.generateContent()` is supported. Function calls in responses are captured and can be rendered in the MLflow UI. The TypeScript SDK is natively async.

To request support for additional APIs, please open a [feature request](https://github.com/mlflow/mlflow/issues) on GitHub.

## Basic Example[​](#basic-example "Direct link to Basic Example")

* Python
* JS / TS

python

```
import mlflow
import google.genai as genai
import os

# Turn on auto tracing for Gemini
mlflow.gemini.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Gemini")


# Configure the SDK with your API key.
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Use the generate_content method to generate responses to your prompts.
response = client.models.generate_content(
    model="gemini-1.5-flash", contents="The opposite of hot is"
)
```

typescript

```
import { GoogleGenAI } from "@google/genai";
import { tracedGemini } from "mlflow-gemini";

const client = tracedGemini(new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }));

const response = await client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "What is the capital of France?"
});
```

## Multi-turn chat interactions[​](#multi-turn-chat-interactions "Direct link to Multi-turn chat interactions")

MLflow support tracing multi-turn conversations with Gemini:

python

```
import mlflow

mlflow.gemini.autolog()

chat = client.chats.create(model="gemini-1.5-flash")
response = chat.send_message(
    "In one sentence, explain how a computer works to a young child."
)
print(response.text)
response = chat.send_message(
    "Okay, how about a more detailed explanation to a high schooler?"
)
print(response.text)
```

## Async[​](#async "Direct link to Async")

MLflow Tracing supports asynchronous API of the Gemini SDK since MLflow 3.2.0. The usage is same as the synchronous API.

* Python
* JS / TS

python

```
# Configure the SDK with your API key.
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Async API is invoked through the `aio` namespace.
response = await client.aio.models.generate_content(
    model="gemini-1.5-flash", contents="The opposite of hot is"
)
```

Gemini Typescript / Javascript SDK is natively async. See the basic example above.

## Embeddings[​](#embeddings "Direct link to Embeddings")

MLflow Tracing for Gemini SDK supports embeddings API (Python only):

python

```
result = client.models.embed_content(model="text-embedding-004", contents="Hello world")
```

## Token usage[​](#token-usage "Direct link to Token usage")

MLflow >= 3.4.0 supports token usage tracking for Gemini. The token usage for each LLM call will be logged in the `mlflow.chat.tokenUsage` attribute. The total token usage throughout the trace will be available in the `token_usage` field of the trace info object.

* Python
* JS / TS

python

```
import json
import mlflow

mlflow.gemini.autolog()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Use the generate_content method to generate responses to your prompts.
response = client.models.generate_content(
    model="gemini-1.5-flash", contents="The opposite of hot is"
)

# Get the trace object just created
trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

# Print the token usage
total_usage = trace.info.token_usage
print("== Total token usage: ==")
print(f"  Input tokens: {total_usage['input_tokens']}")
print(f"  Output tokens: {total_usage['output_tokens']}")
print(f"  Total tokens: {total_usage['total_tokens']}")

# Print the token usage for each LLM call
print("\n== Detailed usage for each LLM call: ==")
for span in trace.data.spans:
    if usage := span.get_attribute("mlflow.chat.tokenUsage"):
        print(f"{span.name}:")
        print(f"  Input tokens: {usage['input_tokens']}")
        print(f"  Output tokens: {usage['output_tokens']}")
        print(f"  Total tokens: {usage['total_tokens']}")
```

typescript

```
import * as mlflow from "mlflow-tracing";

// After your Gemini call completes, flush and fetch the trace
await mlflow.flushTraces();
const lastTraceId = mlflow.getLastActiveTraceId();

if (lastTraceId) {
  const client = new mlflow.MlflowClient({ trackingUri: "http://localhost:5000" });
  const trace = await client.getTrace(lastTraceId);

  // Total token usage on the trace
  console.log("== Total token usage: ==");
  console.log(trace.info.tokenUsage); // { input_tokens, output_tokens, total_tokens }

  // Per-span usage (if provided by the provider)
  console.log("\n== Detailed usage for each LLM call: ==");
  for (const span of trace.data.spans) {
    const usage = span.attributes?.["mlflow.chat.tokenUsage"];
    if (usage) {
      console.log(`${span.name}:`, usage);
    }
  }
}
```

bash

```
== Total token usage: ==
  Input tokens: 5
  Output tokens: 2
  Total tokens: 7

== Detailed usage for each LLM call: ==
Models.generate_content:
  Input tokens: 5
  Output tokens: 2
  Total tokens: 7
Models._generate_content:
  Input tokens: 5
  Output tokens: 2
  Total tokens: 7
```

Token usage tracking is supported for both Python and TypeScript/JavaScript implementations.

### Disable auto-tracing[​](#disable-auto-tracing "Direct link to Disable auto-tracing")

Auto tracing for Gemini can be disabled globally by calling `mlflow.gemini.autolog(disable=True)` or `mlflow.autolog(disable=True)`.
