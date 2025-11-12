# Tracing Quickstart (TS/JS)

info

The Python quickstart is available [here](/mlflow-website/docs/latest/genai/tracing/quickstart/python-openai.md).

This quickstart guide will walk you through setting up a simple GenAI application with MLflow Tracing. In less than 10 minutes, you'll enable tracing, run a basic application, and explore the generated traces in the MLflow UI.

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

Install the required packages by running the following command:

bash

```bash
npm install mlflow-openai

```

tip

MLflow's official TypeScript SDK only contains tracing functionality. If you want to use the full experiment tracking experience, you can combine it with [mlflow.js](https://github.com/open-source-labs/mlflow-js), a **community-maintained** package that provides experiment tracking APIs compatible with Python SDK, built on top of MLflow's REST APIs.

info

The code example in this guide uses the OpenAI SDK; however, the contents of this guide are applicable to any other LLM providers, such as Anthropic, Google, Bedrock, and more.

For other LLM providers that are not natively integrated with auto-tracing, install the `mlflow-tracing` package and wrap the LLM call with the `mlflow.trace` wrapper function instead.

Example of wrapping Anthropic API calls to generate a trace

typescript

```typescript
import * as mlflow from "mlflow-tracing";
import Anthropic from '@anthropic-ai/sdk';

// Instantiate the Anthropic client
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// Define a traced function that wraps the Anthropic API call.
const generate = mlflow.trace(
    (prompt: string) => {
        return anthropic.messages.create({
            model: "claude-sonnet-4-latest",
            max_tokens: 1024,
            messages: [{ role: "user", content: prompt }],
        });
    },
    { spanType: mlflow.SpanType.LLM }
);

// Call the wrapped function as usual.
const response = await generate("Hello, Claude");

```

## Step 1: Set up your environment[​](#step-1-set-up-your-environment "Direct link to Step 1: Set up your environment")

### Connect to MLflow[​](#connect-to-mlflow "Direct link to Connect to MLflow")

MLflow logs Traces in a tracking server. Connect your application to the tracking server by one of the following methods.

* Local (pip)
* Local (docker)
* Remote MLflow Server
* Databricks

For the fastest setup, you can install the [mlflow](https://pypi.org/project/mlflow/) Python package and run MLflow locally:

bash

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

```

This will start the server at port 5000 on your local machine. Connect your notebook/IDE to the server by setting the tracking URI. You can also access to the MLflow UI at <http://localhost:5000>.

python

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

```

You can also brows the MLflow UI at <http://localhost:5000>.

MLflow provides a Docker Compose file to start a local MLflow server with a postgres database and a minio server.

bash

```bash
git clone https://github.com/mlflow/mlflow.git
cd docker-compose
cp .env.dev.example .env
docker compose up -d

```

This will start the server at port 5000 on your local machine. Connect your notebook/IDE to the server by setting the tracking URI. You can also access to the MLflow UI at <http://localhost:5000>.

python

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

```

Refer to the [instruction](https://github.com/mlflow/mlflow/tree/master/docker-compose/README.md) for more details, e.g., overriding the default environment variables.

If you have a remote MLflow tracking server, configure the connection:

python

```python
import os
import mlflow

# Set your MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://your-mlflow-server:5000"
# Or directly in code
mlflow.set_tracking_uri("http://your-mlflow-server:5000")

```

If you have a Databricks account, configure the connection:

python

```python
import mlflow

mlflow.login()

```

This will prompt you for your configuration details (Databricks Host url and a PAT).

tip

If you are unsure about how to set up an MLflow tracking server, you can start with the cloud-based MLflow powered by Databricks: [Sign up for free →](https://login.databricks.com/?destination_url=%2Fml%2Fexperiments-signup%3Fsource%3DTRY_MLFLOW\&dbx_source=TRY_MLFLOW\&signup_experience_step=EXPRESS\&provider=MLFLOW\&utm_source=mlflow_org\&tuuid=a9534f33-78bf-4b81-becc-4334e993251d\&rl_aid=e6685d78-9f85-4fed-b64f-08e247f53547\&intent=SIGN_UP)

### Create a new MLflow Experiment[​](#create-a-new-mlflow-experiment "Direct link to Create a new MLflow Experiment")

Create a new MLflow experiment in the MLflow UI, or choose an existing experiment.

![New Experiment](/mlflow-website/docs/latest/images/llms/tracing/quickstart/mlflow-ui-new-experiment.png)

1. Navigate to the MLflow UI in your browser. For example, if you started a local MLflow server at port 5000, you can navigate to <http://127.0.0.1:5000>.
2. Click on the

   Create

   button on the top right.
3. Enter a name for the experiment and click on "Create".

*You can leave the `Artifact Location` field blank for now. It is an advanced configuration to override where MLflow stores experiment data.*

### Initialize the Tracing SDK[​](#initialize-the-tracing-sdk "Direct link to Initialize the Tracing SDK")

* Self Host
* Remote MLflow Server
* Databricks

Call the `init` function with the tracking URI (e.g., `http://127.0.0.1:5000`) and the experiment ID. You can find the experiment ID by hovering over the

!

icon next to the experiment name within the MLflow UI.

typescript

```typescript
import { init } from "mlflow-tracing";

init({
    trackingUri: "<your-tracking-server-uri>",
    experimentId: "<your-experiment-id>",
});

```

If [authentication](/mlflow-website/docs/latest/self-hosting/security/basic-http-auth.md) is enabled on the tracking server, you can pass the credential through `trackingServerUsername` and `trackingServerPassword`.

typescript

```typescript
import { init } from "mlflow-tracing";

init({
    trackingUri: "<your-tracking-server-uri>",
    experimentId: "<your-experiment-id>",
    trackingServerUsername: "<your-tracking-server-username>",
    trackingServerPassword: "<your-tracking-server-password>",
});

```

Call the `init` function with the URI of your remote MLflow server and the experiment ID. You can find the experiment ID by hovering over the

!

icon next to the experiment name within the MLflow UI.

typescript

```typescript
import { init } from "mlflow-tracing";

init({
    trackingUri: "<remote-tracking-server-uri>",
    experimentId: "<your-experiment-id>",
});

```

If [authentication](/mlflow-website/docs/latest/self-hosting/security/basic-http-auth.md) is enabled on the tracking server, you can pass the credential through `trackingServerUsername` and `trackingServerPassword`.

typescript

```typescript
import { init } from "mlflow-tracing";

init({
    trackingUri: "<remote-tracking-server-uri>",
    experimentId: "<your-experiment-id>",
    trackingServerUsername: "<remote-tracking-server-username>",
    trackingServerPassword: "<remote-tracking-server-password>",
});

```

Call the `init` function with the URI of your remote MLflow server and the experiment ID. You can find the experiment ID by hovering over the

!

icon next to the experiment name within the MLflow UI.

**Method 1: Use Databricks config file for authentication.**

typescript

```typescript
import { init } from "mlflow-tracing";

init({
    trackingUri: "databricks",
    experimentId: "<your-experiment-id>",
    // Optional: Set the Databricks config file path if it is not in the default location
    configPath: "<your-databricks-config-file-path>",
});

```

**Method 2: Use environment variables**

bash

```bash
export DATABRICKS_HOST=<your-databricks-host>
export DATABRICKS_TOKEN=<your-databricks-personal-access-token>

```

or create a `.env` file in the root directory of your project and add the following:

bash

```bash
DATABRICKS_HOST=<your-databricks-host>
DATABRICKS_TOKEN=<your-databricks-personal-access-token>

```

typescript

```typescript
// Load environment variables from .env file
import 'dotenv/config';

```

The `init` function will automatically load the above environment variables when tracking URI is set to `databricks`.

typescript

```typescript
import { init } from "mlflow-tracing";

init({
    trackingUri: "databricks",
    experimentId: "<your-experiment-id>",
});

```

### Set OpenAI API Key (or other LLM providers)[​](#set-openai-api-key-or-other-llm-providers "Direct link to Set OpenAI API Key (or other LLM providers)")

bash

```bash
export OPENAI_API_KEY="your-api-key-here"  # Replace with your actual API key

```

## Step 2: Trace a single LLM call[​](#step-2-trace-a-single-llm-call "Direct link to Step 2: Trace a single LLM call")

Let's start with a simple example of tracing a single LLM call. We first wrap the OpenAI client with the `tracedOpenAI` function. After that, every call to OpenAI API will generate a trace span, capturing the input, output, latency, token counts, and other metadata.

typescript

```typescript
import { OpenAI } from "openai";
import { tracedOpenAI } from "mlflow-openai";

// Wrap the OpenAI client with the tracedOpenAI function
const client = tracedOpenAI(new OpenAI());

// Invoke the client as usual
// If you are running this tutorial as a script, remove the `await` keyword.
await client.chat.completions.create({
    model: "o4-mini",
    messages: [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather like in Seattle?"},
    ],
})

```

After running the code above, go to the MLflow UI and select the "Traces" tab. It should show the newly created trace.

![Single Trace](/mlflow-website/docs/latest/images/llms/tracing/quickstart/single-openai-trace-list.png)

The table view shows the primary metadata of the trace, such as the trace ID, execution duration, token count, source system and status. You can add or remove displayed columns by selecting the columns in the drop down.

By clicking on the request row (the linked request text), you can view the detailed spans in the trace.

![Single Trace Detail](/mlflow-website/docs/latest/images/llms/tracing/quickstart/single-openai-trace-detail.png)

The "Chat" view in the above screenshot shows the full chat messages exchanged between the user and the model. By clicking other tables such as "Inputs / Outputs" or "Attributes", you can see different aspects of the trace, including the raw input payload, token usage breakdown, and more.

## Step 3: Trace a tool calling agent[​](#step-3-trace-a-tool-calling-agent "Direct link to Step 3: Trace a tool calling agent")

Next, let's add a bit more complexity to the application. To get the real-time weather information, we will use an external weather API as a tool. The application will include a tool calling flow, not only a simple LLM call. To instrument that custom Python flow, we will use the `mlflow.trace` wrapper function.

typescript

```typescript
import * as mlflow from "mlflow-tracing";

// Wrap the tool function with the `mlflow.trace` wrapper. The wrapped function will be automatically traced and logged to MLflow.
const getWeather = mlflow.trace(
    // A tool function that fetches the weather information from a weather API
    async function getWeather(latitude: number, longitude: number) {
        const response = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m`);
        const data = await response.json();
        return data['current']['temperature_2m'];
    },
     // Set the span type to TOOL. You can also set other span configuration here.
    { spanType: mlflow.SpanType.TOOL }
);

```

To pass the function as a tool to the LLM, we need to define the JSON schema for the function.

typescript

```typescript
const tools = [{
    type: "function",
    function: {
        name: "get_weather" as const,
        description: "Get current temperature for provided coordinates in celsius.",
        parameters: {
            type: "object",
            properties: {
                latitude: { type: "number" },
                longitude: { type: "number" }
            },
            required: ["latitude", "longitude"],
            additionalProperties: false
        },
        strict: true
    }
}];

```

Lastly, define a simple flow that first asks the LLM to get instructions for calling the tool, then invokes the tool function, and lastly returns the result to the LLM.. This example uses the `mlflow.withSpan` API to create a parent span for the agent flow, but you can also achieve the same by using the `mlflow.trace` API like the previous example.

typescript

```typescript
async function runToolAgent(question: string) {
    console.log(`\n🤖 Running tool agent with question: "${question}"`);

    return await mlflow.withSpan(
        async () => {
            const client = tracedOpenAI(new OpenAI());
            const messages: any[] = [{"role": "user", "content": question}];

            // First LLM call to get tool instructions
            const response = await client.chat.completions.create({
                model: "o4-mini",
                messages: messages,
                tools: tools,
            });

            const aiMsg = response.choices[0].message;
            messages.push(aiMsg);

            // If the model requests tool call(s), invoke the function
            if (aiMsg.tool_calls && aiMsg.tool_calls.length > 0) {
                for (const toolCall of aiMsg.tool_calls) {
                    const functionName = toolCall.function.name;

                    if (functionName === "get_weather") {
                        // Invoke the tool function with the provided arguments
                        const args = JSON.parse(toolCall.function.arguments);
                        const toolResult = await getWeather(args.latitude, args.longitude);
                        messages.push({
                            "role": "tool",
                            "tool_call_id": toolCall.id,
                            "content": String(toolResult),
                        });
                    } else {
                        throw new Error(`Invalid tool returned: ${functionName}`);
                    }
                }

                // Second LLM call with tool results
                const finalResponse = await client.chat.completions.create({
                    model: "o4-mini",
                    messages: messages
                });
                return finalResponse.choices[0].message.content;
            }
            return aiMsg.content;
        },
        {
            name: "run_tool_agent",
            spanType: mlflow.SpanType.AGENT,
        }
    );
}


```

Now we can run the application.

typescript

```typescript
// If you are running this tutorial as a script, remove the `await` keyword.
await runToolAgent("What's the weather like in Seattle?")

```

## Step 4: Explore Traces in the UI[​](#step-4-explore-traces-in-the-ui "Direct link to Step 4: Explore Traces in the UI")

After running the application, you can explore the traces in the MLflow UI.

![Tool Calling Trace](/mlflow-website/docs/latest/images/llms/tracing/quickstart/openai-tool-calling-trace-detail.png)

The trace shows all LLM invocations and tool calls, organized in a tree structure. You can also inspect the timeline breakdown by clicking the timeline icon next to the tree view. This helps you understand where the time is spent in the application.

![Tool Calling Trace Timeline](/mlflow-website/docs/latest/images/llms/tracing/quickstart/openai-tool-calling-trace-timeline.png)

## Step 5: Attach Feedbacks on Traces[​](#step-5-attach-feedbacks-on-traces "Direct link to Step 5: Attach Feedbacks on Traces")

As a last step of this quickstart, let's attach feedback on the generated traces. In real world development, human feedback is critical to improve the quality of any LLM-powered application.

To add a feedback to a trace, you can open the trace detail page and click the "Add new Assessment" button on the top right. It will open an input form where you can provide various feedback values and metadata. For example, we can add feedback called "Quality" with an integer value (1\~5), indicating how good the answer is. We can also put the detailed rationale behind the score for future reference.

![Feedback Input Form](/mlflow-website/docs/latest/images/llms/tracing/quickstart/openai-trace-feedback-input.png)

When you submit the form with "Create" button, the feedback will be attached to the trace.

![Feedback List](/mlflow-website/docs/latest/images/llms/tracing/quickstart/openai-trace-feedback-record.png)

The aggregated score in the experiment can be seen in the Trace list. You can do slice-and-dice by various criteria, such as timestamp, source, tags, and it will update the aggregated score in real-time.

![Feedback Aggregated](/mlflow-website/docs/latest/images/llms/tracing/quickstart/openai-trace-feedback-aggregate.png)

## Summary[​](#summary "Direct link to Summary")

Congratulations! You've successfully:

* ✅ Set up MLflow Tracing for a GenAI application
* ✅ Enabled automatic tracing for OpenAI API calls
* ✅ Generated and explored traces in the MLflow UI
* ✅ Learned how to add custom tracing using decorators
* ✅ Learned how to attach feedback on traces

MLflow Tracing provides powerful observability for your GenAI applications, helping you monitor performance, debug issues, and understand user interactions. Continue exploring the advanced features to get the most out of your tracing setup!

## Next Steps[​](#next-steps "Direct link to Next Steps")

Now that you have basic tracing working, explore these advanced features:

* [Typescript SDK Guide](/mlflow-website/docs/latest/genai/tracing/app-instrumentation/typescript-sdk.md): Learn more about how to use the Typescript SDK.
* [Automatic Evaluation](/mlflow-website/docs/latest/genai/eval-monitor.md): Learn how to set up automatic evaluation for traces using MLflow's GenAI evaluation feature.
