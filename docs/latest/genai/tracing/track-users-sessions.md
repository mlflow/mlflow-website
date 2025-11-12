# Track Users & Sessions

![Traces with session IDs](/mlflow-website/docs/latest/assets/images/chat-sessions-demo-6851f283db36f411076192cf6050cc47.gif)

Many real-world AI applications use session to maintain multi-turn user interactions. MLflow Tracing provides built-in support for associating traces with users and grouping them into sessions. Tracking users and sessions in your GenAI application provides essential context for understanding user behavior, analyzing conversation flows, and improving personalization.

## Store User and Session IDs in Metadata[​](#store-user-and-session-ids-in-metadata "Direct link to Store User and Session IDs in Metadata")

New in MLflow 3

The standard metadata for user and session tracking is only available in MLflow 3 and above. To upgrade, please run `pip install --upgrade mlflow`.

MLflow provides two standard metadata fields for session and user tracking:

* `mlflow.trace.user` - Associates traces with specific users
* `mlflow.trace.session` - Groups traces belonging to multi-turn conversations

When you use these standard metadata fields, MLflow automatically enables filtering and grouping in the UI. Unlike tags, metadata cannot be updated once the trace is logged, making it ideal for immutable identifiers like user and session IDs.

## Basic Usage[​](#basic-usage "Direct link to Basic Usage")

To record user and session information in your application, use the [`mlflow.update_current_trace()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.update_current_trace) API and pass the user and session IDs in the metadata.

* Python
* Typescript

Here's how to add user and session tracking to your application:

python

```python
import mlflow


@mlflow.trace
def chat_completion(message: list[dict], user_id: str, session_id: str):
    """Process a chat message with user and session tracking."""

    # Add user and session context to the current trace
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,  # Links trace to specific user
            "mlflow.trace.session": session_id,  # Groups trace with conversation
        }
    )

    # Your chat logic here
    return generate_response(message)

```

typescript

```typescript
import * as mlflow from "mlflow-tracing";

const chatCompletion = mlflow.trace(
    (message: list[dict], user_id: str, session_id: str) => {
        // Add user and session context to the current trace
        mlflow.updateCurrentTrace({
            metadata: {
                "mlflow.trace.user": user_id,
                "mlflow.trace.session": session_id,
            },
        });

        // Your chat logic here
        return generate_response(message);
    },
    { name: "chat_completion" }
);

```

## Web Application Example[​](#web-application-example "Direct link to Web Application Example")

* Python (FastAPI)
* Typescript (Node.js)

python

```python
import mlflow
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id="3044868363145534")
mlflow.openai.autolog()


class ChatRequest(BaseModel):
    message: str


@mlflow.trace
def process_chat(message: str, user_id: str, session_id: str):
    # Update trace with user and session context
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": session_id,
            "mlflow.trace.user": user_id,
        }
    )

    # Process chat message using OpenAI API
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


@app.post("/chat")
def handle_chat(request: Request, chat_request: ChatRequest):
    session_id = request.headers.get("X-Session-ID", "default-session")
    user_id = request.headers.get("X-User-ID", "default-user")
    response_text = process_chat(chat_request.message, user_id, session_id)
    return {"response": response_text}


@app.get("/")
async def root():
    return {"message": "FastAPI MLflow Tracing Example"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

```

**Example request:**

bash

```bash
python app.py

curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: session-123" \
    -H "X-User-ID: user-456" \
    -d '{"message": "Hello, how are you?"}'

```

```python
```

**Example request:**

bash

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -H "X-Session-ID: session-def-456" \
     -H "X-User-ID: user-jane-doe-12345" \
     -d '{"message": "What is my account balance?"}'

```

## Querying[​](#querying "Direct link to Querying")

* MLflow UI Search
* Programmatic Analysis

Filter traces in the MLflow UI using these search queries:

text

```text
# Find all traces for a specific user
metadata.`mlflow.trace.user` = 'user-123'

# Find all traces in a session
metadata.`mlflow.trace.session` = 'session-abc-456'

# Find traces for a user within a specific session
metadata.`mlflow.trace.user` = 'user-123' AND metadata.`mlflow.trace.session` = 'session-abc-456'

```

Analyze user behavior patterns programmatically:

python

```python
import mlflow
import pandas as pd

# Search for all traces from a specific user
user_traces_df: pd.DataFrame = mlflow.search_traces(
    filter_string=f"metadata.`mlflow.trace.user` = '{user_id}'",
)

# Calculate key metrics
total_interactions = len(user_traces_df)
unique_sessions = user_traces_df["metadata.mlflow.trace.session"].nunique()
avg_response_time = user_traces_df["info.execution_time_ms"].mean()
success_rate = user_traces_df["info.state"].value_counts()["OK"] / total_interactions

# Display the results
print(f"User has {total_interactions} interactions across {unique_sessions} sessions")
print(f"Average response time: {avg_response_time} ms")
print(f"Success rate: {success_rate}")

```

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Search Traces](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

[Master advanced filtering techniques for user and session analysis](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

[Learn search →](/mlflow-website/docs/latest/genai/tracing/search-traces.md)

### [Production Monitoring](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)

[Set up comprehensive production observability with user context](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)

[Monitor production →](/mlflow-website/docs/latest/genai/tracing/prod-tracing.md)
