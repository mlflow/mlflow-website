---
title: Genie Conversation Tracing Pipeline
slug: genie-tracing-pipeline
description: Pull conversations from a Genie space and log each one as an MLflow trace for inspection and evaluation.
tags: [databricks, genie, tracing, agents]
---

This cookbook pulls conversations from a Databricks [Genie space](https://docs.databricks.com/en/genie/index.html) and logs each one as an MLflow trace. Each trace captures the user's question, the SQL Genie generated, and the answer it returned.

<!-- truncate -->

Once conversations are stored as traces, you can inspect them in the MLflow UI and run automated evaluation in the [next cookbook](/cookbook/genie-evaluation-judges).

## Prerequisites

You need a Databricks [Genie space](https://docs.databricks.com/en/genie/set-up.html) with at least a few conversations. To create one, open your Databricks workspace, click **Genie** in the sidebar, and follow the setup wizard to connect [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html) tables and add instructions.

```bash
pip install "mlflow[genai]" databricks-sdk
```

## Step 1: Configure

Set `SPACE_ID` to your Genie space ID. You can find this in the Genie space URL.

```python
import mlflow
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

SPACE_ID = "your-genie-space-id"
EXPERIMENT_NAME = f"/Users/{w.current_user.me().user_name}/genie_eval"

mlflow.set_experiment(EXPERIMENT_NAME)
```

## Step 2: Pull Conversations and Log as Traces

Pull conversations from the Genie space, extract the question, generated SQL, and text response from each message, and log them as MLflow traces. Messages that have already been traced are skipped so you can safely re-run this pipeline as new conversations come in.

```python
# 1. Collect Genie message IDs that have already been traced so we
#    can skip them and safely re-run this pipeline as new
#    conversations come in.
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
existing_traces = mlflow.search_traces(
    locations=[experiment.experiment_id], return_type="list"
)
already_traced = {
    t.info.tags.get("message_id")
    for t in existing_traces
    if t.info.tags.get("message_id")
}

# 2. Pull every conversation from the Genie space.
conversations = w.genie.list_conversations(
    space_id=SPACE_ID, include_all=True
)

# 3. Loop through Genie messages, skip duplicates, and log each
#    new message as an MLflow trace with the question, SQL, and
#    response.
traced = 0
for convo in conversations.conversations or []:
    messages = w.genie.list_conversation_messages(
        space_id=SPACE_ID, conversation_id=convo.conversation_id
    )
    for msg in messages.messages or []:
        if not msg.content:
            continue
        if msg.message_id in already_traced:
            continue

        # 3a. Extract the SQL query and text response from the
        #     Genie message attachments.
        attachments = msg.attachments or []
        sql_att = next((a for a in attachments if a.query), None)
        text_att = next((a for a in attachments if a.text), None)

        # 3b. Log the question, SQL, and response as an MLflow
        #     trace for inspection and evaluation.
        with mlflow.start_span(name="genie_interaction") as span:
            span.set_inputs({"question": msg.content})
            span.set_outputs({
                "response": (
                    text_att.text.content if text_att else None
                ),
                "generated_sql": (
                    sql_att.query.query if sql_att else None
                ),
                "error": str(msg.error) if msg.error else None,
            })
            # 3c. Tag the trace with the Genie message ID so
            #     future runs know this message has already been
            #     traced.
            mlflow.update_current_trace(
                tags={"message_id": msg.message_id}
            )

        traced += 1

print(f"Logged {traced} new traces to experiment: {EXPERIMENT_NAME}")
```

## Results

Open the MLflow experiment to inspect your traces. Each row is one Genie message with the question, generated SQL, and response.

![Genie conversation traces logged in MLflow](/img/cookbook/databricks-genie/tracing-traces-logged.png)

Click a trace to see the full detail, including the `text_to_sql`, `sql_execution`, and `response_generation` spans.

![Trace detail showing spans and outputs](/img/cookbook/databricks-genie/tracing-trace-detail.png)

## Next Steps

- [Evaluation with LLM Judges](/cookbook/genie-evaluation-judges) -Score the traces to find quality issues.
- [Space Improvement Generator](/cookbook/genie-space-analyzer) -Generate fixes you can apply back to the Genie space.
