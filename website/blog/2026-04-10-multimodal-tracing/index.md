---
title: "See What Your AI Sees: Multimodal Tracing for Images, Audio, and Files"
tags: [genai, tracing, multimodal, observability, attachments]
slug: multimodal-tracing
authors: [mlflow-maintainers]
thumbnail: /img/blog/multimodal-tracing-thumbnail.png
image: /img/blog/multimodal-tracing-thumbnail.png
---

Your agent analyzes images, transcribes audio, and processes PDFs. But when something goes wrong, your traces show nothing but opaque base64 strings — megabytes of `iVBORw0KGgo...` buried in JSON. You can see that an image was sent, but not what was in it. You can see audio was returned, but you can't play it. And every one of those multi-megabyte strings is stored directly in your trace database, bloating storage costs and slowing down queries.

Today we're announcing **multimodal tracing** in MLflow — binary content is automatically extracted from traces, stored efficiently as artifacts, and rendered inline in the UI exactly as your model saw it.

![A multimodal trace showing an image rendered inline alongside the model's text response in MLflow's chat view](./chat-view-hero.png)

## Why Text-Only Traces Fall Short

As LLM applications move beyond text — vision models analyzing photos, audio models transcribing calls, agents generating images — three problems compound:

- **Database bloat:** A single image generation response embeds ~1.8MB of base64 directly in your span JSON. Across thousands of traces, that's gigabytes of binary data stored in your tracking database — data that was never meant to live in a relational store.
- **Slow queries and UI:** Loading a trace list means fetching all that inline binary. Trace search slows down, the UI lags, and browsing production traces becomes painful.
- **Blind debugging:** When a vision model misclassifies an image, you need to see the image alongside the model's response, not a wall of encoded bytes. Text-only traces make multimodal debugging impossible.

Multimodal tracing solves all three.

## How It Works: Extract, Store, Render

MLflow's approach separates binary content from trace metadata at the earliest possible point:

1. **Extract** — When a span's inputs or outputs are set, MLflow scans for base64-encoded data patterns (data URIs, OpenAI audio, image generation output, Anthropic images, Gemini inline data, and more). Detected binary is decoded and pulled out of the JSON.

2. **Store** — Extracted content is saved as individual artifact files in your configured artifact store (S3, Azure Blob, GCS, DBFS, or local filesystem) — the same storage layer MLflow already uses for model artifacts. The span JSON retains only a lightweight `mlflow-attachment://` reference (~120 characters) instead of the original megabytes of base64.

3. **Render** — The UI fetches attachments on demand when you view a trace. Images display as thumbnails with click-to-expand, audio plays inline, PDFs render in an embedded viewer, and other file types show as download links.

The result: your trace database stays lean and fast, binary content lives in purpose-built object storage, and the UI renders rich media without ever storing it in the DB.

![Architecture diagram showing the extract, store, and render flow for multimodal trace attachments](./architecture-diagram.png)

## Auto-Extraction: Zero Code Changes

If you're already using MLflow's autologging for OpenAI, Anthropic, Gemini, Bedrock, or LangChain, multimodal tracing works out of the box. No code changes, no configuration — MLflow detects and extracts binary content automatically.

```python
import mlflow
import openai

mlflow.openai.autolog()
client = openai.OpenAI()

# Image data is automatically extracted — no code changes needed
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ],
    }],
)
```

MLflow recognizes 8 multimodal data patterns across providers:

| Pattern | Provider | Content Type |
|---|---|---|
| Data URIs (`data:image/png;base64,...`) | All | Images, audio |
| `input_audio` | OpenAI | Audio input |
| `b64_json` | OpenAI | Generated images |
| Audio output (`audio.data`) | OpenAI | Audio response |
| Anthropic image blocks | Anthropic | Images |
| Bedrock image format | AWS Bedrock | Images |
| Gemini `inline_data` | Google Gemini | Images, audio |
| Responses API `image_generation_call` | OpenAI | Generated images |

## Manual Attachments for Custom Content

For content that doesn't flow through autologging — PDFs, custom file types, or images you generate yourself — use the `Attachment` class:

```python
from mlflow.tracing.attachments import Attachment

with mlflow.start_span(name="analyze_document") as span:
    pdf = Attachment.from_file("report.pdf")
    span.set_inputs({"document": pdf, "question": "Summarize the key findings"})

    result = analyze(pdf_bytes)
    span.set_outputs({"summary": result})
```

`Attachment` objects get the same treatment as auto-extracted content: the binary is stored as an artifact, and the span JSON contains only the reference URI.

## Rich Rendering in the Trace UI

Multimodal traces render across both the **Summary** and **Details & Timeline** views:

- **Images** display as compact thumbnails. Click to expand to a fullscreen preview.
- **Audio** plays inline with standard browser audio controls.
- **PDFs** render in an embedded viewer.
- **Other file types** show as download links.

![A trace showing multiple content types rendered inline — image thumbnail, audio player, and text fields](./mixed-content-trace.png)

The chat view also renders multimodal content inline — vision model inputs show the image alongside the text prompt, and audio responses include a playable player below the transcript.

## Controlling Extraction

Auto-extraction is enabled by default. To disable it — for example, if you need raw base64 in your traces for programmatic access downstream:

```python
import os
os.environ["MLFLOW_TRACE_EXTRACT_ATTACHMENTS"] = "false"
```

When extraction is disabled, binary data stays inline in the span JSON (the pre-3.11 behavior).

## Getting Started

Multimodal tracing is available in MLflow 3.11+. To start capturing multimodal content in your traces:

1. **Upgrade MLflow:** `pip install --upgrade mlflow`
2. **Enable autologging** for your provider (`mlflow.openai.autolog()`, `mlflow.anthropic.autolog()`, etc.) — multimodal extraction happens automatically.
3. **View traces** in the MLflow UI — images, audio, and files render inline.

For manual attachment creation and the full list of supported patterns, see the [Multimodal Content and Attachments documentation](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/multimodal/).

If you find this useful, give us a star on GitHub: **[github.com/mlflow/mlflow](https://github.com/mlflow/mlflow)** ⭐️

Have questions or feedback? [Open an issue](https://github.com/mlflow/mlflow/issues) or join the conversation on [Slack](https://mlflow.org/slack).
