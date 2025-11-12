# Tracing LiteLLM🚄

![LiteLLM Tracing via autolog](/mlflow-website/docs/latest/assets/images/litellm-tracing-39a2a3e58fdb3d8cce0ecdba1f4f70e8.png)

[LiteLLM](https://www.litellm.ai/) is an open-source LLM Gateway that allow accessing 100+ LLMs in the unified interface.

[MLflow Tracing](/mlflow-website/docs/latest/genai/tracing/integrations.md) provides automatic tracing capability for LiteLLM. By enabling auto tracing for LiteLLM by calling the [`mlflow.litellm.autolog()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.litellm.html#mlflow.litellm.autolog) function, MLflow will capture traces for LLM invocation and log them to the active MLflow Experiment.

python

```python
import mlflow

mlflow.litellm.autolog()

```

MLflow trace automatically captures the following information about LiteLLM calls:

* Prompts and completion responses
* Latencies
* Metadata about the LLM provider, such as model name and endpoint URL
* Token usages and cost
* Cache hit
* Any exception if raised

### Basic Example[​](#basic-example "Direct link to Basic Example")

python

```python
import mlflow
import litellm

# Enable auto-tracing for LiteLLM
mlflow.litellm.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LiteLLM")

# Call Anthropic API via LiteLLM
response = litellm.completion(
    model="claude-3-5-sonnet-20240620",
    messages=[{"role": "user", "content": "Hey! how's it going?"}],
)

```

### Async API[​](#async-api "Direct link to Async API")

MLflow supports tracing LiteLLM's async APIs:

python

```python
mlflow.litellm.autolog()

response = await litellm.acompletion(
    model="claude-3-5-sonnet-20240620",
    messages=[{"role": "user", "content": "Hey! how's it going?"}],
)

```

### Streaming[​](#streaming "Direct link to Streaming")

MLflow supports tracing LiteLLM's sync and async streaming APIs:

python

```python
mlflow.litellm.autolog()

response = litellm.completion(
    model="claude-3-5-sonnet-20240620",
    messages=[{"role": "user", "content": "Hey! how's it going?"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="|")

```

MLflow will record concatenated outputs from the stream chunks as a span output.

### Disable auto-tracing[​](#disable-auto-tracing "Direct link to Disable auto-tracing")

Auto tracing for LiteLLM can be disabled globally by calling `mlflow.litellm.autolog(disable=True)` or `mlflow.autolog(disable=True)`.
