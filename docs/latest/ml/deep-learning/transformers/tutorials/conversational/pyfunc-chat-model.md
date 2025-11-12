# Deploying a Transformer model as an OpenAI-compatible Chatbot

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/transformers/tutorials/conversational/pyfunc-chat-model.ipynb)

Welcome to our tutorial on using Transformers and MLflow to create an OpenAI-compatible chat model. In MLflow 2.11 and up, MLflow's Transformers flavors support special task type `llm/v1/chat`, which turns thousands of [text-generation](https://huggingface.co/models?pipeline_tag=text-generation) models on Hugging Face into conversational chat bots that are interoperable with OpenAI models. This enables you to seamlessly swap out your chat app's backing LLM or to easily evaluate different models without having to edit your client-side code.

If you haven't already seen it, you may find it helpful to go through our [introductory notebook on chat and Transformers](https://mlflow.org/docs/latest/ml/deep-learning/transformers/tutorials/conversational/conversational-model.html) before proceeding with this one, as this notebook is slightly higher-level and does not delve too deeply into the inner workings of Transformers or MLflow Tracking.

**Note**: This page covers how to deploy a **Transformers** models as a chatbot. If you are using a different framework or a custom python model, use [ChatModel](https://mlflow.org/docs/latest/genai/flavors/chat-model-intro/index.html) instead to build an OpenAI-compatible chat bot.

### Learning objectives[​](#learning-objectives "Direct link to Learning objectives")

In this tutorial, you will:

* Create an OpenAI-compatible chat model using TinyLLama-1.1B-Chat
* Log the model to MLflow and load it back for local inference.
* Serve the model with MLflow Model Serving

python

```python
%pip install mlflow>=2.11.0 -q -U
# OpenAI-compatible chat model support is available for Transformers 4.34.0 and above
%pip install transformers>=4.34.0 -q -U

```

python

```python
# Disable tokenizers warnings when constructing pipelines
%env TOKENIZERS_PARALLELISM=false

import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

```

### Building a Chat Model[​](#building-a-chat-model "Direct link to Building a Chat Model")

MLflow's native Transformers integration allows you to specify the `task` param when saving or logging your pipelines. Originally, this param accepts any of the [Transformers pipeline task types](https://huggingface.co/tasks), but the `mlflow.transformers` flavor adds a few more MLflow-specific keys for `text-generation` pipeline types.

For `text-generation` pipelines, instead of specifying `text-generation` as the task type, you can provide one of two string literals conforming to the [MLflow AI Gateway's endpoint\_type specification](https://mlflow.org/docs/latest/genai/governance/ai-gateway/#deployments-configuration-details) ("llm/v1/embeddings" can be specified as a task on models saved with `mlflow.sentence_transformers`):

* "llm/v1/chat" for chat-style applications
* "llm/v1/completions" for generic completions

When one of these keys is specified, MLflow will automatically handle everything required to serve a chat or completions model. This includes:

* Setting a chat/completions compatible signature on the model
* Performing data pre- and post-processing to ensure the inputs and outputs conform to the [Chat/Completions API spec](https://mlflow.org/docs/latest/genai/serving/responses-agent#openai-api-compatibility), which is compatible with OpenAI's API spec.

Note that these modifications only apply when the model is loaded with `mlflow.pyfunc.load_model()` (e.g. when serving the model with the `mlflow models serve` CLI tool). If you want to load just the base pipeline, you can always do so via `mlflow.transformers.load_model()`.

In the next few cells, we'll learn how serve a chat model with a local Transformers pipeline and MLflow, using TinyLlama-1.1B-Chat as an example.

To begin, let's go through the original flow of saving a text generation pipeline:

python

```python
from transformers import pipeline

import mlflow

generator = pipeline(
  "text-generation",
  model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

# save the model using the vanilla `text-generation` task type
mlflow.transformers.save_model(
  path="tinyllama-text-generation", transformers_model=generator, task="text-generation"
)

```

Now, let's load the model and use it for inference. Our loaded model is a `text-generation` pipeline, and let's take a look at its signature to see its expected inputs and outputs.

python

```python
# load the model for inference
model = mlflow.pyfunc.load_model("tinyllama-text-generation")

model.metadata.signature

```

Unfortunately, it only accepts `string` as input, which isn't directly compatible with a chat interface. When interacting with OpenAI's API, for example, we expect to simply be able to input a list of messages. In order to do this with our current model, we'll have to write some additional boilerplate:

python

```python
# first, apply the tokenizer's chat template, since the
# model is tuned to accept prompts in a chat format. this
# also converts the list of messages to a string.
messages = [{"role": "user", "content": "Write me a hello world program in python"}]
prompt = generator.tokenizer.apply_chat_template(
  messages, tokenize=False, add_generation_prompt=True
)

model.predict(prompt)

```

Now we're getting somewhere, but formatting our messages prior to inference is cumbersome.

Additionally, the output format isn't compatible with the OpenAI API spec either--it's just a list of strings. If we were looking to evaluate different model backends for our chat app, we'd have to rewrite some of our client-side code to both format the input, and to parse this new response.

To simplify all this, let's just pass in `"llm/v1/chat"` as the task param when saving the model.

python

```python
# save the model using the `"llm/v1/chat"`
# task type instead of `text-generation`
mlflow.transformers.save_model(
  path="tinyllama-chat", transformers_model=generator, task="llm/v1/chat"
)

```

Once again, let's load the model and inspect the signature:

python

```python
model = mlflow.pyfunc.load_model("tinyllama-chat")

model.metadata.signature

```

Now when performing inference, we can pass our messages in a dict as we'd expect to do when interacting with the OpenAI API. Furthermore, the response we receive back from the model also conforms to the spec.

python

```python
messages = [{"role": "user", "content": "Write me a hello world program in python"}]

model.predict({"messages": messages})

```

### Serving the Chat Model[​](#serving-the-chat-model "Direct link to Serving the Chat Model")

To take this example further, let's use MLflow to serve our chat model, so we can interact with it like a web API. To do this, we can use the `mlflow models serve` CLI tool.

In a terminal shell, run:

text

```text
$ mlflow models serve -m tinyllama-chat

```

When the server has finished initializing, you should be able to interact with the model via HTTP requests. The input format is almost identical to the format described in the [MLflow Deployments Server docs](https://mlflow.org/docs/latest/ml/deployment/index.html#chat), with the exception that `temperature` defaults to `1.0` instead of `0.0`.

Here's a quick example:

python

```python
%%sh
curl http://127.0.0.1:5000/invocations   -H 'Content-Type: application/json'   -d '{ "messages": [{"role": "user", "content": "Write me a hello world program in python"}] }'   | jq

```

It's that easy!

You can also call the API with a few optional inference params to adjust the model's responses. These map to Transformers pipeline params, and are passed in directly at inference time.

* `max_tokens` (maps to `max_new_tokens`): The maximum number of new tokens the model should generate.
* `temperature` (maps to `temperature`): Controls the creativity of the model's response. Note that this is not guaranteed to be supported by all models, and in order for this param to have an effect, the pipeline must have been created with `do_sample=True`.
* `stop` (maps to `stopping_criteria`): A list of tokens at which to stop generation.

Note: `n` does not have an equivalent Transformers pipeline param, and is not supported in queries. However, you can implement a model that consumes the `n` param using Custom Pyfunc (details below).

## Conclusion[​](#conclusion "Direct link to Conclusion")

In this tutorial, you learned how to create an OpenAI-compatible chat model by specifying "llm/v1/chat" as the task when saving Transformers pipelines.

### What's next?[​](#whats-next "Direct link to What's next?")

* [Learn about custom ChatModel](https://mlflow.org/docs/latest/llms/chat-model-intro/index.html). If you're looking for futrher customization or models outside Transformers, the linked page provides a hand-on guidance for how to build a chat bot with MLflow's `ChatModel` class.
* [More on MLflow AI Gateway](https://mlflow.org/docs/latest/ml/deployment/index.html). In this tutorial, we saw how to deploy a model using a local server, but MLflow provides many other ways to deploy your models to production. Check out this page to learn more about the different options.
* [More on MLflow's Transformers Integration](https://mlflow.org/docs/latest/ml/deep-learning/transformers/index.html). This page provides a comprehensive overview on MLflow's Transformers integrations, along with lots of hands-on guides and notebooks. Learn how to fine-tune models, use prompt templates, and more!
* [Other LLM Integrations](https://mlflow.org/docs/latest/genai/index.html). Aside from Transformers, MLflow has integrations with many other popular LLM libraries, such as Langchain and OpenAI.
