# Migrating from Legacy LLM Evaluation

info

This is a migration guide for users who are using the legacy LLM evaluation capability through `mlflow.evaluate` API and see the following warning while migrating to MLflow 3.

> The mlflow\.evaluate API has been deprecated as of MLflow 3.0.0.

If you are new to MLflow or its evaluation capabilities, start from the [MLflow 3 GenAI Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/index.html) guide instead.

## Why Migrate?[​](#why-migrate "Direct link to Why Migrate?")

MLflow 3 introduces a [new evaluation suite](/mlflow-website/docs/latest/genai/eval-monitor.md) that are optimized for evaluating LLMs and GenAI applications. Compared to the legacy evaluation through the `mlflow.evaluate` API, the new suite offers the following benefits:

##### 1. Richer evaluation results[​](#1-richer-evaluation-results "Direct link to 1. Richer evaluation results")

MLflow 3 displays the evaluation results with intuitive visualizations. Each prediction is recorded with a trace, which allows you to further investigate the result in details and identify the root cause of low quality predictions.

| Old Results | ![Legacy Evaluation](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/legacy-eval-result.png)     |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| New Results | ![New Evaluation](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/prompt-evaluation-compare.png) |

##### 2. More powerful and flexible LLM-as-a-Judge[​](#2-more-powerful-and-flexible-llm-as-a-judge "Direct link to 2. More powerful and flexible LLM-as-a-Judge")

A rich set of built-in [LLM Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md) and a flexible toolset to build your own LLM-as-a-Judge supports you to evaluate various aspects of your LLM applications. Furthermore, the new [Agents-as-a-Judge](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges.md) capability evaluates complex trace with minimum context window consumption and boilerplate code.

##### 3. Integration with other MLflow GenAI capabilities[​](#3-integration-with-other-mlflow-genai-capabilities "Direct link to 3. Integration with other MLflow GenAI capabilities")

The new evaluation suite is tightly integrated with other MLflow GenAI capabilities, such as [tracing](/mlflow-website/docs/latest/genai/tracing.md), [prompt management](/mlflow-website/docs/latest/genai/prompt-registry.md), [prompt optimization](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts.md), making it an end-to-end solution for building high-quality LLM applications.

##### 4. Better future support[​](#4-better-future-support "Direct link to 4. Better future support")

MLflow is rapidly evolving ([changelog](https://github.com/mlflow/mlflow/releases)) and will continue strengthening its evaluation capabilities with the north star of **Deliver production-ready AI**. Migrating your workload to the new evaluation suite will ensure you have instant access to the latest and greatest features.

## Migration Steps[​](#migration-steps "Direct link to Migration Steps")

#### Wrap your model in a function

If you are evaluating an MLflow Model, wrap the model in a function and pass it to the new evaluation API.

#### Update dataset format

Update the inputs and ground truth format to match the new evaluation dataset format.

#### Migrate metrics

Update the metrics to use the new built-in or custom scorers offered by MLflow 3.

#### Run evaluation

Execute the evaluation and make sure the results are as expected.

Before you start the migration

Before starting the migration, we highly recommend you to visit the [GenAI Evaluation Guide](/mlflow-website/docs/latest/genai/eval-monitor.md) and go through the [Quickstart](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md) to get a sense of the new evaluation suite. Basic understanding of the concepts will help you to migrate your existing workload smoothly.

### 1. Wrap Your Model in a Function[​](#1-wrap-your-model-in-a-function "Direct link to 1. Wrap Your Model in a Function")

The old evaluation API accepts MLflow model URI as an evaluation target. The new evaluation API accepts a callable function as `predict_fn` argument instead, to provide more flexibility and control. This also eliminates the need of logging the model in MLflow before evaluation.

| Old Format                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | New Format                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
# Log the model first before evaluation
with mlflow.start_run() as run:
    logged_model_info = mlflow.openai.log_model(
        model="gpt-5-mini",
        task=openai.chat.completions,
        artifact_path="model",
        messages=[
            {"role": "system", "content": "Answer the following question in two sentences"},
            {"role": "user", "content": "{question}"},
        ],
    )

# Pass the model URI to the evaluation API.
mlflow.evaluate(model=logged_model_info.model_uri, ...)

``` | ```
# Define a function that runs predictions.
def predict_fn(question: str) -> str:
  response = openai.OpenAI().chat.completions.create(
      model="gpt-5-mini",
      messages=[
          {"role": "system", "content": "Answer the following question in two sentences"},
          {"role": "user", "content": question},
      ],
  )
  return response.choices[0].message.content

mlflow.genai.evaluate(predict_fn=predict_fn, ...)

``` |

If you want to evaluate a pre-logged model with the new evaluation API, simply call the loaded model in the function.

python

```python
# IMPORTANT: Load the model outside the predict_fn function. Otherwise the model will be loaded
# for each input in the dataset and significantly slow down the evaluation.
model = mlflow.pyfunc.load_model(model_uri)


def predict_fn(question: str) -> str:
    return model.predict([question])[0]

```

### 2. Update the Dataset Format[​](#2-update-the-dataset-format "Direct link to 2. Update the Dataset Format")

The dataset format has been changed to be more flexible and consistent. The new format requirements are:

* `inputs`: The input to the predict\_fn function. The key(s) must match the parameter name of the predict\_fn function.
* `expectations`: The expected output from the predict\_fn function, namely, ground truth for the answer.
* Optionally, you can pass `outputs` column or `trace` column to evaluate pre-generated outputs and traces.

| Old Format                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | New Format                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```
eval_data = pd.DataFrame(
  {
      "inputs": [
          "What is MLflow?",
          "What is Spark?",
      ],
      "ground_truth": [
          "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle.",
          "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics.",
      ],
      "predictions": [
        "MLflow is an open-source MLOps platform",
        "Apache Spark is an open-source distributed computing engine.",
      ]
  }
)

mlflow.evaluate(
  data=eval_data, # Needed to specify the ground truth and prediction # columns name, otherwise MLflow does not recognize them.
  targets="ground_truth",
  predictions="predictions",
  ...
)

``` | ```
eval_data = [
  {
      "inputs": {"question": "What is MLflow?"},
      "outputs": "MLflow is an open-source MLOps platform",
      "expectations": {"answer": "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle."},
  },
  {
      "inputs": {"question": "What is Spark?"},
      "outputs": "Apache Spark is an open-source distributed computing engine.",
      "expectations": {"answer": "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics."},
  },
]

mlflow.genai.evaluate(
  data=eval_data,
  ...
)

``` |

### 3. Migrate Metrics[​](#3-migrate-metrics "Direct link to 3. Migrate Metrics")

The new evaluation API supports a rich set of built-in and custom LLM-as-a-Judge metrics. The table below shows the mapping between the legacy metrics and the new metrics.

| Metric                          | Before                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | After                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Latency                         | [`latency`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.latency)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Traces record latency and also span-level break down. You don't need to specify a metric to evaluate latency when running the new [`mlflow.genai.evaluate()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate) API.                                                                                                                       |
| Token Count                     | [`token_count`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.token_count)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Traces record token count for LLM calls for most of popular LLM providers. For other cases, you can use a [custom scorer](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md) to calculate the token count.                                                                                                                                                              |
| Heuristic NLP metrics           | [`toxicity`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.toxicity), [`flesch_kincaid_grade_level`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.flesch_kincaid_grade_level), [`ari_grade_level`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.ari_grade_level), [`exact_match`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.exact_match), [`rouge1`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.rouge1), [`rouge2`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.rouge2), [`rougeL`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.rougeL), [`rougeLsum`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.rougeLsum) | Use a [Code-based Scorer](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md) to implement the equivalent metrics. See the example below for reference.                                                                                                                                                                                                                  |
| Retrieval metrics               | [`precision_at_k`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.precision_at_k), [`recall_at_k`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.recall_at_k), [`ndcg_at_k`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.ndcg_at_k)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Use the new [built-in retrieval metrics](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#available-judges) or define a custom code-based scorer.                                                                                                                                                                                                        |
| Built-in LLM-as-a-Judge metrics | [`answer_similarity`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.genai.answer_similarity), [`answer_correctness`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.genai.answer_correctness), [`answer_relevance`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.genai.answer_relevance), [`relevance`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.genai.relevance), [`faithfulness`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.genai.faithfulness)                                                                                                                                                                                                                                                                                                 | Use the new [built-in judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md#available-judges). If the metric is not supported out of the box, define a custom LLM-as-a-Judge scorer using the [`make_judge`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.judges.make_judge) API, following the example below. |
| Custom LLM-as-a-Judge metrics   | [`make_genai_metric`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.genai.make_genai_metric), [`make_genai_metric_from_prompt`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.genai.make_genai_metric_from_prompt)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Use the [`make_judge`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.judges.make_judge) API to define a custom LLM-as-a-Judge scorer, following the example below.                                                                                                                                                                               |

#### Example of custom LLM-as-a-Judge metrics[​](#example-of-custom-llm-as-a-judge-metrics "Direct link to Example of custom LLM-as-a-Judge metrics")

The new evaluation API supports defining custom LLM-as-a-Judge metrics from a custom prompt template. This eliminates a lot of complexity and over-abstractions from the previous `make_genai_metric` and `make_genai_metric_from_prompt` APIs.

python

```python
from mlflow.genai import make_judge

answer_similarity = make_judge(
    name="answer_similarity",
    instructions=(
        "Evaluated on the degree of semantic similarity of the provided output to the expected answer.\n\n"
        "Output: {{ outputs }}\n\n"
        "Expected: {{ expectations }}"
    ),
    feedback_value_type=int,
)

# Pass the scorer to the evaluation API.
mlflow.genai.evaluate(scorers=[answer_similarity, ...])

```

See the [LLM Judges](/mlflow-website/docs/latest/genai/eval-monitor/scorers.md#llms-as-judges) guide for more details.

#### Example of custom heuristic metrics[​](#example-of-custom-heuristic-metrics "Direct link to Example of custom heuristic metrics")

Implementing a custom scorer for heuristic metrics is straightforward. You just need to define a function and decorate it with the [`@scorer`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.scorer) decorator. The example below shows how to implement the `exact_match` metric.

python

```python
@scorer
def exact_match(outputs: dict, expectations: dict) -> bool:
    return outputs == expectations["expected_response"]


# Pass the scorer to the evaluation API.
mlflow.genai.evaluate(scorers=[exact_match, ...])

```

See the [Code-based Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md) guide for more details.

### 4. Run Evaluation[​](#4-run-evaluation "Direct link to 4. Run Evaluation")

Now you have migrated all components of the legacy evaluation API and are ready to run the evaluation!

python

```python
mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[answer_similarity, exact_match, ...],
)

```

To view the evaluation results, click the link in the console output, or navigate to the **Evaluations** tab in the MLflow UI.

[](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/evaluation-result-video.mp4)

## Other Changes[​](#other-changes "Direct link to Other Changes")

* When using Databricks Model Serving endpoint as a LLM-judge model, use `databricks:/<endpoint-name>` as model provider, rather than `endpoints:/<endpoint-name>`
* The evaluation results are shown in the `Evaluations` tab in the MLflow UI.
* Lots of configuration knobs such as `model_type`, `targets`, `feature_names`, `env_manager`, are removed in the new evaluation API.

## FAQ[​](#faq "Direct link to FAQ")

### Q: The feature I want is not supported in the new evaluation suite.[​](#q-the-feature-i-want-is-not-supported-in-the-new-evaluation-suite "Direct link to Q: The feature I want is not supported in the new evaluation suite.")

Please open an feature request in [GitHub](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml).

### Q: Where can I find the documentation for the legacy evaluation API?[​](#q-where-can-i-find-the-documentation-for-the-legacy-evaluation-api "Direct link to Q: Where can I find the documentation for the legacy evaluation API?")

See [MLflow 2 documentation](https://mlflow.org/docs/2.22.1/llms/llm-evaluate) for the legacy evaluation API.

### Q: When will the legacy evaluation API be removed?[​](#q-when-will-the-legacy-evaluation-api-be-removed "Direct link to Q: When will the legacy evaluation API be removed?")

It will likely be removed in MLflow 3.7.0 or a few releases after that.

### Q: Should I migrate non-GenAI workloads to the new evaluation suite?[​](#q-should-i-migrate-non-genai-workloads-to-the-new-evaluation-suite "Direct link to Q: Should I migrate non-GenAI workloads to the new evaluation suite?")

No. The new evaluation suite is only for GenAI workloads. If you are not using GenAI, you should use the [`mlflow.models.evaluate()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.models.html#mlflow.models.evaluate) API, which offers perfect compatibility with `mlflow.evaluate` API but drops the GenAI-specific features.
