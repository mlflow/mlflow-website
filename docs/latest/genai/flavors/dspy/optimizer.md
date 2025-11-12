# DSPy Optimizer Autologging

A [DSPy optimizer](https://dspy.ai/learn/optimization/optimizers/) is an algorithm that tunes the parameters of a DSPy program (i.e., the prompts and/or the LM weights) to maximize the metrics you specify. However, optimizers have the following challenges due to its black box nature:

1. **Parameters and Score of individual trial**: Program parameters (e.g. proposed instructions and selected demonstrations) and performance of each trial during optimization are not saved.
2. **Intermediate Artifacts**: Prompts and model responses for the intermediate programs are not available. While some of them are printed out, there is no easy way to store the information in a structured manner.
3. **Optimization Trajectory**: It is hard to understand the overview of the optimization trajectory from log texts (e.g. score progress over time).

The MLflow DSPy flavor provides a convenient way to automatically track the optimization process and results. This allows you to analyze the optimization process and results more efficiently.

## Enable Autologging[​](#enable-autologging "Direct link to Enable Autologging")

To enable autologging for DSPy optimization, call [`mlflow.dspy.autolog()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.dspy.html#mlflow.dspy.autolog) at the beginning of your script or notebook with the following parameters. This will automatically log traces for your program execution as well as other metrics and artifacts such as program states, train datasets, and evaluation results depending on the configuration. Note that the optimizer autologging feature is available since MLflow `2.21.1`.

python

```python
import mlflow

mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)

# Your DSPy code here
...

```

| Target                     | Default | Parameter                 | Description                                                                                                                                                    |
| -------------------------- | ------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Traces                     | `true`  | `log_traces`              | Whether to generate and log traces for the program. See [MLflow Tracing](/mlflow-website/docs/latest/genai/tracing.md) for more details about tracing feature. |
| Traces during Optimization | `false` | `log_traces_from_compile` | MLflow does not generate traces for program calls during optimization by default. Set `True` to see traces for programs calls during optimization.             |
| Traces during Evaluation   | `True`  | `log_traces_from_eval`    | If set `True`, MLflow generates traces for program calls during evaluation.                                                                                    |
| Optimization               | `false` | `log_compiles`            | If set to `True`, a MLflow run is created for each `Teleprompter.compile` call, and metrics and artifacts for the optimization are logged.                     |
| Evaluation                 | `false` | `log_evals`               | If set to `True`, a MLflow run is created for each `Evaluate.__call__` call, and metrics and artifacts for the evaluation are logged.                          |

## Example Code of DSPy Optimizer Autologging[​](#example-code-of-dspy-optimizer-autologging "Direct link to Example Code of DSPy Optimizer Autologging")

python

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
import mlflow

# Enabling tracing for DSPy
mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")

lm = dspy.LM(model="openai/gpt-3.5-turbo", max_tokens=250)
dspy.configure(lm=lm)

gsm8k = GSM8K()

trainset = gsm8k.train
devset = gsm8k.dev[:50]

program = dspy.ChainOfThought("question -> answer")

# define a teleprompter
teleprompter = dspy.teleprompt.MIPROv2(
    metric=gsm8k_metric,
    auto="light",
)
# run the optimizer
optimized_program = teleprompter.compile(
    program,
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    requires_permission_to_run=False,
)

```

## What Gets Logged?[​](#what-gets-logged "Direct link to What Gets Logged?")

MLflow optimizer autologging logs the following information:

* **Optimizer Parameters**: The hyper parameters of the optimizer (e.g. number of demonstrates).
* **Optimized Program**: MLflow automatically saves the states of the optimized program as a json artifact.
* **Datasets**: The train and evaluation datasets used in the optimization.
* **Overall Progression of Metric**: The progression of the metric over time is captured as step-wise metrics of a compile run.
* **Intermediate Program States and Metric**: The states of the program and the performance metric at each evaluation are captured in an evaluation run.
* **Traces**: The traces of each intermediate program are captured in an evaluation run.

When both `log_compiles` and `log_evals` are set to `True`, MLflow creates a run for `Teleprompter.compile` with child runs for `Evaluate.__call__` calls. In the MLflow UI, they are displayed hierarchically as shown below:

![Experiment Page](https://i.imgur.com/eD0lA6V.png)

In the run page for the parent run (compile call), the optimizer parameters are displayed as run parameters. The progression of program performance is displayed as model metrics and the datasets used in the optimization are displayed as artifacts. When there are multiple type of evaluation calls (e.g. full dataset evaluation and mini-batch evaluation), the evaluation results are logged separately.

![Parent Run](https://i.imgur.com/lXKINPQ.png)

In the run page for the child run (evaluation call), the intermediate program states are displayed as run parameters, the performance metric are displayed as model metrics and the traces of the program are available in the trace tab.

![Child Run](https://i.imgur.com/XAC9hND.png)

## FAQ[​](#faq "Direct link to FAQ")

### How to log my optimization and evaluation into a same run?[​](#how-to-log-my-optimization-and-evaluation-into-a-same-run "Direct link to How to log my optimization and evaluation into a same run?")

To log both optimization and evaluation into a same run, you can manually create a parent run using `mlflow.start_run` and run your optimization and evaluation within the run.

python

```python
with mlflow.start_run(run_name="My Optimization Run") as run:
    optimized_program = teleprompter.compile(
        program,
        trainset=trainset,
    )
    evaluation = dspy.Evaluate(devset=devset, metric=metric)
    evaluation(optimized_program)

```
