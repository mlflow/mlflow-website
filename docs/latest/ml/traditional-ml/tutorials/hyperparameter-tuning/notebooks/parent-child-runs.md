# Leveraging Child Runs in MLflow for Hyperparameter Tuning

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/parent-child-runs.ipynb)

In the world of machine learning, the task of hyperparameter tuning is central to model optimization. This process involves performing multiple runs with varying parameters to identify the most effective combination, ultimately enhancing model performance. However, this can lead to a large volume of runs, making it challenging to track, organize, and compare these experiments effectively.

**MLflow** incorporates the ability to simplify the large-data-volume issue by offering a structured approach to manage this complexity. In this notebook, we will explore the concept of **Parent and Child Runs** in MLflow, a feature that provides a hierarchical structure to organize runs. This hierarchy allows us to bundle a set of runs under a parent run, making it much more manageable and intuitive to analyze and compare the results of different hyperparameter combinations. This structure proves to be especially beneficial in understanding and visualizing the outcomes of hyperparameter tuning processes.

Throughout this notebook, we will:

* Understand the usage and benefits of parent and child runs in MLflow.
* Walk through a practical example demonstrating the organization of runs without and with child runs.
* Observe how child runs aid in effectively tracking and comparing the results of different parameter combinations.
* Demonstrate a further refinement by having the parent run maintain the state of the best conditions from child run iterations.

### Starting Without Child Runs[​](#starting-without-child-runs "Direct link to Starting Without Child Runs")

Before diving into the structured world of parent and child runs, let's begin by observing the scenario without utilizing child runs in MLflow. In this section, we perform multiple runs with different parameters and metrics without associating them as child runs of a parent run.

Below is the code executing five hyperparameter tuning runs. These runs are not organized as child runs, and hence, each run is treated as an independent entity in MLflow. We will observe the challenges this approach poses in tracking and comparing runs, setting the stage for the introduction of child runs in the subsequent sections.

After running the above code, you can proceed to the MLflow UI to view the logged runs. Observing the organization (or lack thereof) of these runs will help in appreciating the structured approach offered by using child runs, which we will explore in the next sections of this notebook.

python

```
import random
from functools import partial
from itertools import starmap

from more_itertools import consume

import mlflow


# Define a function to log parameters and metrics
def log_run(run_name, test_no):
  with mlflow.start_run(run_name=run_name):
      mlflow.log_param("param1", random.choice(["a", "b", "c"]))
      mlflow.log_param("param2", random.choice(["d", "e", "f"]))
      mlflow.log_metric("metric1", random.uniform(0, 1))
      mlflow.log_metric("metric2", abs(random.gauss(5, 2.5)))


# Generate run names
def generate_run_names(test_no, num_runs=5):
  return (f"run_{i}_test_{test_no}" for i in range(num_runs))


# Execute tuning function
def execute_tuning(test_no):
  # Partial application of the log_run function
  log_current_run = partial(log_run, test_no=test_no)
  # Generate run names and apply log_current_run function to each run name
  runs = starmap(log_current_run, ((run_name,) for run_name in generate_run_names(test_no)))
  # Consume the iterator to execute the runs
  consume(runs)


# Set the tracking uri and experiment
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("No Child Runs")

# Execute 5 hyperparameter tuning runs
consume(starmap(execute_tuning, ((x,) for x in range(5))))
```

#### Iterative development simulation[​](#iterative-development-simulation "Direct link to Iterative development simulation")

It is very rare that a tuning run will be conducted in isolation. Typically, we will run many iterations of combinations of parameters, refining our search space to achieve the best possible potential results in the shortest amount of execution time.

In order to arrive at this limited set of selection parameters ranges and conditions, we will be executing many such tests.

python

```
# What if we need to run this again?
consume(starmap(execute_tuning, ((x,) for x in range(5))))
```

### Using Child Runs for Improved Organization[​](#using-child-runs-for-improved-organization "Direct link to Using Child Runs for Improved Organization")

As we proceed, the spotlight now shifts to the utilization of **Child Runs in MLflow**. This feature brings forth an organized structure, inherently solving the challenges we observed in the previous section. The child runs are neatly nested under a parent run, providing a clear, hierarchical view of all the runs, making it exceptionally convenient to analyze and compare the outcomes.

#### Benefits of Using Child Runs:[​](#benefits-of-using-child-runs "Direct link to Benefits of Using Child Runs:")

* **Structured View:** The child runs, grouped under a parent run, offer a clean and structured view in the MLflow UI.
* **Efficient Filtering:** The hierarchical organization facilitates efficient filtering and selection, enhancing the usability of the MLflow UI and search APIs.
* **Distinct Naming:** Utilizing visually distinct naming for runs aids in effortless identification and selection within the UI.

In this section, the code is enhanced to use child runs. Each `execute_tuning` function call creates a parent run, under which multiple child runs are nested. These child runs are performed with different parameters and metrics. Additionally, we incorporate tags to further enhance the search and filter capabilities in MLflow.

Notice the inclusion of the `nested=True` parameter in the `mlflow.start_run()` function, indicating the creation of a child run. The addition of tags, using the `mlflow.set_tag()` function, provides an extra layer of information, useful for filtering and searching runs effectively.

Let's dive into the code and observe the seamless organization and enhanced functionality brought about by the use of child runs in MLflow.

python

```
# Define a function to log parameters and metrics and add tag
# logging for search_runs functionality
def log_run(run_name, test_no, param1_choices, param2_choices, tag_ident):
  with mlflow.start_run(run_name=run_name, nested=True):
      mlflow.log_param("param1", random.choice(param1_choices))
      mlflow.log_param("param2", random.choice(param2_choices))
      mlflow.log_metric("metric1", random.uniform(0, 1))
      mlflow.log_metric("metric2", abs(random.gauss(5, 2.5)))
      mlflow.set_tag("test_identifier", tag_ident)


# Generate run names
def generate_run_names(test_no, num_runs=5):
  return (f"run_{i}_test_{test_no}" for i in range(num_runs))


# Execute tuning function, allowing for param overrides,
# run_name disambiguation, and tagging support
def execute_tuning(
  test_no, param1_choices=("a", "b", "c"), param2_choices=("d", "e", "f"), test_identifier=""
):
  ident = "default" if not test_identifier else test_identifier
  # Use a parent run to encapsulate the child runs
  with mlflow.start_run(run_name=f"parent_run_test_{ident}_{test_no}"):
      # Partial application of the log_run function
      log_current_run = partial(
          log_run,
          test_no=test_no,
          param1_choices=param1_choices,
          param2_choices=param2_choices,
          tag_ident=ident,
      )
      mlflow.set_tag("test_identifier", ident)
      # Generate run names and apply log_current_run function to each run name
      runs = starmap(log_current_run, ((run_name,) for run_name in generate_run_names(test_no)))
      # Consume the iterator to execute the runs
      consume(runs)


# Set the tracking uri and experiment
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Nested Child Association")

# Define custom parameters
param_1_values = ["x", "y", "z"]
param_2_values = ["u", "v", "w"]

# Execute hyperparameter tuning runs with custom parameter choices
consume(starmap(execute_tuning, ((x, param_1_values, param_2_values) for x in range(5))))
```

### Tailoring the Hyperparameter Tuning Process[​](#tailoring-the-hyperparameter-tuning-process "Direct link to Tailoring the Hyperparameter Tuning Process")

In this segment, we are taking a step further in our iterative process of hyperparameter tuning. Observe the execution of additional hyperparameter tuning runs, where we introduce **custom parameter choices** and a unique identifier for tagging.

#### What Are We Doing?[​](#what-are-we-doing "Direct link to What Are We Doing?")

* **Custom Parameter Choices:** We are now employing different parameter values (`param_1_values` as `["x", "y", "z"]` and `param_2_values` as `["u", "v", "w"]`) for the runs.
* **Unique Identifier for Tagging:** A distinct identifier (`ident`) is used for tagging, which provides an easy and efficient way to filter and search these specific runs in the MLflow UI.

#### How Does It Apply to Hyperparameter Tuning?[​](#how-does-it-apply-to-hyperparameter-tuning "Direct link to How Does It Apply to Hyperparameter Tuning?")

* **Parameter Sensitivity Analysis:** This step allows us to analyze the sensitivity of the model to different parameter values, aiding in a more informed and effective tuning process.
* **Efficient Search and Filter:** The use of a unique identifier for tagging facilitates an efficient and quick search for these specific runs among a multitude of others, enhancing the user experience in the MLflow UI.

This approach, employing custom parameters and tagging, enhances the clarity and efficiency of the hyperparameter tuning process, contributing to building a more robust and optimized model.

Let's execute this section of the code and delve deeper into the insights and improvements it offers in the hyperparameter tuning process.

python

```
# Execute additional hyperparameter tuning runs with custom parameter choices
param_1_values = ["x", "y", "z"]
param_2_values = ["u", "v", "w"]
ident = "params_test_2"
consume(starmap(execute_tuning, ((x, param_1_values, param_2_values, ident) for x in range(5))))
```

### Refining the Hyperparameter Search Space[​](#refining-the-hyperparameter-search-space "Direct link to Refining the Hyperparameter Search Space")

In this phase, we focus on **refining the hyperparameter search space**. This is a crucial step in the hyperparameter tuning process. After a broad exploration of the parameter space, we are now narrowing down our search to a subset of parameter values.

#### What Are We Doing?[​](#what-are-we-doing-1 "Direct link to What Are We Doing?")

* **Sub-setting Parameter Values:** We are focusing on a more specific set of parameter values (`param_1_values` as `["b", "c"]` and `param_2_values` as `["d", "f"]`) based on insights gathered from previous runs.
* **Tagging the Runs:** Using a unique identifier (`ident`) for tagging ensures easy filtering and searching of these runs in the MLflow UI.

#### How Does It Apply to Hyperparameter Tuning?[​](#how-does-it-apply-to-hyperparameter-tuning-1 "Direct link to How Does It Apply to Hyperparameter Tuning?")

* **Focused Search:** This narrowed search allows us to deeply explore the interactions and impacts of a specific set of parameter values, potentially leading to more optimized models.
* **Efficient Resource Utilization:** It enables more efficient use of computational resources by focusing the search on promising areas of the parameter space.

#### Caution[​](#caution "Direct link to Caution")

While this approach is a common tactic in hyperparameter tuning, it's crucial to acknowledge the implications. Comparing results from the narrowed search space directly with those from the original, broader search space can be misleading.

#### Why Is It Invalid to Compare?[​](#why-is-it-invalid-to-compare "Direct link to Why Is It Invalid to Compare?")

* **Nature of Bayesian Tuning Algorithms:** Bayesian optimization and other tuning algorithms often depend on the exploration of a broad parameter space to make informed decisions. Restricting the parameter space can influence the behavior of these algorithms, leading to biased or suboptimal results.
* **Interaction of Hyperparameter Selection Values:** Different parameter values have different interactions and impacts on the model performance. A narrowed search space may miss out on capturing these interactions, leading to incomplete or skewed insights.

In conclusion, while refining the search space is essential for efficient and effective hyperparameter tuning, it's imperative to approach the comparison of results with caution, acknowledging the intricacies and potential biases involved.

python

```
param_1_values = ["b", "c"]
param_2_values = ["d", "f"]
ident = "params_test_3"
consume(starmap(execute_tuning, ((x, param_1_values, param_2_values, ident) for x in range(5))))
```

### Challenge: Logging Best Metrics and Parameters[​](#challenge-logging-best-metrics-and-parameters "Direct link to Challenge: Logging Best Metrics and Parameters")

In the real world of machine learning, it is crucial to keep track of the best performing models and their corresponding parameters for easy comparison and reproduction. **Your challenge is to enhance the `execute_tuning` function to log the best metrics and parameters from the child runs in each parent run.** This way, you can easily compare the best-performing models across different parent runs within the MLflow UI.

#### Your Task:[​](#your-task "Direct link to Your Task:")

1. Modify the `execute_tuning` function such that for each parent run, it logs the best (minimum) `metric1` found among all its child runs.
2. Alongside the best `metric1`, also log the parameters `param1` and `param2` that yielded this best `metric1`.
3. Ensure that the `execute_tuning` function can accept a `num_child_runs` parameter to specify how many child iterations to perform per parent run.

This is a common practice that allows you to keep your MLflow experiments organized and easily retrievable, making the model selection process smoother and more efficient.

**Hint:** You might want to return values from the `log_run` function and use these returned values in the `execute_tuning` function to keep track of the best metrics and parameters.

#### Note:[​](#note "Direct link to Note:")

Before moving on to the solution below, **give it a try yourself!** This exercise is a great opportunity to familiarize yourself with advanced features of MLflow and improve your MLOps skills. If you get stuck or want to compare your solution, you can scroll down to see a possible implementation.

python

```
# Define a function to log parameters and metrics and add tag
# logging for search_runs functionality
def log_run(run_name, test_no, param1_choices, param2_choices, tag_ident):
  with mlflow.start_run(run_name=run_name, nested=True) as run:
      param1 = random.choice(param1_choices)
      param2 = random.choice(param2_choices)
      metric1 = random.uniform(0, 1)
      metric2 = abs(random.gauss(5, 2.5))

      mlflow.log_param("param1", param1)
      mlflow.log_param("param2", param2)
      mlflow.log_metric("metric1", metric1)
      mlflow.log_metric("metric2", metric2)
      mlflow.set_tag("test_identifier", tag_ident)

      return run.info.run_id, metric1, param1, param2


# Generate run names
def generate_run_names(test_no, num_runs=5):
  return (f"run_{i}_test_{test_no}" for i in range(num_runs))


# Execute tuning function, allowing for param overrides,
# run_name disambiguation, and tagging support
def execute_tuning(
  test_no,
  param1_choices=("a", "b", "c"),
  param2_choices=("d", "e", "f"),
  test_identifier="",
  num_child_runs=5,
):
  ident = "default" if not test_identifier else test_identifier
  best_metric1 = float("inf")
  best_params = None
  # Use a parent run to encapsulate the child runs
  with mlflow.start_run(run_name=f"parent_run_test_{ident}_{test_no}"):
      # Partial application of the log_run function
      log_current_run = partial(
          log_run,
          test_no=test_no,
          param1_choices=param1_choices,
          param2_choices=param2_choices,
          tag_ident=ident,
      )
      mlflow.set_tag("test_identifier", ident)
      # Generate run names and apply log_current_run function to each run name
      results = list(
          starmap(
              log_current_run,
              ((run_name,) for run_name in generate_run_names(test_no, num_child_runs)),
          )
      )

      for _, metric1, param1, param2 in results:
          if metric1 < best_metric1:
              best_metric1 = metric1
              best_params = (param1, param2)

      mlflow.log_metric("best_metric1", best_metric1)
      mlflow.log_param("best_param1", best_params[0])
      mlflow.log_param("best_param2", best_params[1])
      # Consume the iterator to execute the runs
      consume(results)


# Set the tracking uri and experiment
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Parent Child Association Challenge")

param_1_values = ["a", "b"]
param_2_values = ["d", "f"]

# Execute hyperparameter tuning runs with custom parameter choices
consume(
  starmap(
      execute_tuning, ((x, param_1_values, param_2_values, "subset_test", 25) for x in range(5))
  )
)
```
