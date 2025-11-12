# MLflow with Optuna: Hyperparameter Optimization and Tracking

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs.ipynb)

A critical part of building production-grade models is ensuring that a given model's parameters are selected to create the best inference set possible. However, the sheer number of combinations and their resultant metrics can become overwhelming to track manually. That's where tools like MLflow and Optuna come into play.

### Objective:[​](#objective "Direct link to Objective:")

In this notebook, you'll learn how to integrate MLflow with Optuna for hyperparameter optimization. We'll guide you through the process of:

* Setting up your environment with MLflow tracking.
* Generating our training and evaluation data sets.
* Defining a partial function that fits a machine learning model.
* Using Optuna for hyperparameter tuning.
* Leveraging child runs within MLflow to keep track of each iteration during the hyperparameter tuning process.

### Why Optuna?[​](#why-optuna "Direct link to Why Optuna?")

Optuna is an open-source hyperparameter optimization framework in Python. It provides an efficient approach to searching over hyperparameters, incorporating the latest research and techniques. With its integration into MLflow, every trial can be systematically recorded.

### Child Runs in MLflow:[​](#child-runs-in-mlflow "Direct link to Child Runs in MLflow:")

One of the core features we will be emphasizing is the concept of 'child runs' in MLflow. When performing hyperparameter tuning, each iteration (or trial) in Optuna can be considered a 'child run'. This allows us to group all the runs under one primary 'parent run', ensuring that the MLflow UI remains organized and interpretable. Each child run will track the specific hyperparameters used and the resulting metrics, providing a consolidated view of the entire optimization process.

### What's Ahead?[​](#whats-ahead "Direct link to What's Ahead?")

**Data Preparation**: We'll start by loading and preprocessing our dataset.

**Model Definition**: Defining a machine learning model that we aim to optimize.

**Optuna Study**: Setting up an Optuna study to find the best hyperparameters for our model.

**MLflow Integration**: Tracking each Optuna trial as a child run in MLflow.

**Analysis**: Reviewing the tracked results in the MLflow UI.

By the end of this notebook, you'll have hands-on experience in setting up an advanced hyperparameter tuning workflow, emphasizing best practices and clean organization using MLflow and Optuna.

**Let's dive in!**

python

```python
import math
from datetime import datetime, timedelta

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow

```

### Configure the tracking server uri[​](#configure-the-tracking-server-uri "Direct link to Configure the tracking server uri")

Depending on where you are running this notebook, your configuration may vary for how you initialize the interface with the MLflow Tracking Server.

For this example, we're using a locally running tracking server, but other options are available (The easiest is to use the free managed service within [Databricks Free Trial](https://mlflow.org/docs/latest/getting-started/databricks-trial.html)).

Please see [the guide to running notebooks here](https://www.mlflow.org/docs/latest/getting-started/running-notebooks/) for more information on setting the tracking server uri and configuring access to either managed or self-managed MLflow tracking servers.

python

```python
# NOTE: review the links mentioned above for guidance on connecting to a managed tracking server, such as the Databricks managed MLflow

mlflow.set_tracking_uri("http://localhost:8080")

```

### Generate our synthetic training data[​](#generate-our-synthetic-training-data "Direct link to Generate our synthetic training data")

If you've followed along with the introductory tutorial "Logging your first model with MLflow", then you're familiar with the apples sales data generator that we created for that tutorial.

Here, we're expanding upon the data to create a slightly more complex dataset that should have improved correlation effects between the features and the target variable (the demand).

python

```python
def generate_apple_sales_data_with_promo_adjustment(
  base_demand: int = 1000,
  n_rows: int = 5000,
  competitor_price_effect: float = -50.0,
):
  """
  Generates a synthetic dataset for predicting apple sales demand with multiple
  influencing factors.

  This function creates a pandas DataFrame with features relevant to apple sales.
  The features include date, average_temperature, rainfall, weekend flag, holiday flag,
  promotional flag, price_per_kg, competitor's price, marketing intensity, stock availability,
  and the previous day's demand. The target variable, 'demand', is generated based on a
  combination of these features with some added noise.

  Args:
      base_demand (int, optional): Base demand for apples. Defaults to 1000.
      n_rows (int, optional): Number of rows (days) of data to generate. Defaults to 5000.
      competitor_price_effect (float, optional): Effect of competitor's price being lower
                                                 on our sales. Defaults to -50.

  Returns:
      pd.DataFrame: DataFrame with features and target variable for apple sales prediction.

  Example:
      >>> df = generate_apple_sales_data_with_promo_adjustment(base_demand=1200, n_rows=6000)
      >>> df.head()
  """

  # Set seed for reproducibility
  np.random.seed(9999)

  # Create date range
  dates = [datetime.now() - timedelta(days=i) for i in range(n_rows)]
  dates.reverse()

  # Generate features
  df = pd.DataFrame(
      {
          "date": dates,
          "average_temperature": np.random.uniform(10, 35, n_rows),
          "rainfall": np.random.exponential(5, n_rows),
          "weekend": [(date.weekday() >= 5) * 1 for date in dates],
          "holiday": np.random.choice([0, 1], n_rows, p=[0.97, 0.03]),
          "price_per_kg": np.random.uniform(0.5, 3, n_rows),
          "month": [date.month for date in dates],
      }
  )

  # Introduce inflation over time (years)
  df["inflation_multiplier"] = 1 + (df["date"].dt.year - df["date"].dt.year.min()) * 0.03

  # Incorporate seasonality due to apple harvests
  df["harvest_effect"] = np.sin(2 * np.pi * (df["month"] - 3) / 12) + np.sin(
      2 * np.pi * (df["month"] - 9) / 12
  )

  # Modify the price_per_kg based on harvest effect
  df["price_per_kg"] = df["price_per_kg"] - df["harvest_effect"] * 0.5

  # Adjust promo periods to coincide with periods lagging peak harvest by 1 month
  peak_months = [4, 10]  # months following the peak availability
  df["promo"] = np.where(
      df["month"].isin(peak_months),
      1,
      np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
  )

  # Generate target variable based on features
  base_price_effect = -df["price_per_kg"] * 50
  seasonality_effect = df["harvest_effect"] * 50
  promo_effect = df["promo"] * 200

  df["demand"] = (
      base_demand
      + base_price_effect
      + seasonality_effect
      + promo_effect
      + df["weekend"] * 300
      + np.random.normal(0, 50, n_rows)
  ) * df["inflation_multiplier"]  # adding random noise

  # Add previous day's demand
  df["previous_days_demand"] = df["demand"].shift(1)
  df["previous_days_demand"].fillna(method="bfill", inplace=True)  # fill the first row

  # Introduce competitor pricing
  df["competitor_price_per_kg"] = np.random.uniform(0.5, 3, n_rows)
  df["competitor_price_effect"] = (
      df["competitor_price_per_kg"] < df["price_per_kg"]
  ) * competitor_price_effect

  # Stock availability based on past sales price (3 days lag with logarithmic decay)
  log_decay = -np.log(df["price_per_kg"].shift(3) + 1) + 2
  df["stock_available"] = np.clip(log_decay, 0.7, 1)

  # Marketing intensity based on stock availability
  # Identify where stock is above threshold
  high_stock_indices = df[df["stock_available"] > 0.95].index

  # For each high stock day, increase marketing intensity for the next week
  for idx in high_stock_indices:
      df.loc[idx : min(idx + 7, n_rows - 1), "marketing_intensity"] = np.random.uniform(0.7, 1)

  # If the marketing_intensity column already has values, this will preserve them;
  #  if not, it sets default values
  fill_values = pd.Series(np.random.uniform(0, 0.5, n_rows), index=df.index)
  df["marketing_intensity"].fillna(fill_values, inplace=True)

  # Adjust demand with new factors
  df["demand"] = df["demand"] + df["competitor_price_effect"] + df["marketing_intensity"]

  # Drop temporary columns
  df.drop(
      columns=[
          "inflation_multiplier",
          "harvest_effect",
          "month",
          "competitor_price_effect",
          "stock_available",
      ],
      inplace=True,
  )

  return df

```

python

```python
df = generate_apple_sales_data_with_promo_adjustment(base_demand=1_000, n_rows=5000)
df

```

|      | date                       | average\_temperature | rainfall  | weekend | holiday | price\_per\_kg | promo | demand      | previous\_days\_demand | competitor\_price\_per\_kg | marketing\_intensity |
| ---- | -------------------------- | -------------------- | --------- | ------- | ------- | -------------- | ----- | ----------- | ---------------------- | -------------------------- | -------------------- |
| 0    | 2010-01-14 11:52:20.662955 | 30.584727            | 1.199291  | 0       | 0       | 1.726258       | 0     | 851.375336  | 851.276659             | 1.935346                   | 0.098677             |
| 1    | 2010-01-15 11:52:20.662954 | 15.465069            | 1.037626  | 0       | 0       | 0.576471       | 0     | 906.855943  | 851.276659             | 2.344720                   | 0.019318             |
| 2    | 2010-01-16 11:52:20.662954 | 10.786525            | 5.656089  | 1       | 0       | 2.513328       | 0     | 1108.304909 | 906.836626             | 0.998803                   | 0.409485             |
| 3    | 2010-01-17 11:52:20.662953 | 23.648154            | 12.030937 | 1       | 0       | 1.839225       | 0     | 1099.833810 | 1157.895424            | 0.761740                   | 0.872803             |
| 4    | 2010-01-18 11:52:20.662952 | 13.861391            | 4.303812  | 0       | 0       | 1.531772       | 0     | 983.949061  | 1148.961007            | 2.123436                   | 0.820779             |
| ...  | ...                        | ...                  | ...       | ...     | ...     | ...            | ...   | ...         | ...                    | ...                        | ...                  |
| 4995 | 2023-09-18 11:52:20.659592 | 21.643051            | 3.821656  | 0       | 0       | 2.391010       | 0     | 1140.210762 | 1563.064082            | 1.504432                   | 0.756489             |
| 4996 | 2023-09-19 11:52:20.659591 | 13.808813            | 1.080603  | 0       | 1       | 0.898693       | 0     | 1285.149505 | 1189.454273            | 1.343586                   | 0.742145             |
| 4997 | 2023-09-20 11:52:20.659590 | 11.698227            | 1.911000  | 0       | 0       | 2.839860       | 0     | 965.171368  | 1284.407359            | 2.771896                   | 0.742145             |
| 4998 | 2023-09-21 11:52:20.659589 | 18.052081            | 1.000521  | 0       | 0       | 1.188440       | 0     | 1368.369501 | 1014.429223            | 2.564075                   | 0.742145             |
| 4999 | 2023-09-22 11:52:20.659584 | 17.017294            | 0.650213  | 0       | 0       | 2.131694       | 0     | 1261.301286 | 1367.627356            | 0.785727                   | 0.833140             |

5000 rows × 11 columns

### Examining Feature-Target Correlations[​](#examining-feature-target-correlations "Direct link to Examining Feature-Target Correlations")

Before delving into the model building process, it's essential to understand the relationships between our features and the target variable. The upcoming function will display a plot indicating the correlation coefficient for each feature in relation to the target. Here's why this step is crucial:

1. **Avoiding Data Leakage**: We must ensure that no feature perfectly correlates with the target (a correlation coefficient of 1.0). If such a correlation exists, it's a sign that our dataset might be "leaking" information about the target. Using such data for hyperparameter tuning would mislead the model, as it could easily achieve a perfect score without genuinely learning the underlying patterns.

2. **Ensuring Meaningful Relationships**: Ideally, our features should have some degree of correlation with the target. If all features have correlation coefficients close to zero, it suggests a weak linear relationship. Although this doesn't automatically render the features useless, it does introduce challenges:

   * *Predictive Power*: The model might struggle to make accurate predictions.
   * *Overfitting Risk*: With weak correlations, there's a heightened risk of the model fitting to noise rather than genuine patterns, leading to overfitting.
   * *Complexity*: Demonstrating non-linear relationships or interactions between features would necessitate more intricate visualizations and evaluations.

3. **Auditing and Traceability**: Logging this correlation visualization with our main MLflow run ensures traceability. It provides a snapshot of the data characteristics at the time of the model training, which is invaluable for auditing and replicability purposes.

As we proceed, remember that while understanding correlations is a powerful tool, it's just one piece of the puzzle. Let's visualize these relationships to gain more insights!

python

```python
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_with_demand(df, save_path=None):
  """
  Plots the correlation of each variable in the dataframe with the 'demand' column.

  Args:
  - df (pd.DataFrame): DataFrame containing the data, including a 'demand' column.
  - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

  Returns:
  - None (Displays the plot on a Jupyter window)
  """

  # Compute correlations between all variables and 'demand'
  correlations = df.corr()["demand"].drop("demand").sort_values()

  # Generate a color palette from red to green
  colors = sns.diverging_palette(10, 130, as_cmap=True)
  color_mapped = correlations.map(colors)

  # Set Seaborn style
  sns.set_style(
      "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
  )  # Light grey background and thicker grid lines

  # Create bar plot
  fig = plt.figure(figsize=(12, 8))
  plt.barh(correlations.index, correlations.values, color=color_mapped)

  # Set labels and title with increased font size
  plt.title("Correlation with Demand", fontsize=18)
  plt.xlabel("Correlation Coefficient", fontsize=16)
  plt.ylabel("Variable", fontsize=16)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(axis="x")

  plt.tight_layout()

  # Save the plot if save_path is specified
  if save_path:
      plt.savefig(save_path, format="png", dpi=600)

  # prevent matplotlib from displaying the chart every time we call this function
  plt.close(fig)

  return fig


# Test the function
correlation_plot = plot_correlation_with_demand(df, save_path="correlation_plot.png")

```

#### Investigating Feature Correlation with the Target Variable[​](#investigating-feature-correlation-with-the-target-variable "Direct link to Investigating Feature Correlation with the Target Variable")

In the code above, we've intentionally disabled the automatic display of the plot generated by Matplotlib. During machine learning experimentation, it's often not useful to display figures directly within the notebook for several reasons. Instead, we aim to associate this plot with the results of an iterative experiment run. To achieve this, we'll save the plot to our MLflow tracking system. This provides us with a detailed record, linking the state of the dataset to the logged model, its parameters, and performance metrics.

##### Why Not Display Plots Directly in the Notebook?[​](#why-not-display-plots-directly-in-the-notebook "Direct link to Why Not Display Plots Directly in the Notebook?")

Choosing not to display plots within the notebook is a deliberate decision, and the reasons are multiple. Some of the key points include:

* **Ephemerality of Notebooks**: Notebooks are inherently transient; they are not designed to be a permanent record of your work.

* **Risk of Obsolescence**: If you rerun portions of your notebook, the plot displayed could become outdated or misleading, posing a risk when interpreting results.

* **Loss of Previous State**: If you happen to rerun the entire notebook, the plot will be lost. While some plugins can recover previous cell states, setting this up can be cumbersome and time-consuming.

In contrast, logging the plot to MLflow ensures that we have a permanent, easily accessible record that correlates directly with other experiment data. This is invaluable for maintaining the integrity and reproducibility of your machine learning experiments.

##### Displaying the Plot for This Guide[​](#displaying-the-plot-for-this-guide "Direct link to Displaying the Plot for This Guide")

For the purposes of this guide, we'll still take a moment to examine the plot. We can do this by explicitly calling the figure object we've returned from our function.

python

```python
correlation_plot

```

![](data:image/png;base64,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)

#### Visualizing Model Residuals for Diagnostic Insights[​](#visualizing-model-residuals-for-diagnostic-insights "Direct link to Visualizing Model Residuals for Diagnostic Insights")

The `plot_residuals` function serves to visualize the residuals—the differences between the model's predictions and the actual values in the validation set. Residual plots are crucial diagnostic tools in machine learning, as they can reveal patterns that suggest our model is either failing to capture some aspect of the data or that there's a systematic issue with the model itself.

##### Why Residual Plots?[​](#why-residual-plots "Direct link to Why Residual Plots?")

Residual plots offer several advantages:

* **Identifying Bias**: If residuals show a trend (not centered around zero), it might indicate that your model is systematically over- or under-predicting the target variable.

* **Heteroskedasticity**: Varying spread of residuals across the range of the predicted values can indicate 'Heteroskedasticity,' which can violate assumptions in some modeling techniques.

* **Outliers**: Points far away from the zero line can be considered as outliers and might warrant further investigation.

##### Auto-saving the Plot[​](#auto-saving-the-plot "Direct link to Auto-saving the Plot")

Just like with the correlation plot, this function allows you to save the residual plot to a specific path. This feature aligns with our broader strategy of logging important figures to MLflow for more effective model tracking and auditing.

##### Plot Structure[​](#plot-structure "Direct link to Plot Structure")

In the scatter plot, each point represents the residual for a specific observation in the validation set. The red horizontal line at zero serves as a reference, indicating where residuals would lie if the model's predictions were perfect.

For the sake of this guide, we will be generating this plot, but not examining it until later when we see it within the MLflow UI.

python

```python
def plot_residuals(model, dvalid, valid_y, save_path=None):
  """
  Plots the residuals of the model predictions against the true values.

  Args:
  - model: The trained XGBoost model.
  - dvalid (xgb.DMatrix): The validation data in XGBoost DMatrix format.
  - valid_y (pd.Series): The true values for the validation set.
  - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

  Returns:
  - None (Displays the residuals plot on a Jupyter window)
  """

  # Predict using the model
  preds = model.predict(dvalid)

  # Calculate residuals
  residuals = valid_y - preds

  # Set Seaborn style
  sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

  # Create scatter plot
  fig = plt.figure(figsize=(12, 8))
  plt.scatter(valid_y, residuals, color="blue", alpha=0.5)
  plt.axhline(y=0, color="r", linestyle="-")

  # Set labels, title and other plot properties
  plt.title("Residuals vs True Values", fontsize=18)
  plt.xlabel("True Values", fontsize=16)
  plt.ylabel("Residuals", fontsize=16)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(axis="y")

  plt.tight_layout()

  # Save the plot if save_path is specified
  if save_path:
      plt.savefig(save_path, format="png", dpi=600)

  # Show the plot
  plt.close(fig)

  return fig

```

#### Understanding Feature Importance with XGBoost[​](#understanding-feature-importance-with-xgboost "Direct link to Understanding Feature Importance with XGBoost")

The `plot_feature_importance` function is designed to visualize the importance of each feature used in our XGBoost model. Understanding feature importance can offer critical insights into the model's decision-making process and can aid in feature selection, engineering, and interpretation.

##### Types of Feature Importance[​](#types-of-feature-importance "Direct link to Types of Feature Importance")

XGBoost offers multiple ways to interpret feature importance. This function supports:

* **Weight**: Number of times a feature appears in a tree across the ensemble of trees (for `gblinear` booster).
* **Gain**: Average gain (or improvement to the model) of the feature when it is used in trees (for other booster types).

We automatically select the appropriate importance type based on the booster used in the XGBoost model.

##### Why Feature Importance Matters[​](#why-feature-importance-matters "Direct link to Why Feature Importance Matters")

Understanding feature importance offers several advantages:

* **Interpretability**: Knowing which features are most influential helps us understand the model better.
* **Feature Selection**: Unimportant features can potentially be dropped to simplify the model.
* **Domain Understanding**: Aligns the model's importance scale with domain-specific knowledge or intuition.

##### Saving and Accessing the Plot[​](#saving-and-accessing-the-plot "Direct link to Saving and Accessing the Plot")

This function returns a Matplotlib figure object that you can further manipulate or save. Like the previous plots, it is advisable to log this plot in MLflow for an immutable record of your model's interpretive characteristics.

##### Navigating the Plot[​](#navigating-the-plot "Direct link to Navigating the Plot")

In the resulting plot, each bar represents a feature used in the model. The length of the bar corresponds to the feature's importance, as calculated by the selected importance type.

We need a model to be trained in order to generate this plot. As such, we'll be generating, but not displaying the plot when we train the model. The resulting figure will be logged to MLflow and visible within the UI.

python

```python
def plot_feature_importance(model, booster):
  """
  Plots feature importance for an XGBoost model.

  Args:
  - model: A trained XGBoost model

  Returns:
  - fig: The matplotlib figure object
  """
  fig, ax = plt.subplots(figsize=(10, 8))
  importance_type = "weight" if booster == "gblinear" else "gain"
  xgb.plot_importance(
      model,
      importance_type=importance_type,
      ax=ax,
      title=f"Feature Importance based on {importance_type}",
  )
  plt.tight_layout()
  plt.close(fig)

  return fig

```

### Setting Up the MLflow Experiment[​](#setting-up-the-mlflow-experiment "Direct link to Setting Up the MLflow Experiment")

Before we start our hyperparameter tuning process, we need to designate a specific "experiment" within MLflow to track and log our results. An experiment in MLflow is essentially a named set of runs. Each run within an experiment tracks its own parameters, metrics, tags, and artifacts.

#### Why create a new experiment?[​](#why-create-a-new-experiment "Direct link to Why create a new experiment?")

1. **Organization**: It helps in keeping our runs organized under a specific task or project, making it easier to compare and analyze results.
2. **Isolation**: By isolating different tasks or projects into separate experiments, we prevent accidental overwrites or misinterpretations of results.

The `get_or_create_experiment` function we've defined below aids in this process. It checks if an experiment with the specified name already exists. If yes, it retrieves its ID. If not, it creates a new experiment and returns its ID.

#### How will we use the experiment\_id?[​](#how-will-we-use-the-experiment_id "Direct link to How will we use the experiment_id?")

The retrieved or created experiment\_id becomes crucial when we initiate our hyperparameter tuning. As we start the parent run for tuning, the experiment\_id ensures that the run, along with its nested child runs, gets logged under the correct experiment. It provides a structured way to navigate, compare, and analyze our tuning results within the MLflow UI.

When we want to try additional parameter ranges, different parameters, or a slightly modified dataset, we can use this Experiment to log all parent runs to keep our MLflow Tracking UI clean and easy to navigate.

Let's proceed and set up our experiment!

python

```python
def get_or_create_experiment(experiment_name):
  """
  Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

  This function checks if an experiment with the given name exists within MLflow.
  If it does, the function returns its ID. If not, it creates a new experiment
  with the provided name and returns its ID.

  Parameters:
  - experiment_name (str): Name of the MLflow experiment.

  Returns:
  - str: ID of the existing or newly created MLflow experiment.
  """

  if experiment := mlflow.get_experiment_by_name(experiment_name):
      return experiment.experiment_id
  else:
      return mlflow.create_experiment(experiment_name)

```

#### Create an experiment for our hyperparameter tuning runs[​](#create-an-experiment-for-our-hyperparameter-tuning-runs "Direct link to Create an experiment for our hyperparameter tuning runs")

python

```python
experiment_id = get_or_create_experiment("Apples Demand")

```

We can view the `experiment_id` that was either generated or fetched to see how this unique reference key looks. The value generated here is also visible within the MLflow UI.

python

```python
experiment_id

```

#### Setting Up MLflow and Data Preprocessing for Model Training[​](#setting-up-mlflow-and-data-preprocessing-for-model-training "Direct link to Setting Up MLflow and Data Preprocessing for Model Training")

This section of the code accomplishes two major tasks: initializing an MLflow experiment for usage in run tracking and preparing the dataset for model training and validation.

##### MLflow Initialization[​](#mlflow-initialization "Direct link to MLflow Initialization")

We start by setting the MLflow experiment using the `set_experiment` function. The `experiment_id` serves as a unique identifier for the experiment, allowing us to segregate and manage different runs and their associated data efficiently.

##### Data Preprocessing[​](#data-preprocessing "Direct link to Data Preprocessing")

The next steps involve preparing the dataset for model training:

1. **Feature Selection**: We drop the columns 'date' and 'demand' from our DataFrame, retaining only the feature columns in `X`.

2. **Target Variable**: The 'demand' column is designated as our target variable `y`.

3. **Data Splitting**: We split the dataset into training (`train_x`, `train_y`) and validation (`valid_x`, `valid_y`) sets using a 75-25 split.

4. **XGBoost Data Format**: Finally, we convert the training and validation datasets into XGBoost's DMatrix format. This optimized data structure speeds up the training process and is required for using XGBoost's advanced functionalities.

##### Why These Steps Matter[​](#why-these-steps-matter "Direct link to Why These Steps Matter")

* **MLflow Tracking**: Initializing the MLflow experiment ensures that all subsequent model runs, metrics, and artifacts are logged under the same experiment, making it easier to compare and analyze different models. While we are using the `fluent API` to do this here, you can also specify the `experiment_id` within a `start_run()` context.

* **Data Preparation**: Properly preparing your data ensures that the model training process will proceed without issues and that the results will be as accurate as possible.

In the next steps, we'll proceed to model training and evaluation, and all these preparation steps will come into play.

python

```python
# Set the current active MLflow experiment
mlflow.set_experiment(experiment_id=experiment_id)

# Preprocess the dataset
X = df.drop(columns=["date", "demand"])
y = df["demand"]
train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)

```

#### Hyperparameter Tuning and Model Training using Optuna and MLflow[​](#hyperparameter-tuning-and-model-training-using-optuna-and-mlflow "Direct link to Hyperparameter Tuning and Model Training using Optuna and MLflow")

The `objective` function serves as the core of our hyperparameter tuning process using Optuna. Additionally, it trains an XGBoost model using the selected hyperparameters and logs metrics and parameters to MLflow.

##### MLflow Nested Runs[​](#mlflow-nested-runs "Direct link to MLflow Nested Runs")

The function starts a new nested run in MLflow. Nested runs are useful for organizing hyperparameter tuning experiments as they allow you to group individual runs under a parent run.

##### Defining Hyperparameters[​](#defining-hyperparameters "Direct link to Defining Hyperparameters")

Optuna's `trial.suggest_*` methods are used to define a range of possible values for hyperparameters. Here's what each hyperparameter does:

* `objective` and `eval_metric`: Define the loss function and evaluation metric.
* `booster`: Type of boosting to be used (`gbtree`, `gblinear`, or `dart`).
* `lambda` and `alpha`: Regularization parameters.
* Additional parameters like `max_depth`, `eta`, and `gamma` are specific to tree-based models (`gbtree` and `dart`).

##### Model Training[​](#model-training "Direct link to Model Training")

An XGBoost model is trained using the chosen hyperparameters and the preprocessed training dataset (`dtrain`). Predictions are made on the validation set (`dvalid`), and the mean squared error (`mse`) is calculated.

##### Logging with MLflow[​](#logging-with-mlflow "Direct link to Logging with MLflow")

All the selected hyperparameters and metrics (`mse` and `rmse`) are logged to MLflow for later analysis and comparison.

* `mlflow.log_params`: Logs the hyperparameters.
* `mlflow.log_metric`: Logs the metrics.

##### Why This Function is Important[​](#why-this-function-is-important "Direct link to Why This Function is Important")

* **Automated Tuning**: Optuna automates the process of finding the best hyperparameters.
* **Experiment Tracking**: MLflow allows us to keep track of each run's hyperparameters and performance metrics, making it easier to analyze, compare, and reproduce experiments later.

In the next step, this objective function will be used by Optuna to find the optimal set of hyperparameters for our XGBoost model.

#### Housekeeping: Streamlining Logging for Optuna Trials[​](#housekeeping-streamlining-logging-for-optuna-trials "Direct link to Housekeeping: Streamlining Logging for Optuna Trials")

As we embark on our hyperparameter tuning journey with Optuna, it's essential to understand that the process can generate a multitude of runs. In fact, so many that the standard output (stdout) from the default logger can quickly become inundated, producing pages upon pages of log reports.

While the verbosity of the default logging configuration is undeniably valuable during the code development phase, initiating a full-scale trial can result in an overwhelming amount of information. Considering this, logging every single detail to stdout becomes less practical, especially when we have dedicated tools like MLflow to meticulously track our experiments.

To strike a balance, we'll utilize callbacks to tailor our logging behavior.

##### Implementing a Logging Callback:[​](#implementing-a-logging-callback "Direct link to Implementing a Logging Callback:")

The callback we're about to introduce will modify the default reporting behavior. Instead of logging every trial, we'll only receive updates when a new hyperparameter combination yields an improvement over the best metric value recorded thus far.

This approach offers two salient benefits:

1. **Enhanced Readability**: By filtering out the extensive log details and focusing only on the trials that show improvement, we can gauge the efficacy of our hyperparameter search. For instance, if we observe a diminishing frequency of 'best result' reports early on, it might suggest that fewer iterations would suffice to pinpoint an optimal hyperparameter set. On the other hand, a consistent rate of improvement might indicate that our feature set requires further refinement.

2. **Progress Indicators**: Especially pertinent for extensive trials that span hours or even days, receiving periodic updates provides assurance that the process is still in motion. These 'heartbeat' notifications affirm that our system is diligently at work, even if it's not flooding stdout with every minute detail.

Moreover, MLflow's user interface (UI) complements this strategy. As each trial concludes, MLflow logs the child run, making it accessible under the umbrella of the parent run.

In the ensuing code, we:

1. Adjust Optuna's logging level to report only errors, ensuring a decluttered stdout.
2. Define a `champion_callback` function, tailored to log only when a trial surpasses the previously recorded best metric.

Let's dive into the implementation:

python

```python
# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


def champion_callback(study, frozen_trial):
  """
  Logging callback that will report when a new trial iteration improves upon existing
  best trial values.

  Note: This callback is not intended for use in distributed computing systems such as Spark
  or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
  workers or agents.
  The race conditions with file system state management for distributed trials will render
  inconsistent values with this callback.
  """

  winner = study.user_attrs.get("winner", None)

  if study.best_value and winner != study.best_value:
      study.set_user_attr("winner", study.best_value)
      if winner:
          improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
          print(
              f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
              f"{improvement_percent: .4f}% improvement"
          )
      else:
          print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

```

python

```python
def objective(trial):
  with mlflow.start_run(nested=True):
      # Define hyperparameters
      params = {
          "objective": "reg:squarederror",
          "eval_metric": "rmse",
          "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
          "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
          "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
      }

      if params["booster"] == "gbtree" or params["booster"] == "dart":
          params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
          params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
          params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
          params["grow_policy"] = trial.suggest_categorical(
              "grow_policy", ["depthwise", "lossguide"]
          )

      # Train XGBoost model
      bst = xgb.train(params, dtrain)
      preds = bst.predict(dvalid)
      error = mean_squared_error(valid_y, preds)

      # Log to MLflow
      mlflow.log_params(params)
      mlflow.log_metric("mse", error)
      mlflow.log_metric("rmse", math.sqrt(error))

  return error

```

#### Orchestrating Hyperparameter Tuning, Model Training, and Logging with MLflow[​](#orchestrating-hyperparameter-tuning-model-training-and-logging-with-mlflow "Direct link to Orchestrating Hyperparameter Tuning, Model Training, and Logging with MLflow")

This section of the code serves as the orchestration layer, bringing together Optuna for hyperparameter tuning and MLflow for experiment tracking.

##### Initiating Parent Run[​](#initiating-parent-run "Direct link to Initiating Parent Run")

We begin by starting a parent MLflow run with the name "Best Run". All subsequent operations, including Optuna's trials, are nested under this parent run, providing a structured way to organize our experiments.

##### Hyperparameter Tuning with Optuna[​](#hyperparameter-tuning-with-optuna "Direct link to Hyperparameter Tuning with Optuna")

* `study = optuna.create_study(direction='minimize')`: We create an Optuna study object aiming to minimize our objective function.
* `study.optimize(objective, n_trials=10)`: The `objective` function is optimized over 10 trials.

##### Logging Best Parameters and Metrics[​](#logging-best-parameters-and-metrics "Direct link to Logging Best Parameters and Metrics")

After Optuna finds the best hyperparameters, we log these, along with the best mean squared error (`mse`) and root mean squared error (`rmse`), to MLflow.

##### Logging Additional Metadata[​](#logging-additional-metadata "Direct link to Logging Additional Metadata")

Using `mlflow.set_tags`, we log additional metadata like the project name, optimization engine, model family, and feature set version. This helps in better categorizing and understanding the context of the model run.

##### Model Training and Artifact Logging[​](#model-training-and-artifact-logging "Direct link to Model Training and Artifact Logging")

* We train an XGBoost model using the best hyperparameters.
* Various plots—correlation with demand, feature importance, and residuals—are generated and logged as artifacts in MLflow.

##### Model Serialization and Logging[​](#model-serialization-and-logging "Direct link to Model Serialization and Logging")

Finally, the trained model is logged to MLflow using `mlflow.xgboost.log_model`, along with an example input and additional metadata. The model is stored in a specified artifact path and its URI is retrieved.

##### Why This Block is Crucial[​](#why-this-block-is-crucial "Direct link to Why This Block is Crucial")

* **End-to-End Workflow**: This code block represents an end-to-end machine learning workflow, from hyperparameter tuning to model evaluation and logging.
* **Reproducibility**: All details about the model, including hyperparameters, metrics, and visual diagnostics, are logged, ensuring that the experiment is fully reproducible.
* **Analysis and Comparison**: With all data logged in MLflow, it becomes easier to analyze the performance of various runs and choose the best model for deployment.

In the next steps, we'll explore how to retrieve and use the logged model for inference.

#### Setting a Descriptive Name for the Model Run[​](#setting-a-descriptive-name-for-the-model-run "Direct link to Setting a Descriptive Name for the Model Run")

Before proceeding with model training and hyperparameter tuning, it's beneficial to assign a descriptive name to our MLflow run. This name serves as a human-readable identifier, making it easier to track, compare, and analyze different runs.

##### The Importance of Naming Runs:[​](#the-importance-of-naming-runs "Direct link to The Importance of Naming Runs:")

* **Reference by Name**: While MLflow provides unique identifying keys like `run_id` for each run, having a descriptive name allows for more intuitive referencing, especially when using particular APIs and navigating the MLflow UI.

* **Clarity and Context**: A well-chosen run name can provide context about the hypothesis being tested or the specific modifications made, aiding in understanding the purpose and rationale of a particular run.

* **Automatic Naming**: If you don't specify a run name, MLflow will generate a unique fun name for you. However, this might lack the context and clarity of a manually chosen name.

##### Best Practices:[​](#best-practices "Direct link to Best Practices:")

When naming your runs, consider the following:

1. **Relevance to Code Changes**: The name should reflect any code or parameter modifications made for that run.
2. **Iterative Runs**: If you're executing multiple runs iteratively, it's a good idea to update the run name for each iteration to avoid confusion.

In the subsequent steps, we will set a name for our parent run. Remember, if you execute the model training multiple times, consider updating the run name for clarity.

python

```python
run_name = "first_attempt"

```

python

```python
# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
  # Initialize the Optuna study
  study = optuna.create_study(direction="minimize")

  # Execute the hyperparameter optimization trials.
  # Note the addition of the `champion_callback` inclusion to control our logging
  study.optimize(objective, n_trials=500, callbacks=[champion_callback])

  mlflow.log_params(study.best_params)
  mlflow.log_metric("best_mse", study.best_value)
  mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

  # Log tags
  mlflow.set_tags(
      tags={
          "project": "Apple Demand Project",
          "optimizer_engine": "optuna",
          "model_family": "xgboost",
          "feature_set_version": 1,
      }
  )

  # Log a fit model instance
  model = xgb.train(study.best_params, dtrain)

  # Log the correlation plot
  mlflow.log_figure(figure=correlation_plot, artifact_file="correlation_plot.png")

  # Log the feature importances plot
  importances = plot_feature_importance(model, booster=study.best_params.get("booster"))
  mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

  # Log the residuals plot
  residuals = plot_residuals(model, dvalid, valid_y)
  mlflow.log_figure(figure=residuals, artifact_file="residuals.png")

  artifact_path = "model"

  mlflow.xgboost.log_model(
      xgb_model=model,
      name=artifact_path,
      input_example=train_x.iloc[[0]],
      model_format="ubj",
      metadata={"model_data_version": 1},
  )

  # Get the logged model uri so that we can load it from the artifact store
  model_uri = mlflow.get_artifact_uri(artifact_path)

```

#### Understanding the Artifact URI in MLflow[​](#understanding-the-artifact-uri-in-mlflow "Direct link to Understanding the Artifact URI in MLflow")

The output 'mlflow-artifacts\:/908436739760555869/c8d64ce51f754eb698a3c09239bcdcee/artifacts/model' represents a unique Uniform Resource Identifier (URI) for the trained model artifacts within MLflow. This URI is a crucial component of MLflow's architecture, and here's why:

##### Simplified Access to Model Artifacts[​](#simplified-access-to-model-artifacts "Direct link to Simplified Access to Model Artifacts")

The `model_uri` abstracts away the underlying storage details, providing a consistent and straightforward way to reference model artifacts, regardless of where they are stored. Whether your artifacts are on a local filesystem, in a cloud storage bucket, or on a network mount, the URI remains a consistent reference point.

##### Abstraction of Storage Details[​](#abstraction-of-storage-details "Direct link to Abstraction of Storage Details")

MLflow is designed to be storage-agnostic. This means that while you might switch the backend storage from, say, a local directory to an Amazon S3 bucket, the way you interact with MLflow remains consistent. The URI ensures that you don't need to know the specifics of the storage backend; you only need to reference the model's URI.

##### Associated Information and Metadata[​](#associated-information-and-metadata "Direct link to Associated Information and Metadata")

Beyond just the model files, the URI provides access to associated metadata, the model artifact, and other logged artifacts (files and images). This ensures that you have a comprehensive set of information about the model, aiding in reproducibility, analysis, and deployment.

##### In Summary[​](#in-summary "Direct link to In Summary")

The `model_uri` serves as a consistent, abstracted reference to your model and its associated data. It simplifies interactions with MLflow, ensuring that users don't need to worry about the specifics of underlying storage mechanisms and can focus on the machine learning workflow.

python

```python
model_uri

```

#### Loading the Trained Model with MLflow[​](#loading-the-trained-model-with-mlflow "Direct link to Loading the Trained Model with MLflow")

With the line:

python

```python
loaded = mlflow.xgboost.load_model(model_uri)

```

we're leveraging MLflow's native model loader for XGBoost. Instead of using the generic pyfunc loader, which provides a universal Python function interface for models, we're using the XGBoost-specific loader.

##### Benefits of Native Loading:[​](#benefits-of-native-loading "Direct link to Benefits of Native Loading:")

* **Fidelity**: Loading the model using the native loader ensures that you're working with the exact same model object as it was during training. This means all nuances, specifics, and intricacies of the original model are preserved.

* **Functionality**: With the native model object in hand, you can utilize all of its inherent methods and properties. This allows for more flexibility, especially when you need advanced features or fine-grained control during inference.

* **Performance**: Using the native model object might offer performance benefits, especially when performing batch inference or deploying the model in environments optimized for the specific machine learning framework.

In essence, by loading the model natively, we ensure maximum compatibility and functionality, allowing for a seamless transition from training to inference.

python

```python
loaded = mlflow.xgboost.load_model(model_uri)

```

#### Example: Batch Inference Using the Loaded Model[​](#example-batch-inference-using-the-loaded-model "Direct link to Example: Batch Inference Using the Loaded Model")

After loading the model natively, performing batch inference is straightforward.

In the following cell, we're going to perform a prediction based on the entire source feature set. Although doing an inference action on the entire training and validation dataset features is of very limited utility in a real-world application, we'll use our generated synthetic data here to illustrate using the native model for inference.

#### Performing Batch Inference and Augmenting Data[​](#performing-batch-inference-and-augmenting-data "Direct link to Performing Batch Inference and Augmenting Data")

In this section, we're taking our entire dataset and performing batch inference using our loaded XGBoost model. We'll then append these predictions back into our original dataset to compare, analyze, or further process.

##### Steps Explained:[​](#steps-explained "Direct link to Steps Explained:")

1. **Creating a DMatrix**: `batch_dmatrix = xgb.DMatrix(X)`: We first convert our features (`X`) into XGBoost's optimized DMatrix format. This data structure is specifically designed for efficiency and speed in XGBoost.

2. **Predictions**: `inference = loaded.predict(batch_dmatrix)`: Using the previously loaded model (`loaded`), we perform batch inference on the entire dataset.

3. **Creating a New DataFrame**: `infer_df = df.copy()`: We create a copy of the original DataFrame to ensure that we're not modifying our original data.

4. **Appending Predictions**: `infer_df["predicted_demand"] = inference`: The predictions are then added as a new column, `predicted_demand`, to this DataFrame.

##### Best Practices:[​](#best-practices-1 "Direct link to Best Practices:")

* **Always Copy Data**: When augmenting or modifying datasets, it's generally a good idea to work with a copy. This ensures that the original data remains unchanged, preserving data integrity.

* **Batch Inference**: When predicting on large datasets, using batch inference (as opposed to individual predictions) can offer significant speed improvements.

* **DMatrix Conversion**: While converting to DMatrix might seem like an extra step, it's crucial for performance when working with XGBoost. It ensures that predictions are made as quickly as possible.

In the subsequent steps, we can further analyze the differences between the actual demand and our model's predicted demand, potentially visualizing the results or calculating performance metrics.

python

```python
batch_dmatrix = xgb.DMatrix(X)

inference = loaded.predict(batch_dmatrix)

infer_df = df.copy()

infer_df["predicted_demand"] = inference

```

#### Visualizing the Augmented DataFrame[​](#visualizing-the-augmented-dataframe "Direct link to Visualizing the Augmented DataFrame")

Below, we display the `infer_df` DataFrame. This augmented dataset now includes both the actual demand (`demand`) and the model's predictions (`predicted_demand`). By examining this table, we can get a quick sense of how well our model's predictions align with the actual demand values.

python

```python
infer_df

```

|      | date                       | average\_temperature | rainfall  | weekend | holiday | price\_per\_kg | promo | demand      | previous\_days\_demand | competitor\_price\_per\_kg | marketing\_intensity | predicted\_demand |
| ---- | -------------------------- | -------------------- | --------- | ------- | ------- | -------------- | ----- | ----------- | ---------------------- | -------------------------- | -------------------- | ----------------- |
| 0    | 2010-01-14 11:52:20.662955 | 30.584727            | 1.199291  | 0       | 0       | 1.726258       | 0     | 851.375336  | 851.276659             | 1.935346                   | 0.098677             | 953.708496        |
| 1    | 2010-01-15 11:52:20.662954 | 15.465069            | 1.037626  | 0       | 0       | 0.576471       | 0     | 906.855943  | 851.276659             | 2.344720                   | 0.019318             | 1013.409973       |
| 2    | 2010-01-16 11:52:20.662954 | 10.786525            | 5.656089  | 1       | 0       | 2.513328       | 0     | 1108.304909 | 906.836626             | 0.998803                   | 0.409485             | 1152.382446       |
| 3    | 2010-01-17 11:52:20.662953 | 23.648154            | 12.030937 | 1       | 0       | 1.839225       | 0     | 1099.833810 | 1157.895424            | 0.761740                   | 0.872803             | 1352.879272       |
| 4    | 2010-01-18 11:52:20.662952 | 13.861391            | 4.303812  | 0       | 0       | 1.531772       | 0     | 983.949061  | 1148.961007            | 2.123436                   | 0.820779             | 1121.233032       |
| ...  | ...                        | ...                  | ...       | ...     | ...     | ...            | ...   | ...         | ...                    | ...                        | ...                  | ...               |
| 4995 | 2023-09-18 11:52:20.659592 | 21.643051            | 3.821656  | 0       | 0       | 2.391010       | 0     | 1140.210762 | 1563.064082            | 1.504432                   | 0.756489             | 1070.676636       |
| 4996 | 2023-09-19 11:52:20.659591 | 13.808813            | 1.080603  | 0       | 1       | 0.898693       | 0     | 1285.149505 | 1189.454273            | 1.343586                   | 0.742145             | 1156.580688       |
| 4997 | 2023-09-20 11:52:20.659590 | 11.698227            | 1.911000  | 0       | 0       | 2.839860       | 0     | 965.171368  | 1284.407359            | 2.771896                   | 0.742145             | 1086.527710       |
| 4998 | 2023-09-21 11:52:20.659589 | 18.052081            | 1.000521  | 0       | 0       | 1.188440       | 0     | 1368.369501 | 1014.429223            | 2.564075                   | 0.742145             | 1085.064087       |
| 4999 | 2023-09-22 11:52:20.659584 | 17.017294            | 0.650213  | 0       | 0       | 2.131694       | 0     | 1261.301286 | 1367.627356            | 0.785727                   | 0.833140             | 1047.954102       |

5000 rows × 12 columns

#### Wrapping Up: Reflecting on Our Comprehensive Machine Learning Workflow[​](#wrapping-up-reflecting-on-our-comprehensive-machine-learning-workflow "Direct link to Wrapping Up: Reflecting on Our Comprehensive Machine Learning Workflow")

Throughout this guide, we embarked on a detailed exploration of an end-to-end machine learning workflow. We began with data preprocessing, delved deeply into hyperparameter tuning with Optuna, leveraged MLflow for structured experiment tracking, and concluded with batch inference.

##### Key Takeaways:[​](#key-takeaways "Direct link to Key Takeaways:")

* **Hyperparameter Tuning with Optuna**: We harnessed the power of Optuna to systematically search for the best hyperparameters for our XGBoost model, aiming to optimize its performance.

* **Structured Experiment Tracking with MLflow**: MLflow's capabilities shone through as we logged experiments, metrics, parameters, and artifacts. We also explored the benefits of nested child runs, allowing us to logically group and structure our experiment iterations.

* **Model Interpretation**: Various plots and metrics equipped us with insights into our model's behavior. We learned to appreciate its strengths and identify potential areas for refinement.

* **Batch Inference**: The nuances of batch predictions on extensive datasets were explored, alongside methods to seamlessly integrate these predictions back into our primary data.

* **Logging Visual Artifacts**: A significant portion of our journey emphasized the importance of logging visual artifacts, like plots, to MLflow. These visuals serve as invaluable references, capturing the state of the model, its performance, and any alterations to the feature set that might sway the model's performance metrics.

By the end of this guide, you should possess a robust understanding of a well-structured machine learning workflow. This foundation not only empowers you to craft effective models but also ensures that each step, from data wrangling to predictions, is transparent, reproducible, and efficient.

We're grateful you accompanied us on this comprehensive journey. The practices and insights gleaned will undoubtedly be pivotal in all your future machine learning endeavors!
