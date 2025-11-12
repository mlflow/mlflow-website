# Logging Visualizations with MLflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow.ipynb)

In this part of the guide, we emphasize the **importance of logging visualizations with MLflow**. Retaining visualizations alongside trained models enhances model interpretability, auditing, and provenance, ensuring a robust and transparent machine learning lifecycle.

#### What Are We Doing?[​](#what-are-we-doing "Direct link to What Are We Doing?")

* **Storing Visual Artifacts:** We are logging various plots as visual artifacts in MLflow, ensuring that they are always accessible and aligned with the corresponding model and run data.
* **Enhancing Model Interpretability:** These visualizations aid in understanding and explaining model behavior, contributing to improved model transparency and accountability.

#### How Does It Apply to MLflow?[​](#how-does-it-apply-to-mlflow "Direct link to How Does It Apply to MLflow?")

* **Integrated Visualization Logging:** MLflow seamlessly integrates facilities for logging and accessing visual artifacts, enhancing the ease and efficiency of handling visual context and insights.
* **Convenient Access:** Logged figures are displayable within the Runs view pane in the MLflow UI, ensuring quick and easy access for analysis and review.

#### Caution[​](#caution "Direct link to Caution")

While MLflow offers simplicity and convenience for logging visualizations, it's crucial to ensure the consistency and relevance of the visual artifacts with the corresponding model data, maintaining the integrity and comprehensiveness of the model information.

#### Why Is Consistent Logging Important?[​](#why-is-consistent-logging-important "Direct link to Why Is Consistent Logging Important?")

* **Auditing and Provenance:** Consistent and comprehensive logging of visualizations is pivotal for auditing purposes, ensuring that every model is accompanied by relevant visual insights for thorough analysis and review.
* **Enhanced Model Understanding:** Proper visual context enhances the understanding of model behavior, aiding in effective model evaluation, and validation.

In conclusion, MLflow's capabilities for visualization logging play an invaluable role in ensuring a comprehensive, transparent, and efficient machine learning lifecycle, reinforcing model interpretability, auditing, and provenance.

### Generating Synthetic Apple Sales Data[​](#generating-synthetic-apple-sales-data "Direct link to Generating Synthetic Apple Sales Data")

In this next section, we dive into **generating synthetic data for apple sales demand prediction** using the `generate_apple_sales_data_with_promo_adjustment` function. This function simulates a variety of features relevant to apple sales, providing a rich dataset for exploration and modeling.

#### What Are We Doing?[​](#what-are-we-doing-1 "Direct link to What Are We Doing?")

* **Simulating Realistic Data:** Generating a dataset with features like date, average temperature, rainfall, weekend flag, and more, simulating realistic scenarios for apple sales.
* **Incorporating Various Effects:** The function incorporates effects like promotional adjustments, seasonality, and competitor pricing, contributing to the 'demand' target variable.

#### How Does It Apply to Data Generation?[​](#how-does-it-apply-to-data-generation "Direct link to How Does It Apply to Data Generation?")

* **Comprehensive Dataset:** The synthetic dataset provides a comprehensive set of features and interactions, ideal for exploring diverse aspects and dimensions for demand prediction.
* **Freedom and Flexibility:** The synthetic nature allows for unconstrained exploration and analysis, devoid of real-world data sensitivities and constraints.

#### Caution[​](#caution-1 "Direct link to Caution")

While synthetic data offers numerous advantages for exploration and learning, it's crucial to acknowledge its limitations in capturing real-world complexities and nuances.

#### Why Is Acknowledging Limitations Important?[​](#why-is-acknowledging-limitations-important "Direct link to Why Is Acknowledging Limitations Important?")

* **Real-World Complexities:** Synthetic data may not capture all the intricate patterns and anomalies present in real-world data, potentially leading to over-simplified models and insights.
* **Transferability to Real-World Scenarios:** Ensuring that insights and models derived from synthetic data are transferable to real-world scenarios requires careful consideration and validation.

In conclusion, the `generate_apple_sales_data_with_promo_adjustment` function offers a robust tool for generating a comprehensive synthetic dataset for apple sales demand prediction, facilitating extensive exploration, and analysis while acknowledging the limitations of synthetic data.

python

```python
import math
import pathlib
from datetime import datetime, timedelta

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import (
  mean_absolute_error,
  mean_squared_error,
  mean_squared_log_error,
  median_absolute_error,
  r2_score,
)
from sklearn.model_selection import train_test_split

import mlflow


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

### Generating Apple Sales Data[​](#generating-apple-sales-data "Direct link to Generating Apple Sales Data")

In this cell, we call the `generate_apple_sales_data_with_promo_adjustment` function to generate a dataset of apple sales.

#### Parameters Used:[​](#parameters-used "Direct link to Parameters Used:")

* `base_demand`: Set to 1000, representing the baseline demand for apples.
* `n_rows`: Set to 10,000, determining the number of rows or data points in the generated dataset.
* `competitor_price_effect`: Set to -25.0, representing the impact on our sales when the competitor's price is lower.

By running this cell, we obtain a dataset `my_data`, which holds the synthetic apple sales data with the aforementioned configurations. This dataset will be used for further exploration and analysis in subsequent steps of this notebook.

You can see the data in the cell after the generation cell.

python

```python
my_data = generate_apple_sales_data_with_promo_adjustment(
  base_demand=1000, n_rows=10_000, competitor_price_effect=-25.0
)

```

python

```python
my_data

```

|      | date                       | average\_temperature | rainfall  | weekend | holiday | price\_per\_kg | promo | demand      | previous\_days\_demand | competitor\_price\_per\_kg | marketing\_intensity |
| ---- | -------------------------- | -------------------- | --------- | ------- | ------- | -------------- | ----- | ----------- | ---------------------- | -------------------------- | -------------------- |
| 0    | 1996-05-11 13:10:40.689999 | 30.584727            | 1.831006  | 1       | 0       | 1.578387       | 1     | 1301.647352 | 1326.324266            | 0.755725                   | 0.323086             |
| 1    | 1996-05-12 13:10:40.689999 | 15.465069            | 0.761303  | 1       | 0       | 1.965125       | 0     | 1143.972638 | 1326.324266            | 0.913934                   | 0.030371             |
| 2    | 1996-05-13 13:10:40.689998 | 10.786525            | 1.427338  | 0       | 0       | 1.497623       | 0     | 890.319248  | 1168.942267            | 2.879262                   | 0.354226             |
| 3    | 1996-05-14 13:10:40.689997 | 23.648154            | 3.737435  | 0       | 0       | 1.952936       | 0     | 811.206168  | 889.965021             | 0.826015                   | 0.953000             |
| 4    | 1996-05-15 13:10:40.689997 | 13.861391            | 5.598549  | 0       | 0       | 2.059993       | 0     | 822.279469  | 835.253168             | 1.130145                   | 0.953000             |
| ...  | ...                        | ...                  | ...       | ...     | ...     | ...            | ...   | ...         | ...                    | ...                        | ...                  |
| 9995 | 2023-09-22 13:10:40.682895 | 23.358868            | 7.061220  | 0       | 0       | 1.556829       | 1     | 1981.195884 | 2089.644454            | 0.560507                   | 0.889971             |
| 9996 | 2023-09-23 13:10:40.682895 | 14.859048            | 0.868655  | 1       | 0       | 1.632918       | 0     | 2180.698138 | 2005.305913            | 2.460766                   | 0.884467             |
| 9997 | 2023-09-24 13:10:40.682894 | 17.941035            | 13.739986 | 1       | 0       | 0.827723       | 1     | 2675.093671 | 2179.813671            | 1.321922                   | 0.884467             |
| 9998 | 2023-09-25 13:10:40.682893 | 14.533862            | 1.610512  | 0       | 0       | 0.589172       | 0     | 1703.287285 | 2674.209204            | 2.604095                   | 0.812706             |
| 9999 | 2023-09-26 13:10:40.682889 | 13.048549            | 5.287508  | 0       | 0       | 1.794122       | 1     | 1971.029266 | 1702.474579            | 1.261635                   | 0.750458             |

10000 rows × 11 columns

### Time Series Visualization of Demand[​](#time-series-visualization-of-demand "Direct link to Time Series Visualization of Demand")

In this section, we're creating a time series plot to visualize the demand data alongside its rolling average.

#### Why is this Important?[​](#why-is-this-important "Direct link to Why is this Important?")

Visualizing time series data is crucial for identifying patterns, understanding variability, and making more informed decisions. By plotting the rolling average alongside, we can smooth out short-term fluctuations and highlight longer-term trends or cycles. This visual aid is essential for understanding the data and making more accurate and informed predictions and decisions.

#### Structure of the Code:[​](#structure-of-the-code "Direct link to Structure of the Code:")

* **Input Verification**: The code first ensures the data is a pandas DataFrame.
* **Date Conversion**: It converts the 'date' column to a datetime format for accurate plotting.
* **Rolling Average Calculation**: It calculates the rolling average of the 'demand' with a specified window size (`window_size`), defaulting to 7 days.
* **Plotting**: It plots both the original demand data and the calculated rolling average on the same plot for comparison. The original demand data is plotted with low alpha to appear "ghostly," ensuring the rolling average stands out.
* **Labels and Legend**: Adequate labels and legends are added for clarity.

#### Why Return a Figure?[​](#why-return-a-figure "Direct link to Why Return a Figure?")

We return the figure object (`fig`) instead of rendering it directly so that each iteration of a model training event can consume the figure as a logged artifact to MLflow. This approach allows us to persist the state of the data visualization with precisely the state of the data that was used for training. MLflow can store this figure object, enabling easy retrieval and rendering within the MLflow UI, ensuring that the visualization is always accessible and paired with the relevant model and data information.

python

```python
def plot_time_series_demand(data, window_size=7, style="seaborn", plot_size=(16, 12)):
  if not isinstance(data, pd.DataFrame):
      raise TypeError("df must be a pandas DataFrame.")

  df = data.copy()

  df["date"] = pd.to_datetime(df["date"])

  # Calculate the rolling average
  df["rolling_avg"] = df["demand"].rolling(window=window_size).mean()

  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      # Plot the original time series data with low alpha (transparency)
      ax.plot(df["date"], df["demand"], "b-o", label="Original Demand", alpha=0.15)
      # Plot the rolling average
      ax.plot(
          df["date"],
          df["rolling_avg"],
          "r",
          label=f"{window_size}-Day Rolling Average",
      )

      # Set labels and title
      ax.set_title(
          f"Time Series Plot of Demand with {window_size} day Rolling Average",
          fontsize=14,
      )
      ax.set_xlabel("Date", fontsize=12)
      ax.set_ylabel("Demand", fontsize=12)

      # Add legend to explain the lines
      ax.legend()
      plt.tight_layout()

  plt.close(fig)
  return fig

```

### Visualizing Demand on Weekends vs. Weekdays with Box Plots[​](#visualizing-demand-on-weekends-vs-weekdays-with-box-plots "Direct link to Visualizing Demand on Weekends vs. Weekdays with Box Plots")

In this section, we're utilizing box plots to visualize the distribution of demand on weekends versus weekdays. This visualization assists in understanding the variability and central tendency of demand based on the day of the week.

#### Why is this Important?[​](#why-is-this-important-1 "Direct link to Why is this Important?")

Understanding how demand differs between weekends and weekdays is crucial for making informed decisions regarding inventory, staffing, and other operational aspects. It helps identify the periods of higher demand, allowing for better resource allocation and planning.

#### Structure of the Code:[​](#structure-of-the-code-1 "Direct link to Structure of the Code:")

* **Box Plot**: The code uses Seaborn to create a box plot that shows the distribution of demand on weekends (1) and weekdays (0). The box plot provides insights into the median, quartiles, and possible outliers in the demand data for both categories.
* **Adding Individual Data Points**: To provide more context, individual data points are overlayed on the box plot as a strip plot. They are jittered for better visualization and color-coded based on the day type.
* **Styling**: The plot is styled for clarity, and unnecessary legends are removed to enhance readability.

#### Why Return a Figure?[​](#why-return-a-figure-1 "Direct link to Why Return a Figure?")

As with the time series plot, this function also returns the figure object (`fig`) instead of displaying it directly.

python

```python
def plot_box_weekend(df, style="seaborn", plot_size=(10, 8)):
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      sns.boxplot(data=df, x="weekend", y="demand", ax=ax, color="lightgray")
      sns.stripplot(
          data=df,
          x="weekend",
          y="demand",
          ax=ax,
          hue="weekend",
          palette={0: "blue", 1: "green"},
          alpha=0.15,
          jitter=0.3,
          size=5,
      )

      ax.set_title("Box Plot of Demand on Weekends vs. Weekdays", fontsize=14)
      ax.set_xlabel("Weekend (0: No, 1: Yes)", fontsize=12)
      ax.set_ylabel("Demand", fontsize=12)
      for i in ax.get_xticklabels() + ax.get_yticklabels():
          i.set_fontsize(10)
      ax.legend_.remove()
      plt.tight_layout()
  plt.close(fig)
  return fig

```

### Exploring the Relationship Between Demand and Price per Kg[​](#exploring-the-relationship-between-demand-and-price-per-kg "Direct link to Exploring the Relationship Between Demand and Price per Kg")

In this visualization, we're creating a scatter plot to investigate the relationship between the `demand` and `price_per_kg`. Understanding this relationship is crucial for pricing strategy and demand forecasting.

#### Why is this Important?[​](#why-is-this-important-2 "Direct link to Why is this Important?")

* **Insight into Pricing Strategy:** This visualization helps reveal how demand varies with the price per kg, providing valuable insights for setting prices to optimize sales and revenue.
* **Understanding Demand Elasticity:** It aids in understanding the elasticity of demand concerning price, helping in making informed and data-driven decisions for promotions and discounts.

#### Structure of the Code:[​](#structure-of-the-code-2 "Direct link to Structure of the Code:")

* **Scatter Plot:** The code generates a scatter plot, where each point's position is determined by the `price_per_kg` and `demand`, and the color indicates whether the day is a weekend or a weekday. This color-coding helps in quickly identifying patterns specific to weekends or weekdays.
* **Transparency and Jitter:** Points are plotted with transparency (`alpha=0.15`) to handle overplotting, allowing the visualization of the density of points.
* **Regression Line:** For each subgroup (weekend and weekday), a separate regression line is fitted and plotted on the same axes. These lines provide a clear visual indication of the trend of demand concerning the price per kg for each group.

python

```python
def plot_scatter_demand_price(df, style="seaborn", plot_size=(10, 8)):
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      # Scatter plot with jitter, transparency, and color-coded based on weekend
      sns.scatterplot(
          data=df,
          x="price_per_kg",
          y="demand",
          hue="weekend",
          palette={0: "blue", 1: "green"},
          alpha=0.15,
          ax=ax,
      )
      # Fit a simple regression line for each subgroup
      sns.regplot(
          data=df[df["weekend"] == 0],
          x="price_per_kg",
          y="demand",
          scatter=False,
          color="blue",
          ax=ax,
      )
      sns.regplot(
          data=df[df["weekend"] == 1],
          x="price_per_kg",
          y="demand",
          scatter=False,
          color="green",
          ax=ax,
      )

      ax.set_title("Scatter Plot of Demand vs Price per kg with Regression Line", fontsize=14)
      ax.set_xlabel("Price per kg", fontsize=12)
      ax.set_ylabel("Demand", fontsize=12)
      for i in ax.get_xticklabels() + ax.get_yticklabels():
          i.set_fontsize(10)
      plt.tight_layout()
  plt.close(fig)
  return fig

```

### Visualizing Demand Density: Weekday vs. Weekend[​](#visualizing-demand-density-weekday-vs-weekend "Direct link to Visualizing Demand Density: Weekday vs. Weekend")

This visualization allows us to observe the distribution of `demand` separately for weekdays and weekends.

#### Why is this Important?[​](#why-is-this-important-3 "Direct link to Why is this Important?")

* **Demand Distribution Insight:** Understanding the distribution of demand on weekdays versus weekends can inform inventory management and staffing needs.
* **Informing Business Strategy:** This insight is vital for making data-driven decisions regarding promotions, discounts, and other strategies that might be more effective on specific days.

#### Structure of the Code:[​](#structure-of-the-code-3 "Direct link to Structure of the Code:")

* **Density Plot:** The code generates a density plot for `demand`, separated into weekdays and weekends.
* **Color-Coded Groups:** The two groups (weekday and weekend) are color-coded (blue and green respectively), making it easy to distinguish between them.
* **Transparency and Filling:** The areas under the density curves are filled with a light, transparent color (`alpha=0.15`) for easy visualization while avoiding visual clutter.

#### What are the Visual Elements?[​](#what-are-the-visual-elements "Direct link to What are the Visual Elements?")

* **Two Density Curves:** The plot comprises two density curves, one for weekdays and another for weekends. These curves provide a clear visual representation of the distribution of demand for each group.
* **Legend:** A legend is added to help identify which curve corresponds to which group (weekday or weekend).

python

```python
def plot_density_weekday_weekend(df, style="seaborn", plot_size=(10, 8)):
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)

      # Plot density for weekdays
      sns.kdeplot(
          df[df["weekend"] == 0]["demand"],
          color="blue",
          label="Weekday",
          ax=ax,
          fill=True,
          alpha=0.15,
      )

      # Plot density for weekends
      sns.kdeplot(
          df[df["weekend"] == 1]["demand"],
          color="green",
          label="Weekend",
          ax=ax,
          fill=True,
          alpha=0.15,
      )

      ax.set_title("Density Plot of Demand by Weekday/Weekend", fontsize=14)
      ax.set_xlabel("Demand", fontsize=12)
      ax.legend(fontsize=12)
      for i in ax.get_xticklabels() + ax.get_yticklabels():
          i.set_fontsize(10)

      plt.tight_layout()
  plt.close(fig)
  return fig

```

### Visualization of Model Coefficients[​](#visualization-of-model-coefficients "Direct link to Visualization of Model Coefficients")

In this section, we're utilizing a bar plot to visualize the coefficients of the features from the trained model.

#### Why is this Important?[​](#why-is-this-important-4 "Direct link to Why is this Important?")

Understanding the magnitude and direction of the coefficients is essential for interpreting the model. It helps in identifying the most significant features that influence the prediction. This insight is crucial for feature selection, engineering, and ultimately improving the model performance.

#### Structure of the Code:[​](#structure-of-the-code-4 "Direct link to Structure of the Code:")

* **Context Setting**: The code initiates by setting the plot style to 'seaborn' for aesthetic enhancement.
* **Figure Initialization**: It creates a figure and axes for plotting.
* **Bar Plot**: It uses a horizontal bar plot (`barh`) for visualizing each feature's coefficient. The y-axis represents the feature names, and the x-axis represents the coefficient values. This visualization makes it easy to compare the coefficients, providing insight into their relative importance and impact on the target variable.
* **Title and Labels**: It sets an appropriate title ("Coefficient Plot") and labels for the x ("Coefficient Value") and y ("Features") axes to ensure clarity and understandability.

By visualizing the coefficients, we can gain a deeper understanding of the model, making it easier to explain the model's predictions and make more informed decisions regarding feature importance and impact.

python

```python
def plot_coefficients(model, feature_names, style="seaborn", plot_size=(10, 8)):
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      ax.barh(feature_names, model.coef_)
      ax.set_title("Coefficient Plot", fontsize=14)
      ax.set_xlabel("Coefficient Value", fontsize=12)
      ax.set_ylabel("Features", fontsize=12)
      plt.tight_layout()
  plt.close(fig)
  return fig

```

### Visualization of Residuals[​](#visualization-of-residuals "Direct link to Visualization of Residuals")

In this section, we're creating a plot to visualize the residuals of the model, which are the differences between the observed and predicted values.

#### Why is this Important?[​](#why-is-this-important-5 "Direct link to Why is this Important?")

A residual plot is a fundamental diagnostic tool in regression analysis used to investigate the unpredictability in the relationship between the predictor variable and the response variable. It helps in identifying non-linearity, heteroscedasticity, and outliers. This plot assists in validating the assumption that the errors are normally distributed and have constant variance, crucial for the reliability of the regression model's predictions.

#### Structure of the Code:[​](#structure-of-the-code-5 "Direct link to Structure of the Code:")

* **Residual Calculation**: The code begins by calculating the residuals as the difference between the actual (`y_test`) and predicted (`y_pred`) values.
* **Context Setting**: The code sets the plot style to 'seaborn' for a visually appealing plot.
* **Figure Initialization**: It creates a figure and axes for plotting.
* **Residual Plotting**: It utilizes the `residplot` from Seaborn to create the residual plot, with a lowess (locally weighted scatterplot smoothing) line to highlight the trend in the residuals.
* **Zero Line**: It adds a dashed line at zero to serve as a reference for observing the residuals. Residuals above the line indicate under-prediction, while those below indicate over-prediction.
* **Title and Labels**: It sets an appropriate title ("Residual Plot") and labels for the x ("Predicted values") and y ("Residuals") axes to ensure clarity and understandability.

By examining the residual plot, we can make better-informed decisions on the model's adequacy and the possible need for further refinement or additional complexity.

python

```python
def plot_residuals(y_test, y_pred, style="seaborn", plot_size=(10, 8)):
  residuals = y_test - y_pred

  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      sns.residplot(
          x=y_pred,
          y=residuals,
          lowess=True,
          ax=ax,
          line_kws={"color": "red", "lw": 1},
      )

      ax.axhline(y=0, color="black", linestyle="--")
      ax.set_title("Residual Plot", fontsize=14)
      ax.set_xlabel("Predicted values", fontsize=12)
      ax.set_ylabel("Residuals", fontsize=12)

      for label in ax.get_xticklabels() + ax.get_yticklabels():
          label.set_fontsize(10)

      plt.tight_layout()

  plt.close(fig)
  return fig

```

### Visualization of Prediction Errors[​](#visualization-of-prediction-errors "Direct link to Visualization of Prediction Errors")

In this section, we're creating a plot to visualize the prediction errors, showcasing the discrepancies between the actual and predicted values from our model.

#### Why is this Important?[​](#why-is-this-important-6 "Direct link to Why is this Important?")

Understanding the prediction errors is crucial for assessing the performance of a model. A prediction error plot provides insight into the error distribution and helps identify trends, biases, or outliers. This visualization is a critical component for model evaluation, helping in identifying areas where the model may need improvement, and ensuring it generalizes well to new data.

#### Structure of the Code:[​](#structure-of-the-code-6 "Direct link to Structure of the Code:")

* **Context Setting**: The code sets the plot style to 'seaborn' for a clean and attractive plot.
* **Figure Initialization**: It initializes a figure and axes for plotting.
* **Scatter Plot**: The code plots the predicted values against the errors (actual values - predicted values). Each point on the plot represents a specific observation, and its position on the y-axis indicates the magnitude and direction of the error (above zero for under-prediction and below zero for over-prediction).
* **Zero Line**: A red dashed line at y=0 is plotted as a reference, helping in easily identifying the errors. Points above this line are under-predictions, and points below are over-predictions.
* **Title and Labels**: It adds a title ("Prediction Error Plot") and labels for the x ("Predicted Values") and y ("Errors") axes for better clarity and understanding.

By analyzing the prediction error plot, practitioners can gain valuable insights into the model's performance, helping in the further refinement and enhancement of the model for better and more reliable predictions.

python

```python
def plot_prediction_error(y_test, y_pred, style="seaborn", plot_size=(10, 8)):
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      ax.scatter(y_pred, y_test - y_pred)
      ax.axhline(y=0, color="red", linestyle="--")
      ax.set_title("Prediction Error Plot", fontsize=14)
      ax.set_xlabel("Predicted Values", fontsize=12)
      ax.set_ylabel("Errors", fontsize=12)
      plt.tight_layout()
  plt.close(fig)
  return fig

```

### Visualization of Quantile-Quantile Plot (QQ Plot)[​](#visualization-of-quantile-quantile-plot-qq-plot "Direct link to Visualization of Quantile-Quantile Plot (QQ Plot)")

In this section, we will generate a QQ plot to visualize the distribution of the residuals from our model predictions.

#### Why is this Important?[​](#why-is-this-important-7 "Direct link to Why is this Important?")

A QQ plot is essential for assessing if the residuals from the model follow a normal distribution, a fundamental assumption in linear regression models. If the points in the QQ plot do not follow the line closely and show a pattern, this indicates that the residuals may not be normally distributed, which could imply issues with the model such as heteroscedasticity or non-linearity.

#### Structure of the Code:[​](#structure-of-the-code-7 "Direct link to Structure of the Code:")

* **Residual Calculation**: The code first calculates the residuals by subtracting the predicted values from the actual test values.
* **Context Setting**: The plot style is set to 'seaborn' for aesthetic appeal.
* **Figure Initialization**: A figure and axes are initialized for plotting.
* **QQ Plot Generation**: The `stats.probplot` function is used to generate the QQ plot. It plots the quantiles of the residuals against the quantiles of a normal distribution.
* **Title Addition**: A title ("QQ Plot") is added to the plot for clarity.

By closely analyzing the QQ plot, we can ensure our model's residuals meet the normality assumption. If not, it may be beneficial to explore other model types or transformations to improve the model's performance and reliability.

python

```python
def plot_qq(y_test, y_pred, style="seaborn", plot_size=(10, 8)):
  residuals = y_test - y_pred
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      stats.probplot(residuals, dist="norm", plot=ax)
      ax.set_title("QQ Plot", fontsize=14)
      plt.tight_layout()
  plt.close(fig)
  return fig

```

### Feature Correlation Matrix[​](#feature-correlation-matrix "Direct link to Feature Correlation Matrix")

In this section, we're generating a **feature correlation matrix** to visualize the relationships between different features in the dataset.

**NOTE:** Unlike the other plots in this notebook, we're saving a local copy of the plot to disk to show an alternative logging mechanism for arbitrary files, the `log_artifact()` API. Within the main model training and logging section below, you will see how this plot is added to the MLflow run.

#### Why is this Important?[​](#why-is-this-important-8 "Direct link to Why is this Important?")

Understanding the correlation between different features is essential for:

* Identifying multicollinearity, which can affect model performance and interpretability.
* Gaining insights into relationships between variables, which can inform feature engineering and selection.
* Uncovering potential causality or interaction between different features, which can inform domain understanding and further analysis.

#### Structure of the Code:[​](#structure-of-the-code-8 "Direct link to Structure of the Code:")

* **Correlation Calculation**: The code first calculates the correlation matrix for the provided DataFrame.
* **Masking**: A mask is created for the upper triangle of the correlation matrix, as the matrix is symmetrical, and we don't need to visualize the duplicate information.
* **Heatmap Generation**: A heatmap is generated to visualize the correlation coefficients. The color gradient and annotations provide clear insights into the relationships between variables.
* **Title Addition**: A title is added for clear identification of the plot.

By analyzing the correlation matrix, we can make more informed decisions about feature selection and understand the relationships within our dataset better.

python

```python
def plot_correlation_matrix_and_save(
  df, style="seaborn", plot_size=(10, 8), path="/tmp/corr_plot.png"
):
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)

      # Calculate the correlation matrix
      corr = df.corr()

      # Generate a mask for the upper triangle
      mask = np.triu(np.ones_like(corr, dtype=bool))

      # Draw the heatmap with the mask and correct aspect ratio
      sns.heatmap(
          corr,
          mask=mask,
          cmap="coolwarm",
          vmax=0.3,
          center=0,
          square=True,
          linewidths=0.5,
          annot=True,
          fmt=".2f",
      )

      ax.set_title("Feature Correlation Matrix", fontsize=14)
      plt.tight_layout()

  plt.close(fig)
  # convert to filesystem path spec for os compatibility
  save_path = pathlib.Path(path)
  fig.savefig(save_path)

```

### Detailed Overview of Main Execution for Model Training and Visualization[​](#detailed-overview-of-main-execution-for-model-training-and-visualization "Direct link to Detailed Overview of Main Execution for Model Training and Visualization")

This section delves deeper into the comprehensive workflow executed for model training, prediction, error calculation, and visualization. The significance of each step and the reason for specific choices are thoroughly discussed.

#### The Benefits of Structured Execution[​](#the-benefits-of-structured-execution "Direct link to The Benefits of Structured Execution")

Executing all crucial steps of model training and evaluation in a structured manner is fundamental. It provides a framework that ensures every aspect of the modeling process is considered, offering a more reliable and robust model. This streamlined execution aids in avoiding overlooked errors or biases and guarantees that the model is evaluated on all necessary fronts.

#### Importance of Logging Visualizations to MLflow[​](#importance-of-logging-visualizations-to-mlflow "Direct link to Importance of Logging Visualizations to MLflow")

Logging visualizations to MLflow offers several key benefits:

* **Permanence**: Unlike the ephemeral state of notebooks where cells can be run out of order leading to potential misinterpretation, logging plots to MLflow ensures that the visualizations are stored permanently with the specific run. This permanence assures that the visual context of the model training and evaluation is preserved, eliminating confusion and ensuring clarity in interpretation.

* **Provenance**: By logging visualizations, the exact state and relationships in the data at the time of model training are captured. This practice is crucial for models trained a significant time ago. It offers a reliable reference point to understand the model's behavior and the data characteristics at the time of training, ensuring that insights and interpretations remain valid and reliable over time.

* **Accessibility**: Storing visualizations in MLflow makes them easily accessible to all team members or stakeholders involved. This centralized storage of visualizations enhances collaboration, allowing diverse team members to easily view, analyze, and interpret the visualizations, leading to more informed and collective decision-making.

#### Detailed Structure of the Code:[​](#detailed-structure-of-the-code "Direct link to Detailed Structure of the Code:")

1. **Setting up MLflow**:

   * The tracking URI for MLflow is defined.
   * An experiment named "Visualizations Demo" is set up, under which all runs and logs will be stored.

2. **Data Preparation**:

   * `X` and `y` are defined as features and target variables, respectively.
   * The dataset is split into training and testing sets to ensure the model's performance is evaluated on unseen data.

3. **Initial Plot Generation**:

   * Initial plots including time series, box plot, scatter plot, and density plot are generated.
   * These plots offer a preliminary insight into the data and its characteristics.

4. **Model Definition and Training**:

   * A Ridge regression model is defined with an `alpha` of 1.0.
   * The model is trained on the training data, learning the relationships and patterns in the data.

5. **Prediction and Error Calculation**:

   * The trained model is used to make predictions on the test data.
   * Various error metrics including MSE, RMSE, MAE, R2, MSLE, and MedAE are calculated to evaluate the model's performance.

6. **Additional Plot Generation**:

   * Additional plots including residuals plot, coefficients plot, prediction error plot, and QQ plot are generated.
   * These plots offer further insight into the model's performance, residuals behavior, and the distribution of errors.

7. **Logging to MLflow**:

   * The trained model, calculated metrics, defined parameter (`alpha`), and all the generated plots are logged to MLflow.
   * This logging ensures that all information and visualizations related to the model are stored in a centralized, accessible location.

#### Conclusion:[​](#conclusion "Direct link to Conclusion:")

By executing this comprehensive and structured code, we ensure that every aspect of model training, evaluation, and interpretation is covered. The practice of logging all relevant information and visualizations to MLflow further enhances the reliability, accessibility, and interpretability of the model and its performance, contributing to more informed and reliable model deployment and utilization.

python

```python
mlflow.set_tracking_uri("http://127.0.0.1:8080")

mlflow.set_experiment("Visualizations Demo")

X = my_data.drop(columns=["demand", "date"])
y = my_data["demand"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

fig1 = plot_time_series_demand(my_data, window_size=28)
fig2 = plot_box_weekend(my_data)
fig3 = plot_scatter_demand_price(my_data)
fig4 = plot_density_weekday_weekend(my_data)

# Execute the correlation plot, saving the plot to a local temporary directory
plot_correlation_matrix_and_save(my_data)

# Define our Ridge model
model = Ridge(alpha=1.0)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
msle = mean_squared_log_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# Generate prediction-dependent plots
fig5 = plot_residuals(y_test, y_pred)
fig6 = plot_coefficients(model, X_test.columns)
fig7 = plot_prediction_error(y_test, y_pred)
fig8 = plot_qq(y_test, y_pred)

# Start an MLflow run for logging metrics, parameters, the model, and our figures
with mlflow.start_run() as run:
  # Log the model
  mlflow.sklearn.log_model(sk_model=model, input_example=X_test, name="model")

  # Log the metrics
  mlflow.log_metrics(
      {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "msle": msle, "medae": medae}
  )

  # Log the hyperparameter
  mlflow.log_param("alpha", 1.0)

  # Log plots
  mlflow.log_figure(fig1, "time_series_demand.png")
  mlflow.log_figure(fig2, "box_weekend.png")
  mlflow.log_figure(fig3, "scatter_demand_price.png")
  mlflow.log_figure(fig4, "density_weekday_weekend.png")
  mlflow.log_figure(fig5, "residuals_plot.png")
  mlflow.log_figure(fig6, "coefficients_plot.png")
  mlflow.log_figure(fig7, "prediction_errors.png")
  mlflow.log_figure(fig8, "qq_plot.png")

  # Log the saved correlation matrix plot by referring to the local file system location
  mlflow.log_artifact("/tmp/corr_plot.png")

```
