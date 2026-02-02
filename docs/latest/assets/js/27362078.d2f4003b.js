"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["2583"],{58173(e,t,i){i.r(t),i.d(t,{metadata:()=>n,default:()=>g,frontMatter:()=>c,contentTitle:()=>h,toc:()=>u,assets:()=>p});var n=JSON.parse('{"id":"traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow-ipynb","title":"Logging Visualizations with MLflow","description":"Download this notebook","source":"@site/docs/classic-ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow-ipynb.mdx","sourceDirName":"traditional-ml/tutorials/hyperparameter-tuning/notebooks","slug":"/traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow","draft":false,"unlisted":false,"editUrl":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow.ipynb","tags":[],"version":"current","frontMatter":{"custom_edit_url":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow.ipynb","slug":"logging-plots-in-mlflow"},"sidebar":"classicMLSidebar","previous":{"title":"MLflow with Optuna: Hyperparameter Optimization and Tracking","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs"},"next":{"title":"Leveraging Child Runs in MLflow for Hyperparameter Tuning","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/parent-child-runs"}}'),a=i(74848),s=i(28453),r=i(75940),o=i(75453),l=i(66354),d=i(42676);let c={custom_edit_url:"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow.ipynb",slug:"logging-plots-in-mlflow"},h="Logging Visualizations with MLflow",p={},u=[{value:"What Are We Doing?",id:"what-are-we-doing",level:4},{value:"How Does It Apply to MLflow?",id:"how-does-it-apply-to-mlflow",level:4},{value:"Caution",id:"caution",level:4},{value:"Why Is Consistent Logging Important?",id:"why-is-consistent-logging-important",level:4},{value:"Generating Synthetic Apple Sales Data",id:"generating-synthetic-apple-sales-data",level:3},{value:"What Are We Doing?",id:"what-are-we-doing-1",level:4},{value:"How Does It Apply to Data Generation?",id:"how-does-it-apply-to-data-generation",level:4},{value:"Caution",id:"caution-1",level:4},{value:"Why Is Acknowledging Limitations Important?",id:"why-is-acknowledging-limitations-important",level:4},{value:"Generating Apple Sales Data",id:"generating-apple-sales-data",level:3},{value:"Parameters Used:",id:"parameters-used",level:4},{value:"Time Series Visualization of Demand",id:"time-series-visualization-of-demand",level:3},{value:"Why is this Important?",id:"why-is-this-important",level:4},{value:"Structure of the Code:",id:"structure-of-the-code",level:4},{value:"Why Return a Figure?",id:"why-return-a-figure",level:4},{value:"Visualizing Demand on Weekends vs. Weekdays with Box Plots",id:"visualizing-demand-on-weekends-vs-weekdays-with-box-plots",level:3},{value:"Why is this Important?",id:"why-is-this-important-1",level:4},{value:"Structure of the Code:",id:"structure-of-the-code-1",level:4},{value:"Why Return a Figure?",id:"why-return-a-figure-1",level:4},{value:"Exploring the Relationship Between Demand and Price per Kg",id:"exploring-the-relationship-between-demand-and-price-per-kg",level:3},{value:"Why is this Important?",id:"why-is-this-important-2",level:4},{value:"Structure of the Code:",id:"structure-of-the-code-2",level:4},{value:"Visualizing Demand Density: Weekday vs. Weekend",id:"visualizing-demand-density-weekday-vs-weekend",level:3},{value:"Why is this Important?",id:"why-is-this-important-3",level:4},{value:"Structure of the Code:",id:"structure-of-the-code-3",level:4},{value:"What are the Visual Elements?",id:"what-are-the-visual-elements",level:4},{value:"Visualization of Model Coefficients",id:"visualization-of-model-coefficients",level:3},{value:"Why is this Important?",id:"why-is-this-important-4",level:4},{value:"Structure of the Code:",id:"structure-of-the-code-4",level:4},{value:"Visualization of Residuals",id:"visualization-of-residuals",level:3},{value:"Why is this Important?",id:"why-is-this-important-5",level:4},{value:"Structure of the Code:",id:"structure-of-the-code-5",level:4},{value:"Visualization of Prediction Errors",id:"visualization-of-prediction-errors",level:3},{value:"Why is this Important?",id:"why-is-this-important-6",level:4},{value:"Structure of the Code:",id:"structure-of-the-code-6",level:4},{value:"Visualization of Quantile-Quantile Plot (QQ Plot)",id:"visualization-of-quantile-quantile-plot-qq-plot",level:3},{value:"Why is this Important?",id:"why-is-this-important-7",level:4},{value:"Structure of the Code:",id:"structure-of-the-code-7",level:4},{value:"Feature Correlation Matrix",id:"feature-correlation-matrix",level:3},{value:"Why is this Important?",id:"why-is-this-important-8",level:4},{value:"Structure of the Code:",id:"structure-of-the-code-8",level:4},{value:"Detailed Overview of Main Execution for Model Training and Visualization",id:"detailed-overview-of-main-execution-for-model-training-and-visualization",level:3},{value:"The Benefits of Structured Execution",id:"the-benefits-of-structured-execution",level:4},{value:"Importance of Logging Visualizations to MLflow",id:"importance-of-logging-visualizations-to-mlflow",level:4},{value:"Detailed Structure of the Code:",id:"detailed-structure-of-the-code",level:4},{value:"Conclusion:",id:"conclusion",level:4}];function f(e){let t={code:"code",h1:"h1",h3:"h3",h4:"h4",header:"header",li:"li",ol:"ol",p:"p",strong:"strong",ul:"ul",...(0,s.R)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(t.header,{children:(0,a.jsx)(t.h1,{id:"logging-visualizations-with-mlflow",children:"Logging Visualizations with MLflow"})}),"\n",(0,a.jsx)(d.O,{href:"https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/logging-plots-in-mlflow.ipynb",children:"Download this notebook"}),"\n",(0,a.jsxs)(t.p,{children:["In this part of the guide, we emphasize the ",(0,a.jsx)(t.strong,{children:"importance of logging visualizations with MLflow"}),". Retaining visualizations alongside trained models enhances model interpretability, auditing, and provenance, ensuring a robust and transparent machine learning lifecycle."]}),"\n",(0,a.jsx)(t.h4,{id:"what-are-we-doing",children:"What Are We Doing?"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Storing Visual Artifacts:"})," We are logging various plots as visual artifacts in MLflow, ensuring that they are always accessible and aligned with the corresponding model and run data."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Enhancing Model Interpretability:"})," These visualizations aid in understanding and explaining model behavior, contributing to improved model transparency and accountability."]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"how-does-it-apply-to-mlflow",children:"How Does It Apply to MLflow?"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Integrated Visualization Logging:"})," MLflow seamlessly integrates facilities for logging and accessing visual artifacts, enhancing the ease and efficiency of handling visual context and insights."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Convenient Access:"})," Logged figures are displayable within the Runs view pane in the MLflow UI, ensuring quick and easy access for analysis and review."]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"caution",children:"Caution"}),"\n",(0,a.jsx)(t.p,{children:"While MLflow offers simplicity and convenience for logging visualizations, it's crucial to ensure the consistency and relevance of the visual artifacts with the corresponding model data, maintaining the integrity and comprehensiveness of the model information."}),"\n",(0,a.jsx)(t.h4,{id:"why-is-consistent-logging-important",children:"Why Is Consistent Logging Important?"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Auditing and Provenance:"})," Consistent and comprehensive logging of visualizations is pivotal for auditing purposes, ensuring that every model is accompanied by relevant visual insights for thorough analysis and review."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Enhanced Model Understanding:"})," Proper visual context enhances the understanding of model behavior, aiding in effective model evaluation, and validation."]}),"\n"]}),"\n",(0,a.jsx)(t.p,{children:"In conclusion, MLflow's capabilities for visualization logging play an invaluable role in ensuring a comprehensive, transparent, and efficient machine learning lifecycle, reinforcing model interpretability, auditing, and provenance."}),"\n",(0,a.jsx)(t.h3,{id:"generating-synthetic-apple-sales-data",children:"Generating Synthetic Apple Sales Data"}),"\n",(0,a.jsxs)(t.p,{children:["In this next section, we dive into ",(0,a.jsx)(t.strong,{children:"generating synthetic data for apple sales demand prediction"})," using the ",(0,a.jsx)(t.code,{children:"generate_apple_sales_data_with_promo_adjustment"})," function. This function simulates a variety of features relevant to apple sales, providing a rich dataset for exploration and modeling."]}),"\n",(0,a.jsx)(t.h4,{id:"what-are-we-doing-1",children:"What Are We Doing?"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Simulating Realistic Data:"})," Generating a dataset with features like date, average temperature, rainfall, weekend flag, and more, simulating realistic scenarios for apple sales."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Incorporating Various Effects:"})," The function incorporates effects like promotional adjustments, seasonality, and competitor pricing, contributing to the 'demand' target variable."]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"how-does-it-apply-to-data-generation",children:"How Does It Apply to Data Generation?"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Comprehensive Dataset:"})," The synthetic dataset provides a comprehensive set of features and interactions, ideal for exploring diverse aspects and dimensions for demand prediction."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Freedom and Flexibility:"})," The synthetic nature allows for unconstrained exploration and analysis, devoid of real-world data sensitivities and constraints."]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"caution-1",children:"Caution"}),"\n",(0,a.jsx)(t.p,{children:"While synthetic data offers numerous advantages for exploration and learning, it's crucial to acknowledge its limitations in capturing real-world complexities and nuances."}),"\n",(0,a.jsx)(t.h4,{id:"why-is-acknowledging-limitations-important",children:"Why Is Acknowledging Limitations Important?"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Real-World Complexities:"})," Synthetic data may not capture all the intricate patterns and anomalies present in real-world data, potentially leading to over-simplified models and insights."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Transferability to Real-World Scenarios:"})," Ensuring that insights and models derived from synthetic data are transferable to real-world scenarios requires careful consideration and validation."]}),"\n"]}),"\n",(0,a.jsxs)(t.p,{children:["In conclusion, the ",(0,a.jsx)(t.code,{children:"generate_apple_sales_data_with_promo_adjustment"})," function offers a robust tool for generating a comprehensive synthetic dataset for apple sales demand prediction, facilitating extensive exploration, and analysis while acknowledging the limitations of synthetic data."]}),"\n",(0,a.jsx)(r.d,{executionCount:2,children:`import math
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

  return df`}),"\n",(0,a.jsx)(t.h3,{id:"generating-apple-sales-data",children:"Generating Apple Sales Data"}),"\n",(0,a.jsxs)(t.p,{children:["In this cell, we call the ",(0,a.jsx)(t.code,{children:"generate_apple_sales_data_with_promo_adjustment"})," function to generate a dataset of apple sales."]}),"\n",(0,a.jsx)(t.h4,{id:"parameters-used",children:"Parameters Used:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.code,{children:"base_demand"}),": Set to 1000, representing the baseline demand for apples."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.code,{children:"n_rows"}),": Set to 10,000, determining the number of rows or data points in the generated dataset."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.code,{children:"competitor_price_effect"}),": Set to -25.0, representing the impact on our sales when the competitor's price is lower."]}),"\n"]}),"\n",(0,a.jsxs)(t.p,{children:["By running this cell, we obtain a dataset ",(0,a.jsx)(t.code,{children:"my_data"}),", which holds the synthetic apple sales data with the aforementioned configurations. This dataset will be used for further exploration and analysis in subsequent steps of this notebook."]}),"\n",(0,a.jsx)(t.p,{children:"You can see the data in the cell after the generation cell."}),"\n",(0,a.jsx)(r.d,{executionCount:3,children:`my_data = generate_apple_sales_data_with_promo_adjustment(
  base_demand=1000, n_rows=10_000, competitor_price_effect=-25.0
)`}),"\n",(0,a.jsx)(r.d,{executionCount:4,children:"my_data"}),"\n",(0,a.jsx)(l.Q,{children:(0,a.jsx)("div",{dangerouslySetInnerHTML:{__html:`<div>
<style scoped>
  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: right;
  }
</style>
<table border="1" class="dataframe">
<thead>
  <tr style="text-align: right;">
    <th></th>
    <th>date</th>
    <th>average_temperature</th>
    <th>rainfall</th>
    <th>weekend</th>
    <th>holiday</th>
    <th>price_per_kg</th>
    <th>promo</th>
    <th>demand</th>
    <th>previous_days_demand</th>
    <th>competitor_price_per_kg</th>
    <th>marketing_intensity</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>0</th>
    <td>1996-05-11 13:10:40.689999</td>
    <td>30.584727</td>
    <td>1.831006</td>
    <td>1</td>
    <td>0</td>
    <td>1.578387</td>
    <td>1</td>
    <td>1301.647352</td>
    <td>1326.324266</td>
    <td>0.755725</td>
    <td>0.323086</td>
  </tr>
  <tr>
    <th>1</th>
    <td>1996-05-12 13:10:40.689999</td>
    <td>15.465069</td>
    <td>0.761303</td>
    <td>1</td>
    <td>0</td>
    <td>1.965125</td>
    <td>0</td>
    <td>1143.972638</td>
    <td>1326.324266</td>
    <td>0.913934</td>
    <td>0.030371</td>
  </tr>
  <tr>
    <th>2</th>
    <td>1996-05-13 13:10:40.689998</td>
    <td>10.786525</td>
    <td>1.427338</td>
    <td>0</td>
    <td>0</td>
    <td>1.497623</td>
    <td>0</td>
    <td>890.319248</td>
    <td>1168.942267</td>
    <td>2.879262</td>
    <td>0.354226</td>
  </tr>
  <tr>
    <th>3</th>
    <td>1996-05-14 13:10:40.689997</td>
    <td>23.648154</td>
    <td>3.737435</td>
    <td>0</td>
    <td>0</td>
    <td>1.952936</td>
    <td>0</td>
    <td>811.206168</td>
    <td>889.965021</td>
    <td>0.826015</td>
    <td>0.953000</td>
  </tr>
  <tr>
    <th>4</th>
    <td>1996-05-15 13:10:40.689997</td>
    <td>13.861391</td>
    <td>5.598549</td>
    <td>0</td>
    <td>0</td>
    <td>2.059993</td>
    <td>0</td>
    <td>822.279469</td>
    <td>835.253168</td>
    <td>1.130145</td>
    <td>0.953000</td>
  </tr>
  <tr>
    <th>...</th>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
  <tr>
    <th>9995</th>
    <td>2023-09-22 13:10:40.682895</td>
    <td>23.358868</td>
    <td>7.061220</td>
    <td>0</td>
    <td>0</td>
    <td>1.556829</td>
    <td>1</td>
    <td>1981.195884</td>
    <td>2089.644454</td>
    <td>0.560507</td>
    <td>0.889971</td>
  </tr>
  <tr>
    <th>9996</th>
    <td>2023-09-23 13:10:40.682895</td>
    <td>14.859048</td>
    <td>0.868655</td>
    <td>1</td>
    <td>0</td>
    <td>1.632918</td>
    <td>0</td>
    <td>2180.698138</td>
    <td>2005.305913</td>
    <td>2.460766</td>
    <td>0.884467</td>
  </tr>
  <tr>
    <th>9997</th>
    <td>2023-09-24 13:10:40.682894</td>
    <td>17.941035</td>
    <td>13.739986</td>
    <td>1</td>
    <td>0</td>
    <td>0.827723</td>
    <td>1</td>
    <td>2675.093671</td>
    <td>2179.813671</td>
    <td>1.321922</td>
    <td>0.884467</td>
  </tr>
  <tr>
    <th>9998</th>
    <td>2023-09-25 13:10:40.682893</td>
    <td>14.533862</td>
    <td>1.610512</td>
    <td>0</td>
    <td>0</td>
    <td>0.589172</td>
    <td>0</td>
    <td>1703.287285</td>
    <td>2674.209204</td>
    <td>2.604095</td>
    <td>0.812706</td>
  </tr>
  <tr>
    <th>9999</th>
    <td>2023-09-26 13:10:40.682889</td>
    <td>13.048549</td>
    <td>5.287508</td>
    <td>0</td>
    <td>0</td>
    <td>1.794122</td>
    <td>1</td>
    <td>1971.029266</td>
    <td>1702.474579</td>
    <td>1.261635</td>
    <td>0.750458</td>
  </tr>
</tbody>
</table>
<p>10000 rows \xd7 11 columns</p>
</div>`}})}),"\n",(0,a.jsx)(t.h3,{id:"time-series-visualization-of-demand",children:"Time Series Visualization of Demand"}),"\n",(0,a.jsx)(t.p,{children:"In this section, we're creating a time series plot to visualize the demand data alongside its rolling average."}),"\n",(0,a.jsx)(t.h4,{id:"why-is-this-important",children:"Why is this Important?"}),"\n",(0,a.jsx)(t.p,{children:"Visualizing time series data is crucial for identifying patterns, understanding variability, and making more informed decisions. By plotting the rolling average alongside, we can smooth out short-term fluctuations and highlight longer-term trends or cycles. This visual aid is essential for understanding the data and making more accurate and informed predictions and decisions."}),"\n",(0,a.jsx)(t.h4,{id:"structure-of-the-code",children:"Structure of the Code:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Input Verification"}),": The code first ensures the data is a pandas DataFrame."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Date Conversion"}),": It converts the 'date' column to a datetime format for accurate plotting."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Rolling Average Calculation"}),": It calculates the rolling average of the 'demand' with a specified window size (",(0,a.jsx)(t.code,{children:"window_size"}),"), defaulting to 7 days."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Plotting"}),': It plots both the original demand data and the calculated rolling average on the same plot for comparison. The original demand data is plotted with low alpha to appear "ghostly," ensuring the rolling average stands out.']}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Labels and Legend"}),": Adequate labels and legends are added for clarity."]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"why-return-a-figure",children:"Why Return a Figure?"}),"\n",(0,a.jsxs)(t.p,{children:["We return the figure object (",(0,a.jsx)(t.code,{children:"fig"}),") instead of rendering it directly so that each iteration of a model training event can consume the figure as a logged artifact to MLflow. This approach allows us to persist the state of the data visualization with precisely the state of the data that was used for training. MLflow can store this figure object, enabling easy retrieval and rendering within the MLflow UI, ensuring that the visualization is always accessible and paired with the relevant model and data information."]}),"\n",(0,a.jsx)(r.d,{executionCount:5,children:`def plot_time_series_demand(data, window_size=7, style="seaborn", plot_size=(16, 12)):
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
  return fig`}),"\n",(0,a.jsx)(t.h3,{id:"visualizing-demand-on-weekends-vs-weekdays-with-box-plots",children:"Visualizing Demand on Weekends vs. Weekdays with Box Plots"}),"\n",(0,a.jsx)(t.p,{children:"In this section, we're utilizing box plots to visualize the distribution of demand on weekends versus weekdays. This visualization assists in understanding the variability and central tendency of demand based on the day of the week."}),"\n",(0,a.jsx)(t.h4,{id:"why-is-this-important-1",children:"Why is this Important?"}),"\n",(0,a.jsx)(t.p,{children:"Understanding how demand differs between weekends and weekdays is crucial for making informed decisions regarding inventory, staffing, and other operational aspects. It helps identify the periods of higher demand, allowing for better resource allocation and planning."}),"\n",(0,a.jsx)(t.h4,{id:"structure-of-the-code-1",children:"Structure of the Code:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Box Plot"}),": The code uses Seaborn to create a box plot that shows the distribution of demand on weekends (1) and weekdays (0). The box plot provides insights into the median, quartiles, and possible outliers in the demand data for both categories."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Adding Individual Data Points"}),": To provide more context, individual data points are overlayed on the box plot as a strip plot. They are jittered for better visualization and color-coded based on the day type."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Styling"}),": The plot is styled for clarity, and unnecessary legends are removed to enhance readability."]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"why-return-a-figure-1",children:"Why Return a Figure?"}),"\n",(0,a.jsxs)(t.p,{children:["As with the time series plot, this function also returns the figure object (",(0,a.jsx)(t.code,{children:"fig"}),") instead of displaying it directly."]}),"\n",(0,a.jsx)(r.d,{executionCount:6,children:`def plot_box_weekend(df, style="seaborn", plot_size=(10, 8)):
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
  return fig`}),"\n",(0,a.jsx)(t.h3,{id:"exploring-the-relationship-between-demand-and-price-per-kg",children:"Exploring the Relationship Between Demand and Price per Kg"}),"\n",(0,a.jsxs)(t.p,{children:["In this visualization, we're creating a scatter plot to investigate the relationship between the ",(0,a.jsx)(t.code,{children:"demand"})," and ",(0,a.jsx)(t.code,{children:"price_per_kg"}),". Understanding this relationship is crucial for pricing strategy and demand forecasting."]}),"\n",(0,a.jsx)(t.h4,{id:"why-is-this-important-2",children:"Why is this Important?"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Insight into Pricing Strategy:"})," This visualization helps reveal how demand varies with the price per kg, providing valuable insights for setting prices to optimize sales and revenue."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Understanding Demand Elasticity:"})," It aids in understanding the elasticity of demand concerning price, helping in making informed and data-driven decisions for promotions and discounts."]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"structure-of-the-code-2",children:"Structure of the Code:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Scatter Plot:"})," The code generates a scatter plot, where each point's position is determined by the ",(0,a.jsx)(t.code,{children:"price_per_kg"})," and ",(0,a.jsx)(t.code,{children:"demand"}),", and the color indicates whether the day is a weekend or a weekday. This color-coding helps in quickly identifying patterns specific to weekends or weekdays."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Transparency and Jitter:"})," Points are plotted with transparency (",(0,a.jsx)(t.code,{children:"alpha=0.15"}),") to handle overplotting, allowing the visualization of the density of points."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Regression Line:"})," For each subgroup (weekend and weekday), a separate regression line is fitted and plotted on the same axes. These lines provide a clear visual indication of the trend of demand concerning the price per kg for each group."]}),"\n"]}),"\n",(0,a.jsx)(r.d,{executionCount:7,children:`def plot_scatter_demand_price(df, style="seaborn", plot_size=(10, 8)):
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
  return fig`}),"\n",(0,a.jsx)(t.h3,{id:"visualizing-demand-density-weekday-vs-weekend",children:"Visualizing Demand Density: Weekday vs. Weekend"}),"\n",(0,a.jsxs)(t.p,{children:["This visualization allows us to observe the distribution of ",(0,a.jsx)(t.code,{children:"demand"})," separately for weekdays and weekends."]}),"\n",(0,a.jsx)(t.h4,{id:"why-is-this-important-3",children:"Why is this Important?"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Demand Distribution Insight:"})," Understanding the distribution of demand on weekdays versus weekends can inform inventory management and staffing needs."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Informing Business Strategy:"})," This insight is vital for making data-driven decisions regarding promotions, discounts, and other strategies that might be more effective on specific days."]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"structure-of-the-code-3",children:"Structure of the Code:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Density Plot:"})," The code generates a density plot for ",(0,a.jsx)(t.code,{children:"demand"}),", separated into weekdays and weekends."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Color-Coded Groups:"})," The two groups (weekday and weekend) are color-coded (blue and green respectively), making it easy to distinguish between them."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Transparency and Filling:"})," The areas under the density curves are filled with a light, transparent color (",(0,a.jsx)(t.code,{children:"alpha=0.15"}),") for easy visualization while avoiding visual clutter."]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"what-are-the-visual-elements",children:"What are the Visual Elements?"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Two Density Curves:"})," The plot comprises two density curves, one for weekdays and another for weekends. These curves provide a clear visual representation of the distribution of demand for each group."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Legend:"})," A legend is added to help identify which curve corresponds to which group (weekday or weekend)."]}),"\n"]}),"\n",(0,a.jsx)(r.d,{executionCount:8,children:`def plot_density_weekday_weekend(df, style="seaborn", plot_size=(10, 8)):
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
  return fig`}),"\n",(0,a.jsx)(t.h3,{id:"visualization-of-model-coefficients",children:"Visualization of Model Coefficients"}),"\n",(0,a.jsx)(t.p,{children:"In this section, we're utilizing a bar plot to visualize the coefficients of the features from the trained model."}),"\n",(0,a.jsx)(t.h4,{id:"why-is-this-important-4",children:"Why is this Important?"}),"\n",(0,a.jsx)(t.p,{children:"Understanding the magnitude and direction of the coefficients is essential for interpreting the model. It helps in identifying the most significant features that influence the prediction. This insight is crucial for feature selection, engineering, and ultimately improving the model performance."}),"\n",(0,a.jsx)(t.h4,{id:"structure-of-the-code-4",children:"Structure of the Code:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Context Setting"}),": The code initiates by setting the plot style to 'seaborn' for aesthetic enhancement."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Figure Initialization"}),": It creates a figure and axes for plotting."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Bar Plot"}),": It uses a horizontal bar plot (",(0,a.jsx)(t.code,{children:"barh"}),") for visualizing each feature's coefficient. The y-axis represents the feature names, and the x-axis represents the coefficient values. This visualization makes it easy to compare the coefficients, providing insight into their relative importance and impact on the target variable."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Title and Labels"}),': It sets an appropriate title ("Coefficient Plot") and labels for the x ("Coefficient Value") and y ("Features") axes to ensure clarity and understandability.']}),"\n"]}),"\n",(0,a.jsx)(t.p,{children:"By visualizing the coefficients, we can gain a deeper understanding of the model, making it easier to explain the model's predictions and make more informed decisions regarding feature importance and impact."}),"\n",(0,a.jsx)(r.d,{executionCount:9,children:`def plot_coefficients(model, feature_names, style="seaborn", plot_size=(10, 8)):
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      ax.barh(feature_names, model.coef_)
      ax.set_title("Coefficient Plot", fontsize=14)
      ax.set_xlabel("Coefficient Value", fontsize=12)
      ax.set_ylabel("Features", fontsize=12)
      plt.tight_layout()
  plt.close(fig)
  return fig`}),"\n",(0,a.jsx)(t.h3,{id:"visualization-of-residuals",children:"Visualization of Residuals"}),"\n",(0,a.jsx)(t.p,{children:"In this section, we're creating a plot to visualize the residuals of the model, which are the differences between the observed and predicted values."}),"\n",(0,a.jsx)(t.h4,{id:"why-is-this-important-5",children:"Why is this Important?"}),"\n",(0,a.jsx)(t.p,{children:"A residual plot is a fundamental diagnostic tool in regression analysis used to investigate the unpredictability in the relationship between the predictor variable and the response variable. It helps in identifying non-linearity, heteroscedasticity, and outliers. This plot assists in validating the assumption that the errors are normally distributed and have constant variance, crucial for the reliability of the regression model's predictions."}),"\n",(0,a.jsx)(t.h4,{id:"structure-of-the-code-5",children:"Structure of the Code:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Residual Calculation"}),": The code begins by calculating the residuals as the difference between the actual (",(0,a.jsx)(t.code,{children:"y_test"}),") and predicted (",(0,a.jsx)(t.code,{children:"y_pred"}),") values."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Context Setting"}),": The code sets the plot style to 'seaborn' for a visually appealing plot."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Figure Initialization"}),": It creates a figure and axes for plotting."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Residual Plotting"}),": It utilizes the ",(0,a.jsx)(t.code,{children:"residplot"})," from Seaborn to create the residual plot, with a lowess (locally weighted scatterplot smoothing) line to highlight the trend in the residuals."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Zero Line"}),": It adds a dashed line at zero to serve as a reference for observing the residuals. Residuals above the line indicate under-prediction, while those below indicate over-prediction."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Title and Labels"}),': It sets an appropriate title ("Residual Plot") and labels for the x ("Predicted values") and y ("Residuals") axes to ensure clarity and understandability.']}),"\n"]}),"\n",(0,a.jsx)(t.p,{children:"By examining the residual plot, we can make better-informed decisions on the model's adequacy and the possible need for further refinement or additional complexity."}),"\n",(0,a.jsx)(r.d,{executionCount:10,children:`def plot_residuals(y_test, y_pred, style="seaborn", plot_size=(10, 8)):
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
  return fig`}),"\n",(0,a.jsx)(t.h3,{id:"visualization-of-prediction-errors",children:"Visualization of Prediction Errors"}),"\n",(0,a.jsx)(t.p,{children:"In this section, we're creating a plot to visualize the prediction errors, showcasing the discrepancies between the actual and predicted values from our model."}),"\n",(0,a.jsx)(t.h4,{id:"why-is-this-important-6",children:"Why is this Important?"}),"\n",(0,a.jsx)(t.p,{children:"Understanding the prediction errors is crucial for assessing the performance of a model. A prediction error plot provides insight into the error distribution and helps identify trends, biases, or outliers. This visualization is a critical component for model evaluation, helping in identifying areas where the model may need improvement, and ensuring it generalizes well to new data."}),"\n",(0,a.jsx)(t.h4,{id:"structure-of-the-code-6",children:"Structure of the Code:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Context Setting"}),": The code sets the plot style to 'seaborn' for a clean and attractive plot."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Figure Initialization"}),": It initializes a figure and axes for plotting."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Scatter Plot"}),": The code plots the predicted values against the errors (actual values - predicted values). Each point on the plot represents a specific observation, and its position on the y-axis indicates the magnitude and direction of the error (above zero for under-prediction and below zero for over-prediction)."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Zero Line"}),": A red dashed line at y=0 is plotted as a reference, helping in easily identifying the errors. Points above this line are under-predictions, and points below are over-predictions."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Title and Labels"}),': It adds a title ("Prediction Error Plot") and labels for the x ("Predicted Values") and y ("Errors") axes for better clarity and understanding.']}),"\n"]}),"\n",(0,a.jsx)(t.p,{children:"By analyzing the prediction error plot, practitioners can gain valuable insights into the model's performance, helping in the further refinement and enhancement of the model for better and more reliable predictions."}),"\n",(0,a.jsx)(r.d,{executionCount:11,children:`def plot_prediction_error(y_test, y_pred, style="seaborn", plot_size=(10, 8)):
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      ax.scatter(y_pred, y_test - y_pred)
      ax.axhline(y=0, color="red", linestyle="--")
      ax.set_title("Prediction Error Plot", fontsize=14)
      ax.set_xlabel("Predicted Values", fontsize=12)
      ax.set_ylabel("Errors", fontsize=12)
      plt.tight_layout()
  plt.close(fig)
  return fig`}),"\n",(0,a.jsx)(t.h3,{id:"visualization-of-quantile-quantile-plot-qq-plot",children:"Visualization of Quantile-Quantile Plot (QQ Plot)"}),"\n",(0,a.jsx)(t.p,{children:"In this section, we will generate a QQ plot to visualize the distribution of the residuals from our model predictions."}),"\n",(0,a.jsx)(t.h4,{id:"why-is-this-important-7",children:"Why is this Important?"}),"\n",(0,a.jsx)(t.p,{children:"A QQ plot is essential for assessing if the residuals from the model follow a normal distribution, a fundamental assumption in linear regression models. If the points in the QQ plot do not follow the line closely and show a pattern, this indicates that the residuals may not be normally distributed, which could imply issues with the model such as heteroscedasticity or non-linearity."}),"\n",(0,a.jsx)(t.h4,{id:"structure-of-the-code-7",children:"Structure of the Code:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Residual Calculation"}),": The code first calculates the residuals by subtracting the predicted values from the actual test values."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Context Setting"}),": The plot style is set to 'seaborn' for aesthetic appeal."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Figure Initialization"}),": A figure and axes are initialized for plotting."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"QQ Plot Generation"}),": The ",(0,a.jsx)(t.code,{children:"stats.probplot"})," function is used to generate the QQ plot. It plots the quantiles of the residuals against the quantiles of a normal distribution."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Title Addition"}),': A title ("QQ Plot") is added to the plot for clarity.']}),"\n"]}),"\n",(0,a.jsx)(t.p,{children:"By closely analyzing the QQ plot, we can ensure our model's residuals meet the normality assumption. If not, it may be beneficial to explore other model types or transformations to improve the model's performance and reliability."}),"\n",(0,a.jsx)(r.d,{executionCount:12,children:`def plot_qq(y_test, y_pred, style="seaborn", plot_size=(10, 8)):
  residuals = y_test - y_pred
  with plt.style.context(style=style):
      fig, ax = plt.subplots(figsize=plot_size)
      stats.probplot(residuals, dist="norm", plot=ax)
      ax.set_title("QQ Plot", fontsize=14)
      plt.tight_layout()
  plt.close(fig)
  return fig`}),"\n",(0,a.jsx)(t.h3,{id:"feature-correlation-matrix",children:"Feature Correlation Matrix"}),"\n",(0,a.jsxs)(t.p,{children:["In this section, we're generating a ",(0,a.jsx)(t.strong,{children:"feature correlation matrix"})," to visualize the relationships between different features in the dataset."]}),"\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"NOTE:"})," Unlike the other plots in this notebook, we're saving a local copy of the plot to disk to show an alternative logging mechanism for arbitrary files, the ",(0,a.jsx)(t.code,{children:"log_artifact()"})," API. Within the main model training and logging section below, you will see how this plot is added to the MLflow run."]}),"\n",(0,a.jsx)(t.h4,{id:"why-is-this-important-8",children:"Why is this Important?"}),"\n",(0,a.jsx)(t.p,{children:"Understanding the correlation between different features is essential for:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsx)(t.li,{children:"Identifying multicollinearity, which can affect model performance and interpretability."}),"\n",(0,a.jsx)(t.li,{children:"Gaining insights into relationships between variables, which can inform feature engineering and selection."}),"\n",(0,a.jsx)(t.li,{children:"Uncovering potential causality or interaction between different features, which can inform domain understanding and further analysis."}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"structure-of-the-code-8",children:"Structure of the Code:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Correlation Calculation"}),": The code first calculates the correlation matrix for the provided DataFrame."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Masking"}),": A mask is created for the upper triangle of the correlation matrix, as the matrix is symmetrical, and we don't need to visualize the duplicate information."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Heatmap Generation"}),": A heatmap is generated to visualize the correlation coefficients. The color gradient and annotations provide clear insights into the relationships between variables."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Title Addition"}),": A title is added for clear identification of the plot."]}),"\n"]}),"\n",(0,a.jsx)(t.p,{children:"By analyzing the correlation matrix, we can make more informed decisions about feature selection and understand the relationships within our dataset better."}),"\n",(0,a.jsx)(r.d,{executionCount:13,children:`def plot_correlation_matrix_and_save(
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
  fig.savefig(save_path)`}),"\n",(0,a.jsx)(t.h3,{id:"detailed-overview-of-main-execution-for-model-training-and-visualization",children:"Detailed Overview of Main Execution for Model Training and Visualization"}),"\n",(0,a.jsx)(t.p,{children:"This section delves deeper into the comprehensive workflow executed for model training, prediction, error calculation, and visualization. The significance of each step and the reason for specific choices are thoroughly discussed."}),"\n",(0,a.jsx)(t.h4,{id:"the-benefits-of-structured-execution",children:"The Benefits of Structured Execution"}),"\n",(0,a.jsx)(t.p,{children:"Executing all crucial steps of model training and evaluation in a structured manner is fundamental. It provides a framework that ensures every aspect of the modeling process is considered, offering a more reliable and robust model. This streamlined execution aids in avoiding overlooked errors or biases and guarantees that the model is evaluated on all necessary fronts."}),"\n",(0,a.jsx)(t.h4,{id:"importance-of-logging-visualizations-to-mlflow",children:"Importance of Logging Visualizations to MLflow"}),"\n",(0,a.jsx)(t.p,{children:"Logging visualizations to MLflow offers several key benefits:"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Permanence"}),": Unlike the ephemeral state of notebooks where cells can be run out of order leading to potential misinterpretation, logging plots to MLflow ensures that the visualizations are stored permanently with the specific run. This permanence assures that the visual context of the model training and evaluation is preserved, eliminating confusion and ensuring clarity in interpretation."]}),"\n"]}),"\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Provenance"}),": By logging visualizations, the exact state and relationships in the data at the time of model training are captured. This practice is crucial for models trained a significant time ago. It offers a reliable reference point to understand the model's behavior and the data characteristics at the time of training, ensuring that insights and interpretations remain valid and reliable over time."]}),"\n"]}),"\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Accessibility"}),": Storing visualizations in MLflow makes them easily accessible to all team members or stakeholders involved. This centralized storage of visualizations enhances collaboration, allowing diverse team members to easily view, analyze, and interpret the visualizations, leading to more informed and collective decision-making."]}),"\n"]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"detailed-structure-of-the-code",children:"Detailed Structure of the Code:"}),"\n",(0,a.jsxs)(t.ol,{children:["\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Setting up MLflow"}),":"]}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsx)(t.li,{children:"The tracking URI for MLflow is defined."}),"\n",(0,a.jsx)(t.li,{children:'An experiment named "Visualizations Demo" is set up, under which all runs and logs will be stored.'}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Data Preparation"}),":"]}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.code,{children:"X"})," and ",(0,a.jsx)(t.code,{children:"y"})," are defined as features and target variables, respectively."]}),"\n",(0,a.jsx)(t.li,{children:"The dataset is split into training and testing sets to ensure the model's performance is evaluated on unseen data."}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Initial Plot Generation"}),":"]}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsx)(t.li,{children:"Initial plots including time series, box plot, scatter plot, and density plot are generated."}),"\n",(0,a.jsx)(t.li,{children:"These plots offer a preliminary insight into the data and its characteristics."}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Model Definition and Training"}),":"]}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:["A Ridge regression model is defined with an ",(0,a.jsx)(t.code,{children:"alpha"})," of 1.0."]}),"\n",(0,a.jsx)(t.li,{children:"The model is trained on the training data, learning the relationships and patterns in the data."}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Prediction and Error Calculation"}),":"]}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsx)(t.li,{children:"The trained model is used to make predictions on the test data."}),"\n",(0,a.jsx)(t.li,{children:"Various error metrics including MSE, RMSE, MAE, R2, MSLE, and MedAE are calculated to evaluate the model's performance."}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Additional Plot Generation"}),":"]}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsx)(t.li,{children:"Additional plots including residuals plot, coefficients plot, prediction error plot, and QQ plot are generated."}),"\n",(0,a.jsx)(t.li,{children:"These plots offer further insight into the model's performance, residuals behavior, and the distribution of errors."}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(t.li,{children:["\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.strong,{children:"Logging to MLflow"}),":"]}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsxs)(t.li,{children:["The trained model, calculated metrics, defined parameter (",(0,a.jsx)(t.code,{children:"alpha"}),"), and all the generated plots are logged to MLflow."]}),"\n",(0,a.jsx)(t.li,{children:"This logging ensures that all information and visualizations related to the model are stored in a centralized, accessible location."}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,a.jsx)(t.h4,{id:"conclusion",children:"Conclusion:"}),"\n",(0,a.jsx)(t.p,{children:"By executing this comprehensive and structured code, we ensure that every aspect of model training, evaluation, and interpretation is covered. The practice of logging all relevant information and visualizations to MLflow further enhances the reliability, accessibility, and interpretability of the model and its performance, contributing to more informed and reliable model deployment and utilization."}),"\n",(0,a.jsx)(r.d,{executionCount:14,children:`mlflow.set_tracking_uri("http://127.0.0.1:8080")

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
  mlflow.log_artifact("/tmp/corr_plot.png")`}),"\n",(0,a.jsx)(o.p,{isStderr:!0,children:`2023/09/26 13:10:41 INFO mlflow.tracking.fluent: Experiment with name 'Visualizations Demo' does not exist. Creating a new experiment.
/Users/benjamin.wilson/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/mlflow/models/signature.py:333: UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See \`Handling Integers With Missing Values <https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values>\`_ for more details.
input_schema = _infer_schema(input_ex)
/Users/benjamin.wilson/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils.
warnings.warn("Setuptools is replacing distutils.")`})]})}function g(e={}){let{wrapper:t}={...(0,s.R)(),...e.components};return t?(0,a.jsx)(t,{...e,children:(0,a.jsx)(f,{...e})}):f(e)}},75453(e,t,i){i.d(t,{p:()=>a});var n=i(74848);let a=({children:e,isStderr:t})=>(0,n.jsx)("pre",{style:{margin:0,borderRadius:0,background:"none",fontSize:"0.85rem",flexGrow:1,padding:"var(--padding-sm)"},children:e})},75940(e,t,i){i.d(t,{d:()=>s});var n=i(74848),a=i(37449);let s=({children:e,executionCount:t})=>(0,n.jsx)("div",{style:{flexGrow:1,minWidth:0,marginTop:"var(--padding-md)",width:"100%"},children:(0,n.jsx)(a.A,{className:"codeBlock_oJcR",language:"python",children:e})})},42676(e,t,i){i.d(t,{O:()=>r});var n=i(74848),a=i(96540);let s="3.9.1.dev0";function r({children:e,href:t}){let i=(0,a.useCallback)(async e=>{if(e.preventDefault(),window.gtag)try{window.gtag("event","notebook-download",{href:t})}catch{}s.includes("dev")||(t=t.replace(/\/master\//,`/v${s}/`));let i=await fetch(t),n=await i.blob(),a=window.URL.createObjectURL(n),r=document.createElement("a");r.style.display="none",r.href=a,r.download=t.split("/").pop(),document.body.appendChild(r),r.click(),window.URL.revokeObjectURL(a),document.body.removeChild(r)},[t]);return(0,n.jsx)("a",{className:"button button--primary",style:{marginBottom:"1rem",display:"block",width:"min-content"},href:t,download:!0,onClick:i,children:e})}},66354(e,t,i){i.d(t,{Q:()=>a});var n=i(74848);let a=({children:e})=>(0,n.jsx)("div",{style:{flexGrow:1,minWidth:0,fontSize:"0.8rem",width:"100%"},children:e})},52915(e,t,i){i.d(t,{A:()=>h});var n=i(74848);i(96540);var a=i(34164),s=i(71643),r=i(66697),o=i(92949),l=i(64560),d=i(47819);function c({language:e}){return(0,n.jsxs)("div",{className:(0,a.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,n.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,n.jsx)(d.A,{})]})}function h({className:e}){let{metadata:t}=(0,s.Ph)(),i=t.language||"text";return(0,n.jsxs)(r.A,{as:"div",className:(0,a.A)(e,t.className),children:[t.title&&(0,n.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,n.jsx)(o.A,{children:t.title})}),(0,n.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,n.jsx)(c,{language:i}),(0,n.jsx)(l.A,{})]})]})}}}]);