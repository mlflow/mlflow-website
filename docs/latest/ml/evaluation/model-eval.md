# Model Evaluation

This guide covers MLflow's core model evaluation capabilities for classification and regression tasks, showing how to comprehensively assess model performance with automated metrics, visualizations, and diagnostic tools.

## Quick Start: Evaluating a Classification Model[​](#quick-start-evaluating-a-classification-model "Direct link to Quick Start: Evaluating a Classification Model")

The simplest way to evaluate a model is with MLflow's unified evaluation API:

python

```
import mlflow
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

# Load the UCI Adult Dataset
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train model
model = xgb.XGBClassifier().fit(X_train, y_train)

# Create evaluation dataset
eval_data = X_test.copy()
eval_data["label"] = y_test

with mlflow.start_run():
    # Log model with signature
    signature = infer_signature(X_test, model.predict(X_test))
    model_info = mlflow.sklearn.log_model(model, name="model", signature=signature)

    # Comprehensive evaluation
    result = mlflow.models.evaluate(
        model_info.model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )

    print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
    print(f"F1 Score: {result.metrics['f1_score']:.3f}")
    print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")
```

This single call automatically generates:

* **Performance Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC
* **Visualizations**: Confusion matrix, ROC curve, precision-recall curve
* **Feature Importance**: SHAP values and feature contribution analysis
* **Model Artifacts**: All plots and diagnostic information saved to MLflow

## Supported Model Types[​](#supported-model-types "Direct link to Supported Model Types")

MLflow supports different model types, each with specialized metrics and evaluations:

* **`classifier`** - Binary and multiclass classification models
* **`regressor`** - Regression models for continuous target prediction

- Classification
- Regression

For classification tasks, MLflow automatically computes comprehensive metrics:

python

```
# Binary Classification
result = mlflow.models.evaluate(
    model_uri,
    eval_data,
    targets="label",
    model_type="classifier",  # Automatically detects binary vs multiclass
    evaluators=["default"],
)

# Access classification-specific metrics
metrics = result.metrics
print(f"Precision: {metrics['precision_score']:.3f}")
print(f"Recall: {metrics['recall_score']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
print(f"ROC AUC: {metrics['roc_auc']:.3f}")
```

**Automatic Classification Metrics:**

* Accuracy, Precision, Recall, F1-Score
* ROC-AUC and Precision-Recall AUC
* Log Loss and Brier Score
* Confusion Matrix and Classification Report

For regression tasks, MLflow provides comprehensive error analysis:

python

```
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load regression dataset
housing = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Train regression model
reg_model = LinearRegression().fit(X_train, y_train)

# Create evaluation dataset
eval_data = X_test.copy()
eval_data["target"] = y_test

with mlflow.start_run():
    # Log and evaluate regression model
    signature = infer_signature(X_train, reg_model.predict(X_train))
    mlflow.sklearn.log_model(reg_model, name="model", signature=signature)
    model_uri = mlflow.get_artifact_uri("model")

    result = mlflow.models.evaluate(
        model_uri,
        eval_data,
        targets="target",
        model_type="regressor",
        evaluators=["default"],
    )

    print(f"MAE: {result.metrics['mean_absolute_error']:.3f}")
    print(f"RMSE: {result.metrics['root_mean_squared_error']:.3f}")
    print(f"R² Score: {result.metrics['r2_score']:.3f}")
```

**Automatic Regression Metrics:**

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE) and Root MSE
* R² Score and Adjusted R²
* Mean Absolute Percentage Error (MAPE)
* Residual plots and distribution analysis

## Advanced Evaluation Configurations[​](#advanced-evaluation-configurations "Direct link to Advanced Evaluation Configurations")

### Specifying Evaluators[​](#specifying-evaluators "Direct link to Specifying Evaluators")

Control which evaluators run during assessment:

python

```
# Run only default metrics (fastest)
result = mlflow.models.evaluate(
    model_uri,
    eval_data,
    targets="label",
    model_type="classifier",
    evaluators=["default"],
)

# Include SHAP explainer for feature importance
result = mlflow.models.evaluate(
    model_uri,
    eval_data,
    targets="label",
    model_type="classifier",
    evaluators=["default"],
    evaluator_config={"log_explainer": True},
)
```

Configuration Options Reference

#### SHAP Configuration[​](#shap-configuration "Direct link to SHAP Configuration")

* `log_explainer`: Whether to log the SHAP explainer as a model
* `explainer_type`: Type of SHAP explainer ("exact", "permutation", "partition")
* `max_error_examples`: Maximum number of error examples to analyze
* `log_model_explanations`: Whether to log individual prediction explanations

#### Performance Options[​](#performance-options "Direct link to Performance Options")

* `pos_label`: Positive class label for binary classification metrics
* `average`: Averaging strategy for multiclass metrics ("macro", "micro", "weighted")
* `sample_weights`: Sample weights for weighted metrics
* `normalize`: Normalization for confusion matrix ("true", "pred", "all")

## Custom Metrics and Artifacts[​](#custom-metrics-and-artifacts "Direct link to Custom Metrics and Artifacts")

* Custom Metrics
* Custom Artifacts

Classic System Only

The `make_metric` function and `EvaluationMetric` class are part of MLflow's classic evaluation system.

For GenAI/LLM custom evaluation metrics, use the [@scorer decorator](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md) instead.

MLflow provides a powerful framework for defining custom evaluation metrics using the `make_metric` function:

python

```
import mlflow
import numpy as np
from mlflow.models import make_metric


def weighted_accuracy(predictions, targets, metrics, sample_weights=None):
    """Custom weighted accuracy metric."""
    if sample_weights is None:
        return (predictions == targets).mean()
    else:
        correct = predictions == targets
        return np.average(correct, weights=sample_weights)


# Create custom metric
custom_accuracy = make_metric(
    eval_fn=weighted_accuracy, greater_is_better=True, name="weighted_accuracy"
)

# Use in evaluation
result = mlflow.models.evaluate(
    model_uri,
    eval_data,
    targets="label",
    model_type="classifier",
    extra_metrics=[custom_accuracy],
)
```

Create custom visualization and analysis artifacts:

python

```
import matplotlib.pyplot as plt
import os


def create_residual_plot(eval_df, builtin_metrics, artifacts_dir):
    """Create custom residual plot for regression models."""

    residuals = eval_df["target"] - eval_df["prediction"]

    plt.figure(figsize=(10, 6))
    plt.scatter(eval_df["prediction"], residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plot_path = os.path.join(artifacts_dir, "residual_plot.png")
    plt.savefig(plot_path)
    plt.close()

    return {"residual_plot": plot_path}


# Use custom artifact
result = mlflow.models.evaluate(
    model_uri,
    eval_data,
    targets="target",
    model_type="regressor",
    custom_artifacts=[create_residual_plot],
)
```

## Working with Evaluation Results[​](#working-with-evaluation-results "Direct link to Working with Evaluation Results")

The evaluation result object provides comprehensive access to all generated metrics and artifacts:

python

```
# Run evaluation
result = mlflow.models.evaluate(
    model_uri, eval_data, targets="label", model_type="classifier"
)

# Access metrics
print("All Metrics:")
for metric_name, value in result.metrics.items():
    print(f"  {metric_name}: {value}")

# Access artifacts (plots, tables, etc.)
print("\nGenerated Artifacts:")
for artifact_name, path in result.artifacts.items():
    print(f"  {artifact_name}: {path}")

# Access evaluation dataset
eval_table = result.tables["eval_results_table"]
print(f"\nEvaluation table shape: {eval_table.shape}")
print(f"Columns: {list(eval_table.columns)}")
```

## Model Comparison and Advanced Workflows[​](#model-comparison-and-advanced-workflows "Direct link to Model Comparison and Advanced Workflows")

* Model Comparison
* Cross-Validation
* Automated Selection

Compare multiple models systematically:

python

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define models to compare
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(random_state=42),
    "svm": SVC(probability=True, random_state=42),
}

# Evaluate each model
results = {}

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"eval_{model_name}"):
        # Train model
        model.fit(X_train, y_train)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")

        # Evaluate model
        result = mlflow.models.evaluate(
            model_uri, eval_data, targets="label", model_type="classifier"
        )

        results[model_name] = result.metrics

        # Log comparison metrics
        mlflow.log_metrics(
            {
                "accuracy": result.metrics["accuracy_score"],
                "f1": result.metrics["f1_score"],
                "roc_auc": result.metrics["roc_auc"],
            }
        )

# Compare results
comparison_df = pd.DataFrame(results).T
print("Model Comparison:")
print(comparison_df[["accuracy_score", "f1_score", "roc_auc"]].round(3))
```

Combine MLflow evaluation with cross-validation:

python

```
from sklearn.model_selection import cross_val_score, StratifiedKFold


def evaluate_with_cv(model, X, y, eval_data, cv_folds=5):
    """Evaluate model with cross-validation and final test evaluation."""

    with mlflow.start_run():
        # Cross-validation scores
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")

        # Log CV results
        mlflow.log_metrics(
            {"cv_mean_f1": cv_scores.mean(), "cv_std_f1": cv_scores.std()}
        )

        # Train on full dataset
        model.fit(X, y)

        # Final evaluation
        signature = infer_signature(X, model.predict(X))
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")

        result = mlflow.models.evaluate(
            model_uri, eval_data, targets="label", model_type="classifier"
        )

        # Compare CV and test performance
        test_f1 = result.metrics["f1_score"]
        cv_f1 = cv_scores.mean()

        mlflow.log_metrics(
            {
                "cv_vs_test_diff": abs(cv_f1 - test_f1),
                "potential_overfit": cv_f1 - test_f1 > 0.05,
            }
        )

        return result


# Usage
result = evaluate_with_cv(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train,
    y_train,
    eval_data,
)
```

Automated model selection based on evaluation metrics:

python

```
def evaluate_and_select_best_model(
    models, X_train, y_train, eval_data, metric="f1_score"
):
    """Evaluate multiple models and select the best performer."""

    results = {}
    best_score = -1
    best_model_name = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"candidate_{model_name}"):
            # Train and evaluate
            model.fit(X_train, y_train)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, name="model", signature=signature)
            model_uri = mlflow.get_artifact_uri("model")

            result = mlflow.models.evaluate(
                model_uri, eval_data, targets="label", model_type="classifier"
            )

            score = result.metrics[metric]
            results[model_name] = score

            # Track best model
            if score > best_score:
                best_score = score
                best_model_name = model_name

            # Log selection metrics
            mlflow.log_metrics(
                {"selection_score": score, "is_best": score == best_score}
            )

    print(f"Best model: {best_model_name} (Score: {best_score:.3f})")
    return best_model_name, results


# Use automated selection
best_model, all_scores = evaluate_and_select_best_model(
    models, X_train, y_train, eval_data, metric="f1_score"
)
```

## Model Validation and Quality Gates[​](#model-validation-and-quality-gates "Direct link to Model Validation and Quality Gates")

attention

MLflow 2.18.0 has moved the model validation functionality from the [`mlflow.models.evaluate()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.models.html#mlflow.models.evaluate) API to a dedicated [`mlflow.validate_evaluation_results()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.validate_evaluation_results) API. The relevant parameters, such as baseline\_model, are deprecated and will be removed from the older API in future versions.

With the [`mlflow.validate_evaluation_results()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.validate_evaluation_results) API, you can validate metrics generated during model evaluation to assess the quality of your model against a baseline.

python

```
from mlflow.models import MetricThreshold

# Evaluate your model first
result = mlflow.models.evaluate(
    model_uri, eval_data, targets="label", model_type="classifier"
)

# Define static performance thresholds
static_thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.85, greater_is_better=True  # Must achieve 85% accuracy
    ),
    "precision_score": MetricThreshold(
        threshold=0.80, greater_is_better=True  # Must achieve 80% precision
    ),
    "recall_score": MetricThreshold(
        threshold=0.75, greater_is_better=True  # Must achieve 75% recall
    ),
}

# Validate against static thresholds
try:
    mlflow.validate_evaluation_results(
        candidate_result=result,
        baseline_result=None,  # No baseline comparison
        validation_thresholds=static_thresholds,
    )
    print("✅ Model meets all static performance thresholds.")
except mlflow.exceptions.ModelValidationFailedException as e:
    print(f"❌ Model failed static validation: {e}")
```

More information on model validation behavior and outputs can be found in the [`mlflow.validate_evaluation_results()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.validate_evaluation_results) API documentation.

## Error Analysis and Debugging[​](#error-analysis-and-debugging "Direct link to Error Analysis and Debugging")

* Error Investigation
* Feature Analysis

Analyze model errors in detail:

python

```
def analyze_model_errors(result, eval_data, targets, top_n=20):
    """Analyze model errors in detail."""

    # Load evaluation results
    eval_table = result.tables["eval_results_table"]

    # Identify errors
    errors = eval_table[eval_table["prediction"] != eval_table[targets]]

    if len(errors) > 0:
        print(f"Total errors: {len(errors)} out of {len(eval_table)} predictions")
        print(f"Error rate: {len(errors) / len(eval_table) * 100:.2f}%")

        # Most confident wrong predictions
        if "prediction_score" in errors.columns:
            confident_errors = errors.nlargest(top_n, "prediction_score")
            print(f"\nTop {top_n} most confident errors:")
            print(confident_errors[["prediction", targets, "prediction_score"]].head())

        # Error patterns by true class
        error_by_class = errors.groupby(targets).size()
        print(f"\nErrors by true class:")
        print(error_by_class)

    return errors


# Usage
errors = analyze_model_errors(result, eval_data, "label")
```

Analyze how model errors relate to input features:

python

```
def analyze_errors_by_features(model_uri, eval_data, targets, feature_columns):
    """Analyze how model errors relate to input features."""

    # Get model predictions
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(eval_data[feature_columns])

    # Create analysis dataframe
    analysis_df = eval_data.copy()
    analysis_df["prediction"] = predictions
    analysis_df["is_error"] = analysis_df["prediction"] != analysis_df[targets]

    # Feature statistics for errors vs correct predictions
    feature_stats = {}

    for feature in feature_columns:
        if analysis_df[feature].dtype in ["int64", "float64"]:
            # Numerical features
            correct_mean = analysis_df[~analysis_df["is_error"]][feature].mean()
            error_mean = analysis_df[analysis_df["is_error"]][feature].mean()

            feature_stats[feature] = {
                "correct_mean": correct_mean,
                "error_mean": error_mean,
                "difference": abs(error_mean - correct_mean),
                "relative_difference": abs(error_mean - correct_mean) / correct_mean
                if correct_mean != 0
                else 0,
            }

    # Sort features by impact on errors
    numerical_features = [
        (k, v["relative_difference"])
        for k, v in feature_stats.items()
        if "relative_difference" in v
    ]
    numerical_features.sort(key=lambda x: x[1], reverse=True)

    print("Features most associated with errors:")
    for feature, diff in numerical_features[:5]:
        print(f"  {feature}: {diff:.3f}")

    return feature_stats, analysis_df


# Usage
feature_stats, analysis = analyze_errors_by_features(
    model_uri,
    eval_data,
    "label",
    feature_columns=eval_data.drop(columns=["label"]).columns.tolist(),
)
```

## Best Practices and Optimization[​](#best-practices-and-optimization "Direct link to Best Practices and Optimization")

* Best Practices
* Performance Optimization
* Reproducible Evaluation

Complete evaluation workflow with best practices:

python

```
def comprehensive_model_evaluation(
    model, X_train, y_train, eval_data, targets, model_type
):
    """Complete evaluation workflow with best practices."""

    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)

        # Log training info
        mlflow.log_params(
            {"model_class": model.__class__.__name__, "training_samples": len(X_train)}
        )

        # Log model with signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")

        # Comprehensive evaluation
        result = mlflow.models.evaluate(
            model_uri,
            eval_data,
            targets=targets,
            model_type=model_type,
            evaluators=["default"],
            evaluator_config={
                "log_explainer": True,
                "explainer_type": "exact",
                "log_model_explanations": True,
            },
        )

        return result
```

Optimize evaluation for large datasets and complex models:

python

```
# Optimize evaluation performance
result = mlflow.models.evaluate(
    model_uri,
    eval_data.sample(n=10000, random_state=42),  # Sample for faster evaluation
    targets="label",
    model_type="classifier",
    evaluators=["default"],
    evaluator_config={
        "log_explainer": False,  # Skip SHAP for speed
        "max_error_examples": 50,  # Reduce error analysis
    },
)


# For very large datasets - evaluate in batches
def evaluate_in_batches(model_uri, large_eval_data, targets, batch_size=1000):
    """Evaluate large datasets in batches to manage memory."""

    all_predictions = []
    all_targets = []

    for i in range(0, len(large_eval_data), batch_size):
        batch = large_eval_data.iloc[i : i + batch_size]

        # Get predictions for batch
        model = mlflow.pyfunc.load_model(model_uri)
        batch_predictions = model.predict(batch.drop(columns=[targets]))

        all_predictions.extend(batch_predictions)
        all_targets.extend(batch[targets].values)

    # Create final evaluation dataset
    final_eval_data = pd.DataFrame(
        {"prediction": all_predictions, "target": all_targets}
    )

    # Evaluate using static dataset approach
    result = mlflow.models.evaluate(
        data=final_eval_data,
        predictions="prediction",
        targets="target",
        model_type="classifier",
    )

    return result
```

Ensure consistent evaluation results:

python

```
def reproducible_evaluation(model, eval_data, targets, random_seed=42):
    """Ensure reproducible evaluation results."""

    # Set random seeds
    np.random.seed(random_seed)

    with mlflow.start_run():
        # Log evaluation configuration
        mlflow.log_params(
            {
                "eval_random_seed": random_seed,
                "eval_data_size": len(eval_data),
                "eval_timestamp": pd.Timestamp.now().isoformat(),
            }
        )

        # Consistent data ordering
        eval_data_sorted = eval_data.sort_values(
            by=eval_data.columns.tolist()
        ).reset_index(drop=True)

        # Run evaluation
        result = mlflow.models.evaluate(
            model,
            eval_data_sorted,
            targets=targets,
            model_type="classifier",
            evaluator_config={"random_seed": random_seed},
        )

        return result
```

## Conclusion[​](#conclusion "Direct link to Conclusion")

MLflow's model evaluation capabilities provide a comprehensive framework for assessing model performance across classification and regression tasks. The unified API simplifies complex evaluation workflows while providing deep insights into model behavior through automated metrics, visualizations, and diagnostic tools.

Key benefits of MLflow model evaluation include:

* **Comprehensive Assessment**: Automated generation of task-specific metrics and visualizations
* **Reproducible Workflows**: Consistent evaluation processes with complete tracking and versioning
* **Advanced Analysis**: Error investigation, feature impact analysis, and model comparison capabilities
* **Production Integration**: Seamless integration with MLflow tracking for experiment organization and reporting

Whether you're evaluating a single model or comparing multiple candidates, MLflow's evaluation framework provides the tools needed to make informed decisions about model performance and production readiness.
