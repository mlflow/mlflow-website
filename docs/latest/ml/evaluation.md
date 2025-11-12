# MLflow Evaluation

Classic ML Evaluation System

This documentation covers MLflow's **classic evaluation system** (`mlflow.evaluate`) which uses `EvaluationMetric` and `make_metric` for custom metrics.

**For GenAI/LLM evaluation**, please use the system at [GenAI Evaluation](/mlflow-website/docs/latest/genai/eval-monitor.md) which uses:

* `mlflow.genai.evaluate()` instead of `mlflow.evaluate()`
* `Scorer` objects instead of `EvaluationMetric`
* Built-in LLM judges and scorers

**Important**: These two systems are **not interoperable**. `EvaluationMetric` objects cannot be used with `mlflow.genai.evaluate()`, and `Scorer` objects cannot be used with `mlflow.evaluate()`.

## Introduction[​](#introduction "Direct link to Introduction")

**Model evaluation** is the cornerstone of reliable machine learning, transforming trained models into trustworthy, production-ready systems. MLflow's comprehensive evaluation framework goes beyond simple accuracy metrics, providing deep insights into model behavior, performance characteristics, and real-world readiness through automated testing, visualization, and validation pipelines.

MLflow's evaluation capabilities democratize advanced model assessment, making sophisticated evaluation techniques accessible to teams of all sizes. From rapid prototyping to enterprise deployment, MLflow evaluation ensures your models meet the highest standards of reliability, fairness, and performance.

Why Comprehensive Model Evaluation Matters

#### Beyond Basic Metrics[​](#beyond-basic-metrics "Direct link to Beyond Basic Metrics")

* 📊 **Holistic Assessment**: Performance metrics, visualizations, and explanations in one unified framework
* 🎯 **Task-Specific Evaluation**: Specialized evaluators for classification, regression, and LLM tasks
* 🔍 **Model Interpretability**: SHAP integration for understanding model decisions and feature importance
* ⚖️ **Fairness Analysis**: Bias detection and ethical AI validation across demographic groups

#### Production Readiness[​](#production-readiness "Direct link to Production Readiness")

* 🚀 **Automated Validation**: Threshold-based model acceptance with customizable criteria
* 📈 **Performance Monitoring**: Track model degradation and drift over time
* 🔄 **A/B Testing Support**: Compare candidate models against production baselines
* 📋 **Audit Trails**: Complete evaluation history for regulatory compliance and model governance

## Why MLflow Evaluation?[​](#why-mlflow-evaluation "Direct link to Why MLflow Evaluation?")

MLflow's evaluation framework provides a comprehensive solution for model assessment and validation:

* ⚡ **One-Line Evaluation**: Comprehensive model assessment with `mlflow.evaluate()` - minimal configuration required
* 🎛️ **Flexible Evaluation Modes**: Evaluate models, functions, or static datasets with the same unified API
* 📊 **Rich Visualizations**: Automatic generation of performance plots, confusion matrices, and diagnostic charts
* 🔧 **Custom Metrics**: Define domain-specific evaluation criteria with easy-to-use metric builders
* 🧠 **Built-in Explainability**: SHAP integration for model interpretation and feature importance analysis
* 👥 **Team Collaboration**: Share evaluation results and model comparisons through MLflow's tracking interface
* 🏭 **Enterprise Integration**: Plugin architecture for specialized evaluation frameworks like Giskard and Trubrics

## Core Evaluation Capabilities[​](#core-evaluation-capabilities "Direct link to Core Evaluation Capabilities")

### Automated Model Assessment[​](#automated-model-assessment "Direct link to Automated Model Assessment")

MLflow evaluation transforms complex model assessment into simple, reproducible workflows:

python

```python
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Load and prepare data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create evaluation dataset
eval_data = X_test
eval_data["target"] = y_test

with mlflow.start_run():
    # Log model
    mlflow.sklearn.log_model(model, name="model")

    # Comprehensive evaluation with one line
    result = mlflow.models.evaluate(
        model="models:/my-model/1",
        data=eval_data,
        targets="target",
        model_type="classifier",
        evaluators=["default"],
    )

```

What Gets Automatically Generated

#### Performance Metrics[​](#performance-metrics "Direct link to Performance Metrics")

* 📊 **Classification**: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices
* 📈 **Regression**: MAE, MSE, RMSE, R², residual analysis, prediction vs actual plots
* 🎯 **Custom Metrics**: Domain-specific measures defined with simple Python functions

#### Visual Diagnostics[​](#visual-diagnostics "Direct link to Visual Diagnostics")

* 📊 **Performance Plots**: ROC curves, precision-recall curves, calibration plots
* 📈 **Feature Importance**: SHAP values, permutation importance, feature interactions

#### Model Explanations[​](#model-explanations "Direct link to Model Explanations")

* 🧠 **Global Explanations**: Overall model behavior and feature contributions (with `shap`)
* 🔍 **Local Explanations**: Individual prediction explanations and decision paths (with `shap`)

### Flexible Evaluation Modes[​](#flexible-evaluation-modes "Direct link to Flexible Evaluation Modes")

MLflow supports multiple evaluation approaches to fit your workflow:

Comprehensive Evaluation Options

#### Model Evaluation[​](#model-evaluation "Direct link to Model Evaluation")

* 🤖 **Logged Models**: Evaluate models that have been logged to MLflow
* 🔄 **Live Models**: Direct evaluation of in-memory model objects
* 📦 **Pipeline Evaluation**: End-to-end assessment of preprocessing and modeling pipelines

#### Function Evaluation[​](#function-evaluation "Direct link to Function Evaluation")

* ⚡ **Lightweight Assessment**: Evaluate Python functions without model logging overhead
* 🔧 **Custom Predictions**: Assess complex prediction logic and business rules
* 🎯 **Rapid Prototyping**: Quick evaluation during model development

#### Dataset Evaluation[​](#dataset-evaluation "Direct link to Dataset Evaluation")

* 📊 **Static Analysis**: Evaluate pre-computed predictions without re-running models
* 🔄 **Batch Processing**: Assess large-scale inference results efficiently
* 📈 **Historical Analysis**: Evaluate model performance on past predictions

## Specialized Evaluation Areas[​](#specialized-evaluation-areas "Direct link to Specialized Evaluation Areas")

Our comprehensive evaluation framework is organized into specialized areas, each designed for specific aspects of model assessment:

[Model Evaluation](/mlflow-website/docs/latest/ml/evaluation/model-eval.md)

[Core model evaluation workflows for classification and regression tasks with automated metrics, visualizations, and performance assessment.](/mlflow-website/docs/latest/ml/evaluation/model-eval.md)

[Dataset Evaluation](/mlflow-website/docs/latest/ml/evaluation/dataset-eval.md)

[Evaluate static datasets and pre-computed predictions without re-running models, perfect for batch processing and historical analysis.](/mlflow-website/docs/latest/ml/evaluation/dataset-eval.md)

[Function Evaluation](/mlflow-website/docs/latest/ml/evaluation/function-eval.md)

[Lightweight evaluation of Python functions and custom prediction logic without the overhead of model logging and registration.](/mlflow-website/docs/latest/ml/evaluation/function-eval.md)

[Custom Metrics & Visualizations](/mlflow-website/docs/latest/ml/evaluation/metrics-visualizations.md)

[Define domain-specific evaluation criteria, custom metrics, and specialized visualizations tailored to your business requirements.](/mlflow-website/docs/latest/ml/evaluation/metrics-visualizations.md)

[SHAP Integration](/mlflow-website/docs/latest/ml/evaluation/shap.md)

[Deep model interpretation with SHAP values, feature importance analysis, and explainable AI capabilities for transparent ML.](/mlflow-website/docs/latest/ml/evaluation/shap.md)

[Plugin Evaluators](/mlflow-website/docs/latest/ml/evaluation/plugin-evaluators.md)

[Extend evaluation capabilities with specialized plugins like Giskard for vulnerability scanning and Trubrics for advanced validation.](/mlflow-website/docs/latest/ml/evaluation/plugin-evaluators.md)

## Advanced Evaluation Features[​](#advanced-evaluation-features "Direct link to Advanced Evaluation Features")

### Enterprise Integration[​](#enterprise-integration "Direct link to Enterprise Integration")

Production-Grade Evaluation

#### Model Governance[​](#model-governance "Direct link to Model Governance")

* 📋 **Audit Trails**: Complete evaluation history for regulatory compliance
* 🔒 **Access Control**: Role-based evaluation permissions and result visibility
* 📊 **Executive Dashboards**: High-level model performance summaries for stakeholders
* 🔄 **Automated Reporting**: Scheduled evaluation reports and performance alerts

#### MLOps Integration[​](#mlops-integration "Direct link to MLOps Integration")

* 🚀 **CI/CD Pipelines**: Automated evaluation gates in deployment workflows
* 📈 **Performance Monitoring**: Continuous evaluation of production models
* 🔄 **A/B Testing**: Statistical comparison of model variants in production
* 📊 **Drift Detection**: Automated alerts for model performance degradation

## Real-World Applications[​](#real-world-applications "Direct link to Real-World Applications")

MLflow evaluation excels across diverse machine learning applications:

* 🏦 **Financial Services**: Credit scoring model validation, fraud detection performance assessment, and regulatory compliance evaluation
* 🏥 **Healthcare**: Medical AI model validation, diagnostic accuracy assessment, and safety-critical model certification
* 🛒 **E-commerce**: Recommendation system evaluation, search relevance assessment, and personalization effectiveness measurement
* 🚗 **Autonomous Systems**: Safety-critical model validation, edge case analysis, and robustness testing for self-driving vehicles
* 🎯 **Marketing Technology**: Campaign effectiveness measurement, customer segmentation validation, and attribution model assessment
* 🏭 **Manufacturing**: Quality control model validation, predictive maintenance assessment, and process optimization evaluation
* 📱 **Technology Platforms**: Content moderation effectiveness, user behavior prediction accuracy, and system performance optimization

## Getting Started[​](#getting-started "Direct link to Getting Started")

Ready to elevate your model evaluation practices with MLflow? Choose the evaluation approach that best fits your current needs:

Quick Start Recommendations

#### For Data Scientists[​](#for-data-scientists "Direct link to For Data Scientists")

Start with [Model Evaluation](/mlflow-website/docs/latest/ml/evaluation/model-eval.md) to understand comprehensive performance assessment, then explore **Custom Metrics** for domain-specific requirements.

#### For ML Engineers[​](#for-ml-engineers "Direct link to For ML Engineers")

Begin with [Function Evaluation](/mlflow-website/docs/latest/ml/evaluation/function-eval.md) for lightweight testing, then advance to **Model Validation** for production readiness assessment.

#### For ML Researchers[​](#for-ml-researchers "Direct link to For ML Researchers")

Explore [SHAP Integration](/mlflow-website/docs/latest/ml/evaluation/shap.md) for model interpretability, then investigate **Plugin Evaluators** for specialized analysis capabilities.

#### For Enterprise Teams[​](#for-enterprise-teams "Direct link to For Enterprise Teams")

Start with [Model Validation](/mlflow-website/docs/latest/ml/evaluation/metrics-visualizations.md) for governance requirements, then implement [Dataset Evaluation](/mlflow-website/docs/latest/ml/evaluation/dataset-eval.md) for large-scale assessment workflows.

Whether you're validating your first model or implementing enterprise-scale evaluation frameworks, MLflow's comprehensive evaluation suite provides the tools and insights needed to build trustworthy, reliable machine learning systems that deliver real business value with confidence.
