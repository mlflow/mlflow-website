# MLflow for Deep Learning

**Deep learning** has revolutionized artificial intelligence, enabling breakthrough capabilities in computer vision, natural language processing, generative AI, and countless other domains. As models grow more sophisticated, managing the complexity of deep learning experiments becomes increasingly challenging.

MLflow provides a comprehensive solution for tracking, managing, and deploying deep learning models across all major frameworks. Whether you're fine-tuning transformers, training computer vision models, or developing custom neural networks, MLflow's powerful toolkit simplifies your workflow from experiment to production.

Why Deep Learning Needs MLflow

#### The Challenges of Modern Deep Learning[​](#the-challenges-of-modern-deep-learning "Direct link to The Challenges of Modern Deep Learning")

* 🔄 **Iterative Development**: Deep learning requires extensive experimentation with architectures, hyperparameters, and training regimes
* 📊 **Complex Metrics**: Models generate numerous metrics across training steps that must be tracked and compared
* 💾 **Large Artifacts**: Models, checkpoints, and visualizations need systematic storage and versioning
* 🧩 **Framework Diversity**: Teams often work across PyTorch, TensorFlow, Keras, and other specialized libraries
* 🔬 **Reproducibility Crisis**: Without proper tracking, recreating results becomes nearly impossible
* 👥 **Team Collaboration**: Multiple researchers need visibility into experiments and the ability to build on each other's work
* 🚀 **Deployment Complexities**: Moving from successful experiments to production introduces new challenges

MLflow addresses these challenges with a framework-agnostic platform that brings structure and clarity to the entire deep learning lifecycle.

## Key Features for Deep Learning[​](#key-features-for-deep-learning "Direct link to Key Features for Deep Learning")

### 📊 Comprehensive Experiment Tracking[​](#-comprehensive-experiment-tracking "Direct link to 📊 Comprehensive Experiment Tracking")

MLflow's tracking capabilities are tailor-made for the iterative nature of deep learning:

* **One-Line Autologging** for PyTorch, TensorFlow, and Keras
* **Step-Based Metrics** capture training dynamics across epochs and batches
* **Hyperparameter Tracking** for architecture choices and training configurations
* **Resource Monitoring** tracks GPU utilization, memory consumption, and training time

Advanced Tracking Capabilities

#### Beyond Basic Metrics[​](#beyond-basic-metrics "Direct link to Beyond Basic Metrics")

MLflow's tracking system supports the specialized needs of deep learning workflows:

* **Model Architecture Logging**: Automatically capture neural network structures and parameter counts
* **Dataset Tracking**: Record dataset versions, preprocessing steps, and augmentation parameters
* **Visual Debugging**: Store sample predictions, attention maps, and other visual artifacts
* **Distributed Training**: Monitor metrics across multiple nodes in distributed training setups
* **Custom Artifacts**: Log confusion matrices, embedding projections, and other specialized visualizations
* **Hardware Profiling**: Track GPU/TPU utilization, memory consumption, and throughput metrics
* **Early Stopping Points**: Record when early stopping occurred and store the best model states

- Chart Comparison
- Chart Customization
- Run Comparison
- Statistical Evaluation
- Realtime Tracking
- Model Comparison

#### Compare Training Convergence at a Glance[​](#compare-training-convergence-at-a-glance "Direct link to Compare Training Convergence at a Glance")

Visualize multiple deep learning runs to quickly identify which configurations achieve superior performance across training iterations.

![Training convergence comparison](/mlflow-website/docs/latest/assets/images/dl-run-selection-fa090ee9b5d7cdae1517faa7b017914d.gif)

#### Customize Visualizations for Deeper Insights[​](#customize-visualizations-for-deeper-insights "Direct link to Customize Visualizations for Deeper Insights")

Tailor charts to focus on critical metrics and training phases, helping you pinpoint exactly when and why certain models outperform others.

![Chart customization](/mlflow-website/docs/latest/assets/images/dl-run-navigation-153cf4ab2723199ffc34894fbb19e450.gif)

#### Analyze Parameter Relationships[​](#analyze-parameter-relationships "Direct link to Analyze Parameter Relationships")

Explore parameter interactions and their effects on model performance through MLflow's comprehensive comparison views.

![Parameter comparison](/mlflow-website/docs/latest/assets/images/dl-run-comparison-c122a700dec3ba9d5d1cee658837bf4a.gif)

#### Statistical Insights into Hyperparameters[​](#statistical-insights-into-hyperparameters "Direct link to Statistical Insights into Hyperparameters")

Use boxplot visualizations to quickly determine which hyperparameter values consistently lead to better performance.

![Statistical evaluation](/mlflow-website/docs/latest/assets/images/dl-boxplot-277c13455e58ed73ab47d268310a7d33.gif)

#### Monitor Training in Real-Time[​](#monitor-training-in-real-time "Direct link to Monitor Training in Real-Time")

Watch your deep learning models train with live-updating metrics, eliminating the need for manual progress checks.

![Realtime tracking](/mlflow-website/docs/latest/assets/images/dl-tracking-e0cb67a5ce4e271bb3a5675ef4003c95.gif)

#### Model Comparison[​](#model-comparison "Direct link to Model Comparison")

Track your all your DL checkpoints across epochs using the MLflow UI. Compare performance and quickly find the best checkpoints based on any metrics.

![](/mlflow-website/docs/latest/assets/images/dl-model-comparison-9a386d268043b379cec2bbf328d634da.gif)

### 🏆 Streamlined Model Management[​](#-streamlined-model-management "Direct link to 🏆 Streamlined Model Management")

Deep learning models are valuable assets that require careful management:

* **Versioned Model Registry** provides a central repository for all your models
* **Model Lineage** tracks the complete history from data to deployment
* **Metadata Annotations** store architecture details, training datasets, and performance metrics
* **Stage Transitions** manage models through development, staging, and production phases
* **Team Permissions** control who can view, modify, and deploy models
* **Dependency Management** ensures all required packages are tracked with the model

Model Registry for Teams

#### Collaborative Model Development[​](#collaborative-model-development "Direct link to Collaborative Model Development")

The MLflow Model Registry enhances team productivity through:

* **Transition Requests**: Team members can request model promotion with documented justifications
* **Approval Workflows**: Implement governance with required approvals for production deployments (managed MLflow only)
* **Performance Baselines**: Set threshold requirements before models can advance to production
* **Rollback Capabilities**: Quickly revert to previous versions if issues arise
* **Activity Feeds**: Track who made changes to models and when (managed MLflow only)
* **Webhook Integration**: Trigger CI/CD pipelines and notifications based on registry events (managed MLflow only)
* **Model Documentation**: Store comprehensive documentation alongside model artifacts

### 🚀 Simplified Deployment[​](#-simplified-deployment "Direct link to 🚀 Simplified Deployment")

Move from successful experiments to production with ease:

* **Consistent Inference APIs** across all deep learning frameworks
* **GPU-Ready Deployments** for compute-intensive models
* **Batch and Real-Time Serving** options for different application needs
* **Docker Containerization** for portable, isolated environments
* **Serverless Deployments** for scalable, cost-effective serving within your cloud provider infrastructure
* **Edge Deployment** support for mobile and IoT applications

Advanced Deployment Options

#### Beyond Basic Serving[​](#beyond-basic-serving "Direct link to Beyond Basic Serving")

MLflow supports sophisticated deployment scenarios for deep learning:

* **Model Ensembling**: Deploy multiple models with voting or averaging mechanisms
* **Custom Preprocessing/Postprocessing**: Attach data transformation pipelines to your model
* **Optimized Inference**: Support for quantization, pruning, and other optimization techniques
* **Monitoring Integration**: Connect to observability platforms for production tracking
* **Hardware Acceleration**: Leverage GPU/TPU resources for high-throughput inference in cloud provider infrastructure
* **Scalable Architecture**: Handle variable loads with auto-scaling capabilities (managed MLflow only)
* **Multi-Framework Deployment**: Mix models from different frameworks in the same serving environment

## Framework Integrations[​](#framework-integrations "Direct link to Framework Integrations")

MLflow provides native support for all major deep learning frameworks, allowing you to use your preferred tools while gaining the benefits of unified experiment tracking and model management.

[![TensorFlow Logo](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDI1LjAuMSwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMCIgaWQ9ImthdG1hbl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCIKCSB2aWV3Qm94PSIwIDAgMzM4IDIwMCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMzM4IDIwMDsiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8c3R5bGUgdHlwZT0idGV4dC9jc3MiPgoJLnN0MHtjbGlwLXBhdGg6dXJsKCNTVkdJRF8yXyk7fQoJLnN0MXtmaWxsOnVybCgjU1ZHSURfM18pO30KCS5zdDJ7Y2xpcC1wYXRoOnVybCgjU1ZHSURfNV8pO30KCS5zdDN7ZmlsbDp1cmwoI1NWR0lEXzZfKTt9Cgkuc3Q0e2ZpbGw6IzQyNTA2Njt9Cjwvc3R5bGU+CjxnPgoJPGc+CgkJPGc+CgkJCTxkZWZzPgoJCQkJPHBvbHlnb24gaWQ9IlNWR0lEXzFfIiBwb2ludHM9IjczLjUsODUuNiA1MSw3Mi44IDUxLDEyNS40IDYwLDEyMC4yIDYwLDEwNS40IDY2LjgsMTA5LjMgNjYuNyw5OS4yIDYwLDk1LjMgNjAsODkuNCA3My41LDk3LjMgCgkJCQkJCQkJCSIvPgoJCQk8L2RlZnM+CgkJCTxjbGlwUGF0aCBpZD0iU1ZHSURfMl8iPgoJCQkJPHVzZSB4bGluazpocmVmPSIjU1ZHSURfMV8iICBzdHlsZT0ib3ZlcmZsb3c6dmlzaWJsZTsiLz4KCQkJPC9jbGlwUGF0aD4KCQkJPGcgY2xhc3M9InN0MCI+CgkJCQkKCQkJCQk8bGluZWFyR3JhZGllbnQgaWQ9IlNWR0lEXzNfIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgeDE9IjI0LjQiIHkxPSItMjAxLjA1IiB4Mj0iNzkuNiIgeTI9Ii0yMDEuMDUiIGdyYWRpZW50VHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAtMTAyKSI+CgkJCQkJPHN0b3AgIG9mZnNldD0iMCIgc3R5bGU9InN0b3AtY29sb3I6I0ZGNkYwMCIvPgoJCQkJCTxzdG9wICBvZmZzZXQ9IjEiIHN0eWxlPSJzdG9wLWNvbG9yOiNGRkE4MDAiLz4KCQkJCTwvbGluZWFyR3JhZGllbnQ+CgkJCQk8cGF0aCBjbGFzcz0ic3QxIiBkPSJNMjQuNCw3Mi42aDU1LjJ2NTIuOUgyNC40VjcyLjZ6Ii8+CgkJCTwvZz4KCQk8L2c+Cgk8L2c+CjwvZz4KPGc+Cgk8Zz4KCQk8Zz4KCQkJPGRlZnM+CgkJCQk8cG9seWdvbiBpZD0iU1ZHSURfNF8iIHBvaW50cz0iMjYuNSw4NS42IDQ5LDcyLjggNDksMTI1LjQgNDAsMTIwLjIgNDAsODkuNCAyNi41LDk3LjMgCQkJCSIvPgoJCQk8L2RlZnM+CgkJCTxjbGlwUGF0aCBpZD0iU1ZHSURfNV8iPgoJCQkJPHVzZSB4bGluazpocmVmPSIjU1ZHSURfNF8iICBzdHlsZT0ib3ZlcmZsb3c6dmlzaWJsZTsiLz4KCQkJPC9jbGlwUGF0aD4KCQkJPGcgY2xhc3M9InN0MiI+CgkJCQkKCQkJCQk8bGluZWFyR3JhZGllbnQgaWQ9IlNWR0lEXzZfIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgeDE9IjI0LjEiIHkxPSItMjAxLjA1IiB4Mj0iNzkuMyIgeTI9Ii0yMDEuMDUiIGdyYWRpZW50VHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAtMTAyKSI+CgkJCQkJPHN0b3AgIG9mZnNldD0iMCIgc3R5bGU9InN0b3AtY29sb3I6I0ZGNkYwMCIvPgoJCQkJCTxzdG9wICBvZmZzZXQ9IjEiIHN0eWxlPSJzdG9wLWNvbG9yOiNGRkE4MDAiLz4KCQkJCTwvbGluZWFyR3JhZGllbnQ+CgkJCQk8cGF0aCBjbGFzcz0ic3QzIiBkPSJNMjQuMSw3Mi42aDU1LjJ2NTIuOUgyNC4xVjcyLjZ6Ii8+CgkJCTwvZz4KCQk8L2c+Cgk8L2c+CjwvZz4KPHBhdGggY2xhc3M9InN0NCIgZD0iTTExNC4yLDg5LjFoLTEwdjI3LjdoLTUuNlY4OS4xaC0xMHYtNC41aDI1LjZWODkuMXoiLz4KPHBhdGggY2xhc3M9InN0NCIgZD0iTTEyMC45LDExNy4yYy0zLjQsMC02LjItMS4xLTguMy0zLjJjLTIuMS0yLjEtMy4yLTUtMy4yLTguNnYtMC43YzAtMi4yLDAuNC00LjQsMS40LTYuNAoJYzAuOS0xLjgsMi4yLTMuMywzLjktNC40YzEuNy0xLjEsMy42LTEuNiw1LjYtMS42YzMuMywwLDUuOCwxLDcuNiwzLjFzMi43LDUsMi43LDguOHYyLjJoLTE1LjdjMC4xLDEuOCwwLjgsMy40LDIsNC43CgljMS4yLDEuMiwyLjcsMS44LDQuNCwxLjdjMi40LDAuMSw0LjYtMS4xLDYtM2wyLjksMi44Yy0xLDEuNC0yLjMsMi42LTMuOCwzLjNDMTI0LjYsMTE2LjksMTIyLjgsMTE3LjMsMTIwLjksMTE3LjJ6IE0xMjAuMyw5Ni43CgljLTEuNC0wLjEtMi43LDAuNS0zLjYsMS41Yy0xLDEuMi0xLjYsMi43LTEuNyw0LjNoMTAuM3YtMC40Yy0wLjEtMS44LTAuNi0zLjItMS40LTQuMUMxMjIuOSw5Ny4yLDEyMS42LDk2LjYsMTIwLjMsOTYuN3oKCSBNMTM5LjcsOTIuOGwwLjIsMi44YzEuNy0yLjEsNC4zLTMuMyw3LTMuMmM1LDAsNy41LDIuOSw3LjYsOC42djE1LjhoLTUuNHYtMTUuNWMwLTEuNS0wLjMtMi42LTEtMy40Yy0wLjctMC43LTEuNy0xLjEtMy4yLTEuMQoJYy0yLjEtMC4xLTQsMS4xLTQuOSwyLjl2MTdoLTUuNHYtMjRDMTM0LjYsOTIuNywxMzkuNyw5Mi44LDEzOS43LDkyLjh6IE0xNzEuOSwxMTAuM2MwLTAuOS0wLjQtMS43LTEuMi0yLjIKCWMtMS4yLTAuNy0yLjYtMS4xLTMuOS0xLjNjLTEuNi0wLjMtMy4xLTAuOC00LjYtMS41Yy0yLjctMS4zLTQtMy4yLTQtNS42YzAtMiwxLTQsMi42LTUuMmMxLjctMS40LDQtMi4xLDYuNi0yLjEKCWMyLjksMCw1LjIsMC43LDYuOSwyLjFjMS43LDEuMywyLjcsMy40LDIuNiw1LjVoLTUuNGMwLTEtMC40LTEuOS0xLjItMi42Yy0wLjktMC43LTEuOS0xLjEtMy4xLTFjLTEsMC0yLDAuMi0yLjksMC44CgljLTAuNywwLjUtMS4xLDEuMy0xLjEsMi4yYzAsMC44LDAuNCwxLjUsMSwxLjljMC43LDAuNSwyLjEsMC45LDQuMiwxLjRjMS43LDAuMywzLjQsMC45LDUsMS43YzEuMSwwLjUsMiwxLjMsMi43LDIuMwoJYzAuNiwxLDAuOSwyLjEsMC45LDMuM2MwLDIuMS0xLDQtMi43LDUuMmMtMS44LDEuMy00LjEsMi03LDJjLTEuOCwwLTMuNi0wLjMtNS4yLTEuMWMtMS40LTAuNi0yLjctMS42LTMuNi0yLjkKCWMtMC44LTEuMi0xLjMtMi42LTEuMy00aDUuMmMwLDEuMSwwLjUsMi4yLDEuNCwyLjljMSwwLjcsMi4zLDEuMSwzLjUsMWMxLjQsMCwyLjUtMC4zLDMuMi0wLjhDMTcxLjUsMTExLjksMTcxLjksMTExLjEsMTcxLjksMTEwLjMKCUwxNzEuOSwxMTAuM3ogTTE4MCwxMDQuNmMwLTIuMiwwLjQtNC40LDEuNC02LjNjMC45LTEuOCwyLjItMy4zLDMuOS00LjNjMS44LTEsMy44LTEuNiw1LjgtMS41YzMuMiwwLDUuOSwxLDcuOSwzLjEKCWMyLDIuMSwzLjEsNC44LDMuMyw4LjN2MS4zYzAsMi4yLTAuNCw0LjMtMS40LDYuM2MtMC44LDEuOC0yLjIsMy4zLTMuOSw0LjNjLTEuOCwxLTMuOCwxLjYtNS45LDEuNWMtMy40LDAtNi4xLTEuMS04LjEtMy40CgljLTItMi4yLTMtNS4yLTMuMS05TDE4MCwxMDQuNnogTTE4NS4zLDEwNS4xYzAsMi41LDAuNSw0LjQsMS41LDUuOGMxLjgsMi4zLDUuMSwyLjgsNy41LDFjMC40LTAuMywwLjctMC42LDEtMQoJYzEtMS40LDEuNS0zLjUsMS41LTYuMmMwLTIuNC0wLjUtNC4zLTEuNi01LjhjLTEuNy0yLjMtNS0yLjgtNy40LTEuMWMtMC40LDAuMy0wLjgsMC43LTEuMSwxQzE4NS45LDEwMC4yLDE4NS4zLDEwMi4zLDE4NS4zLDEwNS4xegoJIE0yMTguNCw5Ny44Yy0wLjctMC4xLTEuNS0wLjItMi4yLTAuMmMtMi41LDAtNC4xLDAuOS01LDIuOHYxNi40aC01LjR2LTI0aDUuMWwwLjEsMi43YzEuMy0yLjEsMy4xLTMuMSw1LjQtMy4xCgljMC42LDAsMS4zLDAuMSwxLjksMC4zTDIxOC40LDk3LjhMMjE4LjQsOTcuOHogTTI0MC45LDEwMy4xaC0xM3YxMy43aC01LjZWODQuNmgyMC41djQuNWgtMTQuOXY5LjZoMTNWMTAzLjF6IE0yNTEuNiwxMTYuOGgtNS40CglWODQuNWg1LjRDMjUxLjYsODQuNSwyNTEuNiwxMTYuOCwyNTEuNiwxMTYuOHogTTI1NS41LDEwNC42YzAtMi4yLDAuNC00LjQsMS40LTYuM2MwLjktMS44LDIuMi0zLjMsMy45LTQuM2MxLjgtMSwzLjgtMS42LDUuOC0xLjUKCWMzLjIsMCw1LjksMSw3LjksMy4xYzIsMi4xLDMuMSw0LjgsMy4zLDguM3YxLjNjMCwyLjItMC40LDQuMy0xLjMsNi4zYy0wLjgsMS44LTIuMiwzLjMtMy45LDQuM2MtMS44LDEtMy44LDEuNi01LjksMS41CgljLTMuNCwwLTYuMS0xLjEtOC4xLTMuNGMtMi0yLjItMy4xLTUuMi0zLjEtOUwyNTUuNSwxMDQuNkwyNTUuNSwxMDQuNnogTTI2MC45LDEwNS4xYzAsMi41LDAuNSw0LjQsMS41LDUuOHMyLjYsMi4yLDQuMywyLjEKCWMxLjcsMC4xLDMuMy0wLjcsNC4yLTIuMWMxLTEuNCwxLjUtMy41LDEuNS02LjJjMC0yLjQtMC41LTQuMy0xLjYtNS44Yy0xLjctMi4zLTUtMi44LTcuNC0xLjFjLTAuNCwwLjMtMC44LDAuNy0xLjEsMQoJQzI2MS40LDEwMC4yLDI2MC45LDEwMi4zLDI2MC45LDEwNS4xeiBNMzAyLjEsMTA5LjRsMy44LTE2LjVoNS4ybC02LjUsMjRoLTQuNGwtNS4xLTE2LjVsLTUuMSwxNi41aC00LjRsLTYuNi0yNGg1LjNsMy45LDE2LjQKCWw0LjktMTYuNGg0LjFMMzAyLjEsMTA5LjRMMzAyLjEsMTA5LjR6Ii8+Cjwvc3ZnPgo=)](/mlflow-website/docs/latest/ml/deep-learning/tensorflow.md)

[Seamlessly track TensorFlow experiments with one-line autologging. Capture training metrics, model architecture, and TensorBoard visualizations in a centralized repository.](/mlflow-website/docs/latest/ml/deep-learning/tensorflow.md)

[![PyTorch Logo](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMjAiIGhlaWdodD0iNjAiPjxnIHRyYW5zZm9ybT0ibWF0cml4KDEuMjE2NjAxIDAgMCAxLjIxNjYwMSAtMTcuMjc3MTQ5IDExLjQzMTYyMikiIGZpbGw9IiNlZTRjMmMiPjxwYXRoIGQ9Ik00MC44IDkuM2wtMi4xIDIuMWMzLjUgMy41IDMuNSA5LjIgMCAxMi43cy05LjIgMy41LTEyLjcgMC0zLjUtOS4yIDAtMTIuN2w1LjYtNS42LjctLjhWLjhsLTguNSA4LjVhMTEuODkgMTEuODkgMCAwIDAgMCAxNi45IDExLjg5IDExLjg5IDAgMCAwIDE2LjkgMGM0LjgtNC43IDQuOC0xMi4zLjEtMTYuOXoiLz48Y2lyY2xlIGN4PSIzNi42IiBjeT0iNy4xIiByPSIxLjYiLz48L2c+PHBhdGggZD0iTTQ4LjAwOCAzMi4wMjhoLTJ2NS4xNDRoLTEuNDkzVjIyLjU3aDMuNjVjMy44NzIgMCA1LjY5NyAxLjg4IDUuNjk3IDQuNiAwIDMuMjA4LTIuMjY4IDQuODEyLTUuODYzIDQuODY3em0uMS04LjA3NUg0NS45NnY2LjY5M2wyLjEwMi0uMDU1YzIuNzY2LS4wNTUgNC4yNi0xLjE2MiA0LjI2LTMuNDMgMC0yLjA0Ni0xLjQzOC0zLjIwOC00LjIwNC0zLjIwOHpNNjAuNjIgMzcuMTE2bC0uODg1IDIuMzIzYy0uOTk2IDIuNi0yIDMuMzc0LTMuNDg1IDMuMzc0LS44MyAwLTEuNDM4LS4yMi0yLjEwMi0uNDk4bC40NDItMS4zMjdjLjQ5OC4yNzcgMS4wNS40NDIgMS42Ni40NDIuODMgMCAxLjQzOC0uNDQyIDIuMjEyLTIuNWwuNzItMS44OC00LjE0OC0xMC41NjRoMS41NWwzLjM3NCA4Ljg1IDMuMzItOC44NWgxLjQ5M3ptOS4xMjUtMTMuMTA4djEzLjIyaC0xLjQ5M3YtMTMuMjJoLTUuMTQ0VjIyLjU3aDExLjc4djEuMzgzaC01LjE0NHptOS4zNDcgMTMuNDk1Yy0yLjk4NyAwLTUuMi0yLjIxMi01LjItNS42NDJzMi4yNjgtNS42OTcgNS4zLTUuNjk3YzIuOTg3IDAgNS4xNDQgMi4yMTIgNS4xNDQgNS42NDJzLTIuMjY4IDUuNjk3LTUuMjU1IDUuNjk3em0uMDU1LTEwYy0yLjI2OCAwLTMuNzYgMS44MjUtMy43NiA0LjMxNCAwIDIuNiAxLjU1IDQuMzcgMy44MTYgNC4zN3MzLjc2LTEuODI1IDMuNzYtNC4zMTRjMC0yLjYtMS41NS00LjM3LTMuODE2LTQuMzd6bTguOTA2IDkuNzI0aC0xLjQzOHYtMTAuNzNsMS40MzgtLjI3N3YyLjI2OGMuNzItMS4zODMgMS43Ny0yLjI2OCAzLjE1My0yLjI2OGEzLjkyIDMuOTIgMCAwIDEgMS44OC40OThMOTIuNyAyOC4xYy0uNDQyLS4yNzctMS4wNS0uNDQyLTEuNjYtLjQ0Mi0xLjEwNiAwLTIuMTU3LjgzLTMuMDQyIDIuNzY2djYuODAzem0xMC43My4yNzZjLTMuMjA4IDAtNS4yNTUtMi4zMjMtNS4yNTUtNS42NDIgMC0zLjM3NCAyLjIxMi01LjY5NyA1LjI1NS01LjY5NyAxLjMyNyAwIDIuNDM0LjMzMiAzLjM3NC45NGwtLjM4NyAxLjMyN2MtLjgzLS41NTMtMS44MjUtLjg4NS0yLjk4Ny0uODg1LTIuMzIzIDAtMy43NiAxLjcxNS0zLjc2IDQuMjYgMCAyLjYgMS41NSA0LjMxNCAzLjgxNiA0LjMxNGE1LjU3IDUuNTcgMCAwIDAgMi45ODctLjg4NWwuMjc3IDEuMzI3Yy0uOTQuNjA4LTIuMTAyLjk0LTMuMzIuOTR6bTEyLjMzNC0uMjc2di02LjkxNGMwLTEuODgtLjc3NC0yLjctMi4yNjgtMi43LTEuMjE3IDAtMi40MzQuNjA4LTMuMzIgMS41NXY4LjEzaC0xLjQzOHYtMTUuODJsMS40MzgtLjI3N3Y2Ljc0OGMxLjEwNi0xLjEwNiAyLjU0NC0xLjcxNSAzLjcwNi0xLjcxNSAyLjEwMiAwIDMuMzc0IDEuMzI3IDMuMzc0IDMuNjV2Ny4zNTZ6IiBmaWxsPSIjMjUyNTI1Ii8+PC9zdmc+)](/mlflow-website/docs/latest/ml/deep-learning/pytorch.md)

[Integrate MLflow with PyTorch's flexible deep learning ecosystem. Log metrics from custom training loops, save model checkpoints, and simplify deployment for production.](/mlflow-website/docs/latest/ml/deep-learning/pytorch.md)

[![Keras Logo](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzNjQuNjY1MDcgMzY0LjY2NTA3IiBoZWlnaHQ9IjM4OC45NzYiIHdpZHRoPSIzODguOTc2Ij48cGF0aCBmaWxsPSIjZDAwMDAwIiBkPSJNMCAwaDM2NC42NjV2MzY0LjY2NUgweiIvPjxwYXRoIGQ9Ik0xMzUuNTkyIDI4MS40OHYtNjcuN2wyNy40OS0yNy40MDQgNjguOTYzIDEwMS45MSAzMS41ODcuMjQ4IDUuODMyLTExLjkwNS04MC4yNDgtMTE2LjQxNSA3My44NzYtNzUuMTA4LTQuMDktMTEuOTA5SDIyNy40OGwtOTEuODg4IDkxLjg2M1Y4MC4yMWwtNi43MTctNy4wMTNIMTA2LjA2bC02LjcxOCA3LjAxMnYyMDAuOTc2bDcuMDc1IDcuMTkgMjEuOTg1LS4wODh6IiBmaWxsPSIjZmZmIi8+PC9zdmc+)](/mlflow-website/docs/latest/ml/deep-learning/keras.md)

[Harness Keras 3.0's multi-backend capabilities with comprehensive MLflow tracking. Monitor training across TensorFlow, PyTorch, and JAX backends with consistent experiment management.](/mlflow-website/docs/latest/ml/deep-learning/keras.md)

[![spaCy Logo](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE4LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDEuMS8vRU4iICJodHRwOi8vd3d3LnczLm9yZy9HcmFwaGljcy9TVkcvMS4xL0RURC9zdmcxMS5kdGQiPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkViZW5lXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMzA4LjUgNTk1LjMgMjEzIiBlbmFibGUtYmFja2dyb3VuZD0ibmV3IDAgMzA4LjUgNTk1LjMgMjEzIiB4bWw6c3BhY2U9InByZXNlcnZlIj4KPHBhdGggZmlsbD0iIzA5YTNkNSIgZD0iTTczLjcsMzk1LjJjLTEzLjUtMS42LTE0LjUtMTkuNy0zMS44LTE4LjFjLTguNCwwLTE2LjIsMy41LTE2LjIsMTEuMmMwLDExLjYsMTcuOSwxMi43LDI4LjcsMTUuNgoJYzE4LjQsNS42LDM2LjIsOS40LDM2LjIsMjkuNGMwLDI1LjQtMTkuOSwzNC4yLTQ2LjIsMzQuMmMtMjIsMC00NC4zLTcuOC00NC4zLTI4YzAtNS42LDUuNC0xMCwxMC42LTEwYzYuNiwwLDguOSwyLjgsMTEuMiw3LjQKCWM1LjEsOSwxMC44LDEzLjgsMjUsMTMuOGM5LDAsMTguMi0zLjQsMTguMi0xMS4yYzAtMTEuMS0xMS4zLTEzLjUtMjMtMTYuMmMtMjAuNy01LjgtMzguNS04LjgtNDAuNi0zMS44CgljLTIuMi0zOS4yLDc5LjUtNDAuNyw4NC4yLTYuM0M4NS42LDM5MS40LDc5LjgsMzk1LjIsNzMuNywzOTUuMnogTTE3MC45LDM2MC44YzI4LjcsMCw0NSwyNCw0NSw1My42YzAsMjkuNy0xNS44LDUzLjYtNDUsNTMuNgoJYy0xNi4yLDAtMjYuMy02LjktMzMuNi0xNy41djM5LjJjMCwxMS44LTMuOCwxNy41LTEyLjQsMTcuNWMtMTAuNSwwLTEyLjQtNi43LTEyLjQtMTcuNXYtMTE0YzAtOS4zLDMuOS0xNSwxMi40LTE1CgljOCwwLDEyLjQsNi4zLDEyLjQsMTV2My4yQzE0NS40LDM2OC43LDE1NC43LDM2MC44LDE3MC45LDM2MC44eiBNMTY0LjEsNDQ3LjZjMTYuOCwwLDI0LjMtMTUuNSwyNC4zLTMzLjYKCWMwLTE3LjctNy42LTMzLjYtMjQuMy0zMy42Yy0xNy41LDAtMjUuNiwxNC40LTI1LjYsMzMuNkMxMzguNSw0MzIuNywxNDYuNyw0NDcuNiwxNjQuMSw0NDcuNnogTTIzNS40LDM4OC44YzAtMjAuNiwyMy43LTI4LDQ2LjctMjgKCWMzMi4zLDAsNDUuNiw5LjQsNDUuNiw0MC42djMwYzAsNy4xLDQuNCwyMS4zLDQuNCwyNS42YzAsNi41LTYsMTAuNi0xMi40LDEwLjZjLTcuMSwwLTEyLjQtOC40LTE2LjItMTQuNAoJYy0xMC41LDguNC0yMS42LDE0LjQtMzguNiwxNC40Yy0xOC44LDAtMzMuNi0xMS4xLTMzLjYtMjkuNGMwLTE2LjIsMTEuNi0yNS41LDI1LjYtMjguN2MwLDAuMSw0NS0xMC42LDQ1LTEwLjcKCWMwLTEzLjgtNC45LTE5LjktMTkuNC0xOS45Yy0xMi44LDAtMTkuMywzLjUtMjQuMywxMS4yYy00LDUuOC0zLjUsOS4zLTExLjIsOS4zQzI0MC44LDM5OS4zLDIzNS40LDM5NS4xLDIzNS40LDM4OC44eiBNMjczLjgsNDUwLjcKCWMxOS43LDAsMjgtMTAuNCwyOC0zMS4xdi00LjRjLTUuMywxLjgtMjYuNyw3LjEtMzIuNSw4Yy02LjIsMS4yLTEyLjQsNS44LTEyLjQsMTMuMUMyNTcuMSw0NDQuMywyNjUuMyw0NTAuNywyNzMuOCw0NTAuN3oKCSBNNDE4LjUsMzIxLjdjMjcuOCwwLDU3LjksMTYuNiw1Ny45LDQzYzAsNi44LTUuMSwxMi40LTExLjgsMTIuNGMtOS4xLDAtMTAuNC00LjktMTQuNC0xMS44Yy02LjctMTIuMy0xNC42LTIwLjUtMzEuOC0yMC41CgljLTI2LjYtMC4yLTM4LjUsMjIuNi0zOC41LDUxYzAsMjguNiw5LjksNDkuMiwzNy40LDQ5LjJjMTguMywwLDI4LjQtMTAuNiwzMy42LTI0LjNjMi4xLTYuMyw1LjktMTIuNCwxMy44LTEyLjQKCWM2LjIsMCwxMi40LDYuMywxMi40LDEzLjFjMCwyOC0yOC42LDQ3LjQtNTgsNDcuNGMtMzIuMiwwLTUwLjQtMTMuNi02MC40LTM2LjJjLTQuOS0xMC44LTgtMjItOC0zNy40CglDMzUwLjUsMzUxLjgsMzc1LjgsMzIxLjcsNDE4LjUsMzIxLjdMNDE4LjUsMzIxLjd6IE01NzcuNSwzNjAuOGM3LjEsMCwxMS4yLDQuNiwxMS4yLDExLjhjMCwyLjktMi4zLDguNy0zLjIsMTEuOGwtMzQuMiw4OS45CgljLTcuNiwxOS41LTEzLjMsMzMtMzkuMiwzM2MtMTIuMywwLTIzLTEuMS0yMy0xMS44YzAtNi4yLDQuNy05LjMsMTEuMi05LjNjMS4yLDAsMy4yLDAuNiw0LjQsMC42YzEuOSwwLDMuMiwwLjYsNC40LDAuNgoJYzEzLDAsMTQuOC0xMy4zLDE5LjQtMjIuNWwtMzMtODEuN2MtMS45LTQuNC0zLjItNy40LTMuMi0xMGMwLTcuMiw1LjYtMTIuNCwxMy4xLTEyLjRjOC40LDAsMTEuNyw2LjYsMTMuOCwxMy44bDIxLjgsNjQuOAoJbDIxLjgtNTkuOUM1NjYuMSwzNzAuMiw1NjYuNCwzNjAuOCw1NzcuNSwzNjAuOHoiLz4KPC9zdmc+Cg==)](/mlflow-website/docs/latest/ml/model.md#spacyspacy)

[Track and manage spaCy NLP models throughout their lifecycle. Log training metrics, compare model versions, and deploy language processing pipelines to production.](/mlflow-website/docs/latest/ml/model.md#spacyspacy)

## Getting Started[​](#getting-started "Direct link to Getting Started")

Quick Setup Guide

### 1. Install MLflow[​](#1-install-mlflow "Direct link to 1. Install MLflow")

bash

```bash
pip install mlflow

```

Ensure that you have the appropriate DL integration package installed. For example, for PyTorch with image model support:

bash

```bash
pip install torch torchvision

```

### 2. Start Tracking Server (Optional)[​](#2-start-tracking-server-optional "Direct link to 2. Start Tracking Server (Optional)")

bash

```bash
# Start a local tracking server
mlflow server --host 0.0.0.0 --port 5000

```

### 3. Enable Autologging[​](#3-enable-autologging "Direct link to 3. Enable Autologging")

python

```python
import mlflow

# For TensorFlow/Keras
mlflow.tensorflow.autolog()

# For PyTorch Lightning
mlflow.pytorch.autolog()

# For all supported frameworks
mlflow.autolog()

```

### 4. Train Your Model Normally[​](#4-train-your-model-normally "Direct link to 4. Train Your Model Normally")

python

```python
# Your existing training code works unchanged!
model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))

```

### 5. View Results[​](#5-view-results "Direct link to 5. View Results")

Open the MLflow UI to see your tracked experiments:

bash

```bash
mlflow ui

```

Or if using a tracking server:

text

```text
http://localhost:5000

```

## Real-World Applications[​](#real-world-applications "Direct link to Real-World Applications")

Deep learning with MLflow powers a wide range of applications across industries:

* 🖼️ **Computer Vision**: Track performance of object detection, image segmentation, and classification models
* 🔊 **Speech Recognition**: Monitor acoustic model training and compare word error rates across architectures
* 📝 **Natural Language Processing**: Manage fine-tuning of large language models and evaluate performance on downstream tasks
* 🎮 **Reinforcement Learning**: Track agent performance, rewards, and environmental interactions across training runs
* 🧬 **Genomics**: Organize deep learning models analyzing genetic sequences and protein structures
* 📊 **Financial Forecasting**: Compare predictive models for time series analysis and risk assessment
* 🏭 **Manufacturing**: Deploy computer vision models for quality control and predictive maintenance
* 🏥 **Healthcare**: Manage medical imaging models with rigorous versioning and approval workflows

## Advanced Topics[​](#advanced-topics "Direct link to Advanced Topics")

Distributed Training Integration

MLflow integrates seamlessly with distributed training frameworks:

* **Horovod**: Track metrics across distributed TensorFlow and PyTorch training
* **PyTorch DDP**: Monitor distributed data parallel training
* **TensorFlow Distribution Strategies**: Log metrics from multi-GPU and multi-node training
* **Ray**: Integrate with Ray's distributed computing ecosystem

Example with PyTorch DDP:

python

```python
import mlflow
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

mlflow.pytorch.autolog()

# Initialize process group
dist.init_process_group(backend="nccl")

# Create model and move to GPU with DDP wrapper
model = DistributedDataParallel(model.to(rank))

# MLflow tracking works normally with DDP
with mlflow.start_run():
    trainer.fit(model)

```

Hyperparameter Optimization

MLflow integrates with popular hyperparameter optimization frameworks:

* **Optuna**: Track trials and visualize optimization results
* **Ray Tune**: Monitor distributed hyperparameter sweeps
* **Weights & Biases Sweeps**: Synchronize W\&B sweeps with MLflow tracking
* **HyperOpt**: Organize and compare hyperparameter search results

Example with Optuna:

python

```python
import mlflow
import optuna


def objective(trial):
    with mlflow.start_run(nested=True):
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # Log parameters to MLflow
        mlflow.log_params({"lr": lr, "batch_size": batch_size})

        # Train model
        model = create_model(lr)
        result = train_model(model, batch_size)

        # Log results
        mlflow.log_metrics({"accuracy": result["accuracy"]})

        return result["accuracy"]


# Create study
with mlflow.start_run():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Log best parameters
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_accuracy", study.best_value)

```

Transfer Learning Workflows

MLflow helps organize transfer learning and fine-tuning workflows:

* **Base Model Registry**: Maintain a catalog of pre-trained models
* **Fine-Tuning Tracking**: Monitor performance as you adapt models to new tasks
* **Layer Freezing Analysis**: Compare different layer freezing strategies
* **Learning Rate Scheduling**: Track the impact of different learning rate strategies for fine-tuning

Example tracking a fine-tuning run:

python

```python
import mlflow
import torch
from transformers import AutoModelForSequenceClassification

with mlflow.start_run():
    # Log base model information
    base_model_name = "bert-base-uncased"
    mlflow.log_param("base_model", base_model_name)

    # Create and customize model for fine-tuning
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name)

    # Log which layers are frozen
    frozen_layers = ["embeddings", "encoder.layer.0", "encoder.layer.1"]
    mlflow.log_param("frozen_layers", frozen_layers)

    # Freeze specified layers
    for name, param in model.named_parameters():
        if any(layer in name for layer in frozen_layers):
            param.requires_grad = False

    # Log trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    mlflow.log_params(
        {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percentage": trainable_params / total_params,
        }
    )

    # Fine-tune and track results...

```

## Learn More[​](#learn-more "Direct link to Learn More")

Dive deeper into MLflow's capabilities for deep learning in our framework-specific guides:

* **[TensorFlow Guide](/mlflow-website/docs/latest/ml/deep-learning/tensorflow.md)**: Master MLflow's integration with TensorFlow and Keras
* **[PyTorch Guide](/mlflow-website/docs/latest/ml/deep-learning/pytorch.md)**: Learn how to track custom PyTorch training loops
* **[Keras Guide](/mlflow-website/docs/latest/ml/deep-learning/keras.md)**: Explore Keras 3.0's multi-backend capabilities with MLflow
* **[Model Registry](/mlflow-website/docs/latest/ml/model-registry.md)**: Manage model versions and transitions through development stages
* **[MLflow Deployments](/mlflow-website/docs/latest/ml/deployment.md)**: Deploy deep learning models to production
