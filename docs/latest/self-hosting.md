# Self-Hosting MLflow

> #### ***The most vendor-neutral MLOps/LLMOps platform in the world.***[​](#the-most-vendor-neutral-mlopsllmops-platform-in-the-world "Direct link to the-most-vendor-neutral-mlopsllmops-platform-in-the-world")

MLflow is fully open-source. Thousands of users and organizations run their own MLflow instances to meet their specific needs. Being open-source and trusted by the popular cloud providers, MLflow is the best choice for teams/organizations that worry about vendor lock-in.

## The Quickest Path: Run `mlflow` Command[​](#the-quickest-path-run-mlflow-command "Direct link to the-quickest-path-run-mlflow-command")

The easiest way to start MLflow server is to run the `mlflow` CLI command in your terminal. This is suitable for personal use or small teams.

First, install MLflow with:

bash

```bash
pip install mlflow

```

Then, start the server with:

bash

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

```

This will start the server and UI at `http://localhost:5000`. You can connect the client to the server by setting the tracking URI:

python

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Start tracking!
# Open http://localhost:5000 in your browser to view the UI.

```

Now, you are ready to start your experiment!

* [Tracing QuickStart](/mlflow-website/docs/latest/genai/tracing/quickstart/python-openai.md)
* [LLM Evaluation Quickstart](/mlflow-website/docs/latest/genai/eval-monitor/quickstart.md)
* [Prompt Management Quickstart](/mlflow-website/docs/latest/genai/prompt-registry.md#getting-started)
* [Model Training Quickstart](/mlflow-website/docs/latest/ml/tracking/quickstart.md)

tip

The `--backend-store-uri` option is not mandatory, but highly recommended for better performance and reliability. Check out [Backend Store](/mlflow-website/docs/latest/self-hosting/architecture/backend-store.md).

## Other Deployment Options[​](#other-deployment-options "Direct link to Other Deployment Options")

### Docker Compose[​](#docker-compose "Direct link to Docker Compose")

The MLflow repository includes a ready-to-run Compose project under `docker-compose/` that provisions MLflow, PostgreSQL, and MinIO.

bash

```bash
git clone https://github.com/mlflow/mlflow.git
cd docker-compose
cp .env.dev.example .env
docker compose up -d

```

Read the instructions [here](https://github.com/mlflow/mlflow/tree/master/docker-compose) for more details and configuration options for the docker compose bundle.

### Kubernetes[​](#kubernetes "Direct link to Kubernetes")

To deploy on Kubernetes, use the MLflow Helm chart provided by [Bitnami](https://artifacthub.io/packages/helm/bitnami/mlflow) or [Community Helm Charts](https://artifacthub.io/packages/helm/community-charts/mlflow).

### Cloud Services[​](#cloud-services "Direct link to Cloud Services")

If you are looking for production-scale deployments without maintenance costs, MLflow is also available as managed services from popular cloud providers.

* [Databricks](https://www.databricks.com/product/managed-mlflow)
* [AWS Sagemaker](https://aws.amazon.com/sagemaker/ai/experiments/)
* [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
* [Nebius](https://nebius.com/services/managed-mlflow)
* [GCP (GKE)](https://gke-ai-labs.dev/docs/tutorials/frameworks-and-pipelines/mlflow/)

## Architecture[​](#architecture "Direct link to Architecture")

MLflow, at a high level, consists of the following components:

1. **Tracking Server**: The lightweight FastAPI server that serves the MLflow UI and API.
2. **Backend Store**: The Backend Store is relational database (or file system) that stores the metadata of the experiments, runs, traces, etc.
3. **Artifact Store**: The Artifact Store is responsible for storing the large artifacts such as model weights, images, etc.

Each component is designed to be pluggable, so you can customize it to meet your needs. For example, you can start with a single host mode with SQLite backend and local file system for storing artifacts. To scale up, you can switch backend store to PostgreSQL cluster and point artifact store to cloud storage such as S3, GCS, or Azure Blob Storage.

To learn more about the architecture and available backend options, see [Architecture](/mlflow-website/docs/latest/self-hosting/architecture/overview.md).

## Access Control & Security[​](#access-control--security "Direct link to Access Control & Security")

MLflow support [username/password login](/mlflow-website/docs/latest/self-hosting/security/basic-http-auth.md) via basic HTTP authentication, [SSO (Single Sign-On)](/mlflow-website/docs/latest/self-hosting/security/sso.md), and [custom authentication plugins](/mlflow-website/docs/latest/self-hosting/security/custom.md).

MLflow also provides built-in [network protection](/mlflow-website/docs/latest/self-hosting/security/network.md) middleware to protect your tracking server from network exposure.

Try Managed MLflow

Need highly secure MLflow server? Check out [Databricks Managed MLflow](https://www.databricks.com/product/managed-mlflow) to get fully managed MLflow servers with unified governance and security.

## FAQs[​](#faqs "Direct link to FAQs")

See [Troubleshooting & FAQs](/mlflow-website/docs/latest/self-hosting/troubleshooting.md) for more information.
