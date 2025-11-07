# Deploy MLflow Model to Amazon SageMaker

Amazon SageMaker is a fully managed service designed for scaling ML inference containers. MLflow simplifies the deployment process by offering easy-to-use commands without the need for writing container definitions.

If you are new to MLflow model deployment, please read [MLflow Deployment](/mlflow-website/docs/latest/ml/deployment.md) first to understand the basic concepts of MLflow models and deployments.

## How it works[​](#how-it-works "Direct link to How it works")

SageMaker features a capability called [Bring Your Own Container (BYOC)](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-containers.html), which allows you to run custom Docker containers on the inference endpoint. These containers must meet specific requirements, such as running a web server that exposes certain REST endpoints, having a designated container entrypoint, setting environment variables, etc. Writing a Dockerfile and serving script that meets these requirements can be a tedious task.

MLflow automates the process by building a Docker image from the MLflow Model on your behalf. Subsequently, it pushed the image to Elastic Container Registry (ECR) and creates a SageMaker endpoint using this image. It also uploads the model artifact to an S3 bucket and configures the endpoint to download the model from there.

The container provides the same REST endpoints as a local inference server. For instance, the `/invocations` endpoint accepts CSV and JSON input data and returns prediction results. For more details on the endpoints, refer to [Local Inference Server](/mlflow-website/docs/latest/ml/deployment/deploy-model-locally.md#local-inference-server-spec).

## Deploying Model to SageMaker Endpoint[​](#deploying-model-to-sagemaker-endpoint "Direct link to Deploying Model to SageMaker Endpoint")

This section outlines the process of deploying a model to SageMaker using the MLflow CLI. For Python API references and tutorials, see the [Useful links](#deployment-sagemaker-references) section.

### Step 0: Preparation[​](#step-0-preparation "Direct link to Step 0: Preparation")

#### Install Tools[​](#install-tools "Direct link to Install Tools")

Ensure the installation of the following tools if not already done:

* [mlflow](https://pypi.org/project/mlflow)
* [awscli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
* [docker](https://docs.docker.com/get-docker)

#### Permissions Setup[​](#permissions-setup "Direct link to Permissions Setup")

Set up AWS accounts and permissions correctly. You need an IAM role with permissions to create a SageMaker endpoint, access an S3 bucket, and use the ECR repository. This role should also be assumable by the user performing the deployment. Learn more about this setup at [Use an IAM role in the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html).

#### Create an MLflow Model[​](#create-an-mlflow-model "Direct link to Create an MLflow Model")

Before deploying, you must have an MLflow Model. If you don't have one, you can create a sample scikit-learn model by following the [MLflow Tracking Quickstart](/mlflow-website/docs/latest/ml/getting-started.md). Remember to note down the model URI, such as `models:/<model_id>` (or `models:/<model_name>/<model_version>` if you registered the model in the [MLflow Model Registry](/mlflow-website/docs/latest/ml/model-registry.md)).

### Step 1: Test your model locally[​](#step-1-test-your-model-locally "Direct link to Step 1: Test your model locally")

It's recommended to test your model locally before deploying it to a production environment. The [`mlflow deployments run-local`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.sagemaker.html#mlflow.sagemaker.run_local) command deploys the model in a Docker container with an identical image and environment configuration, making it ideal for pre-deployment testing.

bash

```
mlflow deployments run-local -t sagemaker -m models:/<model_id> -p 5000
```

You can then test the model by sending a POST request to the endpoint:

bash

```
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["a","b"],"data":[[1,2]]}' http://localhost:5000/invocations
```

### Step 2: Build a Docker Image and Push to ECR[​](#step-2-build-a-docker-image-and-push-to-ecr "Direct link to Step 2: Build a Docker Image and Push to ECR")

The [mlflow sagemaker build-and-push-container](/mlflow-website/docs/latest/api_reference/cli.html#mlflow-sagemaker-build-and-push-container) command builds a Docker image compatible with SageMaker and uploads it to ECR.

bash

```
$ mlflow sagemaker build-and-push-container  -m models:/<model_id>
```

Alternatively, you can create a custom Docker image using the [official MLflow Docker image](/mlflow-website/docs/latest/ml/docker.md) and manually push it to ECR.

### Step 3: Deploy to SageMaker Endpoint[​](#step-3-deploy-to-sagemaker-endpoint "Direct link to Step 3: Deploy to SageMaker Endpoint")

The [`mlflow deployments create`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.sagemaker.html#mlflow.sagemaker.SageMakerDeploymentClient.create_deployment) command deploys the model to an Amazon SageMaker endpoint. MLflow uploads the Python Function model to S3 and automatically initiates an Amazon SageMaker endpoint serving the model.

Various command-line options are available to customize the deployment, such as instance type, count, IAM role, etc. Refer to the [CLI reference](/mlflow-website/docs/latest/api_reference/cli.html#mlflow-sagemaker) for a complete list of options.

bash

```
$ mlflow deployments create -t sagemaker -m runs:/<run_id>/model \
    -C region_name=<your-region> \
    -C instance-type=ml.m4.xlarge \
    -C instance-count=1 \
    -C env='{"DISABLE_NGINX": "true"}''
```

## API Reference[​](#api-reference "Direct link to API Reference")

You have two options for deploying a model to SageMaker: using the CLI or the Python API.

* [CLI Reference](/mlflow-website/docs/latest/api_reference/cli.html#mlflow-sagemaker)
* [Python API Documentation](/mlflow-website/docs/latest/api_reference/python_api/mlflow.sagemaker.html#mlflow.sagemaker)

## Useful Links[​](#deployment-sagemaker-references "Direct link to Useful Links")

* [MLflow Quickstart Part 2: Serving Models Using Amazon SageMaker](https://docs.databricks.com/en/_extras/notebooks/source/mlflow/mlflow-quick-start-deployment-aws.html) - This step-by-step tutorial demonstrates how to deploy a model to SageMaker using MLflow Python APIs from a Databricks notebook.
* [Managing Your Machine Learning Lifecycle with MLflow and Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/managing-your-machine-learning-lifecycle-with-mlflow-and-amazon-sagemaker) - This comprehensive tutorial covers integrating the entire MLflow lifecycle with SageMaker, from model training to deployment.
