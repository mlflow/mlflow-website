# Connect Your Development Environment to MLflow

This guide shows you how to connect your development environment to an MLflow Experiment. You can run MLflow on your local machine, self-host the open source MLflow service, or use a managed offering, such as Databricks Managed MLflow.

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

* OSS MLflow
* Databricks

- **Python Environment**: Python 3.9+ with pip installed

* **Databricks Workspace**: Access to a Databricks workspace

Authentication Methods

This guide describes using a Databricks Personal Access Token. MLflow also works with the other [Databricks-supported authentication methods](https://docs.databricks.com/aws/en/dev-tools/auth).

## Setup Instructions[​](#setup-instructions "Direct link to Setup Instructions")

* OSS MLflow
* Databricks - Local IDE
* Databricks - Notebook

#### Step 1: Install MLflow[​](#step-1-install-mlflow "Direct link to Step 1: Install MLflow")

bash

```bash
pip install --upgrade "mlflow>=3.1"

```

<br />

#### Step 2: Configure Tracking[​](#step-2-configure-tracking "Direct link to Step 2: Configure Tracking")

MLflow supports different backends for tracking your experiment data. Choose one of the following options to get started. Refer to the [Self Hosting Guide](/mlflow-website/docs/latest/self-hosting.md) for detailed setup and configurations.

**Option A: Database (Recommended)**

Set the tracking URI to a local database URI (e.g., `sqlite:///mlflow.db`). This is recommended option for quickstart and local development.

python

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my-genai-experiment")

```

**Option B: File System**

MLflow will automatically use local file storage if no tracking URI is specified:

python

```python
import mlflow

# Creates local mlruns directory for experiments
mlflow.set_experiment("my-genai-experiment")

```

TO BE DEPRECATED SOON

File system backend is in Keep-the-Light-On (KTLO) mode and will not receive most of the new features in MLflow. We recommend using the database backend instead. Database backend will also be the default option soon.

**Option C: Remote Tracking Server**

Start a remote MLflow tracking server following the [Self Hosting Guide](/mlflow-website/docs/latest/self-hosting.md). Then configure your client to use the remote server:

python

```python
import mlflow

# Connect to remote MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-genai-experiment")

```

Alternatively, you can configure the tracking URI and experiment using environment variables:

bash

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="my-genai-experiment"

```

<br />

#### Step 3: Verify Your Connection[​](#step-3-verify-your-connection "Direct link to Step 3: Verify Your Connection")

Create a test file and run this code:

python

```python
import mlflow

# Print connection information
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active Experiment: {mlflow.get_experiment_by_name('my-genai-experiment')}")

# Test logging
with mlflow.start_run():
    mlflow.log_param("test_param", "test_value")
    print("✓ Successfully connected to MLflow!")

```

<br />

#### Step 4: Access MLflow UI[​](#step-4-access-mlflow-ui "Direct link to Step 4: Access MLflow UI")

If you are using local tracking (option A or B), run the following command and access the MLflow UI at `http://localhost:5000`.

bash

```bash
# For Option A
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# For Option B
mlflow ui --port 5000

```

If you have the remote tracking server running (option C), access the MLflow UI at the same URI.

ACCESS DENIED?

When using the remote tracking server, you may hit an access denied error when accessing the MLflow UI from a browser.

> Invalid Host header - possible DNS rebinding attack detected

This error typically indicates that the tracking server's network security settings need to be configured. The most common causes are:

* **Host validation**: The `--allowed-hosts` flag restricts which Host headers are accepted
* **CORS restrictions**: The `--cors-allowed-origins` flag controls which origins can make API requests

To resolve this, configure your tracking server with the appropriate flags. For example:

bash

```bash
mlflow server --allowed-hosts "mlflow.company.com,localhost:*" \
              --cors-allowed-origins "https://app.company.com"

```

**Note**: These security options are only available with the default FastAPI-based server (uvicorn). They are not supported when using Flask directly or with `--gunicorn-opts` or `--waitress-opts`.

Refer to the [Network Security Guide](/mlflow-website/docs/latest/self-hosting/security/network.md) for detailed configuration options.

#### Step 1: Install MLflow[​](#step-1-install-mlflow-1 "Direct link to Step 1: Install MLflow")

Install MLflow with Databricks connectivity:

bash

```bash
pip install --upgrade "mlflow[databricks]>=3.1"

```

<br />

#### Step 2: Create an MLflow Experiment[​](#step-2-create-an-mlflow-experiment "Direct link to Step 2: Create an MLflow Experiment")

1. Open your Databricks workspace
2. Go to **Experiments** in the left sidebar under **Machine Learning**
3. At the top of the Experiments page, click on **New GenAI Experiment**

<br />

#### Step 3: Configure Authentication[​](#step-3-configure-authentication "Direct link to Step 3: Configure Authentication")

Choose one of the following authentication methods:

**Option A: Environment Variables**

1. In your MLflow Experiment, click **Generate API Key**
2. Copy and run the generated code in your terminal:

bash

```bash
export DATABRICKS_TOKEN=<databricks-personal-access-token>
export DATABRICKS_HOST=https://<workspace-name>.cloud.databricks.com
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_EXPERIMENT_ID=<experiment-id>

```

**Option B: .env File**

1. In your MLflow Experiment, click **Generate API Key**
2. Copy the generated code to a `.env` file in your project root:

bash

```bash
DATABRICKS_TOKEN=<databricks-personal-access-token>
DATABRICKS_HOST=https://<workspace-name>.cloud.databricks.com
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT_ID=<experiment-id>

```

3. Install the `python-dotenv` package:

bash

```bash
pip install python-dotenv

```

4. Load environment variables in your code:

python

```python
# At the beginning of your Python script
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

```

#### Step 4: Verify Your Connection[​](#step-4-verify-your-connection "Direct link to Step 4: Verify Your Connection")

Create a test file and run this code to verify your connection:

python

```python
import mlflow

# Test logging to verify connection
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
with mlflow.start_run():
    print("✓ Successfully connected to MLflow!")

```

#### Step 1: Install MLflow[​](#step-1-install-mlflow-2 "Direct link to Step 1: Install MLflow")

Databricks runtimes include MLflow, but for the best experience with GenAI capabilities, update to the latest version:

bash

```bash
%pip install --upgrade "mlflow[databricks]>=3.1"
dbutils.library.restartPython()

```

<br />

#### Step 2: Create a Notebook[​](#step-2-create-a-notebook "Direct link to Step 2: Create a Notebook")

Creating a Databricks Notebook will create an MLflow Experiment that is the container for your GenAI application. Learn more about Experiments in the [MLflow documentation](/mlflow-website/docs/latest/ml/tracking.md).

1. Open your Databricks workspace
2. Go to **New** at the top of the left sidebar
3. Click **Notebook**

<br />

#### Step 3: Configure Authentication[​](#step-3-configure-authentication-1 "Direct link to Step 3: Configure Authentication")

No additional authentication configuration is needed when working within a Databricks Notebook. The notebook automatically has access to your workspace and the associated MLflow Experiment.

<br />

#### Step 4: Verify Your Connection[​](#step-4-verify-your-connection-1 "Direct link to Step 4: Verify Your Connection")

Run this code in a notebook cell to verify your connection:

python

```python
import mlflow

# Test logging to verify connection
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
with mlflow.start_run():
    print("✓ Successfully connected to MLflow!")

```

## Next Steps[​](#next-steps "Direct link to Next Steps")

Now that your environment is connected to MLflow, try the other GenAI quickstarts:

* **Instrument your app with tracing**: Follow the [quickstart](/mlflow-website/docs/latest/genai/tracing/quickstart/python-openai.md) to instrument your first GenAI app
* **Evaluate your app's quality**: Use the [evaluation quickstart](https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/eval.html) to systematically test and improve your app's quality
