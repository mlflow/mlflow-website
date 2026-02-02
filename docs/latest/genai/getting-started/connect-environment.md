# Set Up MLflow Server

MLflow Assistant

Need help with this setup? Try [MLflow Assistant](/mlflow-website/docs/latest/genai/getting-started/try-assistant.md) - a powerful AI assistant that understands your codebase and can set up MLflow for you.

MLflow is open source, and you can set up the MLflow server using either `pip` or `docker`.

Before you can leverage MLflow for your GenAI application development, you must first start the MLflow server.

* Local (uv)
* Local (pip)
* Local (docker)

Install the Python package manager [uv](https://docs.astral.sh/uv/getting-started/installation/) (that will also install [`uvx` command](https://docs.astral.sh/uv/guides/tools/) to invoke Python tools without installing them).

Start a MLflow server locally.

shell

```shell
uvx mlflow server

```

**Python Environment**: Python 3.10+

Install the `mlflow` Python package via `pip` and start a MLflow server locally.

shell

```shell
pip install --upgrade 'mlflow[genai]'
mlflow server

```

MLflow provides a Docker Compose file to start a local MLflow server with a PostgreSQL database and a MinIO server.

shell

```shell
git clone --depth 1 --filter=blob:none --sparse https://github.com/mlflow/mlflow.git
cd mlflow
git sparse-checkout set docker-compose
cd docker-compose
cp .env.dev.example .env
docker compose up -d

```

Refer to the [instruction](https://github.com/mlflow/mlflow/tree/master/docker-compose/README.md) for more details (e.g., overriding the default environment variables).

This will start the server at port 5000 on your local machine and you can access the MLflow web UI at <http://localhost:5000>.

![MLflow UI Home](/mlflow-website/docs/latest/images/quickstart/quickstart_ui_home.png)

If you are looking for more guidance about self-hosting the MLflow server, please see the [Self-Hosting Guide](/mlflow-website/docs/latest/self-hosting.md) for more details.

info

If you are using MLflow on Databricks, please visit [this](https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/) for environment setup instructions specific to Databricks.

## Next Step[​](#next-step "Direct link to Next Step")

Now that you have started the MLflow server, let's start tracing your GenAI application.

Follow [this quickstart](/mlflow-website/docs/latest/genai/tracing/quickstart.md) to send your GenAI application traces to the MLflow server.
