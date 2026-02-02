# Backend Stores

The backend store is a core component in MLflow that stores metadata for Runs, models, traces, and experiments such as:

* Run ID
* Model ID
* Trace ID
* Tags
* Start & end time
* Parameters
* Metrics

Large model artifacts such as model weight files are stored in the [artifact store](/mlflow-website/docs/latest/self-hosting/architecture/artifact-store.md).

## Types of Backend Stores[​](#types-of-backend-stores "Direct link to Types of Backend Stores")

### Relational Database (**Default**)[​](#relational-database-default "Direct link to relational-database-default")

MLflow supports different databases through SQLAlchemy, including `sqlite`, `postgresql`, `mysql`, and `mssql`. This option provides better performance through indexing and is easier to scale to larger volumes of data than the file system backend.

**SQLite is the default backend store**. When you start MLflow without specifying a backend, it automatically creates and uses `sqlite:///mlflow.db` in the current directory. To use a different database such as PostgreSQL, specify `--backend-store-uri` when starting MLflow (e.g., `--backend-store-uri postgresql://...`).

### Local File System (**Legacy**)[​](#local-file-system-legacy "Direct link to local-file-system-legacy")

The file-based backend stores metadata in local files in the `./mlruns` directory. This was the default backend in earlier versions of MLflow, but is still supported for backward compatibility.

To use file-based storage, specify `--backend-store-uri ./mlruns` when starting the server, or set `MLFLOW_TRACKING_URI=./mlruns`.

TO BE DEPRECATED SOON

File system backend is in Keep-the-Light-On (KTLO) mode and is no longer receiving new feature updates. We strongly recommend using the database backend (now the default) for better performance and reliability.

## Configure Backend Store[​](#configure-backend-store "Direct link to Configure Backend Store")

You can configure a different backend store by passing the desired **tracking URI** to MLflow, via either of the following methods:

* Set the `MLFLOW_TRACKING_URI` environment variable.
* Call [`mlflow.set_tracking_uri()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.set_tracking_uri) in your code.
* If you are running a [Tracking Server](/mlflow-website/docs/latest/self-hosting/architecture/tracking-server.md), you can set the `--backend-store-uri` option when starting the server, like `mlflow server --backend-store-uri postgresql://...`

Continue to the next section for the supported format of tracking URLs. Also visit [this guidance](/mlflow-website/docs/latest/self-hosting/architecture/tracking-server.md) for how to set up the backend store properly for your workflow.

## Supported Store Types[​](#supported-store-types "Direct link to Supported Store Types")

MLflow supports the following types of tracking URI for backend stores:

* Local file path (specified as `file:/my/local/dir`), where data is just directly stored locally to a system disk where your code is executing.
* A Database, encoded as `<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>`. MLflow supports the dialects `mysql`, `mssql`, `sqlite`, and `postgresql`. For more details, see [SQLAlchemy database uri](https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls).
* HTTP server (specified as `https://my-server:5000`), which is a server hosting an [MLflow tracking server](/mlflow-website/docs/latest/self-hosting/architecture/tracking-server.md).
* Databricks workspace (specified as `databricks` or as `databricks://<profileName>`, a [Databricks CLI profile](https://github.com/databricks/databricks-cli#installation)). Refer to Access the MLflow tracking server from outside Databricks [\[AWS\]](http://docs.databricks.com/applications/mlflow/access-hosted-tracking-server.html) [\[Azure\]](http://docs.microsoft.com/azure/databricks/applications/mlflow/access-hosted-tracking-server).

database-requirements

**Database-Backed Store Requirements**

When using database-backed stores, please note:

* **Model Registry Integration**: [Model Registry](/mlflow-website/docs/latest/ml/model-registry.md) functionality requires a database-backed store. See [this FAQ](/mlflow-website/docs/latest/ml/tracking.md#tracking-with-model-registry) for more information.

* **Schema Migrations**: `mlflow server` will fail against a database with an out-of-date schema. Always run `mlflow db upgrade [db_uri]` to upgrade your database schema before starting the server. Schema migrations can result in database downtime and may take longer on larger databases. **Always backup your database before running migrations.**

parameter-limits

In Sep 2023, we increased the max length for params recorded in a Run from 500 to 8k (but we limit param value max length to 6000 internally). [mlflow/2d6e25af4d3e\_increase\_max\_param\_val\_length](https://github.com/mlflow/mlflow/blob/master/mlflow/store/db_migrations/versions/2d6e25af4d3e_increase_max_param_val_length.py) is a non-invertible migration script that increases the cap in existing database to 8k. Please be careful if you want to upgrade and backup your database before upgrading.

## Deletion Behavior[​](#deletion-behavior "Direct link to Deletion Behavior")

In order to allow MLflow Runs to be restored, Run metadata and artifacts are not automatically removed from the backend store or artifact store when a Run is deleted. The [mlflow gc](/mlflow-website/docs/latest/api_reference/cli.html#mlflow-gc) CLI is provided for permanently removing Run metadata and artifacts for deleted runs.

## SQLAlchemy Options[​](#sqlalchemy-options "Direct link to SQLAlchemy Options")

You can inject some [SQLAlchemy connection pooling options](https://docs.sqlalchemy.org/en/latest/core/pooling.html) using environment variables.

| MLflow Environment Variable           | SQLAlchemy QueuePool Option |
| ------------------------------------- | --------------------------- |
| `MLFLOW_SQLALCHEMYSTORE_POOL_SIZE`    | `pool_size`                 |
| `MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE` | `pool_recycle`              |
| `MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW` | `max_overflow`              |

## MySQL SSL Options[​](#mysql-ssl-options "Direct link to MySQL SSL Options")

When connecting to a MySQL database that requires SSL certificates, you can set the following environment variables:

bash

```bash
# Path to SSL CA certificate file
export MLFLOW_MYSQL_SSL_CA=/path/to/ca.pem

# Path to SSL client certificate file (if needed)
export MLFLOW_MYSQL_SSL_CERT=/path/to/client-cert.pem

# Path to SSL client key file (if needed)
export MLFLOW_MYSQL_SSL_KEY=/path/to/client-key.pem

```

Then start the MLflow server with your MySQL URI:

bash

```bash
mlflow server --backend-store-uri="mysql+pymysql://username@hostname:port/database" --default-artifact-root=s3://your-bucket --host=0.0.0.0 --port=5000

```

These environment variables will be used to configure the SSL connection to the MySQL server.

## File Store Performance[​](#file-store-performance "Direct link to File Store Performance")

MLflow will automatically try to use [LibYAML](https://pyyaml.org/wiki/LibYAML) bindings if they are already installed. However, if you notice any performance issues when using *file store* backend, it could mean LibYAML is not installed on your system. On Linux or Mac you can easily install it using your system package manager:

bash

```bash
# On Ubuntu/Debian
apt-get install libyaml-cpp-dev libyaml-dev

# On macOS using Homebrew
brew install yaml-cpp libyaml

```

After installing LibYAML, you need to reinstall PyYAML:

bash

```bash
# Reinstall PyYAML
pip --no-cache-dir install --force-reinstall -I pyyaml

```

note

We generally recommend using a database backend to get better performance.
