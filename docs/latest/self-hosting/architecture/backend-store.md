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

### Relational Database (**Recommended**)[​](#relational-database-recommended "Direct link to relational-database-recommended")

MLflow supports different databases through SQLAlchemy, including `sqlite`, `postgresql`, `mysql`, and `mssql`. This option provides better performance through indexing and is easier to scale to larger volumes of data than the file system backend..

SQLite is the easiest way to use to database backend. To use it, simply add `--backend-store-uri sqlite:///my.db` when starting MLflow. A database file will be created for you and it will be used to store your tracking data.

### Local File System (**Deprecated Soon**)[​](#local-file-system-deprecated-soon "Direct link to local-file-system-deprecated-soon")

By default, MLflow stores metadata in local files in the `./mlruns` directory. This is for the pure sake of simplicity. For better performance and reliability, we always recommend using a database backend.

TO BE DEPRECATED SOON

File system backend is in Keep-the-Light-On (KTLO) mode and will not receive most of the new features in MLflow. We recommend using the database backend instead. Database backend will also be the default option soon.

## Configure Backend Store[​](#configure-backend-store "Direct link to Configure Backend Store")

By default, MLflow stores metadata in local files in the `./mlruns` directory, but MLflow can store metadata to databases as well. You can configure the location by passing the desired **tracking URI** to MLflow, via either of the following methods:

* Set the `MLFLOW_TRACKING_URI` environment variable.
* Call [`mlflow.set_tracking_uri()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.set_tracking_uri) in your code.
* If you are running a [Tracking Server](/mlflow-website/docs/latest/self-hosting/architecture/tracking-server.md), you can set the `tracking_uri` option when starting the server, like `mlflow server --backend-store-uri sqlite:///mydb.sqlite`

Continue to the next section for the supported format of tracking URLs. Also visit [this guidance](/mlflow-website/docs/latest/self-hosting/architecture/tracking-server.md) for how to set up the backend store properly for your workflow.

important

The default file backend works fine for small use cases where you only want to log less than 1000 runs, metrics, and traces. Otherwise, we **highly recommend using a database backend** for better performance and reliability.

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
