# Authentication with Username and Password

MLflow supports basic HTTP authentication to enable access control over experiments, registered models, and scorers. Once enabled, any visitor will be required to login before they can view any resource from the Tracking Server.

MLflow Authentication provides Python and REST API for managing users and permissions.

* [MLflow Authentication Python API](/mlflow-website/docs/latest/api_reference/auth/python-api.html#mlflow.server.auth)
* [MLflow Authentication REST API](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/)

![](data:image/png;base64,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)

## Overview[​](#overview "Direct link to Overview")

First, install all dependencies required for the basic auth app:

bash

```bash
pip install mlflow[auth]

```

note

The basic auth app requires a secret key for CSRF protection. Please set the `MLFLOW_FLASK_SERVER_SECRET_KEY` environment variable before running the `mlflow server` command. For example:

text

```text
export MLFLOW_FLASK_SERVER_SECRET_KEY="my-secret-key"

```

If your setup uses multiple servers, please make sure that this key is consistent between them. Otherwise, you may run into unexpected validation errors.

To enable MLflow authentication, launch the MLflow UI with the following command:

bash

```bash
mlflow server --app-name basic-auth

```

Server admin can choose to disable this feature anytime by restarting the server without the `app-name` flag. Any users and permissions created will be persisted on a SQL database and will be back in service once the feature is re-enabled.

Due to the nature of HTTP authentication, it is only supported on a remote Tracking Server, where users send requests to the server via REST APIs.

## How It Works[​](#how-it-works "Direct link to How It Works")

### Permissions[​](#permissions "Direct link to Permissions")

The available permissions are:

| Permission      | Can read | Can use | Can update | Can delete | Can manage |
| --------------- | -------- | ------- | ---------- | ---------- | ---------- |
| READ            | Yes      | No      | No         | No         | No         |
| USE             | Yes      | Yes     | No         | No         | No         |
| EDIT            | Yes      | Yes     | Yes        | No         | No         |
| MANAGE          | Yes      | Yes     | Yes        | Yes        | Yes        |
| NO\_PERMISSIONS | No       | No      | No         | No         | No         |

The default permission for all users is `READ`. It can be changed in the [configuration](#configuration) file.

Permissions can be granted on individual resources for each user. Supported resources include `Experiment`, `Registered Model`, and `Scorer`. To access an API endpoint, an user must have the required permission. Otherwise, a `403 Forbidden` response will be returned.

Required Permissions for accessing experiments:

| API                                                                                                                       | Endpoint                                    | Method | Required permission |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------ | ------------------- |
| [Create Experiment](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicecreateexperiment)         | `2.0/mlflow/experiments/create`             | `POST` | None                |
| [Get Experiment](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicegetexperiment)               | `2.0/mlflow/experiments/get`                | `GET`  | can\_read           |
| [Get Experiment By Name](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicegetexperimentbyname) | `2.0/mlflow/experiments/get-by-name`        | `GET`  | can\_read           |
| [Delete Experiment](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicedeleteexperiment)         | `2.0/mlflow/experiments/delete`             | `POST` | can\_delete         |
| [Restore Experiment](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicerestoreexperiment)       | `2.0/mlflow/experiments/restore`            | `POST` | can\_delete         |
| [Update Experiment](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowserviceupdateexperiment)         | `2.0/mlflow/experiments/update`             | `POST` | can\_update         |
| [Search Experiments](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicesearchexperiments)       | `2.0/mlflow/experiments/search`             | `POST` | None                |
| [Search Experiments](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicesearchexperiments)       | `2.0/mlflow/experiments/search`             | `GET`  | None                |
| [Set Experiment Tag](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicesetexperimenttag)        | `2.0/mlflow/experiments/set-experiment-tag` | `POST` | can\_update         |
| [Create Run](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicecreaterun)                       | `2.0/mlflow/runs/create`                    | `POST` | can\_update         |
| [Get Run](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicegetrun)                             | `2.0/mlflow/runs/get`                       | `GET`  | can\_read           |
| [Update Run](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowserviceupdaterun)                       | `2.0/mlflow/runs/update`                    | `POST` | can\_update         |
| [Delete Run](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicedeleterun)                       | `2.0/mlflow/runs/delete`                    | `POST` | can\_delete         |
| [Restore Run](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicerestorerun)                     | `2.0/mlflow/runs/restore`                   | `POST` | can\_delete         |
| [Search Runs](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicesearchruns)                     | `2.0/mlflow/runs/search`                    | `POST` | None                |
| [Set Tag](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicesettag)                             | `2.0/mlflow/runs/set-tag`                   | `POST` | can\_update         |
| [Delete Tag](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicedeletetag)                       | `2.0/mlflow/runs/delete-tag`                | `POST` | can\_update         |
| [Log Metric](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicelogmetric)                       | `2.0/mlflow/runs/log-metric`                | `POST` | can\_update         |
| [Log Param](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicelogparam)                         | `2.0/mlflow/runs/log-parameter`             | `POST` | can\_update         |
| [Log Batch](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicelogbatch)                         | `2.0/mlflow/runs/log-batch`                 | `POST` | can\_update         |
| [Log Model](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicelogmodel)                         | `2.0/mlflow/runs/log-model`                 | `POST` | can\_update         |
| [List Artifacts](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicelistartifacts)               | `2.0/mlflow/artifacts/list`                 | `GET`  | can\_read           |
| [Get Metric History](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicegetmetrichistory)        | `2.0/mlflow/metrics/get-history`            | `GET`  | can\_read           |

Required Permissions for accessing prompt optimization jobs:

note

Prompt optimization jobs inherit permissions from their parent experiment. When you create a job within an experiment, the job's permissions are determined by your permissions on that experiment.

| API                             | Endpoint                                              | Method   | Required permission         |
| ------------------------------- | ----------------------------------------------------- | -------- | --------------------------- |
| Create Prompt Optimization Job  | `3.0/mlflow/prompt-optimization/jobs`                 | `POST`   | can\_update (on experiment) |
| Get Prompt Optimization Job     | `3.0/mlflow/prompt-optimization/jobs/{job_id}`        | `GET`    | can\_read                   |
| Search Prompt Optimization Jobs | `3.0/mlflow/prompt-optimization/jobs/search`          | `POST`   | can\_read (on experiment)   |
| Search Prompt Optimization Jobs | `3.0/mlflow/prompt-optimization/jobs/search`          | `GET`    | can\_read (on experiment)   |
| Cancel Prompt Optimization Job  | `3.0/mlflow/prompt-optimization/jobs/{job_id}/cancel` | `POST`   | can\_update                 |
| Delete Prompt Optimization Job  | `3.0/mlflow/prompt-optimization/jobs/{job_id}`        | `DELETE` | can\_delete                 |

Required Permissions for accessing registered models:

| API                                                                                                                                              | Endpoint                                           | Method   | Required permission |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------- | -------- | ------------------- |
| [Create Registered Model](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicecreateregisteredmodel)              | `2.0/mlflow/registered-models/create`              | `POST`   | None                |
| [Rename Registered Model](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicerenameregisteredmodel)              | `2.0/mlflow/registered-models/rename`              | `POST`   | can\_update         |
| [Update Registered Model](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryserviceupdateregisteredmodel)              | `2.0/mlflow/registered-models/update`              | `PATCH`  | can\_update         |
| [Delete Registered Model](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicedeleteregisteredmodel)              | `2.0/mlflow/registered-models/delete`              | `DELETE` | can\_delete         |
| [Get Registered Model](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicegetregisteredmodel)                    | `2.0/mlflow/registered-models/get`                 | `GET`    | can\_read           |
| [Search Registered Models](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicesearchregisteredmodels)            | `2.0/mlflow/registered-models/search`              | `GET`    | None                |
| [Get Latest Versions](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicegetlatestversions)                      | `2.0/mlflow/registered-models/get-latest-versions` | `POST`   | can\_read           |
| [Get Latest Versions](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicegetlatestversions)                      | `2.0/mlflow/registered-models/get-latest-versions` | `GET`    | can\_read           |
| [Set Registered Model Tag](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicesetregisteredmodeltag)             | `2.0/mlflow/registered-models/set-tag`             | `POST`   | can\_update         |
| [Delete Registered Model Tag](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicedeleteregisteredmodeltag)       | `2.0/mlflow/registered-models/delete-tag`          | `DELETE` | can\_update         |
| [Set Registered Model Alias](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicesetregisteredmodelalias)         | `2.0/mlflow/registered-models/alias`               | `POST`   | can\_update         |
| [Delete Registered Model Alias](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicedeleteregisteredmodelalias)   | `2.0/mlflow/registered-models/alias`               | `DELETE` | can\_delete         |
| [Get Model Version By Alias](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicegetmodelversionbyalias)          | `2.0/mlflow/registered-models/alias`               | `GET`    | can\_read           |
| [Create Model Version](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicecreatemodelversion)                    | `2.0/mlflow/model-versions/create`                 | `POST`   | can\_update         |
| [Update Model Version](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryserviceupdatemodelversion)                    | `2.0/mlflow/model-versions/update`                 | `PATCH`  | can\_update         |
| [Transition Model Version Stage](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicetransitionmodelversionstage) | `2.0/mlflow/model-versions/transition-stage`       | `POST`   | can\_update         |
| [Delete Model Version](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicedeletemodelversion)                    | `2.0/mlflow/model-versions/delete`                 | `DELETE` | can\_delete         |
| [Get Model Version](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicegetmodelversion)                          | `2.0/mlflow/model-versions/get`                    | `GET`    | can\_read           |
| [Search Model Versions](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicesearchmodelversions)                  | `2.0/mlflow/model-versions/search`                 | `GET`    | None                |
| [Get Model Version Download Uri](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicegetmodelversiondownloaduri)  | `2.0/mlflow/model-versions/get-download-uri`       | `GET`    | can\_read           |
| [Set Model Version Tag](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicesetmodelversiontag)                   | `2.0/mlflow/model-versions/set-tag`                | `POST`   | can\_update         |
| [Delete Model Version Tag](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicedeletemodelversiontag)             | `2.0/mlflow/model-versions/delete-tag`             | `DELETE` | can\_delete         |

### AI Gateway Permissions[​](#ai-gateway-permissions "Direct link to AI Gateway Permissions")

AI Gateway resources (API keys, model definitions, and endpoints) support the same permission model as experiments and registered models. When a user creates a resource, they automatically receive `MANAGE` permission on it and can grant permissions to other users.

| API                                                                                                                                 | Endpoint                                      | Method   | Required permission                             |
| ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | -------- | ----------------------------------------------- |
| [List API Keys](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicelistgatewaysecretinfos)                 | `3.0/mlflow/gateway/secrets/list`             | `GET`    | None                                            |
| [Get API Key](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicegetgatewaysecretinfo)                     | `3.0/mlflow/gateway/secrets/get`              | `GET`    | can\_read                                       |
| [Create API Key](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicecreategatewaysecret)                   | `3.0/mlflow/gateway/secrets/create`           | `POST`   | None                                            |
| [Update API Key](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowserviceupdategatewaysecret)                   | `3.0/mlflow/gateway/secrets/update`           | `PATCH`  | can\_update                                     |
| [Delete API Key](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicedeletegatewaysecret)                   | `3.0/mlflow/gateway/secrets/delete`           | `DELETE` | can\_delete                                     |
| [List Model Definitions](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicelistgatewaymodeldefinitions)   | `3.0/mlflow/gateway/model-definitions/list`   | `GET`    | None                                            |
| [Get Model Definition](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicegetgatewaymodeldefinition)       | `3.0/mlflow/gateway/model-definitions/get`    | `GET`    | can\_read                                       |
| [Create Model Definition](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicecreategatewaymodeldefinition) | `3.0/mlflow/gateway/model-definitions/create` | `POST`   | None (can\_use on referenced API key)           |
| [Update Model Definition](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowserviceupdategatewaymodeldefinition) | `3.0/mlflow/gateway/model-definitions/update` | `PATCH`  | can\_update                                     |
| [Delete Model Definition](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicedeletegatewaymodeldefinition) | `3.0/mlflow/gateway/model-definitions/delete` | `DELETE` | can\_delete                                     |
| [List Endpoints](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicelistgatewayendpoints)                  | `3.0/mlflow/gateway/endpoints/list`           | `GET`    | None                                            |
| [Get Endpoint](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicegetgatewayendpoint)                      | `3.0/mlflow/gateway/endpoints/get`            | `GET`    | can\_read                                       |
| [Create Endpoint](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicecreategatewayendpoint)                | `3.0/mlflow/gateway/endpoints/create`         | `POST`   | None (can\_use on referenced model definitions) |
| [Update Endpoint](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowserviceupdategatewayendpoint)                | `3.0/mlflow/gateway/endpoints/update`         | `PATCH`  | can\_update                                     |
| [Delete Endpoint](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicedeletegatewayendpoint)                | `3.0/mlflow/gateway/endpoints/delete`         | `DELETE` | can\_delete                                     |
| Query Endpoint                                                                                                                      | `gateway/{endpoint_name}/...`                 | `POST`   | can\_use                                        |

note

When creating AI Gateway resources, the creator automatically receives `MANAGE` permission. To grant access to other users, use the permission management APIs listed below.

Required Permissions for accessing scorers:

| API                  | Endpoint                           | Method | Required permission         |
| -------------------- | ---------------------------------- | ------ | --------------------------- |
| Register Scorer      | `3.0/mlflow/scorers/register`      | `POST` | can\_update (on experiment) |
| List Scorers         | `3.0/mlflow/scorers/list`          | `GET`  | can\_read (on experiment)   |
| Get Scorer           | `3.0/mlflow/scorers/get`           | `GET`  | can\_read                   |
| Delete Scorer        | `3.0/mlflow/scorers/delete`        | `POST` | can\_delete                 |
| List Scorer Versions | `3.0/mlflow/scorers/list-versions` | `GET`  | can\_read                   |

MLflow Authentication provides API endpoints to manage users and permissions.

| API                                                                                                                                                    | Endpoint                                                  | Method   | Required permission         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- | -------- | --------------------------- |
| [Create User](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#create-user)                                                               | `2.0/mlflow/users/create`                                 | `POST`   | None                        |
| [Get User](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#get-user)                                                                     | `2.0/mlflow/users/get`                                    | `GET`    | Only readable by that user  |
| [Update User Password](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#update-user-password)                                             | `2.0/mlflow/users/update-password`                        | `PATCH`  | Only updatable by that user |
| [Update User Admin](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#update-user-admin)                                                   | `2.0/mlflow/users/update-admin`                           | `PATCH`  | Only admin                  |
| [Delete User](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#delete-user)                                                               | `2.0/mlflow/users/delete`                                 | `DELETE` | Only admin                  |
| [Create Experiment Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#create-experiment-permission)                             | `2.0/mlflow/experiments/permissions/create`               | `POST`   | can\_manage                 |
| [Get Experiment Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#get-experiment-permission)                                   | `2.0/mlflow/experiments/permissions/get`                  | `GET`    | can\_manage                 |
| [Update Experiment Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#update-experiment-permission)                             | `2.0/mlflow/experiments/permissions/update`               | `PATCH`  | can\_manage                 |
| [Delete Experiment Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#delete-experiment-permission)                             | `2.0/mlflow/experiments/permissions/delete`               | `DELETE` | can\_manage                 |
| [Create Registered Model Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#create-registered-model-permission)                 | `2.0/mlflow/registered-models/permissions/create`         | `POST`   | can\_manage                 |
| [Get Registered Model Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#get-registered-model-permission)                       | `2.0/mlflow/registered-models/permissions/get`            | `GET`    | can\_manage                 |
| [Update Registered Model Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#update-registered-model-permission)                 | `2.0/mlflow/registered-models/permissions/update`         | `PATCH`  | can\_manage                 |
| [Delete Registered Model Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#delete-registered-model-permission)                 | `2.0/mlflow/registered-models/permissions/delete`         | `DELETE` | can\_manage                 |
| [Create Gateway Endpoint Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#create-gateway-endpoint-permission)                 | `3.0/mlflow/gateway/endpoints/permissions/create`         | `POST`   | can\_manage                 |
| [Get Gateway Endpoint Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#get-gateway-endpoint-permission)                       | `3.0/mlflow/gateway/endpoints/permissions/get`            | `GET`    | can\_manage                 |
| [Update Gateway Endpoint Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#update-gateway-endpoint-permission)                 | `3.0/mlflow/gateway/endpoints/permissions/update`         | `PATCH`  | can\_manage                 |
| [Delete Gateway Endpoint Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#delete-gateway-endpoint-permission)                 | `3.0/mlflow/gateway/endpoints/permissions/delete`         | `DELETE` | can\_manage                 |
| [Create Gateway API Key Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#create-gateway-secret-permission)                    | `3.0/mlflow/gateway/secrets/permissions/create`           | `POST`   | can\_manage                 |
| [Get Gateway API Key Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#get-gateway-secret-permission)                          | `3.0/mlflow/gateway/secrets/permissions/get`              | `GET`    | can\_manage                 |
| [Update Gateway API Key Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#update-gateway-secret-permission)                    | `3.0/mlflow/gateway/secrets/permissions/update`           | `PATCH`  | can\_manage                 |
| [Delete Gateway API Key Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#delete-gateway-secret-permission)                    | `3.0/mlflow/gateway/secrets/permissions/delete`           | `DELETE` | can\_manage                 |
| [Create Gateway Model Definition Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#create-gateway-model-definition-permission) | `3.0/mlflow/gateway/model-definitions/permissions/create` | `POST`   | can\_manage                 |
| [Get Gateway Model Definition Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#get-gateway-model-definition-permission)       | `3.0/mlflow/gateway/model-definitions/permissions/get`    | `GET`    | can\_manage                 |
| [Update Gateway Model Definition Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#update-gateway-model-definition-permission) | `3.0/mlflow/gateway/model-definitions/permissions/update` | `PATCH`  | can\_manage                 |
| [Delete Gateway Model Definition Permission](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#delete-gateway-model-definition-permission) | `3.0/mlflow/gateway/model-definitions/permissions/delete` | `DELETE` | can\_manage                 |

Some APIs will also have their behaviour modified. For example, the creator of an experiment will automatically be granted `MANAGE` permission on that experiment, so that the creator can grant or revoke other users' access to that experiment.

| API                                                                                                                                   | Endpoint                              | Method | Effect                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ------ | ----------------------------------------------------------------------- |
| [Create Experiment](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicecreateexperiment)                     | `2.0/mlflow/experiments/create`       | `POST` | Automatically grants `MANAGE` permission to the creator.                |
| [Create Registered Model](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicecreateregisteredmodel)   | `2.0/mlflow/registered-models/create` | `POST` | Automatically grants `MANAGE` permission to the creator.                |
| [Search Experiments](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicesearchexperiments)                   | `2.0/mlflow/experiments/search`       | `POST` | Only returns experiments which the user has `READ` permission on.       |
| [Search Experiments](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicesearchexperiments)                   | `2.0/mlflow/experiments/search`       | `GET`  | Only returns experiments which the user has `READ` permission on.       |
| [Search Runs](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmlflowservicesearchruns)                                 | `2.0/mlflow`                          | `POST` | Only returns experiments which the user has `READ` permission on.       |
| [Search Registered Models](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicesearchregisteredmodels) | `2.0/mlflow/registered-models/search` | `GET`  | Only returns registered models which the user has `READ` permission on. |
| [Search Model Versions](/mlflow-website/docs/latest/api_reference/rest-api.html/#mlflowmodelregistryservicesearchmodelversions)       | `2.0/mlflow/model-versions/search`    | `GET`  | Only returns registered models which the user has `READ` permission on. |

### Permissions Database[​](#permissions-database "Direct link to Permissions Database")

All users and permissions are stored in a database in `basic_auth.db`, relative to the directory where MLflow server is launched. The location can be changed in the [configuration](#configuration) file. To run migrations, use the following command:

bash

```bash
python -m mlflow.server.auth db upgrade --url <database_url>

```

### Admin Users[​](#admin-users "Direct link to Admin Users")

Admin users have unrestricted access to all MLflow resources, **including creating or deleting users, updating password and admin status of other users, granting or revoking permissions from other users, and managing permissions for all MLflow resources,** even if `NO_PERMISSIONS` is explicitly set to that admin account.

MLflow has a built-in admin user that will be created the first time that the MLflow authentication feature is enabled.

note

It is recommended that you update the default admin password as soon as possible after creation.

The default admin user credentials are as follows:

| Username | Password       |
| -------- | -------------- |
| `admin`  | `password1234` |

Multiple admin users can exist by promoting other users to admin, using the `2.0/mlflow/users/update-admin` endpoint.

Example

bash

```bash
# authenticate as built-in admin user
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

```

python

```python
from mlflow.server import get_app_client

tracking_uri = "http://localhost:5000/"

auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
auth_client.create_user(username="user1", password="pw1")
auth_client.update_user_admin(username="user1", is_admin=True)

```

### Managing Permissions[​](#managing-permissions "Direct link to Managing Permissions")

MLflow provides [REST APIs](/mlflow-website/docs/latest/api_reference/auth/rest-api.html/#create-user) and a client class [`AuthServiceClient`](/mlflow-website/docs/latest/api_reference/auth/python-api.html#mlflow.server.auth.client.AuthServiceClient) to manage users and permissions. To instantiate `AuthServiceClient`, it is recommended that you use [`mlflow.server.get_app_client()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.server.html#mlflow.server.get_app_client).

Example

bash

```bash
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

```

python

```python
from mlflow import MlflowClient
from mlflow.server import get_app_client

tracking_uri = "http://localhost:5000/"

auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
auth_client.create_user(username="user1", password="pw1")
auth_client.create_user(username="user2", password="pw2")

client = MlflowClient(tracking_uri=tracking_uri)
experiment_id = client.create_experiment(name="experiment")

auth_client.create_experiment_permission(
    experiment_id=experiment_id, username="user2", permission="MANAGE"
)

```

## Authenticating to MLflow[​](#authenticating-to-mlflow "Direct link to Authenticating to MLflow")

### Using MLflow UI[​](#using-mlflow-ui "Direct link to Using MLflow UI")

When a user first visits the MLflow UI on a browser, they will be prompted to login. There is no limit to how many login attempts can be made.

Currently, MLflow UI does not display any information about the current user. Once a user is logged in, the only way to log out is to close the browser.

![](/mlflow-website/docs/latest/assets/images/auth_prompt-fae372058246a97a44a2236263de5c75.png)

### Using Environment Variables[​](#using-environment-variables "Direct link to Using Environment Variables")

MLflow provides two environment variables for authentication: `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`. To use basic authentication, you must set both environment variables.

bash

```bash
export MLFLOW_TRACKING_USERNAME=username
export MLFLOW_TRACKING_PASSWORD=password

```

python

```python
import mlflow

mlflow.set_tracking_uri("https://<mlflow_tracking_uri>/")
with mlflow.start_run():
    ...

```

### Using Credentials File[​](#using-credentials-file "Direct link to Using Credentials File")

You can save your credentials in a file to remove the need for setting environment variables every time. The credentials should be saved in `~/.mlflow/credentials` using `INI` format. Note that the password will be stored unencrypted on disk, and is protected only by filesystem permissions.

If the environment variables `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are configured, they override any credentials provided in the credentials file.

Credentials file format

ini

```ini
[mlflow]
mlflow_tracking_username = username
mlflow_tracking_password = password

```

### Using REST API[​](#using-rest-api "Direct link to Using REST API")

A user can authenticate using the HTTP `Authorization` request header. See <https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication> for more information.

In Python, you can use the `requests` library:

python

```python
import requests

response = requests.get(
    "https://<mlflow_tracking_uri>/",
    auth=("username", "password"),
)

```

## Creating a New User[​](#creating-a-new-user "Direct link to Creating a New User")

important

To create a new user, you are required to authenticate with admin privileges.

### Using MLflow UI[​](#using-mlflow-ui-1 "Direct link to Using MLflow UI")

MLflow UI provides a simple page for creating new users at `<tracking_uri>/signup`.

![](data:image/png;base64,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)

### Using REST API[​](#using-rest-api-1 "Direct link to Using REST API")

Alternatively, you can send `POST` requests to the Tracking Server endpoint `2.0/users/create`.

In Python, you can use the `requests` library:

python

```python
import requests

response = requests.post(
    "https://<mlflow_tracking_uri>/api/2.0/mlflow/users/create",
    json={
        "username": "username",
        "password": "password",
    },
)

```

### Using MLflow AuthServiceClient[​](#using-mlflow-authserviceclient "Direct link to Using MLflow AuthServiceClient")

MLflow [`AuthServiceClient`](/mlflow-website/docs/latest/api_reference/auth/python-api.html#mlflow.server.auth.client.AuthServiceClient) provides a function to create new users easily.

python

```python
import mlflow

auth_client = mlflow.server.get_app_client(
    "basic-auth", tracking_uri="https://<mlflow_tracking_uri>/"
)
auth_client.create_user(username="username", password="password")

```

## Configuration[​](#configuration "Direct link to Configuration")

Authentication configuration is located at `mlflow/server/auth/basic_auth.ini`:

| Variable                 | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| `default_permission`     | Default permission on all resources                        |
| `database_uri`           | Database location to store permission and user data        |
| `admin_username`         | Default admin username if the admin is not already created |
| `admin_password`         | Default admin password if the admin is not already created |
| `authorization_function` | Function to authenticate requests                          |

Alternatively, assign the environment variable `MLFLOW_AUTH_CONFIG_PATH` to point to your custom configuration file.

The `authorization_function` setting supports pluggable authentication methods if you want to use another authentication method than HTTP basic auth. The value specifies `module_name:function_name`. The function has the following signature:

python

```python
def authenticate_request() -> Union[Authorization, Response]:
    ...

```

The function should return a `werkzeug.datastructures.Authorization` object if the request is authenticated, or a `Response` object (typically `401: Unauthorized`) if the request is not authenticated. For an example of how to implement a custom authentication method, see `tests/server/auth/jwt_auth.py`. **NOTE:** This example is not intended for production use.

## Connecting to a Centralized Database[​](#connecting-to-a-centralized-database "Direct link to Connecting to a Centralized Database")

By default, MLflow Authentication uses a local SQLite database to store user and permission data. In the case of a multi-node deployment, it is recommended to use a centralized database to store this data.

To connect to a centralized database, you can set the `database_uri` configuration variable to the database URL.

Example: /path/to/my\_auth\_config.ini

ini

```ini
[mlflow]
database_uri = postgresql://username:password@hostname:port/database

```

Then, start the MLflow server with the `MLFLOW_AUTH_CONFIG_PATH` environment variable set to the path of your configuration file.

bash

```bash
MLFLOW_AUTH_CONFIG_PATH=/path/to/my_auth_config.ini mlflow server --app-name basic-auth

```

The database must be created before starting the MLflow server. The database schema will be created automatically when the server starts.

note

Auth migrations use a separate version table (`alembic_version_auth`) from tracking migrations (`alembic_version`). This allows you to safely use the same database for both tracking data and auth data without migration conflicts.

To use separate databases instead, configure `database_uri` to point to a different database than `--backend-store-uri`. By default, auth uses `basic_auth.db` (SQLite) while tracking uses the database specified by `--backend-store-uri`.
