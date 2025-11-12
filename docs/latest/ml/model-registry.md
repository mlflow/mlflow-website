# MLflow Model Registry

The MLflow Model Registry is a centralized model store, set of APIs and a UI designed to collaboratively manage the full lifecycle of a model. It provides lineage (i.e., which MLflow experiment and run produced the model), versioning, aliasing, metadata tagging and annotation support to ensure that you have the full spectrum of information at every stage from development to production deployment.

## Why Model Registry?[​](#why-model-registry "Direct link to Why Model Registry?")

As machine learning projects grow in complexity and scale, managing models manually across different environments, teams, and iterations becomes increasingly error-prone and inefficient. The MLflow Model Registry addresses this challenge by providing a centralized, structured system for organizing and governing ML models throughout their lifecycle.

Using the Model Registry offers the following benefits:

* **🗂️ Version Control**: The registry automatically tracks versions of each model, allowing teams to compare iterations, roll back to previous states, and manage multiple versions in parallel (e.g., staging vs. production).
* **🧬 Model Lineage and Traceability**: Each registered model version is linked to the MLflow run, logged model or notebook that produced it, enabling full reproducibility. You can trace back exactly how a model was trained, with what data and parameters.
* **🚀 Production-Ready Workflows**: Features like model aliases (e.g., @champion) and tags make it easier to manage deployment workflows, promoting models to experimental, staging, or production environments in a controlled and auditable way.
* **🛡️ Governance and Compliance**: With structured metadata, tagging, and role-based access controls (when used with a backend like Databricks or a managed MLflow service), the Model Registry supports governance requirements critical for enterprise-grade ML operations.

Whether you're a solo data scientist or part of a large ML platform team, the Model Registry is a foundational component for scaling reliable and maintainable machine learning systems.

## Concepts[​](#concepts "Direct link to Concepts")

The Model Registry introduces a few concepts that describe and facilitate the full lifecycle of an MLflow Model.

| Concept                      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model                        | An MLflow Model is created with one of the model flavor's **`mlflow.<model_flavor>.log_model()`** methods, or **[`mlflow.create_external_model()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.create_external_model)** API since MLflow 3. Once logged, this model can then be registered with the Model Registry.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Registered Model             | An MLflow Model can be registered with the Model Registry. A registered model has a unique name, contains versions, aliases, tags, and other metadata.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Model Version                | Each registered model can have one or many versions. When a new model is added to the Model Registry, it is added as version 1. Each new model registered to the same model name **increments the version number**. Model versions have tags, which can be useful for tracking attributes of the model version (e.g. *`pre_deploy_checks: "PASSED"`*)                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Model URI                    | You can refer to the registered model by using a URI of this format: `models:/<model-name>/<model-version>`, e.g., if you have a registered model with name "MyModel" and version 1, the URI referring to the model is: `models:/MyModel/1`".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Model Alias                  | Model aliases allow you to assign a mutable, named reference to a particular version of a registered model. By assigning an alias to a specific model version, you can use the alias to refer to that model version via a model URI or the model registry API. For example, you can create an alias named **`champion`** that points to version 1 of a model named **`MyModel`**. You can then refer to version 1 of **`MyModel`** by using the URI **`models:/MyModel@champion`**.Aliases are especially useful for deploying models. For example, you could assign a **`champion`** alias to the model version intended for production traffic and target this alias in production workloads. You can then update the model serving production traffic by reassigning the **`champion`** alias to a different model version. |
| Tags                         | Tags are key-value pairs that you associate with registered models and model versions, allowing you to label and categorize them by function or status. For example, you could apply a tag with key **`"task"`** and value **`"question-answering"`** (displayed in the UI as **`task:question-answering`**) to registered models intended for question answering tasks. At the model version level, you could tag versions undergoing pre-deployment validation with **`validation_status:pending`** and those cleared for deployment with **`validation_status:approved`**.                                                                                                                                                                                                                                                  |
| Annotations and Descriptions | You can annotate the top-level model and each version individually using Markdown, including the description and any relevant information useful for the team such as algorithm descriptions, datasets employed or the overall methodology involved in a given version's modeling approach.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

## Model Registry in practice[​](#model-registry-in-practice "Direct link to Model Registry in practice")

The MLflow Model Registry is available in both open-source (OSS) MLflow and managed platforms like Databricks. Depending on the environment, the registry offers different levels of integration, governance, and collaboration features.

### Model Registry in OSS MLflow[​](#model-registry-in-oss-mlflow "Direct link to Model Registry in OSS MLflow")

In the open-source version of MLflow, the Model Registry provides both a UI and API for managing the lifecycle of machine learning models. You can register models, track versions, add tags and descriptions, and transition models between stages such as Staging and Production.

Register a model in MLflow

* Python APIs
* MLflow UI

#### Register a model with MLflow Python APIs[​](#register-a-model-with-mlflow-python-apis "Direct link to Register a model with MLflow Python APIs")

MLflow provides several ways to register a model version

text

```text
# Option 1: specify `registered_model_name` parameter when logging a model
mlflow.<flavor>.log_model(..., registered_model_name="<YOUR_MODEL_NAME>")

# Option 2: register a logged model
mlflow.register_model(model_uri="<YOUR_MODEL_URI>", name="<YOUR_MODEL_NAME>")

```

After registering the model, you can load it back with the model name and version

text

```text
mlflow.<flavor>.load_model("models:/<YOUR_MODEL_NAME>/<YOUR_MODEL_VERSION>")

```

#### Register a model on MLflow UI[​](#register-a-model-on-mlflow-ui "Direct link to Register a model on MLflow UI")

1. Open the details page for the MLflow Run containing the logged MLflow model you'd like to register. Select the model folder containing the intended MLflow model in the **Artifacts** section.

![](/mlflow-website/docs/latest/assets/images/oss_registry_1_register-a71f2ea36d15265894cf0ea1810dd95f.png)

2. Click the **Register Model** button, which will trigger a modal form to pop up.

3. In the **Model** dropdown menu on the form, you can either select "Create New Model" (which creates a new registered model with your MLflow model as its initial version) or select an existing registered model (which registers your model under it as a new version). The screenshot below demonstrates registering the MLflow model to a new registered model named `"iris_model_testing"`.

![](/mlflow-website/docs/latest/assets/images/oss_registry_2_dialog-1ac2c5e115d621eb507274c577093173.png)

To learn more about the OSS Model Registry, refer to the [tutorial on the model registry](/mlflow-website/docs/latest/ml/model-registry/tutorial.md).

### Model Registry in Databricks[​](#model-registry-in-databricks "Direct link to Model Registry in Databricks")

Databricks extends MLflow's capabilities by integrating the Model Registry with Unity Catalog, enabling centralized governance, fine-grained access control, and cross-workspace collaboration.

Key benefits of Unity Catalog integration include:

* **🛡️ Enhanced governance**: Apply access policies and permission controls to model assets.
* **🌐 Cross-workspace access**: Register models once and access them across multiple Databricks workspaces.
* **🔗 Model lineage**: Track which notebooks, datasets, and experiments were used to create each model.
* **🔍 Discovery and reuse**: Browse and reuse production-grade models from a shared catalog.

Register a model in Databricks UC

* Python APIs
* Databricks UI

#### Register a model to Databricks UC with MLflow Python APIs[​](#register-a-model-to-databricks-uc-with-mlflow-python-apis "Direct link to Register a model to Databricks UC with MLflow Python APIs")

**Prerequisite**: Set tracking uri to Databricks

python

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")

```

Use MLflow APIs to register the model

text

```text
# Option 1: specify `registered_model_name` parameter when logging a model
mlflow.<flavor>.log_model(..., registered_model_name="<YOUR_MODEL_NAME>")

# Option 2: register a logged model
mlflow.register_model(model_uri="<YOUR_MODEL_URI>", name="<YOUR_MODEL_NAME>")

```

warning

ML model versions in UC must have a [model signature](/mlflow-website/docs/latest/ml/model/signatures.md). If you want to set a signature on a model that's already logged or saved, the [`mlflow.models.set_signature()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.models.html#mlflow.models.set_signature) API is available for this purpose.

After registering the model, you can load it back with the model name and version

text

```text
mlflow.<flavor>.load_model("models:/<YOUR_MODEL_NAME>/<YOUR_MODEL_VERSION>")

```

#### Register a model on Databricks UI[​](#register-a-model-on-databricks-ui "Direct link to Register a model on Databricks UI")

1. From the experiment run page or models page, click Register model in the upper-right corner of the UI.

2. In the dialog, select Unity Catalog, and select a destination model from the drop down list.

![](/mlflow-website/docs/latest/assets/images/uc_register_model_1_dialog-dbc7806e79613776eb84159fa6c394e2.png)

3. Click Register.

![](/mlflow-website/docs/latest/assets/images/uc_register_model_2_button-e6b3b94bde6506bda3be82836db5e019.png)

Registering a model can take time. To monitor progress, navigate to the destination model in Unity Catalog and refresh periodically.

For more information, refer to the [Databricks documentation](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle) on managing the model lifecycle.
