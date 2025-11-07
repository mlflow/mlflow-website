# Manage Prompt Lifecycles

Managing changes of prompts is crucial for maintaining quality, improving performance, and ensuring consistency across different environments. This page provides a comprehensive guide to change management in MLflow Prompt Registry.

![Aliases List](/mlflow-website/docs/latest/assets/images/prompt-aliases-list-abcce1795c6c37ec5b7d90afefde7cf3.png)

## Commit-based Versioning[​](#commit-based-versioning "Direct link to Commit-based Versioning")

The design of the prompt registry is inspired by version control systems like Git.

* **🪨 Immutable versions**: Once created, a prompt version cannot be modified. This ensures that the prompt's behavior remains consistent across different applications and experiments.
* **✉️ Commit message**: When creating a new prompt version, you can provide a commit message to document the changes made in the new version. This helps you and your team understand the context of the changes and track the evolution of the prompt over time.
* **🔍 Difference view**: The MLflow UI provides a side-by-side comparison of prompt versions, highlighting the changes between versions. This makes it easy to understand the differences and track the evolution of the prompt.

Why not use Git?

Hard-coding prompt text in source code is indeed a common practice, but it has several limitations. A GenAI application or project often contains multiple prompts for different components/tasks, as well as all software artifacts. Tracking the change of a single prompt with a monotonic Git tree is challenging.

## Compare Prompt Versions[​](#compare-prompt-versions "Direct link to Compare Prompt Versions")

MLflow Prompt Registry UI provides a side-by-side comparison of prompt versions, highlighting the changes between versions. To compare prompt versions in the MLflow UI, click on the **Compare** tab in the prompt details page and select the versions you want to compare.

![Compare Prompt Versions](/mlflow-website/docs/latest/assets/images/compare-prompt-versions-2082121aeaca4be99a0cf968535141ed.png)

## Aliases[​](#aliases "Direct link to Aliases")

Alias is a strong mechanism to managing prompt versions in production systems, without hardcoding version numbers in the application code. You can create an alias for a specific version of a prompt using either the MLflow UI or Python API.

The common use case for aliases is to build a robust **deployment pipeline** for your GenAI applications. For example, you can set a stage name such as `beta`, `staging`, `production`, etc., to refer to the version used in that environment. By switching the alias to a different version, you can easily maintain multiple prompt versions for different environments and perform tasks such as roll-back A/B testing.

### Create an Alias[​](#create-an-alias "Direct link to Create an Alias")

* UI
* Python

![Create Prompt Alias](/mlflow-website/docs/latest/assets/images/create-prompt-alias-eada243c60800b04059e6a5311a5b492.png)

1. Open the existing prompt version in the MLflow UI.
2. Click on the **Add** button next to the **Aliases** section.
3. Choose an existing alias or create a new one by entering the alias name.
4. Click **Save aliases** to apply the changes.

python

```
# Set a production alias for a specific version
mlflow.set_prompt_alias("summarization-prompt", alias="production", version=2)
```

Attached aliases can be viewed in the prompt list page. You can click the pencil icon to edit or delete an alias directly from the list view.

### Load a Prompt using an Alias[​](#load-a-prompt-using-an-alias "Direct link to Load a Prompt using an Alias")

To load a prompt using an alias, use the `prompts:/<prompt_name>@<alias>` format as the prompt URI:

python

```
prompt = mlflow.load_prompt("prompts:/summarization-prompt@production")
```

### Reserved `@latest` alias[​](#reserved-latest-alias "Direct link to reserved-latest-alias")

The `@latest` alias is a reserved alias name and MLflow will automatically find the latest available version of the prompt. This is useful when you want to dynamically load the latest version of a prompt without updating the application code.

python

```
prompt = mlflow.load_prompt("prompts:/summarization-prompt@latest")
```
