---
title: MLflow AI Gateway renamed to MLflow Deployments for LLMs
tags: [ai]
slug: ai-gateway-rename
authors: [mlflow-maintainers]
thumbnail: /img/blog/ai-gateway.png
---

If you are currently using the MLflow AI Gateway, please read this in full to get critically important information about this feature!

# 🔔 Important Update Regarding the MLflow AI Gateway

Please note that the feature previously known as the `MLflow AI Gateway`, which was in an experimental phase, has undergone significant updates and improvements.

<!-- truncate -->

## Introducing "MLflow Deployments for LLMs"

This feature, while still in experimental status, has been renamed and migrated to utilize the `deployments` API.

## 🔑 Key Changes

**New Name, Enhanced Features**: The transition from "MLflow AI Gateway" to "MLflow Deployments for LLMs" reflects not only a change in name but also substantial enhancements in usability and **standardization** for API endpoints for Large Language Models.

**API Changes**: With this move, there are changes to the API endpoints and configurations. Users are encouraged to review the updated API documentation to familiarize themselves with the new structure.

**Migration Path**: For existing projects using "MLflow AI Gateway", a migration guide is available to assist with the transition to "MLflow Deployments for LLMs". This guide provides step-by-step instructions and best practices to ensure a smooth migration.

⚠️ **Action Required**: Users who have been utilizing the experimental "MLflow AI Gateway" should plan to migrate to "MLflow Deployments for LLMs". While we aim to make this transition as seamless as possible, manual changes to your code and deployment configurations will be necessary. This new namespace for deploying the previously-known-as AI Gateway will be released in version 2.9.0. The old AI Gateway references will remain active but will enter a deprecated state. _We will be removing the entire AI Gateway namespace in a future release_.

## 📚 Resources and Support

**Updated Documentation**: Detailed documentation for "MLflow Deployments for LLMs" is available [here](pathname:///docs/latest/llms/deployments/index.html). It includes comprehensive information about the modifications to API interfaces, updates to the input and output structures for queries and responses, API utilization, and the updated configuration options.

**Community and Support**: If you have any questions or need assistance, please reach out to the maintainers [on GitHub](https://github.com/mlflow/mlflow/issues).

We are excited about these advancements and strongly believe that leveraging the deployments API will offer a more robust, efficient, and scalable solution for managing your Large Language Model deployments. Thank you for your continued support and collaboration!
