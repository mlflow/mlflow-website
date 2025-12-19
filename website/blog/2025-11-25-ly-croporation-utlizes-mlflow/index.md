---
title: "Enterprise-Scale MLflow Operations and Security Practices at LY Corporation"
slug: ly-corporation-utilizes-mlflow
tags: [mlflow, authorization]
authors: [motoki-yuhara,  mlflow-maintainers]
thumbnail: img/blog/ly-utilizes-mlflow/AIPF.png
---

# How LY Corporation Uses MLflow: An Overview
I’m responsible for building and operating the Managed MLflow service for the internal AI platform at [LY Corporation](https://www.lycorp.co.jp/en/company/overview/).

LY Corporation is one of Japan’s leading tech companies, operating a wide range of online services including advertising, e-commerce, and digital membership platforms. Through its group companies, LY continues to advance innovation across different digital domains.

At LY Corporation, we have developed an in-house AI platform that supports the entire lifecycle of AI and machine learning–based services (MLOps) across more than 100 products and services.

This platform provides end-to-end MLOPs capabilities from model development to deployment and operation. MLflow is a core component of the platform for managing model training, evaluation, and model lifecycle.
Currently, MLflow is used by approximately 40 services, and it handles more than 600,000 access requests per day on average.

![Aipf](/img/blog/ly-utilizes-mlflow/AIPF.png)

## Managed MLflow Service on Kubernetes
When each user independently runs their own MLflow server, it can lead to inefficient use of computing resources.
To address this, we centralized MLflow as a managed service running on our Kubernetes-based computing infrastructure.

Since multiple services within LY Corporation use MLflow, we needed to ensure both stability and data isolation.
To achieve this, we provide a dedicated MLflow server instance for each service.
However, this also requires strict access control so that only the authorized project members of a given service can use the corresponding MLflow instance.

In addition, model training runs on our Kubernetes-based training environment and the training programs must have access the MLflow API from the pod. This required us to implement a machine-to-machine authentication and authorization flow between services.
# Implementing Service-to-Service Authentication and Authorization Based on OAuth 2.0
In OAuth 2.0, the standard specification for authorization, the Client Credentials Grant Flow is defined as the authentication and authorization mechanism for service-to-service communication.

This flow allows a service to authenticate itself without any end-user involvement, obtain an access token, and use that token to call protected APIs.
The flow works as follows:

```
[Service A] → Authorization Server (authenticates using client_id and client_secret)
        ↓
[Authorization Server] issues an Access Token
        ↓
[Service A] → [Service B (Resource Server)] accesses API (Access Token validation and authorization check)
```

In our environment, the flow works between the model training program and MLflow as follows:

1. The model training program (corresponding to Service A) obtains an access token and attaches it to API requests via the Authorization header.
2. MLflow (corresponding to Service B / Resource Server) validates the access token included in the request and performs authorization checks based on configured policies.

At LY Corporation, we provide an internal OAuth-compliant auth platform built on [Athenz](https://www.athenz.io/), a service-to-service authentication and authorization system co-developed by Yahoo Inc. (US) and LY Corporation. Athenz is currently a [Sandbox project under the Cloud Native Computing Foundation (CNCF)](https://www.cncf.io/projects/athenz/). This platform underpins our service-to-service authentication and authorization for MLflow.

## Obtaining an Access Token from the Authorization Server
In the Client Credentials Grant Flow, the Authorization Server can authenticate Service A not only by using a client_id and client_secret, but also through mTLS-based authentication.

Athenz provides a feature called [Athenz Copper Argos](https://athenz.github.io/athenz/copper_argos/), which issues client certificates in X.509 format based on the SPIFFE (Secure Production Identity Framework For Everyone) specification—an open standard for secure identity management in distributed systems. These certificates serve as proof of a service’s identity.

By using an Instance Certificate issued through Copper Argos, a service can perform mTLS communication with the Authorization Server and prove that it is a legitimate service.

In our model training platform, each Pod uses Athenz Copper Argos to perform mTLS authentication with the Authorization Server at startup and automatically obtains an access token. As a result, model training programs inside the Pod can use the issued access token transparently, without having to implement any token acquisition logic themselves.


## Authorization Checks Based on Access Tokens Using the Authorization Proxy
On the MLflow side, we need a mechanism to validate access tokens and perform authorization checks.

To achieve this, we adopted [Authorization Proxy](https://github.com/AthenZ/authorization-proxy), one of the components provided by Athenz. Authorization Proxy runs as a reverse proxy in a Kubernetes sidecar container, validates access tokens and enforces authorization policies. From an OAuth 2.0 perspective, Authorization Proxy takes the role of performing authentication and authorization checks on behalf of the Resource Server.

Because we regularly upgrade MLflow, our policy is to avoid adding custom features directly to the MLflow OSS codebase, as doing so would increase operational complexity. The Authorization Proxy aligns well with this policy, since it allows us to add authentication and authorization features externally without modifying MLflow itself.

In our deployment, each MLflow Pod includes the Authorization Proxy as a sidecar that enforces authentication and authorization for all MLflow API requests.

OAuth 2.0 leaves the detailed specification of authorization scopes to each implementation. In Athenz, access permissions are assigned to roles based on an RBAC (Role-Based Access Control) model, and these role definitions can be used as OAuth access token scopes.

By leveraging this capability, we are able to apply Athenz role-based access control directly to MLflow API operations.

The overall flow of MLflow API access control combining access tokens and the Authorization Proxy is summarized in the diagram below.

*(Here, Model Trainer Container refers to the container running the model training program.)*

![Authorizationflow](/img/blog/ly-utilizes-mlflow/authorization_flow.png)

## Example: Access Control Verification for MLflow Servers
The following examples show how access control works for MLflow servers that are integrated with Athenz.
The first example demonstrates a successful API request to an MLflow server where the user has access permissions.
The second example shows an unauthorized request that correctly returns a 401 error, indicating that the user does not have permission to access that server.

Example of Access to an Authorized MLflow Server
```
# Example of access to an MLflow server with access permission
% curl -i -X GET ${MLFLOW_SERVER_HAVING_ACCESS_RIGHTS}/api/2.0/mlflow/experiments/search \
    -H 'Content-Type:application/json;'  \
    -H "Authorization: Bearer $(echo $MLFLOW_ACCESS_TOKEN)" \
    --data '{"max_results":1}'
HTTP/2 200
content-length: 321
content-type: application/json
{
  "experiments": [
    {
      "experiment_id": "1857",
      "name": "mlflow-example",
      "artifact_location": "s3://sandbox/1857",
      "lifecycle_stage": "active",
      "last_update_time": 1759381587999,
      "creation_time": 1759381587999
    }
  ],
}
```

Example of Unauthorized Access to an MLflow Server
```
# Example of unauthorized access to an MLflow server
 % curl -i -X GET https://${MLFLOW_SERVER_NOT_HAVING_ACCESS_RIGHTS}/api/2.0/mlflow/experiments/search \
    -H 'Content-Type:application/json;'  \
    -H "Authorization: Bearer $(echo $MLFLOW_ACCESS_TOKEN)" \
    --data '{"max_results":1}'
HTTP/2 401
content-length: 0
```
As shown above, when accessing an MLflow server without the proper permissions, the request correctly returns a 401 Unauthorized response — confirming that access control via Athenz works as expected.

With these mechanisms in place, we have built a managed MLflow environment that integrates seamlessly with our Kubernetes-based training platform while ensuring secure and reliable service-to-service usage.

# MLflow Usage at LY Corporation
To monitor the health and performance of each MLflow server provided as part of the Managed MLflow service, we have installed a Prometheus exporter on each server and built a monitoring system using Prometheus and Alertmanager.
This setup allows us to track service availability and request processing performance.

Since collecting metrics for all API endpoints would have a negative impact on performance, we only collect metrics from a limited set of APIs. However, this means the monitoring metrics alone do not let us understand the complete usage patterns of MLflow.

To compensate for this, we analyze Ingress access logs and identify which MLflow servers and endpoints are being accessed, which gives us an overview of how MLflow is being used at LY Corporation.

The following graph shows the trend of average daily API access counts from April 2024 to the present.
Although usage fluctuates significantly, the overall number of API requests has increased roughly fourfold over the past year and a half — indicating growing adoption of MLflow across our internal AI platform.

![ApiCallDailyAvg](/img/blog/ly-utilizes-mlflow/fig_api_calls_daily_avg.png)

## Trends in Runs, Model Registrations, and Model References
The next chart tracks the number of Runs created, models registered in the MLflow Model Registry, and model references.

Interestingly, while the number of Runs continues to grow, the count of model registrations fluctuates within a relatively stable range.

Digging deeper, we discovered that some services use MLflow primarily to record model evaluation results rather than to persist final training artifacts. In other words, MLflow is increasingly being used beyond simple storage of training outputs—for example, to log and analyze evaluation metrics as part of experimentation workflows.

Another notable trend: although model registrations remain fairly steady, model references are increasing. This suggests that the models recorded in MLflow are being reused and consumed more actively across our platform, indicating growing operational adoption and downstream integration.


![RunAndModel](/img/blog/ly-utilizes-mlflow/fig_log_model_modelref_twinx.png)


# Conclusion
In this article, we introduced how LY Corporation provides a Managed MLflow service within our internal AI platform — including the implementation of custom authentication and authorization based on Athenz, and an overview of MLflow usage trends across our organization.

Beyond what we covered here, we have also developed several custom MLflow plugins to support our internal infrastructure.
For example, we built a plugin that added internal certificates to access the MLflow Model Registry REST API (this has since been deprecated as MLflow OSS now supports certificate configuration natively), as well as a plugin to access our internal object storage system.
We are also working on integrating the MLflow Model Registry with our internal inference platform.

We will continue to enhance our MLflow environment to support the development of AI and machine learning–powered services across LY Corporation.