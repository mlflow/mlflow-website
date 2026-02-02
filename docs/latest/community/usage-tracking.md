# Usage Tracking

Starting with version **3.2.0**, MLflow collects anonymized usage data by default. This data contains no sensitive or personally identifiable information.

important

MLflow does not collect any data that contains personal information, in accordance with GDPR and other privacy regulations. As a Linux Foundation project, MLflow adheres to the [**LF telemetry data collection and usage policy**](https://lfprojects.org/policies/telemetry-data-policy/). This implementation has been reviewed and approved by the Linux Foundation, with the approved proposal documented at the [**Completed Reviews**](https://lfprojects.org/policies/telemetry-data-policy/) section in the official policy. See the [`Data Explanation section`](#data-explanation) below for details on what is collected.

note

Telemetry is only enabled in **Open Source MLflow**. If you're using MLflow through a managed service or distribution, please consult your vendor to determine whether telemetry is enabled in your environment. In all cases, you can choose to opt out by following the guidance provided in our documentation.

## Why is data being collected?[​](#why-is-data-being-collected "Direct link to Why is data being collected?")

MLflow uses anonymous telemetry to understand feature usage, which helps guide development priorities and improve the library. This data helps us identify which features are most valuable and where to focus on bug fixes or enhancements.

### GDPR Compliance[​](#gdpr-compliance "Direct link to GDPR Compliance")

Under the General Data Protection Regulation (GDPR), data controllers and processors are responsible for handling personal data with care, transparency, and accountability.

MLflow complies with GDPR in the following ways:

* **No Personal Data Collected**: The telemetry data collected is fully anonymized and does not include any personal or sensitive information (e.g., usernames, IP addresses, file names, parameters, or model content). MLflow generates a random UUID for each session for aggregating usage events, which cannot be used to identify or track individual users.
* **Purpose Limitation**: Data is only used to improve the MLflow project based on aggregate feature usage patterns.
* **Data Minimization**: Only the minimum necessary metadata is collected to inform project priorities (e.g., feature toggle state, SDK/platform used, version info).
* **User Control**: Users can opt out of telemetry at any time by setting the environment variable **MLFLOW\_DISABLE\_TELEMETRY=true** or **DO\_NOT\_TRACK=true**. MLflow respects these settings immediately without requiring a restart.
* **Transparency**: Telemetry endpoints and behavior are documented publicly, and MLflow users can inspect or block the relevant network calls.

For further inquiries or data protection questions, users can file an issue on the [MLflow GitHub repository](https://github.com/mlflow/mlflow/issues).

## What data is collected?[​](#what-data-is-collected "Direct link to What data is collected?")

MLflow collects only non-sensitive, anonymized data to help us better understand usage patterns. The below section outlines the data currently collected in this version of MLflow. You can view the exact data collected [in the source code](https://github.com/mlflow/mlflow/blob/c71fd0d677c1806ba2d5928398435c4de2c25c0e/mlflow/telemetry/schemas.py).

### Data Explanation[​](#data-explanation "Direct link to Data Explanation")

| Data Element                              | Explanation                                                                                                                                             | Example                                                                                   | Why we track this                                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Unique session ID                         | A randomly generated, non-personally identifiable UUID is created for each session—defined as each time MLflow is imported                              | 45e2751243e84c7e87aca6ac25d75a0d                                                          | As an identifier for the data in current MLflow session                                      |
| Unique installation ID                    | A randomly generated, non-personally identifiable UUID is created for each installation—defined as each time MLflow is imported. Added in MLflow 3.7.0. | 45e2751243e84c7e87aca6ac25d75a0d                                                          | As an identifier for the data in current MLflow installation                                 |
| Source SDK                                | The current used SDK name                                                                                                                               | mlflow \| mlflow-skinny \| mlflow-tracing                                                 | To understand adoption of different MLflow SDKs and identify enhancement areas               |
| MLflow version                            | The current SDK version                                                                                                                                 | 3.2.0                                                                                     | To identify version-specific usage patterns and support, bug fixes, or deprecation decisions |
| Python version                            | The current python version                                                                                                                              | 3.10.16                                                                                   | To ensure compatibility across Python versions and guide testing or upgrade recommendations  |
| Operating System                          | The operating system on which MLflow is running                                                                                                         | macOS-15.4.1-arm64-arm-64bit                                                              | To understand platform-specific usage and detect platform-dependent issues                   |
| Tracking URI Scheme                       | The scheme of the current tracking URI                                                                                                                  | file \| sqlite \| mysql \| postgresql \| mssql \| https \| http \| custom\_scheme \| None | To determine which tracking backends are most commonly used and optimize backend support     |
| Event Name                                | The tracked event name (see [below table](#tracked-events) for what events are tracked)                                                                 | create\_experiment                                                                        | To measure feature usage and improvements                                                    |
| Event Status                              | Whether the event succeeds or not                                                                                                                       | success \| failure \| unknown                                                             | To identify common failure points and improve reliability and error handling                 |
| Timestamp (nanoseconds)                   | Time when the event occurred                                                                                                                            | 1753760188623715000                                                                       | As an identifier for the event                                                               |
| Duration                                  | The time the event call takes, in milliseconds                                                                                                          | 1000                                                                                      | To monitor performance trends and detect regressions in response time                        |
| Parameters (boolean or enumerated values) | See [below table](#tracked-events) for collected parameters for each event                                                                              | create\_logged\_model event: `{"flavor": "langchain"}`                                    | To better understand the usage pattern for each event                                        |

#### Tracked Events[​](#tracked-events "Direct link to Tracked Events")

**No details about the specific model, code, or weights are collected.** Only the parameters listed under the `Tracked Parameters` column are recorded alongside the event; For events with None in the `Tracked Parameters` column, only the event name is recorded. If "MLFLOW\_EXPERIMENT\_ID" environment variable exists, it is tracked as a param. For a comprehensive list of tracked events, please refer to the [source code](https://github.com/mlflow/mlflow/blob/005f8b18186d254286a7d258a564b414f0ee0f75/mlflow/telemetry/events.py).

| Event Name                   | Tracked Parameters                                                                                                                                                                                                                  | Example                                                                                                                                                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| create\_experiment           | Created Experiment ID (random uuid or integer)                                                                                                                                                                                      | `{"experiment_id": "0"}`                                                                                                                                                                           |
| create\_run                  | Imported packages among [MODULES\_TO\_CHECK\_IMPORT](https://github.com/mlflow/mlflow/blob/c71fd0d677c1806ba2d5928398435c4de2c25c0e/mlflow/telemetry/constant.py#L19) are imported or not; experiment ID used when creating the run | `{"imports": ["sklearn"], "experiment_id": "0"}`                                                                                                                                                   |
| create\_logged\_model        | Flavor of the model (e.g. langchain, sklearn)                                                                                                                                                                                       | `{"flavor": "langchain"}`                                                                                                                                                                          |
| get\_logged\_model           | Imported packages among [MODULES\_TO\_CHECK\_IMPORT](https://github.com/mlflow/mlflow/blob/c71fd0d677c1806ba2d5928398435c4de2c25c0e/mlflow/telemetry/constant.py#L19) are imported or not                                           | `{"imports": ["sklearn"]}`                                                                                                                                                                         |
| create\_registered\_model    | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| create\_model\_version       | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| create\_prompt               | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| load\_prompt                 | Whether alias is used                                                                                                                                                                                                               | `{"uses_alias": True}`                                                                                                                                                                             |
| start\_trace                 | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| traces\_received\_by\_server | Type of client (sanitized) that submitted the traces and number of completed traces received                                                                                                                                        | `{"source": "MLFLOW_PYTHON_CLIENT", "count": 3}`                                                                                                                                                   |
| log\_assessment              | Type of the assessment and source                                                                                                                                                                                                   | `{"type": "feedback", "source_type": "CODE"}`                                                                                                                                                      |
| evaluate                     | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| create\_webhook              | Entities of the webhook                                                                                                                                                                                                             | `{"events": ["model_version.created"]}`                                                                                                                                                            |
| genai\_evaluate              | Builtin scorers used during GenAI Evaluate                                                                                                                                                                                          | `{"builtin_scorers": ["relevance_to_query"]}`                                                                                                                                                      |
| prompt\_optimization         | Optimizer type, number of prompts, and number of scorers                                                                                                                                                                            | `{"optimizer_type": True, "prompt_count": 5, "scorer_count": 1}`                                                                                                                                   |
| log\_dataset                 | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| log\_metric                  | Whether synchronous mode is on or not                                                                                                                                                                                               | `{"synchronous": False}`                                                                                                                                                                           |
| log\_param                   | Whether synchronous mode is on or not                                                                                                                                                                                               | `{"synchronous": True}`                                                                                                                                                                            |
| log\_batch                   | Information on whether metrics, parameters, or tags are logged, and the logging mode                                                                                                                                                | `{"metrics": False, "params": True, "tags": False, "synchronous": False}`                                                                                                                          |
| invoke\_custom\_judge\_model | Judge model provider                                                                                                                                                                                                                | `{"model_provider": "databricks"}`                                                                                                                                                                 |
| make\_judge                  | Model provider (extracted from model string if format is provider<!-- -->:model<!-- -->)                                                                                                                                            | `{"model_provider": "openai"}`                                                                                                                                                                     |
| align\_judge                 | Number of traces provided and optimizer type                                                                                                                                                                                        | `{"trace_count": 100, "optimizer_type": "AlignmentOptimizer"}`                                                                                                                                     |
| autologging                  | Flavor and metadata                                                                                                                                                                                                                 | `{"flavor": "openai", "log_traces": True, "disable": False}`                                                                                                                                       |
| ai\_command\_run             | Command key and invocation context (cli or mcp)                                                                                                                                                                                     | `{"command_key": "genai/analyze_experiment", "context": "cli"}`                                                                                                                                    |
| gateway\_start               | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| gateway\_create\_endpoint    | Whether fallback config is set, routing strategy, and number of model configs                                                                                                                                                       | `{"has_fallback_config": true, "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT", "num_model_configs": 2}`                                                                                         |
| gateway\_update\_endpoint    | Whether fallback config is set, routing strategy, and number of model configs (null if not provided)                                                                                                                                | `{"has_fallback_config": false, "routing_strategy": "ROUND_ROBIN", "num_model_configs": 1}`                                                                                                        |
| gateway\_delete\_endpoint    | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| gateway\_get\_endpoint       | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| gateway\_list\_endpoints     | Whether filtering by provider                                                                                                                                                                                                       | `{"filter_by_provider": true}`                                                                                                                                                                     |
| gateway\_create\_secret      | Provider name                                                                                                                                                                                                                       | `{"provider": "openai"}`                                                                                                                                                                           |
| gateway\_update\_secret      | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| gateway\_delete\_secret      | None                                                                                                                                                                                                                                | None                                                                                                                                                                                               |
| gateway\_list\_secrets       | Whether filtering by provider                                                                                                                                                                                                       | `{"filter_by_provider": false}`                                                                                                                                                                    |
| gateway\_invocation          | Whether streaming is enabled and the invocation type                                                                                                                                                                                | `{"is_streaming": true, "invocation_type": "mlflow_chat_completions"}`                                                                                                                             |
| ui\_event                    | A UI interaction event. See the [below table](#ui-interaction-metadata) for a description of the various metadata elements                                                                                                          | `{ "eventType": "onClick", "componentViewId": "88fc9edd-5e9e-4a17-abd2-c543f505b8eb", "componentId": "mlflow.prompts.list.create", "componentType": "button", timestamp_ns: 1765784028467000000 }` |

#### UI Interaction Metadata[​](#ui-interaction-metadata "Direct link to UI Interaction Metadata")

This table describes a list of metadata that may be collected together with a given UI interaction log.

| Metadata Element                        | Explanation                                                                                                                                                                                                                                                                                                                                        | Example                                                                                      |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Component ID of interactive UI elements | An ID string of an interactive element (e.g. button, switch, link, input field) in the UI. A log is generated upon clicking, typing, or otherwise interacting with such elements. A comprehensive list of component ID values can be found by [this search query](https://github.com/search?q=repo%3Amlflow%2Fmlflow%20componentId%3D\&type=code). | `mlflow.prompts.list.create` (identifier for the "Create prompt" button on the prompts page) |
| Event type                              | An enumerated categorical value describing the nature of the interaction                                                                                                                                                                                                                                                                           | `onView`, `onClick`, `onValueChange`                                                         |
| Component type                          | An enumerated categorical value describing the type of component that the interaction happened with                                                                                                                                                                                                                                                | `button`, `alert`, `banner`, `radio`, `input`, ...                                           |
| Component View ID                       | A randomly generated UUID that is regenerated whenever the UI element rerenders                                                                                                                                                                                                                                                                    | `774db636-5cfa-4ce8-8f56-7e7126dc3439`                                                       |
| Timestamp                               | The client-side timestamp of when the interaction occurred                                                                                                                                                                                                                                                                                         | `1765789548484000`.                                                                          |

## Why is MLflow Telemetry Opt-Out?[​](#why-is-mlflow-telemetry-opt-out "Direct link to Why is MLflow Telemetry Opt-Out?")

MLflow uses an opt-out telemetry model to help improve the platform for all users based on real-world usage patterns. Collecting anonymous usage data by default allows us to:

* Understand how MLflow is being used across a wide range of environments and workflows
* Identify common pain points and identify feature improvements area more effectively
* Measure the impact of changes and ensure they improve the experience for the broader community

If telemetry were opt-in, only a small, self-selected subset of users would be represented, leading to biased insights and potentially misaligned priorities. We are committed to transparency and user choice. Telemetry is clearly documented, anonymized, and can be easily disabled at any time through configuration. This approach helps us make MLflow better for everyone, while giving you full control. Check [`what we are doing with this data`](#what-are-we-doing-with-this-data) section for more information.

## How to opt-out?[​](#how-to-opt-out "Direct link to How to opt-out?")

MLflow supports opt-out telemetry through either of the following environment variables:

* **MLFLOW\_DISABLE\_TELEMETRY=true**
* **DO\_NOT\_TRACK=true**

Setting either of these will **immediately disable telemetry**, no need to re-import MLflow or restart your session.

note

MLflow automatically disables telemetry in [**some CI environments**](https://github.com/mlflow/mlflow/blob/de6c11193ce6a68ffec4b33650f75bd163143178/mlflow/telemetry/utils.py#L22). If you'd like support for additional CI environments, please [open an issue on our GitHub repository](https://github.com/mlflow/mlflow/issues).

* CI
* Github Actions
* CircleCI
* GitLab CI/CD
* Jenkins Pipeline
* Travis CI
* Azure Pipelines
* BitBucket
* AWS CodeBuild
* BuildKite
* ...

### Scope of the setting[​](#scope-of-the-setting "Direct link to Scope of the setting")

* The environment variable only takes effect in processes where it is explicitly set or inherited.
* If you spawn subprocesses from a clean environment, those subprocesses may not inherit your shell's environment, and telemetry could remain enabled. e.g. `subprocess.run([...], env={})`
* Setting this environment variable before running `mlflow server` also disables all UI telemetry

Recommendations to ensure telemetry is consistently disabled across all environments:

* Add the variable to your shell startup file (\~/.bashrc, \~/.zshrc, etc.): `export MLFLOW_DISABLE_TELEMETRY=true`
* If you're using subprocesses or isolated environments, use a dotenv manager or explicitly pass the variable when launching.

### How to validate telemetry is disabled?[​](#how-to-validate-telemetry-is-disabled "Direct link to How to validate telemetry is disabled?")

Use the following code to validate telemetry is disabled.

python

```python
from mlflow.telemetry import get_telemetry_client

assert get_telemetry_client() is None, "Telemetry is enabled"

```

### How to opt-out for your organization?[​](#how-to-opt-out-for-your-organization "Direct link to How to opt-out for your organization?")

Aside from setting the environment variables described above, organizations can additionally opt out of telemetry by blocking network access to the `mlflow-telemetry.io` domain. When this domain is unreachable, telemetry will be disabled.

### Opting out of UI telemetry[​](#opting-out-of-ui-telemetry "Direct link to Opting out of UI telemetry")

As described above, the admin of an MLflow server can set the `MLFLOW_DISABLE_TELEMETRY` or `DO_NOT_TRACK` environment variables to disable UI telemetry globally for the server. However, if you are not an admin (i.e. you have no ability to set environment variables), you can still personally opt out from UI telemetry by visiting the "Settings" page in the MLflow UI (introduced in MLflow 3.8.0).

Setting the toggle to "Off" will disable UI telemetry from your device, even if the admin has not opted out server-side.

## What are we doing with this data?[​](#what-are-we-doing-with-this-data "Direct link to What are we doing with this data?")

We aggregate anonymized usage data and plan to share insights with the community through public dashboards. You'll be able to see how MLflow features are used and help improve them by contributing.
