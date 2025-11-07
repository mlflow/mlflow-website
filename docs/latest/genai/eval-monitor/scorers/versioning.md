# Registering and Versioning Scorers

Scorers can be registered to MLflow experiments for version control and team collaboration.

## Supported Scorers[​](#supported-scorers "Direct link to Supported Scorers")

| Scorer Type                                                                                                    | Supported                                                                                       |
| -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [Agent-as-a-Judge](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/agentic-overview.md)       | ✅                                                                                              |
| [Template-based LLM Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/make-judge.md)   | ✅                                                                                              |
| [Code-based Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/custom.md)                         | ✅                                                                                              |
| [Guidelines-based LLM Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines.md) | ❌ (Use [MLflow Prompt Registry](/mlflow-website/docs/latest/genai/prompt-registry.md) instead) |
| [Predefined Scorers](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined.md)           | ❌ (Prompts are hard-coded in MLflow)                                                           |

## Usage[​](#usage "Direct link to Usage")

### Prerequisite[​](#prerequisite "Direct link to Prerequisite")

Judges are registered to an **MLflow Experiment** (not Run-level).

python

```
import mlflow

mlflow.set_tracking_uri("your-tracking-uri")
mlflow.create_experiment("evaluation-judges")
```

Define a sample template-based LLM scorer:

python

```
from mlflow.genai.judges import make_judge

quality_judge = make_judge(
    name="response_quality",
    instructions=("Evaluate if {{ outputs }} is high quality for {{ inputs }}."),
    model="anthropic:/claude-opus-4-1-20250805",
    feedback_value_type=str,
)
```

### Registering a Scorer[​](#registering-a-scorer "Direct link to Registering a Scorer")

To register a judge to the experiment, call the `register` method on the judge instance.

python

```
# Register the judge
registered = quality_judge.register()
# You can pass experiment_id to register the judge to a specific experiment
# registered = quality_judge.register(experiment_id=experiment_id)
```

### Updating a Scorer[​](#updating-a-scorer "Direct link to Updating a Scorer")

Registering a new scorer with the same name will create a new version.

python

```
# Update and register a new version of the judge
quality_judge_v2 = make_judge(
    name="response_quality",  # Same name
    instructions=(
        "Evaluate if {{ outputs }} is high quality, accurate, and complete "
        "for the question in {{ inputs }}."
    ),
    model="anthropic:/claude-3.5-sonnet-20241022",  # Updated model
    feedback_value_type=str,
)

# Register the updated judge
registered_v2 = quality_judge_v2.register(experiment_id=experiment_id)
```

### Loading a Scorer[​](#loading-a-scorer "Direct link to Loading a Scorer")

To load a registered scorer, use the `get_scorer` function.

python

```
from mlflow.genai.scorers import get_scorer

# Get the latest version
latest_judge = get_scorer(name="response_quality")
# or specify experiment_id to get a scorer from a specific experiment
# latest_judge = get_scorer(name="response_quality", experiment_id=experiment_id)
```

### Listing Scorers[​](#listing-scorers "Direct link to Listing Scorers")

The `list_scorers` function returns a list of the scorers registered in the experiment.

python

```
from mlflow.genai.scorers import list_scorers

all_scorers = list_scorers(experiment_id=experiment_id)
for scorer in all_scorers:
    print(f"Scorer: {scorer.name}, Model: {scorer.model}")
```

## UI Support[​](#ui-support "Direct link to UI Support")

Coming soon!
