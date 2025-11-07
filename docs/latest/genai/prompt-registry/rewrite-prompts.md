# Auto-rewrite Prompts for New Models (Experimental)

When migrating to a new language model, you often discover that your carefully crafted prompts don't work as well with the new model. MLflow's [`mlflow.genai.optimize_prompts()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.optimize_prompts) API helps you **automatically rewrite prompts** to maintain output quality when switching models, using your existing application's outputs as training data.

Key Benefits

* **Model Migration**: Seamlessly switch between language models while maintaining output consistency
* **Automatic Optimization**: Automatically rewrites prompts based on your existing data
* **No Ground Truth Requirement**: No human labeling is required if you optimize prompts based on the existing outputs
* **Trace-Aware**: Leverages MLflow tracing to understand prompt usage patterns
* **Flexible**: Works with any function that uses MLflow Prompt Registry

Version Requirements

The `optimize_prompts` API requires **MLflow >= 3.5.0**.

![Model Migration Workflow](data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTAwMCA0NTAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPCEtLSBEZWZpbmUgc3R5bGVzIC0tPgogIDxkZWZzPgogICAgPHN0eWxlPgogICAgICAuYm94IHsgcng6IDg7IHN0cm9rZS13aWR0aDogMjsgfQogICAgICAuaW5wdXQtYm94IHsgZmlsbDogI0UzRjJGRDsgc3Ryb2tlOiAjMTk3NkQyOyB9CiAgICAgIC5wcm9jZXNzLWJveCB7IGZpbGw6ICNGRkY5QzQ7IHN0cm9rZTogI0Y1N0MwMDsgfQogICAgICAub3V0cHV0LWJveCB7IGZpbGw6ICNDOEU2Qzk7IHN0cm9rZTogIzM4OEUzQzsgfQogICAgICAubW9kZWwtYm94IHsgZmlsbDogI0YzRTVGNTsgc3Ryb2tlOiAjN0IxRkEyOyB9CiAgICAgIC50ZXh0LXRpdGxlIHsgZm9udC1mYW1pbHk6IEFyaWFsLCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDE4cHg7IGZvbnQtd2VpZ2h0OiBib2xkOyBmaWxsOiAjMzMzOyB9CiAgICAgIC50ZXh0LWJvZHkgeyBmb250LWZhbWlseTogQXJpYWwsIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTRweDsgZmlsbDogIzU1NTsgfQogICAgICAudGV4dC1zbWFsbCB7IGZvbnQtZmFtaWx5OiBBcmlhbCwgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMnB4OyBmaWxsOiAjNjY2OyB9CiAgICAgIC5hcnJvdyB7IGZpbGw6IG5vbmU7IHN0cm9rZTogIzY2Njsgc3Ryb2tlLXdpZHRoOiAyLjU7IG1hcmtlci1lbmQ6IHVybCgjYXJyb3doZWFkKTsgfQogICAgICAubGFiZWwtYmcgeyBmaWxsOiB3aGl0ZTsgc3Ryb2tlOiAjOTk5OyBzdHJva2Utd2lkdGg6IDE7IHJ4OiA0OyB9CiAgICA8L3N0eWxlPgogICAgPG1hcmtlciBpZD0iYXJyb3doZWFkIiBtYXJrZXJXaWR0aD0iMTAiIG1hcmtlckhlaWdodD0iMTAiIHJlZlg9IjkiIHJlZlk9IjMiIG9yaWVudD0iYXV0byI+CiAgICAgIDxwb2x5Z29uIHBvaW50cz0iMCAwLCAxMCAzLCAwIDYiIGZpbGw9IiM2NjYiIC8+CiAgICA8L21hcmtlcj4KICA8L2RlZnM+CiAgCiAgPCEtLSBTdGFnZSAxOiBJbnB1dHMgLS0+CiAgPGcgaWQ9InN0YWdlMSI+CiAgICA8cmVjdCB4PSI1MCIgeT0iMTIwIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjEwMCIgY2xhc3M9ImJveCBpbnB1dC1ib3giLz4KICAgIDx0ZXh0IHg9IjE0MCIgeT0iMTU1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0idGV4dC10aXRsZSI+SW5wdXRzPC90ZXh0PgogICAgPHRleHQgeD0iMTQwIiB5PSIxODUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJ0ZXh0LXNtYWxsIj5DdXJyZW50IEFwcGxpY2F0aW9uICsgUHJvbXB0czwvdGV4dD4KICAgIDx0ZXh0IHg9IjE0MCIgeT0iMjA1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0idGV4dC1zbWFsbCI+KyBTYW1wbGUgRGF0YTwvdGV4dD4KICA8L2c+CiAgCiAgPCEtLSBBcnJvdyAxIC0tPgogIDxwYXRoIGQ9Ik0gMjMwIDE3MCBMIDI4MCAxNzAiIGNsYXNzPSJhcnJvdyIvPgogIDxyZWN0IHg9IjI0MCIgeT0iMTUwIiB3aWR0aD0iMzAiIGhlaWdodD0iMjUiIGNsYXNzPSJsYWJlbC1iZyIvPgogIDx0ZXh0IHg9IjI1NSIgeT0iMTY3IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0idGV4dC1zbWFsbCI+QnVpbGQ8L3RleHQ+CiAgCiAgPCEtLSBTdGFnZSAyOiBUcmFpbmluZyBEYXRhc2V0IC0tPgogIDxnIGlkPSJzdGFnZTIiPgogICAgPHJlY3QgeD0iMjgwIiB5PSIxMjAiIHdpZHRoPSIxODAiIGhlaWdodD0iMTAwIiBjbGFzcz0iYm94IHByb2Nlc3MtYm94Ii8+CiAgICA8dGV4dCB4PSIzNzAiIHk9IjE1NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgY2xhc3M9InRleHQtdGl0bGUiPlRyYWluaW5nPC90ZXh0PgogICAgPHRleHQgeD0iMzcwIiB5PSIxNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJ0ZXh0LXRpdGxlIj5EYXRhc2V0PC90ZXh0PgogICAgPHRleHQgeD0iMzcwIiB5PSIyMDAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJ0ZXh0LXNtYWxsIj5JbnB1dHMgKyBDdXJyZW50IE91dHB1dHM8L3RleHQ+CiAgPC9nPgogIAogIDwhLS0gQXJyb3cgMiAtLT4KICA8cGF0aCBkPSJNIDQ2MCAxNzAgTCA1MTAgMTcwIiBjbGFzcz0iYXJyb3ciLz4KICA8cmVjdCB4PSI0NzAiIHk9IjE1MCIgd2lkdGg9IjMwIiBoZWlnaHQ9IjI1IiBjbGFzcz0ibGFiZWwtYmciLz4KICA8dGV4dCB4PSI0ODUiIHk9IjE2NyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgY2xhc3M9InRleHQtc21hbGwiPlJ1bjwvdGV4dD4KICAKICA8IS0tIFN0YWdlIDM6IFByb21wdCBPcHRpbWl6YXRpb24gLS0+CiAgPGcgaWQ9InN0YWdlMyI+CiAgICA8cmVjdCB4PSI1MTAiIHk9IjEyMCIgd2lkdGg9IjE4MCIgaGVpZ2h0PSIxMDAiIGNsYXNzPSJib3ggcHJvY2Vzcy1ib3giLz4KICAgIDx0ZXh0IHg9IjYwMCIgeT0iMTU1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0idGV4dC10aXRsZSI+UHJvbXB0PC90ZXh0PgogICAgPHRleHQgeD0iNjAwIiB5PSIxNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJ0ZXh0LXRpdGxlIj5PcHRpbWl6YXRpb248L3RleHQ+CiAgICA8dGV4dCB4PSI2MDAiIHk9IjIwMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgY2xhc3M9InRleHQtc21hbGwiPk9wdGltaXplIHByb21wdHM8L3RleHQ+CiAgPC9nPgogIAogIDwhLS0gQXJyb3cgMyAtLT4KICA8cGF0aCBkPSJNIDY5MCAxNzAgTCA3NDAgMTcwIiBjbGFzcz0iYXJyb3ciLz4KICA8cmVjdCB4PSI3MDAiIHk9IjE1MCIgd2lkdGg9IjMwIiBoZWlnaHQ9IjI1IiBjbGFzcz0ibGFiZWwtYmciLz4KICA8dGV4dCB4PSI3MTUiIHk9IjE2NyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgY2xhc3M9InRleHQtc21hbGwiPkdldDwvdGV4dD4KICAKICA8IS0tIFN0YWdlIDQ6IE9wdGltaXplZCBQcm9tcHRzIC0tPgogIDxnIGlkPSJzdGFnZTQiPgogICAgPHJlY3QgeD0iNzQwIiB5PSIxMjAiIHdpZHRoPSIxODAiIGhlaWdodD0iMTAwIiBjbGFzcz0iYm94IG91dHB1dC1ib3giLz4KICAgIDx0ZXh0IHg9IjgzMCIgeT0iMTU1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0idGV4dC10aXRsZSI+T3B0aW1pemVkPC90ZXh0PgogICAgPHRleHQgeD0iODMwIiB5PSIxNzUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJ0ZXh0LXRpdGxlIj5Qcm9tcHRzPC90ZXh0PgogICAgPHRleHQgeD0iODMwIiB5PSIyMDAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJ0ZXh0LXNtYWxsIj5NaW5pbWFsIERyaWZ0PC90ZXh0PgogIDwvZz4KICAKICA8IS0tIE1vZGVsIGluZGljYXRvcnMgLS0+CiAgPGcgaWQ9ImN1cnJlbnQtbW9kZWwiPgogICAgPHJlY3QgeD0iNTAiIHk9IjI5MCIgd2lkdGg9IjE4MCIgaGVpZ2h0PSI4MCIgY2xhc3M9ImJveCBtb2RlbC1ib3giLz4KICAgIDxjaXJjbGUgY3g9Ijc1IiBjeT0iMzIwIiByPSIxMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjN0IxRkEyIiBzdHJva2Utd2lkdGg9IjIiLz4KICAgIDxjaXJjbGUgY3g9Ijc1IiBjeT0iMzIwIiByPSI4IiBmaWxsPSJub25lIiBzdHJva2U9IiM3QjFGQTIiIHN0cm9rZS13aWR0aD0iMiIvPgogICAgPGNpcmNsZSBjeD0iNzUiIGN5PSIzMjAiIHI9IjQiIGZpbGw9IiM3QjFGQTIiLz4KICAgIDx0ZXh0IHg9IjEwMCIgeT0iMzIwIiBjbGFzcz0idGV4dC1ib2R5IiBmb250LXdlaWdodD0iYm9sZCI+Q3VycmVudCBMTE08L3RleHQ+CiAgICA8dGV4dCB4PSIxMDAiIHk9IjM0MCIgY2xhc3M9InRleHQtc21hbGwiPihlLmcuLCBHUFQtNG8pPC90ZXh0PgogICAgPHRleHQgeD0iMTAwIiB5PSIzNjAiIGNsYXNzPSJ0ZXh0LXNtYWxsIiBmaWxsPSIjOTk5Ij5PcmlnaW5hbCBtb2RlbDwvdGV4dD4KICA8L2c+CiAgCiAgPGcgaWQ9Im5ldy1tb2RlbCI+CiAgICA8cmVjdCB4PSIyODAiIHk9IjI5MCIgd2lkdGg9IjE4MCIgaGVpZ2h0PSI4MCIgY2xhc3M9ImJveCBtb2RlbC1ib3giLz4KICAgIDxjaXJjbGUgY3g9IjMwNSIgY3k9IjMyMCIgcj0iMTIiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzdCMUZBMiIgc3Ryb2tlLXdpZHRoPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIzMDUiIGN5PSIzMjAiIHI9IjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzdCMUZBMiIgc3Ryb2tlLXdpZHRoPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIzMDUiIGN5PSIzMjAiIHI9IjQiIGZpbGw9IiM3QjFGQTIiLz4KICAgIDx0ZXh0IHg9IjMzMCIgeT0iMzIwIiBjbGFzcz0idGV4dC1ib2R5IiBmb250LXdlaWdodD0iYm9sZCI+VGFyZ2V0IExMTTwvdGV4dD4KICAgIDx0ZXh0IHg9IjMzMCIgeT0iMzQwIiBjbGFzcz0idGV4dC1zbWFsbCI+KGUuZy4sIEdQVC00by1taW5pKTwvdGV4dD4KICAgIDx0ZXh0IHg9IjMzMCIgeT0iMzYwIiBjbGFzcz0idGV4dC1zbWFsbCIgZmlsbD0iIzk5OSI+RGVzdGluYXRpb24gbW9kZWw8L3RleHQ+CiAgPC9nPgogIAogIDwhLS0gQXJyb3cgY29ubmVjdGluZyBjdXJyZW50IExMTSB0byBldmFsdWF0aW9uIGRhdGFzZXQgLS0+CiAgPHBhdGggZD0iTSAxNDAgMjkwIEwgMTQwIDI1MCBMIDM3MCAyNTAgTCAzNzAgMjIwIiBjbGFzcz0iYXJyb3ciIHN0cm9rZT0iIzdCMUZBMiIgc3Ryb2tlLXdpZHRoPSIzIi8+CiAgPHJlY3QgeD0iMjEwIiB5PSIyMzUiIHdpZHRoPSIxMDAiIGhlaWdodD0iMzAiIGNsYXNzPSJsYWJlbC1iZyIgZmlsbD0iI0YzRTVGNSIgc3Ryb2tlPSIjN0IxRkEyIi8+CiAgPHRleHQgeD0iMjYwIiB5PSIyNTUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJ0ZXh0LXNtYWxsIiBmaWxsPSIjN0IxRkEyIiBmb250LXdlaWdodD0iYm9sZCI+R2VuZXJhdGUgbGFiZWxzPC90ZXh0PgogIAogIDwhLS0gQXJyb3cgY29ubmVjdGluZyB0YXJnZXQgTExNIHRvIGFkYXB0YXRpb24gLS0+CiAgPHBhdGggZD0iTSAzNzAgMjkwIEwgMzcwIDI3MCBMIDYwMCAyNzAgTCA2MDAgMjIwIiBjbGFzcz0iYXJyb3ciIHN0cm9rZT0iIzdCMUZBMiIgc3Ryb2tlLXdpZHRoPSIzIi8+CiAgPHJlY3QgeD0iNDM1IiB5PSIyMzUiIHdpZHRoPSIxMDAiIGhlaWdodD0iMzAiIGNsYXNzPSJsYWJlbC1iZyIgZmlsbD0iI0YzRTVGNSIgc3Ryb2tlPSIjN0IxRkEyIi8+CiAgPHRleHQgeD0iNDg1IiB5PSIyNTUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJ0ZXh0LXNtYWxsIiBmaWxsPSIjN0IxRkEyIiBmb250LXdlaWdodD0iYm9sZCI+T3B0aW1pemUgZm9yPC90ZXh0Pgo8L3N2Zz4=)

### Example: Simple Prompt → Optimized Prompt[​](#example-simple-prompt--optimized-prompt "Direct link to Example: Simple Prompt → Optimized Prompt")

**Before Optimization:**

text

```
Classify the sentiment. Answer 'positive'
or 'negative' or 'neutral'.

Text: {{text}}
```

**After Optimization:**

text

```
Classify the sentiment of the provided text.
Your response must be one of the following:
- 'positive'
- 'negative'
- 'neutral'

Ensure your response is lowercase and contains
only one of these three words.

Text: {{text}}

Guidelines:
- 'positive': The text expresses satisfaction,
  happiness, or approval
- 'negative': The text expresses dissatisfaction,
  anger, or disapproval
- 'neutral': The text is factual or balanced
  without strong emotion

Your response must match this exact format with
no additional explanation.
```

## When to Use Prompt Rewriting[​](#when-to-use-prompt-rewriting "Direct link to When to Use Prompt Rewriting")

This approach is ideal when:

* **Downgrading Models**: Moving from `gpt-5` → `gpt-4o-mini` to reduce costs
* **Switching Providers**: Changing from OpenAI to Anthropic or vice versa
* **Performance Optimization**: Moving to faster models while maintaining quality
* **You Have Existing Outputs**: Your current system already produces good results

## Quick Start: Model Migration Workflow[​](#quick-start-model-migration-workflow "Direct link to Quick Start: Model Migration Workflow")

Here's a complete example of migrating from `gpt-5` to `gpt-4o-mini` while maintaining output consistency:

### Step 1: Capture Outputs from Original Model[​](#step-1-capture-outputs-from-original-model "Direct link to Step 1: Capture Outputs from Original Model")

First, collect outputs from your existing model using MLflow tracing:

python

```
import mlflow
import openai
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.datasets import create_dataset
from mlflow.genai.scorers import Equivalence

# Register your current prompt
prompt = mlflow.genai.register_prompt(
    name="sentiment",
    template="""Classify the sentiment. Answer 'positive' or 'negative' or 'neutral'.
Text: {{text}}""",
)


# Define your prediction function using the original model and base prompt
@mlflow.trace
def predict_fn_base_model(text: str) -> str:
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-5",  # Original model
        messages=[{"role": "user", "content": prompt.format(text=text)}],
    )
    return completion.choices[0].message.content.lower()


# Example inputs - each record contains an "inputs" dict with the function's input parameters
inputs = [
    {
        "inputs": {
            "text": "This movie was absolutely fantastic! I loved every minute of it."
        }
    },
    {"inputs": {"text": "The service was terrible and the food arrived cold."}},
    {"inputs": {"text": "It was okay, nothing special but not bad either."}},
    {
        "inputs": {
            "text": "I'm so disappointed with this purchase. Complete waste of money."
        }
    },
    {"inputs": {"text": "Best experience ever! Highly recommend to everyone."}},
    {"inputs": {"text": "The product works as described. No complaints."}},
    {"inputs": {"text": "I can't believe how amazing this turned out to be!"}},
    {"inputs": {"text": "Worst customer support I've ever dealt with."}},
    {"inputs": {"text": "It's fine for the price. Gets the job done."}},
    {"inputs": {"text": "This exceeded all my expectations. Truly wonderful!"}},
]

# Collect outputs from original model
with mlflow.start_run() as run:
    for record in inputs:
        predict_fn_base_model(**record["inputs"])
```

### Step 2: Create Training Dataset from Traces[​](#step-2-create-training-dataset-from-traces "Direct link to Step 2: Create Training Dataset from Traces")

Convert the traced outputs into a training dataset:

python

```
# Create dataset
dataset = create_dataset(name="sentiment_migration_dataset")

# Retrieve traces from the run
traces = mlflow.search_traces(return_type="list", run_id=run.info.run_id)

# Merge traces into dataset
dataset.merge_records(traces)
```

This automatically creates a dataset with:

* `inputs`: The input variables (`text` in this case)
* `outputs`: The actual outputs from your original model (`gpt-5`)

You can view the created dataset in the MLflow UI by navigating to:

1. **Experiments** tab → Select your experiment
2. **Evaluations** tab → Select the "Datasets" tab on the left sidebar
3. **Dataset** tab → Inspect the input/output pairs

The dataset view shows all the inputs and outputs collected from your traces, making it easy to verify the training data before optimization.

![](/mlflow-website/docs/latest/assets/images/evaluation_dataset_ui-b4a751f7446218d0f2e7d640f395517d.png)

### Step 3: Switch Model[​](#step-3-switch-model "Direct link to Step 3: Switch Model")

Switch your LM to the target model:

python

```
# Define function using target model
@mlflow.trace
def predict_fn(text: str) -> str:
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",  # Target model
        messages=[{"role": "user", "content": prompt.format(text=text)}],
        temperature=0,
    )
    return completion.choices[0].message.content.lower()
```

You might notice the target model doesn't follow the format as consistently as the original model.

### Step 4: Optimize Prompts for Target Model[​](#step-4-optimize-prompts-for-target-model "Direct link to Step 4: Optimize Prompts for Target Model")

Use the collected dataset to optimize prompts for the target model:

python

```
# Optimize prompts for the target model
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-5"),
    scorers=[Equivalence(model="openai:/gpt-5")],
)

# View the optimized prompt
optimized_prompt = result.optimized_prompts[0]
print(f"Optimized template: {optimized_prompt.template}")
```

The optimized prompt will include additional instructions to help `gpt-4o-mini` match the behavior of `gpt-5`:

text

```
Optimized template:
Classify the sentiment of the provided text. Your response must be one of the following:
- 'positive'
- 'negative'
- 'neutral'

Ensure your response is lowercase and contains only one of these three words.

Text: {{text}}

Guidelines:
- 'positive': The text expresses satisfaction, happiness, or approval
- 'negative': The text expresses dissatisfaction, anger, or disapproval
- 'neutral': The text is factual or balanced without strong emotion

Your response must match this exact format with no additional explanation.
```

### Step 5: Use Optimized Prompt[​](#step-5-use-optimized-prompt "Direct link to Step 5: Use Optimized Prompt")

Deploy the optimized prompt in your application:

python

```
# Load the optimized prompt
optimized = mlflow.genai.load_prompt(optimized_prompt.uri)


# Use in production
@mlflow.trace
def predict_fn_optimized(text: str) -> str:
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": optimized.format(text=text)}],
        temperature=0,
    )
    return completion.choices[0].message.content.lower()


# Test with new inputs
test_result = predict_fn_optimized("This product is amazing!")
print(test_result)  # Output: positive
```

## Best Practices[​](#best-practices "Direct link to Best Practices")

### 1. Collect Sufficient Data[​](#1-collect-sufficient-data "Direct link to 1. Collect Sufficient Data")

For best results, collect outputs from at least 20-50 diverse examples:

python

```
# ✅ Good: Diverse examples
inputs = [
    {"inputs": {"text": "Great product!"}},
    {
        "inputs": {
            "text": "The delivery was delayed by three days and the packaging was damaged. The product itself works fine but the experience was disappointing overall."
        }
    },
    {
        "inputs": {
            "text": "It meets the basic requirements. Nothing more, nothing less."
        }
    },
    # ... more varied examples
]

# ❌ Poor: Too few, too similar
inputs = [
    {"inputs": {"text": "Good"}},
    {"inputs": {"text": "Bad"}},
]
```

### 2. Use Representative Examples[​](#2-use-representative-examples "Direct link to 2. Use Representative Examples")

Include edge cases and challenging inputs:

python

```
inputs = [
    {"inputs": {"text": "Absolutely fantastic!"}},  # Clear positive
    {"inputs": {"text": "It's not bad, I guess."}},  # Ambiguous
    {"inputs": {"text": "The food was good but service terrible."}},  # Mixed sentiment
]
```

### 3. Verify Results[​](#3-verify-results "Direct link to 3. Verify Results")

Always test optimized prompts using [`mlflow.genai.evaluate()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate) before production deployment.

python

```
# Evaluate optimized prompt
results = mlflow.genai.evaluate(
    data=test_dataset,
    predict_fn=predict_fn_optimized,
    scorers=[accuracy_scorer, format_scorer],
)

print(f"Accuracy: {results.metrics['accuracy']}")
print(f"Format compliance: {results.metrics['format_scorer']}")
```

## See Also[​](#see-also "Direct link to See Also")

* [Optimize Prompts](/mlflow-website/docs/latest/genai/prompt-registry/optimize-prompts.md): General prompt optimization guide
* [Create and Edit Prompts](/mlflow-website/docs/latest/genai/prompt-registry/create-and-edit-prompts.md): Prompt Registry basics
* [Evaluate Prompts](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/prompts.md): Evaluate prompt performance
* [MLflow Tracing](/mlflow-website/docs/latest/genai/tracing.md): Understanding MLflow tracing
