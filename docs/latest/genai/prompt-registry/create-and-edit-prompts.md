# Create and Edit Prompts

This guide walks you through the process of creating new prompts and managing their versions within the MLflow Prompt Registry.

## Creating a New Prompt[​](#creating-a-new-prompt "Direct link to Creating a New Prompt")

You can initiate a new prompt in the MLflow Prompt Registry in two primary ways: through the MLflow UI or programmatically using the Python SDK.

* UI
* Python

1. Navigate to the Prompt Registry section in your MLflow instance.
2. Click on the "Create Prompt" (or similar) button.
3. Fill in the prompt details such as name, prompt template text, and commit message (optional)

![Registered Prompt in UI](/mlflow-website/docs/latest/assets/images/registered-prompt-b8d47ff0d061d8703b61a9a6e94a77c3.png)

To create a new prompt programmatically, use the [`mlflow.genai.register_prompt()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.register_prompt) function. This is particularly useful for automating prompt creation or managing prompts as part of a larger script.

python

```
import mlflow

# Use double curly braces for variables in the template
initial_template = """\
Summarize content you are provided with in {{ num_sentences }} sentences.

Sentences: {{ sentences }}
"""

# Chat Style template
initial_template = [
    {
        "role": "system",
        "content": "Summarize content you are provided with in {{ num_sentences }} sentences.",
    },
    {"role": "user", "content": "Sentences: {{ sentences }}"},
]

# Optional Response Format
from pydantic import BaseModel, Field


class ResponseFormat:
    summary: str = Field(..., description="Summary of the content")


# Register a new prompt
prompt = mlflow.genai.register_prompt(
    name="summarization-prompt",
    template=initial_template,
    # Optional: Provide Response Format to get structured output
    response_format=ResponseFormat,
    # Optional: Provide a commit message to describe the changes
    commit_message="Initial commit",
    # Optional: Set tags applies to the prompt (across versions)
    tags={
        "author": "author@example.com",
        "task": "summarization",
        "language": "en",
    },
)

# The prompt object contains information about the registered prompt
print(f"Created prompt '{prompt.name}' (version {prompt.version})")
```

## Editing an Existing Prompt (Creating New Versions)[​](#editing-an-existing-prompt-creating-new-versions "Direct link to Editing an Existing Prompt (Creating New Versions)")

Once a prompt version is created, its template and initial metadata are **immutable**. Editing an existing prompt means creating a *new version* of that prompt with your changes. This Git-like behavior ensures a complete history and allows you to revert to previous versions if needed.

* UI
* Python

1. Navigate to the specific prompt you wish to edit in the Prompt Registry.
2. Select the version you want to base your new version on (often the latest).
3. Look for an "Edit Prompt" or "Create New Version" button.
4. Modify the template, update metadata, or change tags as needed.
5. Provide a new **Commit Message** describing the changes you made for this new version.

![Update Prompt UI](/mlflow-website/docs/latest/assets/images/update-prompt-ui-74a489e65098893bbffe253f43fb210d.png)

To create a new version of an existing prompt, you again use the [`mlflow.genai.register_prompt()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.register_prompt) function, but this time, you provide the `name` of an existing prompt. MLflow will automatically increment the version number.

python

```
import mlflow

new_template = """\
You are an expert summarizer. Condense the following content into exactly {{ num_sentences }} clear and informative sentences that capture the key points.

Sentences: {{ sentences }}

Your summary should:
- Contain exactly {{ num_sentences }} sentences
- Include only the most important information
- Be written in a neutral, objective tone
- Maintain the same level of formality as the original text
"""

# Register a new version of an existing prompt
updated_prompt = mlflow.genai.register_prompt(
    name="summarization-prompt",  # Specify the existing prompt name
    template=new_template,
    commit_message="Improvement",
    tags={
        "author": "author@example.com",
    },
)
```

## Understanding Immutability[​](#understanding-immutability "Direct link to Understanding Immutability")

It's crucial to remember that prompt versions in the MLflow Prompt Registry are immutable. Once `mlflow.genai.register_prompt()` is called and a version is created (or a new version of an existing prompt is made), the template, initial commit message, and initial metadata for *that specific version* cannot be altered. This design choice provides strong guarantees for reproducibility and lineage tracking.

If you need to change a prompt, you always create a new version.

## Comparing Prompt Versions[​](#comparing-prompt-versions "Direct link to Comparing Prompt Versions")

The MLflow UI provides tools to compare different versions of a prompt. This typically includes a side-by-side diff view, allowing you to easily see what changed in the template text, metadata, or tags between versions.

![Compare Prompt Versions](/mlflow-website/docs/latest/assets/images/compare-prompt-versions-2082121aeaca4be99a0cf968535141ed.png)
