# Prompt Registry

**MLflow Prompt Registry** is a powerful tool that streamlines prompt engineering and management in your Generative AI (GenAI) applications. It enables you to version, track, and reuse prompts across your organization, helping maintain consistency and improving collaboration in prompt development.

## Key Benefits[​](#key-benefits "Direct link to Key Benefits")

#### Version Control

Track the evolution of your prompts with Git-inspired commit-based versioning and side-by-side comparison with diff highlighting. Prompt versions in MLflow are immutable, providing strong guarantees for reproducibility.

#### Aliasing

Build robust yet flexible deployment pipelines for prompts, allowing you to isolate prompt versions from main application code and perform tasks such as A/B testing and roll-backs with ease.

#### Lineage

Seamlessly integrate with MLflow's existing features such as model tracking and evaluation for end-to-end GenAI lifecycle management.

#### Collaboration

Share prompts across your organization with a centralized registry, enabling teams to build upon each other's work.

## Getting Started[​](#getting-started "Direct link to Getting Started")

### 1. Create a Prompt[​](#1-create-a-prompt "Direct link to 1. Create a Prompt")

* UI
* Python

![Create Prompt UI](/mlflow-website/docs/latest/assets/images/create-prompt-ui-03c88144e65d28eb7847b2ae5d8dd49a.png)

1. Run `mlflow ui` in your terminal to start the MLflow UI.
2. Navigate to the **Prompts** tab in the MLflow UI.
3. Click on the **Create Prompt** button.
4. Fill in the prompt details such as name, prompt template text, and commit message (optional).
5. Click **Create** to register the prompt.

note

Prompt template text can contain variables in `{{variable}}` format. These variables can be filled with dynamic content when using the prompt in your GenAI application. MLflow also provides the `to_single_brace_format()` API to convert templates into single brace format for frameworks like LangChain or LlamaIndex that require single brace interpolation.

To create a new prompt using the Python API, use [`mlflow.genai.register_prompt()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.register_prompt) API:

python

```
import mlflow

# Use double curly braces for variables in the template
initial_template = """\
Summarize content you are provided with in {{ num_sentences }} sentences.

Sentences: {{ sentences }}
"""

# Register a new prompt
prompt = mlflow.genai.register_prompt(
    name="summarization-prompt",
    template=initial_template,
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

This creates a new prompt with the specified template text and metadata. The prompt is now available in the MLflow UI for further management.

![Registered Prompt in UI](/mlflow-website/docs/latest/assets/images/registered-prompt-b8d47ff0d061d8703b61a9a6e94a77c3.png)

### 2. Update the Prompt with a New Version[​](#2-update-the-prompt-with-a-new-version "Direct link to 2. Update the Prompt with a New Version")

* UI
* Python

![Update Prompt UI](/mlflow-website/docs/latest/assets/images/update-prompt-ui-74a489e65098893bbffe253f43fb210d.png)

1. The previous step leads to the created prompt page. (If you closed the page, navigate to the **Prompts** tab in the MLflow UI and click on the prompt name.)
2. Click on the **Create prompt Version** button.
3. The popup dialog is pre-filled with the existing prompt text. Modify the prompt as you wish.
4. Click **Create** to register the new version.

To update an existing prompt with a new version, use the [`mlflow.genai.register_prompt()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.register_prompt) API with the existing prompt name:

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

### 3. Compare the Prompt Versions[​](#3-compare-the-prompt-versions "Direct link to 3. Compare the Prompt Versions")

Once you have multiple versions of a prompt, you can compare them to understand the changes between versions. To compare prompt versions in the MLflow UI, click on the **Compare** tab in the prompt details page:

![Compare Prompt
Versions](/mlflow-website/docs/latest/assets/images/compare-prompt-versions-2082121aeaca4be99a0cf968535141ed.png)

### 4. Load and Use the Prompt[​](#4-load-and-use-the-prompt "Direct link to 4. Load and Use the Prompt")

To use a prompt in your GenAI application, you can load it with the [`mlflow.genai.load_prompt()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.load_prompt) API and fill in the variables using the [`mlflow.entities.Prompt.format()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Prompt.format) method of the prompt object:

python

```
import mlflow
import openai

target_text = """
MLflow is an open source platform for managing the end-to-end machine learning lifecycle.
It tackles four primary functions in the ML lifecycle: Tracking experiments, packaging ML
code for reuse, managing and deploying models, and providing a central model registry.
MLflow currently offers these functions as four components: MLflow Tracking,
MLflow Projects, MLflow Models, and MLflow Registry.
"""

# Load the prompt
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt/2")

# Use the prompt with an LLM
client = openai.OpenAI()
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt.format(num_sentences=1, sentences=target_text),
        }
    ],
    model="gpt-4o-mini",
)

print(response.choices[0].message.content)
```

### 5. Search Prompts[​](#5-search-prompts "Direct link to 5. Search Prompts")

You can discover prompts by name, tag or other registry fields:

python

```
import mlflow

# Fluent API: returns a flat list of all matching prompts
prompts = mlflow.genai.search_prompts(filter_string="task='summarization'")
print(f"Found {len(prompts)} prompts")

# For pagination control, use the client API:
from mlflow.tracking import MlflowClient

client = MlflowClient()
all_prompts = []
token = None
while True:
    page = client.search_prompts(
        filter_string="task='summarization'",
        max_results=50,
        page_token=token,
    )
    all_prompts.extend(page)
    token = page.token
    if not token:
        break
print(f"Total prompts across pages: {len(all_prompts)}")
```

## Prompt Object[​](#prompt-object "Direct link to Prompt Object")

The `Prompt` object is the core entity in MLflow Prompt Registry. It represents a versioned template text that can contain variables for dynamic content.

Key attributes of a Prompt object:

* `Name`: A unique identifier for the prompt.

* `Template`: The content of the prompt, which can be either:

  <!-- -->

  * A string containing text with variables in `{{variable}}` format (text prompts)
  * A list of dictionaries representing chat messages with 'role' and 'content' keys (chat prompts)

* `Version`: A sequential number representing the revision of the prompt.

* `Commit Message`: A description of the changes made in the prompt version, similar to Git commit messages.

* `Tags`: Optional key-value pairs assigned at the prompt version for categorization and filtering. For example, you may add tags for project name, language, etc, which apply to all versions of the prompt.

* `Alias`: An mutable named reference to the prompt. For example, you can create an alias named `production` to refer to the version used in your production system. See [Aliases](/mlflow-website/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases.md) for more details.

* `is_text_prompt`: A boolean property indicating whether the prompt is a text prompt (True) or chat prompt (False).

* `response_format`: An optional property containing the expected response structure specification, which can be used to validate or structure outputs from LLM calls.

### Prompt Types[​](#prompt-types "Direct link to Prompt Types")

#### Text Prompts[​](#text-prompts "Direct link to Text Prompts")

Text prompts use a simple string template with variables enclosed in double curly braces:

python

```
text_template = "Hello {{ name }}, how are you today?"
```

#### Chat Prompts[​](#chat-prompts "Direct link to Chat Prompts")

Chat prompts use a list of message dictionaries, each with 'role' and 'content' keys:

python

```
chat_template = [
    {"role": "system", "content": "You are a helpful {{ style }} assistant."},
    {"role": "user", "content": "{{ question }}"},
]
```

### Response Format[​](#response-format "Direct link to Response Format")

The `response_format` property allows you to specify the expected structure of responses from LLM calls. This can be either a Pydantic model class or a dictionary defining the schema:

python

```
from pydantic import BaseModel


class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str]
    word_count: int


# Or as a dictionary
response_format_dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "word_count": {"type": "integer"},
    },
}
```

### Manage Prompt and Version Tags[​](#manage-prompt-and-version-tags "Direct link to Manage Prompt and Version Tags")

MLflow lets you modify and inspect tags after a prompt has been registered. Tags can be applied either at the prompt level or to individual prompt versions.

python

```
import mlflow

# Prompt-level tag operations
mlflow.genai.set_prompt_tag("summarization-prompt", "language", "en")
mlflow.genai.get_prompt_tags("summarization-prompt")
mlflow.genai.delete_prompt_tag("summarization-prompt", "language")

# Prompt-version tag operations
mlflow.genai.set_prompt_version_tag("summarization-prompt", 1, "author", "alice")
mlflow.genai.load_prompt("prompts:/summarization-prompt/1").tags
mlflow.genai.delete_prompt_version_tag("summarization-prompt", 1, "author")
```

## FAQ[​](#faq "Direct link to FAQ")

#### Q: How do I delete a prompt version?[​](#q-how-do-i-delete-a-prompt-version "Direct link to Q: How do I delete a prompt version?")

A: You can delete a prompt version using the MLflow UI or Python API:

python

```
import mlflow

# Delete a prompt version
client = mlflow.MlflowClient()
client.delete_prompt_version("summarization-prompt", version=2)
```

To avoid accidental deletion, you can only delete one version at a time via API. If you delete the all versions of a prompt, the prompt itself will be deleted.

#### Q: Can I update the prompt template of an existing prompt version?[​](#q-can-i-update-the-prompt-template-of-an-existing-prompt-version "Direct link to Q: Can I update the prompt template of an existing prompt version?")

A: No, prompt versions are immutable once created. To update a prompt, create a new version with the desired changes.

#### Q: How to dynamically load the latest version of a prompt?[​](#q-how-to-dynamically-load-the-latest-version-of-a-prompt "Direct link to Q: How to dynamically load the latest version of a prompt?")

A: You can load the latest version of a prompt by passing the name only, or using the `@latest` alias. This is the reserved alias name and MLflow will automatically find the latest available version of the prompt.

python

```
prompt = mlflow.genai.load_prompt("summarization-prompt")
# or
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt@latest")
```

#### Q: Can I use prompt templates with frameworks like LangChain or LlamaIndex?[​](#q-can-i-use-prompt-templates-with-frameworks-like-langchain-or-llamaindex "Direct link to Q: Can I use prompt templates with frameworks like LangChain or LlamaIndex?")

A: Yes, you can load prompts from MLflow and use them with any framework. For example, the following example demonstrates how to use a prompt registered in MLflow with LangChain. Also refer to [Logging Prompts with LangChain](/mlflow-website/docs/latest/genai/prompt-registry/log-with-model.md#example-1-logging-prompts-with-langchain) for more details.

python

```
import mlflow
from langchain.prompts import PromptTemplate

# Load prompt from MLflow
prompt = mlflow.genai.load_prompt("question_answering")

# Convert the prompt to single brace format for LangChain (MLflow uses double braces),
# using the `to_single_brace_format` method.
langchain_prompt = PromptTemplate.from_template(prompt.to_single_brace_format())
print(langchain_prompt.input_variables)
# Output: ['num_sentences', 'sentences']
```

#### Q: Is Prompt Registry integrated with the Prompt Engineering UI?[​](#q-is-prompt-registry-integrated-with-the-prompt-engineering-ui "Direct link to Q: Is Prompt Registry integrated with the Prompt Engineering UI?")

A. Direct integration between the Prompt Registry and the Prompt Engineering UI is coming soon. In the meantime, you can iterate on prompt template in the Prompt Engineering UI and register the final version in the Prompt Registry by manually copying the prompt template.
