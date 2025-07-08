---
title: "Leveraging MLflow Prompt Registry and Evaluate for LLM-based OCR"
slug: mlflow-prompt-evaluate
tags: [mlflow, genai, evaluation, prompt-registry, tracing]
authors: [allison-bennett, shyam-sankararaman, michael-berk, mlflow-maintainers]
thumbnail: /img/blog/prompt-evaluate/prompt-evaluate.png
---

Building GenAI tools presents a unique set of challenges. As we evaluate accuracy, iterate on prompts, and enable collaboration, we often encounter bottlenecks that slow down our progress toward production. In this blog, we explore how MLflow's GenAI capabilities, namely MLflow Prompt Registry and MLflow Evaluate, help us streamline development and unlock value for both technical and non-technical contributors. 

In this example, we will build an LLM-based tool that applies Optical Character Recognition (OCR) to scanned documents, demonstrating how these MLflow features support each step.

## What is OCR?

Optical Character Recognition, or OCR, is the process of extracting text from scanned documents and images. The resulting text is machine-encoded, editable, and searchable, unlocking a wide range of downstream use cases.

Here, we leverage multi-modal LLMs to extract formatted text from various forms. Unlike traditional OCR tools such as PyTesseract, LLM-based methods offer greater flexibility for complex layouts, handwritten content, and context-aware extraction.  While these methods may require more computational resources and careful prompt engineering, they provide significant advantages for advanced use cases.

Fun fact: The earliest form of OCR, the Optophone, was introduced in 1914 to help blind individuals read printed text without Braille.
![Optophone](/img/blog/prompt-evaluate/margaret-hogan-used-the-black-sounding-optophone-to-read-a-book.jpg)

## The Challenge with Prompting

Prompt engineering requires both analytical thinking and creativity to maximize the potential of our GenAI applications. To refine, iterate, and automate our process, we need technical expertise and significant additional effort. How do we enable prompt engineers to focus on crafting effective prompts, rather than on maintenance?

Verification and assessment of outputs also present challenges. Does the response provide the appropriate context? Does it introduce bias? We need a robust way to iterate on prompts and assess their accuracy to build production-ready GenAI applications.

## MLflow Key Capabilities

### MLflow Prompt Registry

[MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry) addresses many of the challenges prompt engineers face today. With a Git-inspired setup, prompt engineers have a streamlined way to version, track, and reuse prompts.  

A typical workflow looks like this: 

**1. Create and Register Your Prompt:** A _prompt object_ is a versioned, parameterized template that can be dynamically filled at runtime. The object includes metadata such as name, version #, and aliases.  
Register your prompt using -  
`mlflow.genai.register_prompt()`

**2. Manage Aliases:** An _alias_ is a label that ensures prompts are aligned with their intended use. For example, "Production" and "Staging" aliases tie prompts to their respective environments, enabling robust deployment pipelines and supporting continuous delivery (CD) practices. This is similar to model aliasing, allowing seamless updates and rollbacks without disrupting production workflows.

**3. Load and Use Prompts:** A _loaded prompt_ is a prompt object retrieved by name, version number, or alias. Once loaded, we use it in GenAI applications such as our LLM-backed OCR tool.  
Load your prompt using -  
`mlflow.genai.load_prompt()`

**4. Iterate on Prompts:** _Prompt iteration_ is the process of improving and updating prompts within a centralized repository. This enables us to continuously refine accuracy while maintaining organized version control. Because prompts are registered and versioned, we can make updates asynchronously and safely, supporting agile and collaborative workflows. 

Prompt registry enables non-technical subject matter experts (SMEs) to contribute directly to prompt optimization. By decoupling prompt engineering from the more technical aspects of GenAI development, organizations can outsource prompt iteration and improvement to SMEs, reducing costs and accelerating development.  

### MLflow Evaluate

[MLflow Evaluate](https://mlflow.org/docs/latest/genai/eval-monitor/) is essential for systematically assessing and improving both traditional ML models and LLMs using [built-in and custom metrics](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html).

**Built-in Metrics:** Automatically computed based on model type, including NLP metrics such as BLEU and ROUGE, as well as LLM-judged metrics for accuracy, relevance, and toxicity.

**Custom Metrics:** Define user-specific evaluation functions to validate outputs based on criteria such as formatting, compliance, or business logic. These metrics are particularly valuable for GenAI and agentic workflows, where standard metrics may not capture the full complexity of our use case.   

We pass custom metrics into the `mlflow.evaluate` API to enable targeted, meaningful evaluations that ensure our model meets specific requirements and quality thresholds for production.  All results are logged to an MLflow run, accessible through the MLflow UI or APIs.  

Let’s see MLflow Prompt Registry and MLflow Evaluate in action in our OCR use case.   

## Our Use Case

Our task is to create a document parsing tool for text extraction (OCR) using LLMs. The data consists of scanned documents and their corresponding text extracted as JSON. We use the [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/), which contains around 200 fully annotated forms, structured as semantic entity labels and word groupings.  

Example: 

![Annotated form example](/img/blog/prompt-evaluate/word_grouping_semantic_entity_labeling.png)

All of these annotations are encoded as JSON files like below.

```
{
        "form": [
        {
            "id": 0,
            "text": "Registration No.",
            "box": [94,169,191,186],
            "linking": [
                [0,1]
            ],
            "label": "question",
            "words": [
                {
                    "text": "Registration",
                    "box": [94,169,168,186]
                },
                {
                    "text": "No.",
                    "box": [170,169,191,183]
                }
            ]
        },
        {
            "id": 1,
            "text": "533",
            "box": [209,169,236,182],
            "label": "answer",
            "words": [
                {
                    "box": [209,169,236,182
                    ],
                    "text": "533"
                }
            ],
            "linking": [
                [0,1]
            ]
        }
    ]
    }
```
For a detailed description of each entry, refer to the [original paper](https://arxiv.org/pdf/1905.13538.pdf). 

## Setting it Up

Install the required packages:  

```bash
pip install openai mlflow tiktoken aiohttp
```

**openai:** For interacting with OpenAI models and APIs   
**mlflow:** For experiment tracking, model management, and GenAI workflow tools   
**tiktoken:** For efficient tokenization, especially useful with OpenAI models   
**aiohttp:** For asynchronous HTTP requests, enabling efficient API calls   

For this tutorial we use OpenAI, but the approach extends to other LLM providers.


```python
import os
from utils import _set_openai_api_key_for_demo

if (not _set_openai_api_key_for_demo()) and (not os.getenv("OPENAI_API_KEY")):
    os.environ["OPENAI_API_KEY"] = getpass("Your OpenAI API Key: ")
```

#### 1. Observe Data

Let's read a randomly selected annotated file.

```python
DATA_DIRECTORY = "./data" # for simplicity, a local directory
ANNOTATIONS_DIRECTORY = os.path.join(DATA_DIRECTORY, "annotations")

def _extract_qa_pairs(data: dict) -> dict:
    """Extracts question-answer pairs from OCR-style linked data."""
    qa_pairs = {}

    elements = {item["id"]: item for item in data}
    for item in data:
        if item["label"] != "question" or not item["linking"]:
            continue
        for q_id, a_id in item["linking"]:
            if q_id != item["id"]:
                continue
            answer = elements.get(a_id)
            if answer and answer["label"] == "answer":
                q_text = " ".join(w["text"] for w in item["words"])
                a_text = " ".join(w["text"] for w in answer["words"])
                qa_pairs[q_text] = a_text
    return qa_pairs

def get_json(file_name: str, directory: str = ANNOTATIONS_DIRECTORY) -> dict:
    file_name += ".json" if not file_name.endswith(".json") else file_name
    path = os.path.join(directory, file_name)
    with open(path, "r", encoding="utf-8") as f:
        contents = json.load(f)["form"]
        if isinstance(contents, list):
            if all(isinstance(page, list) for page in contents):
                flat_items = [item for page in contents for item in page]
            else:
                flat_items = contents
            return _extract_qa_pairs(flat_items)
        else:
            return _extract_qa_pairs(contents)

```

The `_extract_qa_pairs` function creates a lookup dictionary of question-answer pairs by identifying "question" items linked to their corresponding "answer" items. This structure allows us to easily retrieve the OCR-identified question/answer pairs from the form image data.

```python

from utils import get_image, get_json, get_random_files
from IPython.display import Image, display

random_file = get_random_files()
image_bytes = get_image(random_file)
get_json(random_file)

```

#### 2. Example LLM Call

Before we make a call to OpenAI, we need to set up [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking) and create an experiment. MLflow Tracking allows us to log, organize, and visualize experiment results. This makes our result set more ingestible and auditable.  

We can automatically log all API calls using `mlflow.openai.autolog()`. [Autolog](https://mlflow.org/docs/latest/ml/tracking/autolog) simplifies the tracking process by reducing the need for manual log statements. Metrics, params, artifacts, and other useful information is logged automatically.

```python
import mlflow

mlflow.set_tracking_uri(os.getcwd() + "/mlruns")
mlflow.set_experiment("quickstart")
mlflow.openai.autolog()
```

On the OpenAI side, initialize the client and provide the prompt to the LLM, instructing it to act as an OCR expert.

```python
client = OpenAI()

system_prompt = """You are an expert at Optical Character Recognition (OCR). Extract the questions and answers from the image."""
base64_image = get_image(random_file, encode_as_str=True)
```

The `get_image()` helper function reads the image file from a specified path, resizes it, and converts it into a JPEG of specified quality. 

```python
import base64
from PIL import Image
from io import BytesIO

def _compress_image(file_path: str, quality: int = 40, max_size: tuple[int, int] = (1000, 1000)) -> bytes:
    """Compresses an image by resizing and converting to JPEG with given quality."""
    with Image.open(file_path) as img:
        img = img.convert("RGB")  # Ensure JPEG compatibility
        img.thumbnail(max_size)  # Resize
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def get_image(
    file_name: str, directory: str = IMAGES_DIRECTORY, encode_as_str: bool = False
) -> bytes:
    file_name += ".png" if not file_name.endswith(".png") else file_name
    path = os.path.join(directory, file_name)
    with open(path, "rb") as f:
        file = f.read()
        compressed = _compress_image(path)

        if encode_as_str:
            return base64.b64encode(compressed).decode("utf-8")
        else:
            return compressed
```

#### 3. Tracing UI

[MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/) provides end-to-end observability for GenAI workflows. We gain a comprehensive view of each step in our GenAI pipeline from prompt construction and model inference to tool calls and final outputs. This level of detail allows us to diagnose issues, optimize performance, and ensure reproducibility across experiments. 

![MLflow UI showing the tracing](/img/blog/prompt-evaluate/TracingUI.png)

### Create Model and Evaluate

In this section, we define the system prompt and extract the contents of the images into lists of "questions" and "answers" using an LLM. These are tracked as MLflow experiment runs.

```python
system_prompt = """You are an expert at Optical Character Recognition (OCR). Extract the questions and answers from the image."""

def get_completion(inputs: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "what's in this image?" },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{inputs}",
                        },
                    },
                ],
            }
        ],
    )

    return completion.choices[0].message.content

with mlflow.start_run() as run:
    predicted = get_completion(images[0])
    print(predicted)

```

[WIP] Add a screenshot for the logged runs of the LLM completion

#### 1. Defining a Custom Gen AI Metric for Evaluation

You can create a custom LLM-as-a-judge metric within MLflow using [`mlflow.metrics.genai.make_genai_metric()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.genai.make_genai_metric/). For simplicity, we define a custom metric called `correct_format`, which returns a boolean value depending on whether the output contains a list of dicts with the required keys "question" and "answer". The `definition` parameter explains the metric, and `grading_prompt` sets the grading criteria for the LLM.

```python
correct_format = mlflow.metrics.genai.make_genai_metric(
    name="correct_format",
    definition=(
        """The answer is a list of dicts where keys are `question` and `answer`."""
    ),
    grading_prompt=(
        """If formatted correctly, return 1. Otherwise, return 0."""
    ),
    model="openai:/gpt-4o-mini",
    greater_is_better=True,
)
```

#### 2. MLflow Evaluate

Pass the defined metric `correct_format` to `mlflow.evaluate()`, which evaluates each selected image. The JSONs created from the OCR extraction using `get_json()` serve as ground truth. 

```python
def batch_completion(df: pd.DataFrame) -> list[str]:
    return [get_completion(image) for image in df["inputs"]]

eval_result = mlflow.evaluate(
    model=batch_completion,
    data=pd.DataFrame({"inputs": images, "truth": jsons}),
    targets="truth",
    model_type="text",
    extra_metrics=[correct_format],
)
```

[WIP] - Add a screenshot of the MLflow UI + explanation

### Applying Prompt Registry and Evaluate to the OCR Use Case

Suppose you are developing an OCR application with different roles contributing to the project. Here's how a prompt engineer and a ML engineer benefit from these MLflow capabilities.

#### Prompt Engineer: Improve the Prompt

As prompt engineers, we iteratively modify prompts. `mlflow.genai.register_prompt()` allows us to version different prompts or system messages, set a template text, and include metadata such as author and project. We can compare different prompt versions and search for specific prompts from the past.

Here's an example of a prompt template, specifically instructing the LLM to generate results in our expected format. 

````python
new_template = """\
You are an expert at key information extraction and OCR.

Format as a list of dictionaries as shown below. They keys should only be `question` and `answer`.

```\
[
    {
        "question": "question field",
        "answer": "answer to question field"

    },
...
]
```\

Question refers to a field in the form that takes in information. Answer refers to the information
that is filled in the field.

Follow these rules:
- Only use the information present in the text.
{{ additional_rules }}
"""
````

```python
# Register a new version of an existing prompt
updated_prompt = mlflow.register_prompt(
    name="ocr-question-answer",
    template=new_template,
    version_metadata={
        "author": "author@example.com",
    },
)

```

#### ML Engineer: Use the Prompt

As ML engineers, we ensure that the OCR application using the LLM is robustly evaluated against ground truth. In production, we typically leverage multiple metrics to properly describe our system's accuracy. For simplicity, this tutorial uses one.

Follow the steps in [Defining a Gen AI metric](#1-defining-a-custom-gen-ai-metric-for-evaluation) and [MLflow Evaluate](#2-mlflow-evaluate) to use `mlflow.metrics.genai.make_genai_metric()` and `mlflow.evaluate()`

## Conclusion and Next Steps

By leveraging MLflow GenAI capabilities, we efficiently manage prompts and evaluate models for our OCR tool. These features enable both technical and non-technical contributors to collaborate, iterate, and deploy AI solutions confidently.

We can take several directions to further enhance our workflow and outcomes:  

**Expand Your Custom Metrics:** Scale out your custom evaluation metrics to more accurately capture the requirements of our specific OCR problem. This allows us to measure what truly matters for the use case, such as domain-specific accuracy, formatting compliance, or business logic adherence.  

**Experiment with Multiple LLMs:** Take advantage of MLflow’s ability to track and compare experiments by iterating with different LLMs. We can view and analyze results side-by-side in the MLflow UI, making it easier to identify which model best fits our needs and to justify model selection with clear, data-driven evidence. 

**Utilize Tracing and Model Logging:** Leverage MLflow’s tracing and model logging features to gain end-to-end visibility into our GenAI workflows. By capturing detailed traces and logs, we can iteratively refine our models and prompts, diagnose issues, and ensure reproducibility—all within the context of our custom metrics.  

**Expand Governance and Access Control**: Implement robust governance practices to ensure secure, compliant, and auditable management of our GenAI assets and workflows. This is especially important for scaling in enterprise or regulated environments.  

These are just a few of the many ways we can build on this solution. Whether we are aiming to improve model performance, streamline collaboration, or scale our solution to new domains, these MLflow capabilities support us in our GenAI development.


## Further Reading

[Practical AI Observability: Getting Started with MLflow Tracing](https://mlflow.org/blog/ai-observability-mlflow-tracing)  
[Beyond Autolog: Add MLflow Tracing to a New LLM Provider](https://mlflow.org/blog/custom-tracing)  
[LLM as Judge](https://mlflow.org/blog/llm-as-judge)  

