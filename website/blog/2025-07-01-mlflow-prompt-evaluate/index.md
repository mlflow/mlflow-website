---
title: "Leveraging MLflow Prompt Registry and Evaluate for LLM-based OCR"
slug: mlflow-prompt-evaluate
tags: [mlflow, genai, evaluation, prompt-registry, tracing]
authors: [allison-bennett, shyam-sankararaman, michael-berk, mlflow-maintainers]
thumbnail: /img/blog/prompt-evaluate/prompt-evaluate.png
---

Building GenAI tools comes with a unique set of challenges. Evaluating accuracy, iterating on prompts, enabling collaboration and more can slow down your prompt and ML engineers when trying to get these tools to production. This blog will showcase how various MLflow capabilities built for GenAI, namely MLflow Prompt Registry and MLflow Evaluate, can streamline this development.

In this example, we will build an LLM-based tool that will apply OCR to scanned documents.

## What is OCR?

Optical Character Recognition, or OCR, is the process of extracting text from scanned documents and images. This outputted text is machine-encoded, editable, and searchable - opening up a variety of downstream use cases.

There are a few types of OCR, ranging from Simple to Intelligent. Today we will be focusing on the latter. This incorporates AI and ML for increased capabilities like handwriting recognition and adaptation to new styles.

Fun fact - the earliest form of OCR was introduced in 1914. It was a device designed for blind indivuduals to read printed text without Braille called the Optophone.
![Optophone](/img/blog/prompt-evaluate/margaret-hogan-used-the-black-sounding-optophone-to-read-a-book.jpg)

## The Challenge with Prompting

A good prompt engineer knows how to combine analytical thinking with creativity in order to get the most out of their GenAI applications. This is all while ensuring alignment with the high level goals of an organization. To refine, iterate, and automate their process, they often need the technical skills, or simply the additional effort, to do so. How can we allow prompt engineers to focus on the crafting of prompts, and less on the maintenance?

There are also an abundance of challenges related verifying and assessing the expected output. Did the prompt address the complexity of the situation? Did it have the appropriate context? Too much context? Did it introduce bias?

Having a robust way to iterate on prompts and assess their accuracy becomes paramount in building production-ready GenAI applications.

## MLflow Key Capabilities

### MLflow Prompt Registry

[MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry) addresses many of the challenges prompt engineers face today. With a Git-inspired set up, prompt engineers have a streamlined way to version, track, and reuse prompts. Let's look at a typical flow.

**1. Create and Register Your Prompt:** Each prompt is registered as a **Prompt Object**. This becomes a versioned, paramaterized template that can be dynamically filled at runtime. The object will include a **name** (unique identifier), **template text**, **version #**, **commit message**, **metadata/tags** (optional), and **aliases**.  
Register your prompt with -
`mlflow.genai.register_prompt()`

**2. Manage Aliases:** Aliases ensure prompts remain aligned with their intended use. For example, "Production" and "Staging" aliases tie them to their respective environments, simplifying traceability, orchestration, and governance of our prompts.

**3. Load and Use Prompts:** We are now ready to load in our prompt and start using it with our GenAI applications, like our LLM-backed OCR tool.  
Load in your prompt with -
`mlflow.genai.load_prompt()`

**4. Iterate on Prompts:** An important step in our process is improving and updating prompts. This can be done any number of times while moving toward the accuracy required, all while staying organized in a centralized repository for prompt management.  
<br/>

### MLflow Evaluate

[MLflow Evaluate](https://mlflow.org/docs/latest/genai/eval-monitor/) is key in our example today as we iterate on and improve our OCR tool. Evaluation techniques within MLflow allow us to systematically asses the performance of both traditional ML models and LLMs by incorporating built-in and custom metrics in our model iteration.

[Built-in metrics](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html) are a great way to get started. They are automatically computed based on your model type. The exact set will depend on your configurations, but for LLM and agent evaluation we are able to follow traditional NLP metrics like BLEU and ROUGE, and LLM-judged metrics that will assess outputs like accuracy, relevance, toxicity, and more.

Custom metrics can enable additional validations for traditional models, but really come in handy for our GenAI and agentic workflows. Custom metric functions are passed into the `mlflow.evaluate` API. The function signature is flexible, allowing us to pass in things like request, response, expected_response, guidelines, trace, tool_calls, etc. This enables you to meet evaluation criteria and scoring thresholds to ensure the model is up to production standards.

Results are all logged in the MLflow run, which can be accessed through the MLflow UI or programatically.

We will see MLflow Evaluate in action as we walk through our OCR use case.

### Additional Concepts

While not central to this example, there are a few other MLflow functionalities that are critical to our GenAI application development lifecycle.

**Tracking:**
[MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking) allows you to log, organize, and visualize experiment results. This makes our result set more ingestable and auditable.

**Tracing:**
[MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing) takes tracking one step further in order to have end-to-end observability for GenAI workflows. Again, this keeps everything organized as we work through and iterate on our workflows.

**Autolog**
Another extension of tracking is [automatic logging](https://mlflow.org/docs/latest/ml/tracking/autolog). This will simplify the tracking process by reducing the need for manual log statements. Metrics, params, artifacts, and other useful information will be logged automatically.

See [Further Reading](#further-reading) for additional blogs digging into these concepts.

## Our Use Case

Our data consists of scanned documents and their corresponding text extracted as JSON. Like mentioned, our goal is to utilize LLMs to build a tool that will handle the text extraction (OCR) for us.

## Setting it Up

We require the following packages

```
openai
mlflow
tiktoken
aiohttp
```

which can be installed in the requirement.txt file:

```bash
pip install -r requirements.txt -qU
```

<br/>
Since we will need to make calls to OpenAI's LLMs, we can utilize helper functions to read the OpenAI API key that is stored in a (known) secure location. We can then set the environment variable ```OPENAI_API_KEY```.

```python
import os

def _set_openai_api_key_for_demo() -> bool:
    """Dummy method for doing easy auth in a notebook environment during the live demo."""
    try:
        with open("/XXXX/openai_api_key.txt", "r") as f: #replace this with your actual directory
            os.environ["OPENAI_API_KEY"] = f.read().strip()
            return True
    except Exception:
        pass

    return False
```

Alternatively, we can prompt the user to type the OpenAI API key without echoing using `getpass()`:

```python
import os
from utils import _set_openai_api_key_for_demo

if (not _set_openai_api_key_for_demo()) and (not os.getenv("OPENAI_API_KEY")):
    os.environ["OPENAI_API_KEY"] = getpass("Your OpenAI API Key: ")
```

### Explore Our Data

We are going to use the [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/) for this exercise, which contains around 200 fully annotated forms, in the form of semantic entity labels and word groupings as shown in the example below:

    ![Annotated form example](/img/blog/prompt-evaluate/word_grouping_semantic_entity_labeling.png)

All of these annotations are encoded as JSON files like below (for detailed description of each entry, refer to the [original paper](https://arxiv.org/pdf/1905.13538.pdf)):

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

#### 1. Observe Data

Let's try to read a random annotated file.

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

What does _\_extract_qa_pairs_ do here? Let's break down it down for a second:

- We create a look-up dictionary _qa_pairs_
- We identify "question" items that have linked answer pair(s) in the form of _(q_id, a_id)_
- Construct question-answer pairs by joining the individual words  
  <br/>
  Subsequently, we can call _get_json_ to get the OCR-identified question/answer structure based on the form image data.

```python

from utils import get_image, get_json, get_random_files
from IPython.display import Image, display

random_file = get_random_files()
image_bytes = get_image(random_file)
get_json(random_file)

```

#### 2. Example LLM Call

Before we make a call to OpenAI, we need to set up MLflow tracking and create and experiment. We can then automatically log all the API calls we make using `autolog()`

```python
import mlflow

mlflow.set_tracking_uri(os.getcwd() + "/mlruns")
mlflow.set_experiment("quickstart")
mlflow.openai.autolog()
```

Our next step will be to set mlruns as the local directory for MLflow tracking. We will create an experiment called quickstart and log the API calls for metrics like etc.

On the OpenAI side, we initialize the OpenAI client and provide the prompt to the LLM (for example, instructing the LLM to act as an OCR expert).

```python
client = OpenAI()

system_prompt = """You are an expert at Optical Character Recognition (OCR). Extract the questions and answers from the image."""
base64_image = get_image(random_file, encode_as_str=True)
```

_get_image()_ can be a neat helper function that can handle

        Reading the image file from a specified path
        Resizing the file and convert it into a JPEG of a specified quality

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

#### 3. Tracing UI [WIP]

    ![MLflow UI showing the tracing](/img/blog/prompt-evaluate/TracingUI.png)

### Create Model and Evaluate

In this section, you'll be defining the system prompt, along with extracting the contents of the images which are structured into "questions" and "answers" using an LLM. You'll then track these as MLflow experiment runs,

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

#### 1. Defining a custom Gen AI metric for evaluation

You can create a custom LLM-as-a-judge metric within MLflow using [mlflow.metrics.genai.make_genai_metric()](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html#mlflow.metrics.genai.make_genai_metric/). For simplicity, we define a (custom) metric called _correct_format_, which returns a boolean value depending on whether the output from the LLM contains a list of dicts with the required keys 'question' and 'answer'. The _definition_ provides an explanation for the new metric and the _grading_prompt_ lets you set the grading criteria, which the LLM can use to compute the metric.

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

In this section, we pass the defined metric _correct_format_ to `mlflow.evaluate()`, which makes for every image selected earlier. The JSONs created from the OCR extraction using `get_json()` can used as the ground truth

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

### Applying Prompt Registry and Evaluate to the OCR use case

Let's assume you are developing an OCR application, with different roles contributing to the project development. Here's how a Prompt engineer and a ML engineer benefit from these MLflow capabilities

#### Prompt Engineer: Improve the Prompt

As a Prompt Engineer, you'll be iteratively modifying the prompts and `mlflow.genai.register_prompt()` lets you version different prompts/ system messages that you craft, with the provision to set a template text, and include prompt metadata like author, project, etc. It is also possible to compare different versions of the prompt, and also search for specific prompts from the past.

Here's an example of a prompt template, instructing the LLM to generate

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

As an ML Engineer, it's your responsibility to ensure that the OCR application using the LLM can be evaluated against ground truth and the performance can be tracked using the custom GenAI metric. In real applications, you'll go with more than one metric which meets the business requirement.

Here you'll be following the same steps as mentioned in the sections [Defining a Gen AI metric](#1-defining-a-custom-gen-ai-metric-for-evaluation) and [MLflow Evaluate](#2-mlflow-evaluate) using `mlflow.metrics.genai.make_genai_metric()` and `mlflow.evaluate()`

## Conclusion and Next Steps

From our new OCR tool, we are able to utilize MLflow GenAI capabilities to manage our prompts and evaluate our model. This can be taken further as we look to create more complex and agentic systems.

## Further Reading

[Practical AI Observability: Getting Started with MLflow Tracing](https://mlflow.org/blog/ai-observability-mlflow-tracing)  
[Beyond Autolog: Add MLflow Tracing to a New LLM Provider](https://mlflow.org/blog/custom-tracing)  
[LLM as Judge](https://mlflow.org/blog/llm-as-judge)  

