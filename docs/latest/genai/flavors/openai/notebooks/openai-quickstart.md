# Introduction to Using the OpenAI Flavor in MLflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/openai/notebooks/openai-quickstart.ipynb)

Welcome to our tutorial on harnessing the power of OpenAI's GPT models through the MLflow `openai` flavor. In this session, we embark on a journey to explore the intriguing world of AI-powered text analysis and modification. As we delve into the capabilities of GPT models, you'll discover the nuances of their API and understand the evolution from the older Completions API to the more advanced ChatCompletions, which offers a conversational style interaction.

### What You Will Learn:[​](#what-you-will-learn "Direct link to What You Will Learn:")

* **Interfacing with GPT Models**: Understand how to interact with different model families like GPT-3.5 and GPT-4.
* **MLflow Integration**: Learn to seamlessly integrate these models within MLflow, allowing you to craft a purpose-built model instance that performs a single specific task in a predictable and repeatable way.
* **Model Definition**: You'll learn how to define a simple single-purpose prompt with the `Completions` endpoint to define a function that you can interact with.

### Backstory: OpenAI and GPT Models[​](#backstory-openai-and-gpt-models "Direct link to Backstory: OpenAI and GPT Models")

OpenAI has revolutionized the field of natural language processing with their Generative Pre-trained Transformer (GPT) models. These models are trained on a diverse range of internet text and have an uncanny ability to generate human-like text, answer questions, summarize passages, and much more. The evolution from GPT-3 to GPT-4 marks significant improvements in understanding context and generating more accurate responses.

### The Completions API[​](#the-completions-api "Direct link to The Completions API")

This legacy API is used for generating text based on a prompt. It's simple, straightforward, and doesn't require a great deal of effort to implememnt apart from the creativity required to craft a useful prompt instruction set.

### Exploring the Tutorial[​](#exploring-the-tutorial "Direct link to Exploring the Tutorial")

In this tutorial, we'll use MLflow to deploy a model that interfaces with the `Completions` API, submitting a prompt that will be used for any call that is made to the model. Within this tutorial, you'll learn the process of creating a prompt, how to save a model with callable parameters, and finally how to load the saved model to use for interactions.

Let's dive into the world of AI-enhanced communication and explore the potential of GPT models in everyday scenarios.

### Prerequisites[​](#prerequisites "Direct link to Prerequisites")

In order to get started with the OpenAI flavor, we're going to need a few things first.

1. An OpenAI API Account. You can [sign up here](https://platform.openai.com/login?launch) to get access in order to start programatically accessing one of the leading highly sophisticated LLM services on the planet.
2. An OpenAI API Key. You can access this once you've created an account by navigating [to the API keys page](https://platform.openai.com/api-keys).
3. The OpenAI SDK. It's [available on PyPI](https://pypi.org/project/openai/) here. For this tutorial, we're going to be using version 0.28.1 (the last release prior to the 1.0 release).

To install the `openai` SDK library that is compatible with this notebook to try this out yourself, as well as the additional `tiktoken` dependency that is required for the MLflow integration with `openai`, simply run:

bash

```bash
pip install 'openai<1' tiktoken

```

### API Key Security Overview[​](#api-key-security-overview "Direct link to API Key Security Overview")

API keys, especially for SaaS Large Language Models (LLMs), are as sensitive as financial information due to their connection to billing.

#### Essential Practices:[​](#essential-practices "Direct link to Essential Practices:")

* **Confidentiality**: Always keep API keys private.
* **Secure Storage**: Prefer environment variables or secure services.
* **Frequent Rotation**: Regularly update keys to avoid unauthorized access.

#### Configuring API Keys[​](#configuring-api-keys "Direct link to Configuring API Keys")

For secure usage, set API keys as environment variables.

**macOS/Linux**: Refer to [Apple's guide on using environment variables in Terminal](https://support.apple.com/en-gb/guide/terminal/apd382cc5fa-4f58-4449-b20a-41c53c006f8f/mac) for detailed instructions.

**Windows**: Follow the steps outlined in [Microsoft's documentation on environment variables](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.4).

### Imports and a quick environment verification step[​](#imports-and-a-quick-environment-verification-step "Direct link to Imports and a quick environment verification step")

Along with the customary imports that we need to run this tutorial, we're also going to verify that our API key has been set and is accessible.

After running the following cell, if an Exception is raised, please recheck the steps to ensure your API key is properly registered in your system's environment variables.

#### Troubleshooting Tips[​](#troubleshooting-tips "Direct link to Troubleshooting Tips")

If you encounter an Exception stating that the `OPENAI_API_KEY` environment variable must be set, consider the following common issues and remedies:

* **Kernel Restart**: If you're using a Jupyter notebook, make sure to restart the kernel after setting the environment variable. This is necessary for the kernel to recognize changes to environment variables.
* **Correct Profile Script**: On macOS and Linux, ensure you've edited the correct profile script (.bashrc for Bash, .zshrc for Zsh) and that you've used the correct syntax.
* **System Restart**: Sometimes, especially on Windows, you may need to restart your system for the changes to environment variables to take effect.
* **Check Spelling and Syntax**: Verify that the variable `OPENAI_API_KEY` is spelled correctly in both your environment settings and your script. Also, ensure that there are no extra spaces or syntax errors in your profile scripts or environment variable settings.

python

```python
import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

```

python

```python
import os

import openai
import pandas as pd
from IPython.display import HTML

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

# Run a quick validation that we have an entry for the OPEN_API_KEY within environment variables
assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable must be set"

```

### Understanding Prompts and Their Engineering[​](#understanding-prompts-and-their-engineering "Direct link to Understanding Prompts and Their Engineering")

#### What is a Prompt?[​](#what-is-a-prompt "Direct link to What is a Prompt?")

A prompt is a text input given to an AI model, particularly language models like GPT-3 and GPT-4, to elicit a specific type of response or output. It guides the model on the expected information or format of the response, setting the stage for the AI's "thought process" and steering it toward the desired outcome.

#### Prompt Engineering[​](#prompt-engineering "Direct link to Prompt Engineering")

Prompt engineering involves crafting these inputs to maximize the AI's response effectiveness and accuracy. It's about fine-tuning the language and structure of the prompt to align with the specific task, improving the quality and relevance of the AI's output by reducing ambiguity and directing the model's response towards the intended application.

### A Fun and Simple Example: The Lyrics Corrector[​](#a-fun-and-simple-example-the-lyrics-corrector "Direct link to A Fun and Simple Example: The Lyrics Corrector")

Imagine a scenario where a group of friends, who enjoy pop music, often end up in passionate, good-natured debates over misremembered song lyrics. To add more fun to these gatherings, we decide to create a game where an impartial judge – an AI model – adjudicates the correct lyrics after someone proposes and the group attempts to guess the correct song and lyrics from a creative interpretation.

#### Why not a Search Engine?[​](#why-not-a-search-engine "Direct link to Why not a Search Engine?")

Typically, one might turn to an internet search engine to settle these lyrical disputes. However, this method has its drawbacks. Depending on the input, search results can be imprecise, leading to time-consuming searches through various web pages to find the actual lyrics. The authenticity for the results found can be quite questionable due to the nature of the contents of some lyrics. Search engines are not designed with use cases like this in mind.

#### Why an LLM is Perfect for This Task[​](#why-an-llm-is-perfect-for-this-task "Direct link to Why an LLM is Perfect for This Task")

This is where a powerful Language Model (LLM) like GPT-4 becomes a game-changer. LLMs, trained on extensive datasets, are adept at understanding and generating human-like text. Their ability to process natural language inputs and provide relevant, accurate responses makes them ideal for this lyrical challenge.

#### Our Solution: The Lyrics Corrector Prompt[​](#our-solution-the-lyrics-corrector-prompt "Direct link to Our Solution: The Lyrics Corrector Prompt")

To leverage the LLM effectively, we craft a specialized prompt for our Lyrics Corrector application. This prompt is designed with two goals in mind:

1. **Correct Misheard Lyrics**: It instructs the AI to identify the actual lyrics of a song, replacing commonly misheard versions.
2. **Add a Humorous Explanation**: More than just correction, the AI also provides a funny explanation for why the misheard lyric is amusingly incorrect, adding an engaging, human-like element to the task.

text

```text
"Here's a misheard lyric: {lyric}. What's the actual lyric, which song does it come from, which artist performed it, and can you give a funny explanation as to why the misheard version doesn't make sense? Also, rate the creativity of the lyric on a scale of 1 to 3, where 3 is good."

```

In this prompt, `{lyric}` is a placeholder for various misheard lyrics. This setup not only showcases the model's ability to process and correct information but also to engage in a more creative, human-like manner.

Through this fun and simple example, we explore the potential of LLMs in real-world applications, demonstrating their capacity to enhance everyday experiences with a blend of accuracy and creativity.

python

```python
lyrics_prompt = (
  "Here's a misheard lyric: {lyric}. What's the actual lyric, which song does it come from, which artist performed it, and can you give a funny "
  "explanation as to why the misheard version doesn't make sense? Also, rate the creativity of the lyric on a scale of 1 to 3, where 3 is good."
)

```

### Setting Up and Logging the Model in MLflow[​](#setting-up-and-logging-the-model-in-mlflow "Direct link to Setting Up and Logging the Model in MLflow")

In this section, we define our model and [log it to MLflow](https://www.mlflow.org/docs/latest/tracking/tracking-api.html). This integrates our prompt that defines the characteristics of what we want the nature of the responses to be with the configuration parameters that dictate how MLflow will interact with the OpenAI SDK in order to select the right model with our desired parameters.

* **MLflow Experiments**: Our first step involves creating or reusing an [MLflow experiment](https://www.mlflow.org/docs/latest/tracking/tracking-api.html#organizing-runs-in-experiments) named "Lyrics Corrector". Experiments in MLflow are crucial for organizing and tracking different model runs, along with their associated data and parameters.

* **Model Logging**: Within an [MLflow run](https://www.mlflow.org/docs/latest/tracking.html#tracking-runs), we log our model, specifying details such as the model type (`gpt-4o-mini`), the task it's intended for (`openai.completions`), and the custom prompt we've designed. This action ensures that MLflow accurately captures the essence and operational context of our model.

* **Model Signature**: Here, we define the input and output schema for our model. We expect a string as input (the misheard lyric) and output a string (the corrected lyric with a humorous explanation). Additional parameters like `max_tokens`, `temperature`, and `best_of` are set to control the model's text generation process.

* **Model Loading**: Finally, we load the logged model as a generic Python function within MLflow. This makes the model readily usable for predictions and further interactions, allowing us to invoke it like a regular Python function with the specified inputs.

This setup not only establishes our Lyrics Corrector model but also demonstrates how MLflow can be effectively used to manage complex AI models, ensuring efficient tracking, management, and deployment in practical applications.

python

```python
# Create a new experiment (or reuse the existing one if we've run this cell more than once)
mlflow.set_experiment("Lyrics Corrector")

# Start our run and log our model
with mlflow.start_run():
  model_info = mlflow.openai.log_model(
      model="gpt-4o-mini",
      task=openai.completions,
      name="model",
      prompt=lyrics_prompt,
      signature=ModelSignature(
          inputs=Schema([ColSpec(type="string", name=None)]),
          outputs=Schema([ColSpec(type="string", name=None)]),
          params=ParamSchema(
              [
                  ParamSpec(name="max_tokens", default=16, dtype="long"),
                  ParamSpec(name="temperature", default=0, dtype="float"),
                  ParamSpec(name="best_of", default=1, dtype="long"),
              ]
          ),
      ),
  )

# Load the model as a generic python function that can be used for completions
model = mlflow.pyfunc.load_model(model_info.model_uri)

```

### Generating and Correcting Misheard Lyrics[​](#generating-and-correcting-misheard-lyrics "Direct link to Generating and Correcting Misheard Lyrics")

Let's have some fun with our Lyrics Corrector model by testing it with a set of humorously misheard lyrics. These phrases are well-known for their amusing misinterpretations and will be a great way to see how the model performs with a touch of humor.

#### Generating Questionable Lyrics[​](#generating-questionable-lyrics "Direct link to Generating Questionable Lyrics")

We've prepared a collection of iconic song lyrics that are often humorously misheard:

* "We built this city on sausage rolls" (a twist on "rock and roll")
* "Hold me closer, Tony Danza" (instead of "tiny dancer")
* "Sweet dreams are made of cheese. Who am I to dis a brie? I cheddar the world and a feta cheese" (a cheesy take on the original lyrics)
* "Excuse me while I kiss this guy" (rather than "the sky")
* "I want to rock and roll all night and part of every day" (changing "every day" to a less committed schedule)
* "Don't roam out tonight, it's bound to take your sight, there's a bathroom on the right." (a creative take on Bad Moon Rising)
* "I think you'll understand, when I say that somethin', I want to take your land" (a dark take on a classic Beatles love song)

These misheard versions add a layer of humor and quirkiness to the original lines, making them perfect candidates for our Lyrics Corrector.

#### Submitting Lyrics to the Model[​](#submitting-lyrics-to-the-model "Direct link to Submitting Lyrics to the Model")

Now, it's time to see how our model interprets these creative takes. We submit the misheard lyrics to the Lyrics Corrector, which will use its AI capabilities to determine the actual lyrics and provide a witty explanation for why the misheard version might be off-base.

#### Viewing the Model's Responses[​](#viewing-the-models-responses "Direct link to Viewing the Model's Responses")

After processing, the model's responses are formatted and displayed in an easy-to-read manner. This step will highlight the model's understanding of the lyrics and its ability to engage humorously with the content. It's a showcase of blending AI's linguistic accuracy with a sense of humor, making for an entertaining and insightful experience.

Let's see what amusing corrections and explanations our Lyrics Corrector comes up with for these classic misheard lyrics!

python

```python
# Generate some questionable lyrics
bad_lyrics = pd.DataFrame(
  {
      "lyric": [
          "We built this city on sausage rolls",
          "Hold me closer, Tony Danza",
          "Sweet dreams are made of cheese. Who am I to dis a brie? I cheddar the world and a feta cheese",
          "Excuse me while I kiss this guy",
          "I want to rock and roll all night and part of every day",
          "Don't roam out tonight, it's bound to take your sight, there's a bathroom on the right.",
          "I think you'll understand, when I say that somethin', I want to take your land",
      ]
  }
)

# Submit our faulty lyrics to the model
fix_my_lyrics = model.predict(bad_lyrics, params={"max_tokens": 500, "temperature": 0})

# See what the response is
formatted_output = "<br>".join(
  [f"<p><strong>{line.strip()}</strong></p>" for line in fix_my_lyrics]
)
display(HTML(formatted_output))

```

**The actual lyric is "We built this city on rock and roll" from the song "We Built This City" by Starship. The misheard version doesn't make sense because sausage rolls are a type of food, not a building material. Perhaps someone misheard the word "rock" as "roll" and their mind automatically went to food. The creativity of the misheard lyric is a 2, as it is a common mistake to mix up similar sounding words.**

<br />

**The actual lyric is "Hold me closer, tiny dancer" from the song "Tiny Dancer" by Elton John. The misheard version is a common one, with many people thinking the line is about the actor Tony Danza instead of a small dancer. The artist, Elton John, is known for his flamboyant and over-the-top performances, so it's not too far-fetched to imagine him singing about being held by Tony Danza. However, it doesn't make much sense in the context of the song, which is about a young girl who dreams of becoming a famous dancer. On a scale of 1 to 3, I would rate the creativity of the misheard lyric a 2. It's a common mistake and not particularly clever, but it does add a humorous twist to the song.**

<br />

**The actual lyric is "Sweet dreams are made of this. Who am I to disagree? I travel the world and the seven seas. Everybody's looking for something." It comes from the song "Sweet Dreams (Are Made of This)" by the Eurythmics. The misheard version doesn't make sense because it replaces the word "this" with "cheese" and changes the rest of the lyrics to be about different types of cheese. It also changes the meaning of the song from a reflection on the search for fulfillment and purpose to a silly ode to cheese. I would rate the creativity of the misheard lyric a 2. It's a clever play on words, but it doesn't quite fit with the original song or make much sense.**

<br />

**The actual lyric is "Excuse me while I kiss the sky" from the song "Purple Haze" by Jimi Hendrix. The misheard version doesn't make sense because it suggests the singer is going to kiss a random guy, which is not what the song is about. Perhaps the singer is feeling a bit confused and disoriented from the "purple haze" and mistakenly thinks there's a guy in front of them. I would rate the creativity of the misheard lyric a 2, as it plays off the similar sounding words but doesn't quite fit with the context of the song.**

<br />

**The actual lyric is "I want to rock and roll all night and party every day" from the song "Rock and Roll All Nite" by Kiss. The misheard version doesn't make sense because it suggests that the person only wants to rock and roll during the day and not at night, which goes against the spirit of the song. It also implies that they only want to party for part of the day, rather than all day and night. I would rate the creativity of the misheard lyric a 2, as it still maintains the overall theme of the song but with a humorous twist.**

<br />

**The actual lyric is "Don't go 'round tonight, it's bound to take your life, there's a bad moon on the rise" from the song "Bad Moon Rising" by Creedence Clearwater Revival. The misheard version doesn't make sense because a bathroom on the right would not pose any danger to someone's sight or life. Perhaps the misheard version is a warning to not use the bathroom at night because it's haunted or cursed. I would rate the creativity of the misheard lyric a 2.**

<br />

**The actual lyric is "I think you'll understand, when I say that something, I want to hold your hand." It comes from the song "I Want to Hold Your Hand" by The Beatles. The misheard version doesn't make sense because wanting to take someone's land is a very aggressive and strange thing to say in a love song. It also doesn't fit with the overall theme of the song, which is about wanting to be close to someone and hold their hand. I would rate the creativity of the misheard lyric a 1, as it doesn't really make sense and doesn't add anything new or interesting to the original lyric.**

### Refining Our Approach with Prompt Engineering[​](#refining-our-approach-with-prompt-engineering "Direct link to Refining Our Approach with Prompt Engineering")

After reviewing the initial results from our Lyrics Corrector model, we find that the responses, while amusing, don't quite hit the mark in terms of creativity scoring. The ratings seem to cluster around the middle of the scale, lacking the differentiation we're aiming for. This observation leads us to the iterative and nuanced process of prompt engineering, a critical step in fine-tuning AI model responses.

#### The Iterative Process of Prompt Engineering[​](#the-iterative-process-of-prompt-engineering "Direct link to The Iterative Process of Prompt Engineering")

Prompt engineering is not a one-shot affair; it's an iterative process. It involves refining the prompt based on the model's responses and adjusting it to more precisely align with our objectives. This process is crucial when working with advanced language models like GPT-3 and GPT-4, which, while powerful, often require detailed guidance to produce specific types of outputs.

#### Achieving a Refined Response[​](#achieving-a-refined-response "Direct link to Achieving a Refined Response")

Our initial prompt provided a basic structure for the task but lacked detailed guidance on how to effectively rate the creativity of the misheard lyrics. To address this, we need to:

1. **Provide Clearer Instructions**: Enhance the prompt with more explicit instructions on what constitutes different levels of creativity.
2. **Incorporate Examples**: Include examples within the prompt that illustrate low, medium, and high creativity ratings.
3. **Clarify Expectations**: Make it clear that the rating should consider not just the humor but also the originality and deviation from the original lyrics.

#### Our Improved Prompt[​](#our-improved-prompt "Direct link to Our Improved Prompt")

In the next cell, you'll see an improved prompt that is designed to elicit more nuanced and varied responses from the model, providing a clearer framework for evaluating the creativity of misheard lyrics. By refining our approach through prompt engineering, we aim to achieve more accurate and diverse ratings that align better with our intended goal for the Lyrics Corrector.

python

```python
# Define our prompt
improved_lyrics_prompt = (
  "Here's a misheard lyric: {lyric}. What's the actual lyric, which song does it come from, which artist performed it, and can "
  "you give a funny explanation as to why the misheard version doesn't make sense? Additionally, please provide an objective rating to the "
  "misheard lyric on a scale of 1 to 3, where 1 is 'not particularly creative' (minimal humor, closely resembles the "
  "original lyrics and the intent of the song) and 3 is 'hilariously creative' (highly original, very humorous, significantly different from "
  "the original). Explain your rating briefly. For example, 'I left my heart in San Francisco' misheard as 'I left my hat in San Francisco' "
  "might be a 1, as it's a simple word swap with minimal humor. Conversely, 'I want to hold your hand' misheard as 'I want to steal your land' "
  "could be a 3, as it significantly changes the meaning in a humorous and unexpected way."
)

```

python

```python
# Create a new experiment for the Improved Version (or reuse the existing one if we've run this cell more than once)
mlflow.set_experiment("Improved Lyrics Corrector")

# Start our run and log our model
with mlflow.start_run():
  model_info = mlflow.openai.log_model(
      model="gpt-4o-mini",
      task=openai.completions,
      name="model",
      prompt=improved_lyrics_prompt,
      signature=ModelSignature(
          inputs=Schema([ColSpec(type="string", name=None)]),
          outputs=Schema([ColSpec(type="string", name=None)]),
          params=ParamSchema(
              [
                  ParamSpec(name="max_tokens", default=16, dtype="long"),
                  ParamSpec(name="temperature", default=0, dtype="float"),
                  ParamSpec(name="best_of", default=1, dtype="long"),
              ]
          ),
      ),
  )

# Load the model as a generic python function that can be used for completions
improved_model = mlflow.pyfunc.load_model(model_info.model_uri)

```

python

```python
# Submit our faulty lyrics to the model
fix_my_lyrics_improved = improved_model.predict(
  bad_lyrics, params={"max_tokens": 500, "temperature": 0.1}
)

# See what the response is
formatted_output = "<br>".join(
  [f"<p><strong>{line.strip()}</strong></p>" for line in fix_my_lyrics_improved]
)
display(HTML(formatted_output))

```

**The actual lyric is "We built this city on rock and roll" from the song "We Built This City" by Starship. The misheard version is a 3 on the scale, as it completely changes the meaning of the song and adds a humorous twist. The misheard version doesn't make sense because it replaces the iconic rock and roll genre with a food item, sausage rolls. This could be interpreted as a commentary on the current state of the music industry, where popular songs are often criticized for being shallow and lacking substance. The misheard version could also be seen as a nod to the British culture, where sausage rolls are a popular snack. Overall, the misheard lyric adds a playful and unexpected element to the song.**

<br />

**The actual lyric is "Hold me closer, tiny dancer" from the song "Tiny Dancer" by Elton John. The misheard version is a common one, with many people thinking the lyric is about the actor Tony Danza instead of a dancer. This misheard version would be a 2 on the scale, as it is a humorous and unexpected interpretation of the original lyrics, but still closely resembles the original and the intent of the song. The misheard version doesn't make sense because Tony Danza is not known for his dancing skills, so it would be odd for someone to want to be held closer to him specifically for his dancing abilities. It also changes the meaning of the song, as the original lyrics are about a small and delicate dancer, while the misheard version is about a well-known actor.**

<br />

**The actual lyric is "Sweet dreams are made of this. Who am I to disagree? I travel the world and the seven seas. Everybody's looking for something." It comes from the song "Sweet Dreams (Are Made of This)" by the Eurythmics. The misheard version is a play on words, replacing "this" with "cheese" and using different types of cheese in place of "seven seas." It doesn't make sense because cheese is not typically associated with dreams or traveling the world. It also changes the meaning of the song from a philosophical exploration of desires and purpose to a silly ode to cheese. I would rate this misheard lyric a 2. It is fairly creative and humorous, but it still closely resembles the original lyrics and doesn't deviate too far from the intent of the song.**

<br />

**The actual lyric is "Excuse me while I kiss the sky" from the song "Purple Haze" by Jimi Hendrix. The misheard version is a common one, with many people thinking Hendrix was singing about kissing a guy instead of the sky. Funny explanation: Maybe the misheard version came about because Hendrix was known for his wild and unpredictable performances, so people thought he might just randomly kiss a guy on stage. Or maybe they thought he was singing about a romantic moment with a male lover in the sky. Objective rating: 2. While the misheard version is a common one and does have some humor to it, it's not particularly original or unexpected. It's a simple word swap that still somewhat makes sense in the context of the song.**

<br />

**The actual lyric is "I want to rock and roll all night and party every day" from the song "Rock and Roll All Nite" by Kiss. The misheard lyric, "I want to rock and roll all night and part of every day," doesn't make sense because it implies that the person only wants to rock and roll for a portion of each day, rather than all night and every day. It also changes the meaning of the lyric from wanting to party all day and night to only wanting to party for part of the day. I would rate this misheard lyric a 2. While it does have some humor and changes the meaning of the original lyric, it is still quite similar to the original and doesn't completely change the intent of the song.**

<br />

**The actual lyric is "There's a bad moon on the rise" from the song "Bad Moon Rising" by Creedence Clearwater Revival. The misheard lyric is a common one, with many people hearing "There's a bathroom on the right" instead of the correct lyrics. This misheard version doesn't make sense because it changes the tone and meaning of the song from a warning about a dangerous situation to a mundane reminder about bathroom locations. Objective rating: 2. While the misheard lyric is not particularly creative, it does add a humorous twist to the song and is a common misinterpretation.**

<br />

**The actual lyric is "I think you'll understand, when I say that somethin', I want to hold your hand" from the song "I Want to Hold Your Hand" by The Beatles. The misheard lyric is a 3 on the scale. The original lyric is a sweet and innocent expression of love, while the misheard version turns it into a bizarre and aggressive desire to take someone's land. It's a complete departure from the original meaning and adds a humorous twist to the song. It also plays on the idea of misheard lyrics often being nonsensical and out of context.**

### Conclusion: The Power of MLflow in Managing AI Model Experiments[​](#conclusion-the-power-of-mlflow-in-managing-ai-model-experiments "Direct link to Conclusion: The Power of MLflow in Managing AI Model Experiments")

As we wrap up our tutorial, let's revisit the significant insights we've gained, particularly focusing on how MLflow enhances the experimentation and deployment of OpenAI's advanced language models.

#### Key Takeaways[​](#key-takeaways "Direct link to Key Takeaways")

* **Prompt Engineering and Experimentation**: This tutorial highlighted the iterative nature of prompt engineering, showcasing how subtle changes in the prompt can lead to significantly different outcomes from an AI model. MLflow plays a pivotal role here, allowing us to track these variations effectively, compare results, and iterate towards the optimal prompt configuration.

* **Simplifying AI Model Management with MLflow**: MLflow's capacity to log models, manage experiments, and handle the nuances of the machine learning lifecycle has been indispensable. It simplifies the complex process of managing and deploying AI models, making these tasks more accessible to both developers and data scientists.

* **Leveraging OpenAI's Advanced Models**: The seamless integration of OpenAI's GPT models within MLflow demonstrates how state-of-the-art AI technology can be applied in real-world scenarios. Our Lyrics Corrector example, built on the GPT-3.5-turbo model, illustrates just one of many potential applications that blend creativity, humor, and advanced language understanding.

* **Advantages of MLflow's pyfunc Implementation**: The pyfunc implementation in MLflow allows for flexible and straightforward access to advanced models like OpenAI's GPT. It enables users to deploy these models as generic Python functions, greatly enhancing their usability and integration into diverse applications.

#### Forward-Looking[​](#forward-looking "Direct link to Forward-Looking")

The integration of MLflow with OpenAI's GPT models opens up a world of possibilities for innovative applications. As AI technology evolves, the versatility and robustness of MLflow will be key in translating these advancements into practical and impactful solutions.

#### Encouragement for Further Exploration[​](#encouragement-for-further-exploration "Direct link to Encouragement for Further Exploration")

We invite you to continue exploring the vast potential of combining MLflow's powerful model management capabilities with the advanced linguistic prowess of OpenAI's models. Whether you're enhancing communication, automating tasks, or creating new AI-driven services, this combination offers a rich platform for your creativity and technical expertise.

Thank you for joining us in this exploration of AI model experimentation and management. We are excited to see how you utilize these powerful tools in your upcoming projects and innovations!

To continue learning about the capabilities of MLflow and OpenAI as they work together, we encourage you to continue your learning with a more advanced example, [the Custom Python Model example for the MLflow OpenAI flavor](https://www.mlflow.org/docs/latest/genai/flavors/openai/notebooks/openai-chat-completions.html).
