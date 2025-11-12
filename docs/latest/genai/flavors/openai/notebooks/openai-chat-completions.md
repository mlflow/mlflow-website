# Introduction: Advancing Communication with GPT-4 and MLflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/openai/notebooks/openai-chat-completions.ipynb)

Welcome to our advanced tutorial, where we delve into the cutting-edge capabilities of OpenAI's GPT-4, particularly exploring its Chat Completions feature. In this session, we will combine the advanced linguistic prowess of GPT-4 with the robust experiment tracking and deployment framework of MLflow to create an innovative application: The Text Message Angel.

### Tutorial Overview[​](#tutorial-overview "Direct link to Tutorial Overview")

In this tutorial, we will:

1. **Set Up and Validate Environment**: Ensure that all necessary configurations, including the `OPENAI_API_KEY`, are in place for our experiments.

2. **Initialize MLflow Experiment**: Set up an MLflow experiment named "Text Message Angel" to track and manage our model's performance and outcomes.

3. **Implement Chat Completions with GPT-4**: Utilize the Chat Completions task of GPT-4 to develop an application that can analyze and respond to text messages. This feature of GPT-4 allows for context-aware, conversational AI applications that can understand and generate human-like text responses.

4. **Model Deployment and Prediction**: Deploy our model using MLflow's `pyfunc` implementation and make predictions on a set of sample text messages. This will demonstrate the practical application of our model in real-world scenarios.

### The Text Message Angel Application[​](#the-text-message-angel-application "Direct link to The Text Message Angel Application")

Our application, the Text Message Angel, aims to enhance everyday text communication. It will analyze SMS responses for tone, appropriateness, and relationship impact. The model will categorize responses as either appropriate ("Good to Go!") or suggest caution ("You might want to read that again before pressing send"). For responses deemed inappropriate, it will also suggest alternative phrasing that maintains a friendly yet witty tone.

### Why GPT-4 and MLflow?[​](#why-gpt-4-and-mlflow "Direct link to Why GPT-4 and MLflow?")

* **GPT-4's Advanced AI**: GPT-4 represents the latest in AI language model development, offering nuanced understanding and response generation capabilities that are ideal for a text-based application like the Text Message Angel.

* **MLflow's Seamless Management**: MLflow simplifies the process of tracking experiments, managing different model versions, and deploying AI models. Its integration with GPT-4 allows us to focus on the creative aspect of our application while efficiently handling the technicalities of model management.

### Engaging with the Tutorial[​](#engaging-with-the-tutorial "Direct link to Engaging with the Tutorial")

As we progress, we encourage you to actively engage with the code and concepts presented. This tutorial is not just about learning the functionalities but also understanding the potential of these technologies when combined creatively.

Let's embark on this journey to harness the synergy of MLflow and GPT-4's Chat Completions to enhance communication and interactions in our digital world.

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

### Implementing the Text Message Angel with GPT-4 and Chat Completions[​](#implementing-the-text-message-angel-with-gpt-4-and-chat-completions "Direct link to Implementing the Text Message Angel with GPT-4 and Chat Completions")

After exploring the humorous world of misheard lyrics in the introductory tutorial to the `openai` flavor, we now shift our focus to a more sophisticated application involving GPT-4 and the Chat Completions feature. This tutorial introduces the "Text Message Angel", an innovative application designed to pre-screen text messages, ensuring they are appropriate and relationship-friendly, especially for those inclined towards sarcasm.

#### Setting Up the Text Message Angel Experiment[​](#setting-up-the-text-message-angel-experiment "Direct link to Setting Up the Text Message Angel Experiment")

We begin by setting up a new MLflow experiment titled "Text Message Angel". This experiment aims to create a service that analyzes text messages and provides guidance on their appropriateness before sending. The goal is to maintain positive communication while allowing for a playful tone.

#### The Role of GPT-4 and Chat Completions[​](#the-role-of-gpt-4-and-chat-completions "Direct link to The Role of GPT-4 and Chat Completions")

GPT-4 represents a massive leap in capability compared with the previous model we used in the introductory tutorial. It brings enhanced understanding and contextual awareness, making it ideal for interpreting and responding to natural language with a high degree of accuracy and nuance. The Chat Completions feature, specifically, enables a more conversational approach, which is perfect for our text message evaluation scenario.

#### Crafting the Prompt for Text Message Evaluation[​](#crafting-the-prompt-for-text-message-evaluation "Direct link to Crafting the Prompt for Text Message Evaluation")

The core of our application is a well-crafted prompt that directs GPT-4 to evaluate text messages based on specific criteria:

* **Content Analysis**: The model determines if a message contains inappropriate elements like humorless sarcasm, passive-aggressive tones, or anything that could harm a relationship.
* **Response Categorization**: Based on its analysis, the model categorizes the message as either "Good to Go!" or advises to "read that again before pressing send."
* **Suggested Corrections**: If a message is deemed inappropriate, the model goes a step further to suggest an alternative version. This corrected message aims to preserve a fun and slightly snarky tone while ensuring it does not harm the relationship.

This setup not only demonstrates the advanced capabilities of GPT-4 in understanding and generating human-like text but also highlights its potential for practical applications in everyday communication scenarios.

python

```python
mlflow.set_experiment("Text Message Angel")

messages = [
  {
      "role": "user",
      "content": (
          "Determine if this is an acceptable response to a friend through SMS. "
          "If the response contains humorless sarcasm, a passive aggressive tone, or could potentially "
          "damage my relationship with them, please respond with 'You might want to read that again before "
          "pressing send.', otherwise respond with 'Good to Go!'. If the response classifies as inappropriate, "
          "please suggest a corrected version following the classification that will help to keep my "
          "relationship with this person intact, yet still maintains a fun and somewhat snarky tone: {text}"
      ),
  }
]

```

### Integrating GPT-4 with MLflow for the Text Message Angel[​](#integrating-gpt-4-with-mlflow-for-the-text-message-angel "Direct link to Integrating GPT-4 with MLflow for the Text Message Angel")

In this crucial step, we're integrating the advanced GPT-4 model with MLflow for our `Text Message Angel` application. This process involves setting up the model within an MLflow run, logging its configuration, and preparing it for practical use.

#### Starting the MLflow Run[​](#starting-the-mlflow-run "Direct link to Starting the MLflow Run")

We initiate an MLflow run, a crucial step in tracking our model's performance, parameters, and outputs. This run encapsulates all the details and metrics related to the GPT-4 model we are using.

#### Logging the GPT-4 Model in MLflow[​](#logging-the-gpt-4-model-in-mlflow "Direct link to Logging the GPT-4 Model in MLflow")

Within this run, we log our GPT-4 model using `mlflow.openai.log_model`. This function call is instrumental in registering our model's specifics in MLflow's tracking system. Here's a breakdown of the parameters we're logging:

* **Model Selection**: We specify `gpt-4`, indicating we are utilizing a far more advanced version of OpenAI's models than the previous example.
* **Task Specification**: The `openai.chat.completions` task is chosen, aligning with our objective of creating a conversational AI capable of analyzing and responding to text messages.
* **Artifact Path**: We define an artifact path where MLflow will store the model-related data.
* **Messages**: The `messages` variable, containing our pre-defined prompt and criteria for evaluating text messages, is passed to the model.
* **Model Signature**: The signature defines the input-output schema and parameters for our model, such as `max_tokens` and `temperature`. These settings are crucial in controlling how the model generates responses.

python

```python
with mlflow.start_run():
  model_info = mlflow.openai.log_model(
      model="gpt-4",
      task=openai.chat.completions,
      name="model",
      messages=messages,
      signature=ModelSignature(
          inputs=Schema([ColSpec(type="string", name=None)]),
          outputs=Schema([ColSpec(type="string", name=None)]),
          params=ParamSchema(
              [
                  ParamSpec(name="max_tokens", default=16, dtype="long"),
                  ParamSpec(name="temperature", default=0, dtype="float"),
              ]
          ),
      ),
  )

```

#### Loading the Model for Use[​](#loading-the-model-for-use "Direct link to Loading the Model for Use")

After logging the model in MLflow, we load it as a generic Python function using `mlflow.pyfunc.load_model`. This step is vital as it transforms our GPT-4 model into a format that's easily callable and usable within our application.

python

```python
model = mlflow.pyfunc.load_model(model_info.model_uri)

```

### Testing the Text Message Angel[​](#testing-the-text-message-angel "Direct link to Testing the Text Message Angel")

With our Text Message Angel application powered by GPT-4 and integrated within MLflow, we are now ready to put it to the test. This section involves creating a set of sample text messages, some potentially containing sarcasm or passive-aggressive tones, and others being more straightforward and friendly.

#### Creating Validation Data[​](#creating-validation-data "Direct link to Creating Validation Data")

We start by creating a DataFrame named `validation_data` with a variety of text messages. These messages are designed to test the model's ability to discern tone and suggest corrections where necessary:

1. A message using humor to mask a critique of a dinner experience.
2. A sarcastic comment expressing reluctance to go to the movies.
3. A straightforward message expressing excitement for a road trip.
4. A simple thank-you message.
5. A sarcastic remark about enjoying someone's singing.

#### Submitting Messages to the Model[​](#submitting-messages-to-the-model "Direct link to Submitting Messages to the Model")

Next, we submit these messages to our Text Message Angel model for evaluation. The model will analyze each message, determining whether it's appropriate or needs a revision. For messages that might strain a relationship, the model will suggest a more suitable version.

#### Displaying the Model's Responses[​](#displaying-the-models-responses "Direct link to Displaying the Model's Responses")

The responses from the model are then formatted for clear and attractive display. This step is crucial for assessing the model's performance in real-time and understanding how its corrections and suggestions align with the intended tone of the messages.

#### Model's Output[​](#models-output "Direct link to Model's Output")

Let's take a look at how the Text Message Angel responded:

1. Suggested a more tactful way to comment on the dinner.
2. Offered a humorous yet softer alternative for declining a movie invitation.
3. Confirmed that the road trip message is appropriate.
4. Validated the thank-you message as suitable.
5. Suggested a playful yet kinder remark about singing.

These responses showcase the model's nuanced understanding of social communication, its ability to maintain a friendly yet fun tone, and its potential in assisting users to communicate more effectively and harmoniously.

python

```python
validation_data = pd.DataFrame(
  {
      "text": [
          "Wow, what an interesting dinner last night! I had no idea that you could use canned "
          "cat food to make a meatloaf.",
          "I'd rather book a 14th century surgical operation than go to the movies with you on Thursday.",
          "Can't wait for the roadtrip this weekend! Love the playlist mixes that you choose!",
          "Thanks for helping out with the move this weekend. I really appreciate it.",
          "You know what part I love most when you sing? The end. It means its over.",
      ]
  }
)

chat_completions_response = model.predict(
  validation_data, params={"max_tokens": 50, "temperature": 0.2}
)

formatted_output = "<br>".join(
  [f"<p><strong>{line.strip()}</strong></p>" for line in chat_completions_response]
)
display(HTML(formatted_output))

```

**You might want to read that again before pressing send. Suggested response: "Wow, dinner last night was certainly unique! Who knew meatloaf could be so... adventurous?"**

<br />

**You might want to read that again before pressing send. Suggested correction: "I'd rather watch a 14th century surgical operation documentary than miss out on the movies with you on Thursday. How's that for a plot twist?"**

<br />

**Good to Go!**

<br />

**Good to Go!**

<br />

**You might want to read that again before pressing send. Suggested response: "You know what part I love most when you sing? The encore. It means I get to hear you again!"**

### Conclusion: Advancing AI Interactions with MLflow and OpenAI's GPT-4[​](#conclusion-advancing-ai-interactions-with-mlflow-and-openais-gpt-4 "Direct link to Conclusion: Advancing AI Interactions with MLflow and OpenAI's GPT-4")

As we reach the end of this tutorial, it's time to reflect on the insights we've gained, especially the remarkable capabilities of GPT-4 in the realm of conversational AI, and how MLflow facilitates the deployment and management of these advanced models.

#### Key Takeaways[​](#key-takeaways "Direct link to Key Takeaways")

* **Deep Dive into ChatCompletions with GPT-4**: This tutorial gave us a hands-on experience with GPT-4's ChatCompletions feature, demonstrating its ability to understand context, maintain conversation flow, and generate human-like responses. The `Text Message Angel` application exemplified how such a model can be used to improve and refine everyday communication.

* **MLflow's Role in Managing Advanced AI**: MLflow has shown its strength not just in handling model logistics, but also in simplifying the experimentation with complex AI models like GPT-4. Its robust tracking and logging capabilities make it easier to manage and iterate over conversational AI models.

* **Real-World Application and Potential**: The `Text Message Angel` illustrated a practical application of GPT-4's advanced capabilities, demonstrating how AI can be leveraged to enhance and safeguard interpersonal communication. It's a glimpse into how conversational AI can be used in customer service, mental health, education, and other domains.

* **The Evolution of AI and MLflow's Adaptability**: The tutorial highlighted how MLflow's flexible framework is well-suited to keep pace with the rapid advancements in AI, particularly in areas like natural language processing and conversational AI.

#### Moving Forward with Conversational AI[​](#moving-forward-with-conversational-ai "Direct link to Moving Forward with Conversational AI")

The combination of MLflow and OpenAI's GPT-4 opens up exciting avenues for developing more intuitive and responsive AI-driven applications. As we continue to witness advancements in AI, MLflow's ability to adapt and manage these complex models becomes increasingly vital.

#### Embarking on Your AI Journey[​](#embarking-on-your-ai-journey "Direct link to Embarking on Your AI Journey")

We encourage you to build upon the foundations laid in this tutorial to explore the vast potential of conversational AI. With MLflow and OpenAI's GPT-4, you are well-equipped to create innovative applications that can converse, understand, and interact in more human-like ways.

Thank you for joining us in exploring the cutting-edge of conversational AI and model management. Your journey into developing AI-enhanced communication tools is just beginning, and we are excited to see where your creativity and skills will lead you next!

To continue your learning journey, see the additional [advanced tutorials for MLflow's OpenAI flavor](https://www.mlflow.org/docs/latest/genai/flavors/openai/index.html#advanced-tutorials).
