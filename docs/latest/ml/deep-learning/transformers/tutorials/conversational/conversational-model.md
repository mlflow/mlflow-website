# Introduction to Conversational AI with MLflow and DialoGPT

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/transformers/tutorials/conversational/conversational-model.ipynb)

Welcome to our tutorial on integrating [Microsoft's DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium) with MLflow's transformers flavor to explore conversational AI.

### Learning Objectives[​](#learning-objectives "Direct link to Learning Objectives")

In this tutorial, you will:

* Set up a conversational AI **pipeline** using DialoGPT from the Transformers library.
* **Log** the DialoGPT model along with its configurations using MLflow.
* Infer the input and output **signature** of the DialoGPT model.
* **Load** a stored DialoGPT model from MLflow for interactive usage.
* Interact with the chatbot model and understand the nuances of conversational AI.

By the end of this tutorial, you will have a solid understanding of managing and deploying conversational AI models with MLflow, enhancing your capabilities in natural language processing.

#### What is DialoGPT?[​](#what-is-dialogpt "Direct link to What is DialoGPT?")

DialoGPT is a conversational model developed by Microsoft, fine-tuned on a large dataset of dialogues to generate human-like responses. Part of the GPT family, DialoGPT excels in natural language understanding and generation, making it ideal for chatbots.

#### Why MLflow with DialoGPT?[​](#why-mlflow-with-dialogpt "Direct link to Why MLflow with DialoGPT?")

Integrating MLflow with DialoGPT enhances conversational AI model development:

* **Experiment Tracking**: Tracks configurations and metrics across experiments.
* **Model Management**: Manages different versions and configurations of chatbot models.
* **Reproducibility**: Ensures the reproducibility of the model's behavior.
* **Deployment**: Simplifies deploying conversational models in production.

python

```python
# Disable tokenizers warnings when constructing pipelines
%env TOKENIZERS_PARALLELISM=false

import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

```

### Setting Up the Conversational Pipeline[​](#setting-up-the-conversational-pipeline "Direct link to Setting Up the Conversational Pipeline")

We begin by setting up a conversational pipeline with DialoGPT using `transformers` and managing it with MLflow.

We start by importing essential libraries. The `transformers` library from Hugging Face offers a rich collection of pre-trained models, including DialoGPT, for various NLP tasks. MLflow, a comprehensive tool for the ML lifecycle, aids in experiment tracking, reproducibility, and deployment.

#### Initializing the Conversational Pipeline[​](#initializing-the-conversational-pipeline "Direct link to Initializing the Conversational Pipeline")

Using the `transformers.pipeline` function, we set up a conversational pipeline. We choose the "`microsoft/DialoGPT-medium`" model, balancing performance and resource efficiency, ideal for conversational AI. This step is pivotal for ensuring the model is ready for interaction and integration into various applications.

#### Inferring the Model Signature with MLflow[​](#inferring-the-model-signature-with-mlflow "Direct link to Inferring the Model Signature with MLflow")

Model signature is key in defining how the model interacts with input data. To infer it, we use a sample input ("`Hi there, chatbot!`") and leverage `mlflow.transformers.generate_signature_output` to understand the model's input-output schema. This process ensures clarity in the model's data requirements and prediction format, crucial for seamless deployment and usage.

This configuration phase sets the stage for a robust conversational AI system, leveraging the strengths of DialoGPT and MLflow for efficient and effective conversational interactions.

python

```python
import transformers

import mlflow

# Define our pipeline, using the default configuration specified in the model card for DialoGPT-medium
conversational_pipeline = transformers.pipeline(model="microsoft/DialoGPT-medium")

# Infer the signature by providing a representnative input and the output from the pipeline inference abstraction in the transformers flavor in MLflow
signature = mlflow.models.infer_signature(
  "Hi there, chatbot!",
  mlflow.transformers.generate_signature_output(conversational_pipeline, "Hi there, chatbot!"),
)

```

### Creating an experiment[​](#creating-an-experiment "Direct link to Creating an experiment")

We create a new MLflow Experiment so that the run we're going to log our model to does not log to the default experiment and instead has its own contextually relevant entry.

python

```python
# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

# mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Set a name for the experiment that is indicative of what the runs being created within it are in regards to
mlflow.set_experiment("Conversational")

```

### Logging the Model with MLflow[​](#logging-the-model-with-mlflow "Direct link to Logging the Model with MLflow")

We'll now use MLflow to log our conversational AI model, ensuring systematic versioning, tracking, and management.

#### Initiating an MLflow Run[​](#initiating-an-mlflow-run "Direct link to Initiating an MLflow Run")

Our first step is to start an MLflow run with `mlflow.start_run()`. This action initiates a new tracking environment, capturing all model-related data under a unique run ID. It's a crucial step to segregate and organize different modeling experiments.

#### Logging the Conversational Model[​](#logging-the-conversational-model "Direct link to Logging the Conversational Model")

We log our DialoGPT conversational model using `mlflow.transformers.log_model`. This specialized function efficiently logs Transformer models and requires several key parameters:

* **transformers\_model**: We pass our DialoGPT conversational pipeline.
* **artifact\_path**: The storage location within the MLflow run, aptly named `"chatbot"`.
* **task**: Set to `"conversational"` to reflect the model's purpose.
* **signature**: The inferred model signature, dictating expected inputs and outputs.
* **input\_example**: A sample prompt, like `"A clever and witty question"`, to demonstrate expected usage.

Through this process, MLflow not only tracks our model but also organizes its metadata, facilitating future retrieval, understanding, and deployment.

python

```python
with mlflow.start_run():
  model_info = mlflow.transformers.log_model(
      transformers_model=conversational_pipeline,
      name="chatbot",
      task="conversational",
      signature=signature,
      input_example="A clever and witty question",
  )

```

### Loading and Interacting with the Chatbot Model[​](#loading-and-interacting-with-the-chatbot-model "Direct link to Loading and Interacting with the Chatbot Model")

Next, we'll load the MLflow-logged chatbot model and interact with it to see it in action.

#### Loading the Model with MLflow[​](#loading-the-model-with-mlflow "Direct link to Loading the Model with MLflow")

We use `mlflow.pyfunc.load_model` to load our conversational AI model. This function is a crucial aspect of MLflow's Python function flavor, offering a versatile way to interact with Python models. By specifying `model_uri=model_info.model_uri`, we precisely target the stored location of our DialoGPT model within MLflow's tracking system.

#### Interacting with the Chatbot[​](#interacting-with-the-chatbot "Direct link to Interacting with the Chatbot")

Once loaded, the model, referenced as `chatbot`, is ready for interaction. We demonstrate its conversational capabilities by:

* **Asking Questions**: Posing a question like "What is the best way to get to Antarctica?" to the chatbot.
* **Capturing Responses**: The chatbot's response, generated through the `predict` method, provides a practical example of its conversational skills. For instance, it might respond with suggestions about reaching Antarctica by boat.

This demonstration highlights the practicality and convenience of deploying and using models logged with MLflow, especially in dynamic and interactive scenarios like conversational AI.

python

```python
# Load the model as a generic python function in order to leverage the integrated Conversational Context
# Note that loading a conversational model with the native flavor (i.e., `mlflow.transformers.load_model()`) will not include anything apart from the
# pipeline itself; if choosing to load in this way, you will need to manage your own Conversational Context instance to maintain state on the
# conversation history.
chatbot = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

# Validate that the model is capable of responding to a question
first = chatbot.predict("What is the best way to get to Antarctica?")

```

python

```python
print(f"Response: {first}")

```

### Continuing the Conversation with the Chatbot[​](#continuing-the-conversation-with-the-chatbot "Direct link to Continuing the Conversation with the Chatbot")

We further explore the MLflow `pyfunc` implementation's conversational contextual statefulness with the DialoGPT chatbot model.

#### Testing Contextual Memory[​](#testing-contextual-memory "Direct link to Testing Contextual Memory")

We pose a follow-up question, "What sort of boat should I use?" to test the chatbot's contextual understanding. The response we get, "A boat that can go to Antarctica," while straightforward, showcases the MLflow pyfunc model's ability to retain and utilize conversation history for coherent responses with `ConversationalPipeline` types of models.

#### Understanding the Response Style[​](#understanding-the-response-style "Direct link to Understanding the Response Style")

The response's style – witty and slightly facetious – reflects the training data's nature, primarily conversational exchanges from Reddit. This training source significantly influences the model's tone and style, leading to responses that can be humorous and diverse.

#### Implications of Training Data[​](#implications-of-training-data "Direct link to Implications of Training Data")

This interaction underlines the importance of the training data's source in shaping the model's responses. When deploying such models in real-world applications, it's essential to understand and consider the training data's influence on the model's conversational style and knowledge base.

python

```python
# Verify that the PyFunc implementation has maintained state on the conversation history by asking a vague follow-up question that requires context
# in order to answer properly
second = chatbot.predict("What sort of boat should I use?")

```

python

```python
print(f"Response: {second}")

```

### Conclusion and Key Takeaways[​](#conclusion-and-key-takeaways "Direct link to Conclusion and Key Takeaways")

In this tutorial, we've explored the integration of MLflow with a conversational AI model, specifically using the DialoGPT model from Microsoft. We've covered several important aspects and techniques that are crucial for anyone looking to work with advanced machine learning models in a practical, real-world setting.

#### Key Takeaways[​](#key-takeaways "Direct link to Key Takeaways")

1. **MLflow for Model Management**: We demonstrated how MLflow can be effectively used for managing and deploying machine learning models. The ability to log models, track experiments, and manage different versions of models is invaluable in a machine learning workflow.

2. **Conversational AI**: By using the DialoGPT model, we delved into the world of conversational AI, showcasing how to set up and interact with a conversational model. This included understanding the nuances of maintaining conversational context and the impact of training data on the model's responses.

3. **Practical Implementation**: Through practical examples, we showed how to log a model in MLflow, infer a model signature, and use the `pyfunc` model flavor for easy deployment and interaction. This hands-on approach is designed to provide you with the skills needed to implement these techniques in your own projects.

4. **Understanding Model Responses**: We emphasized the importance of understanding the nature of the model's training data. This understanding is crucial for interpreting the model's responses and for tailoring the model to specific use cases.

5. **Contextual History**: MLflow's `transformers` `pyfunc` implementation for `ConversationalPipelines` maintains a `Conversation` context without the need for managing state yourself. This enables chat bots to be created with minimal effort, since statefulness is maintained for you.

### Wrapping Up[​](#wrapping-up "Direct link to Wrapping Up")

As we conclude this tutorial, we hope that you have gained a deeper understanding of how to integrate MLflow with conversational AI models and the practical considerations involved in deploying these models. The skills and knowledge acquired here are not only applicable to conversational AI but also to a broader range of machine learning applications.

Remember, the field of machine learning is vast and constantly evolving. Continuous learning and experimentation are key to staying updated and making the most out of these exciting technologies.

Thank you for joining us in this journey through the world of MLflow and conversational AI. We encourage you to take these learnings and apply them to your own unique challenges and projects. Happy coding!
