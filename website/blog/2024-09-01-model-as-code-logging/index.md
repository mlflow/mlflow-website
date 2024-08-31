---
title: Model as Code Logging in MLflow for Enhanced Model Management
tags: [pyfunc, genai]
slug: model_as_code
authors: [awadelrahman-ahmed, daniel-liden]
thumbnail: img/blog/release-candidates.png
---

We all—well, most of us—remember November 2022 when the public release of ChatGPT by OpenAI marked a significant turning point in the world of AI. While Generative AI had been evolving for some time, ChatGPT, built on OpenAI's GPT-3.5 architecture, quickly captured the public’s imagination. This led to an explosion of interest in Generative AI, both within the tech industry and among the general public.

On the tools side, MLflow continues to solidify its position as the favorite tool for MLOps among the ML community. However, the rise of Generative AI has introduced new needs in how we use MLflow. One of these new challenges is how we log model artifacts in MLflow. If you’ve used MLflow before (and I bet you have), you’re probably familiar with the `mlflow.log_model()` function and how it efficiently **pickles** model artifacts (I’ve bolded "pickles" because it’s a key term in this post).

Particularly with GenAI, there’s a new requirement: logging the model "as code," not just serializing it into a pickle file. This post explores how MLflow has adapted to meet this need. And guess what? This feature isn’t limited to GenAI models; it’s implemented at a very abstract level, allowing you to log any model "as code," whether it’s GenAI or not! I like to think of it as a generic approach, with GenAI models being just one of its use cases. So, in this post, I’ll explore this new approach, "model as code."

## What Is Actually Model-as-Code Logging?

In fact, when MLflow announced this feature, it got me thinking in a more abstract way about the concept of a "model"! You might find it interesting, too, if you zoom out and consider a model as a mathematical representation or function that describes the relationship between input and output variables. At this level of abstraction, a model can be many things!

One could realize, at this level, that it can be argued a model as an object (or artifact) is just one form of a model, even if it’s the most popular in the ML community. But if you think about it, at that high level, a model can also be a simple mapping function—just a piece of code—or even code that sends API requests to another service that doesn’t necessarily reside within your "premises" (e.g., OpenAI APIs).

I'll explain the detailed workflow of how to log mode-as-code later in tehe post, but for now, let's consider it at a high level with two main steps: first, writing your model code, and second, logging your model as code.

🔴 It's important to note that when we refer to "model code," we're talking about code that can be treated as a model itself. This means it's **not** your training code that generates a trained model object, but rather the step-by-step code that is executed as a model itself. This will look like the folowing figure:

![High Level Model-as-Code Logging Workflow](model_as_code1.png)

## How Model-as-Code Differs From Model-as-Artifact Logging?

In the previous section, we discussed what is meant by model-as-code logging. In my experience, concepts often become clearer when contrasted with their alternatives—a technique known as _contrast learning_. So, the alternative will be model-as-artifact logging, which is the most commonly used approach for logging models.

You're probably familiar with the process of writing training code, training a model, and then saving the trained model as an artifact to be reused later by loading it back into your application. This what I refer to here as model-as-artifact logging. In its simplest form, this involves calling the function `mlflow.log_model()`, after which MLflow typically handles the serialization process for you. If you're using a Python-based model, this might involve using Pickle or a similar method under the hood to store the model so it can be easily loaded later.

The model-as-artifact logging can be broken down into three high-level steps as in the following figure: first, creating the model as an object (whether by training it or acquiring it), second, serializing it (usually with Pickle or a similar tool), and third, logging it as an artifact.

![High Level Model-as-Artifact Logging Workflow](model_as_code2.png)

🟢 So, the main distinction between the popular model-as-artifact logging and model-as-code logging is that in the former, we log the model object itself—whether it's a model you've trained or a pre-trained model you've acquired. In the latter, however, we log the code that represents your model. In the model-as-artifact approach, the model exists as an object, which you either create through training or acquire as a pre-trained model.

## When Do You Need Model-as-Code Logging?

## How To Implement Model-as-Code Logging With Examples

### A 101 Example

### A GenAI Use Case

## Conclusion
