---
title: Model as Code Logging in MLflow for Enhanced Model Management
tags: [pyfunc, genai]
slug: model_as_code
authors: [awadelrahman-ahmed]
thumbnail: img/blog/release-candidates.png
---

We all—well, most of us—remember November 2022 when the public release of ChatGPT by OpenAI marked a significant turning point in the world of AI. While Generative AI had been evolving for some time, ChatGPT, built on OpenAI's GPT-3.5 architecture, quickly captured the public’s imagination. This led to an explosion of interest in Generative AI, both within the tech industry and among the general public.

On the tools side, MLflow continues to solidify its position as the favorite tool for MLOps among the ML community. However, the rise of Generative AI has introduced new needs in how we use MLflow. One of these new challenges is how we log model artifacts in MLflow. If you’ve used MLflow before (and I bet you have), you’re probably familiar with the `mlflow.log_model()` function and how it efficiently **pickles** model artifacts (I’ve bolded "pickles" because it’s a key term in this post).

Particularly with GenAI, there’s a new requirement: logging the model "as code," not just serializing it into a pickle file. This post explores how MLflow has adapted to meet this need. And guess what? This feature isn’t limited to GenAI models; it’s implemented at a very abstract level, allowing you to log any model "as code," whether it’s GenAI or not! I like to think of it as a generic approach, with GenAI models being just one of its use cases. So, in this post, I’ll explore this new approach, "model as code."

## What Is Actually Model-as-Code Logging?

In fact, when MLflow announced this feature, it got me thinking in a more abstract way about the concept of a "model". You might find it interesting, too, if we zoom out and consider a model as a mathematical representation or function that describes the relationship between input and output variables.

At this level of abstraction, a model can be many things! One could realize at this level that it can be argued that a model as an object (or artifact) is just one form of a model, even if it’s the most popular form in the ML community. But if you think about it, at that high level, a model can also be a simple mapping function—just a piece of code—or even code that sends API requests to another service that doesn’t reside within your "premises" (e.g., OpenAI APIs).

## How Model-as-Code Differs From Model-as-Artifact Logging?

## When Do You Need Model-as-Code Logging?

## How To Implement Model-as-Code Logging With Examples

### A 101 Example

### A GenAI Use Case

## Conclusion
