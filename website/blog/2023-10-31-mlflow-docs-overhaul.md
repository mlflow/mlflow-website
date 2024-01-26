---
title: MLflow Docs Overhaul
tags: [docs]
slug: mlflow-docs-overhaul
authors: [mlflow-maintainers]
thumbnail: img/blog/docs-overhaul.png
---

The MLflow Documentation is getting an upgrade.

## Overhauling the MLflow Docs

We're thrilled to announce a comprehensive overhaul of the MLflow Docs. This initiative is not just about refreshing the look and feel but about reimagining how our users interact with our content. Our primary goal is to enhance clarity, improve navigation, and provide more in-depth resources for our community.

## A Renewed Focus on User Experience

The MLflow documentation has always been an essential resource for our users. Over time, we've received invaluable feedback, and we've listened. The modernization effort is a direct response to the needs and preferences of our community.

<!-- truncate -->

Along with working on covering new cutting-edge features as part of this documentation overhaul, we're working on addressing the complexity of getting started. As the first part of a series of tutorials and guides focusing on the initial learning phase, we've created a new [getting started guide](https://www.mlflow.org/docs/latest/getting-started/logging-first-model/index.html), the first of many in a new series we're working on in an effort to teach the fundamentals of using MLflow. We feel that more in-depth instructional tutorials for learning the concepts and tools of MLflow will help to enhance the user experience for not only new users, but experienced users who need a refresher of how to do certain tasks.

There are more of these coming in the future!

### **Easier Navigation**

Our first order of business is to declutter and reorganize. This is going to be a process, though. With some of the monolithic pages ([Mlflow Models](https://www.mlflow.org/docs/2.7.1/models.html)), this will be more of a marathon than a sprint.

We've introduced a [new main navigation page](https://www.mlflow.org/docs/latest/index.html) in an effort to help steer you to the content that you're looking for based on end-use domain, rather than component of MLflow. We're hoping that this helps to bring new feature content and useful examples to your awareness, limiting the amount of exploratory discovery needed to understand how to use these new features.

Another priority for us was to make major new features easier to discover. While the [release notes](https://github.com/mlflow/mlflow/blob/master/CHANGELOG.md) are useful, particularly for Engineers who are maintaining integrations with, or are managing a deployment of, MLflow, they're not particularly user-friendly for an end-user of MLflow. We felt that a curated list of major new features would help to distill the information in our release notes, so we built the [new features](https://www.mlflow.org/docs/latest/new-features/index.html) page. We sincerely hope it helps to reduce the amount of effort needed to know what new major features have been released.

### **Interactive Learning with Notebooks**

In today's fast-paced tech world, interactive learning is becoming the norm. Recognizing this trend, we're embedding viewable notebooks directly within the docs. But we're not stopping there. These notebooks are downloadable, allowing you to run, modify, and experiment with them locally. It's a hands-on approach to learning, bridging the gap between theory and practice.

### **In-depth Tutorials and Guides**

While our previous documentation provided a solid foundation, we felt there was room for more detailed explorations. We're introducing comprehensive [tutorials](https://www.mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/index.html) and [guides](https://www.mlflow.org/docs/latest/llms/llm-evaluate/index.html) that delve deep into MLflow's features, showing how to solve actual problems. These first new tutorials and guides are just the start. We're going to be spending a lot of time and effort on making much more of MLflow documented in this way, helping to dramatically reduce the amount of time you have to spend figuring out how to leverage features in MLflow.

## Diving Deeper: Expanding on Guides and Tutorials

Our dedication to simplifying the usage of MLflow shines through in our revamped tutorials and guides. We're not just providing instructions; we're offering [deep dives](https://www.mlflow.org/docs/latest/llms/custom-pyfunc-for-llms/notebooks/index.html), [best practices](https://www.mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/index.html), and real-world applications. What you see in the MLflow 2.8.0 release is just the beginning. We're going to be heavily focusing on creating more content, showing the best way to leverage the many features and services within MLflow, all the while endeavoring to make it easier than ever to manage any ML project you're working on.

- **LLMs**: With all of the [new LLM-focused features](https://www.mlflow.org/docs/latest/llms/llm-evaluate/notebooks/rag-evaluation.html) we've been releasing in the past year, we feel the need to create easier getting started guides,
  [in-depth tutorials](https://www.mlflow.org/docs/latest/llms/llm-evaluate/notebooks/question-answering-evaluation.html), runnable examples, and more teaching-oriented step-by-step introductions to these features.

- **Tracking and the MLflow UI**: Our expanded section on tracking will cover everything from setting up your first experiment to advanced tracking techniques. The MLflow UI, an integral part of the platform, will also get its spotlight, ensuring you can make the most of its features.

- **Model Registry**: The model registry is where MLflow truly shines, and our new guides will ensure you can harness its full power. From organizing models to version control, we'll cover it all.

- **Recipes and LLM-focused Features**: MLflow's versatility is one of its strengths. Our new content will explore the breadth of features available, from recipes to LLM-focused tools like the AI Gateway, LLM Evaluation, and the PromptLab UI.

## The Transformative Power of Interactive Notebooks

Interactive notebooks have revolutionized data science and machine learning. By integrating them into our documentation, we aim to provide a holistic learning experience. You can see code in action, understand its impact, and then experiment on their own. It's a dynamic way to grasp complex concepts, ensuring that you not only understand but can also apply your knowledge in your actual project code.

## Join Us on This Journey

The overhaul of the MLflow documentation is a significant milestone, but it's just the beginning. We have a roadmap full of exciting updates, new content, and features. And for those in our community with a passion for sharing knowledge, we have a message: We'd love to [collaborate](https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md)! Whether it's writing tutorials, sharing use-cases, or providing feedback, every contribution enriches the MLflow community.

In conclusion, our commitment to providing top-notch documentation is a new primary focus of the maintainer group. We believe that well-documented features, combined with interactive learning tools, can significantly enhance the experience of using any tool. We want to put in the effort and time to make sure that your journey with using MLflow is as simple and powerful as it can be.

Stay tuned for more updates, and as always, happy coding!
