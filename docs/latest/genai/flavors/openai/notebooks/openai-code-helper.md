# Building a Code Assistant with OpenAI & MLflow

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/openai/notebooks/openai-code-helper.ipynb)

### Overview[​](#overview "Direct link to Overview")

Welcome to this comprehensive tutorial, where you'll embark on a fascinating journey through the integration of OpenAI's powerful language models with MLflow, where we'll be building an actually useful tool that can, with the simple addition of a decorator to any function that we declare, get immediate feedback within an interactive environment on code under active development.

### Learning Objectives[​](#learning-objectives "Direct link to Learning Objectives")

By the end of this tutorial, you will:

1. **Master OpenAI's GPT-4 for Code Assistance**: Understand how to leverage OpenAI's GPT-4 model for providing real-time coding assistance. Learn to harness its capabilities for generating code suggestions, explanations, and improving overall coding efficiency.
2. **Utilize MLflow for Enhanced Model Tracking**: Delve into MLflow's powerful tracking systems to manage machine learning experiments. Learn how to adapt a `pyfunc model` from within MLflow to control how the output of an LLM is displayed from within an interactive coding environment.
3. **Seamlessly Combine OpenAI and MLflow**: Discover the practical steps to integrate OpenAI's AI capabilities with MLflow's tracking and management systems. This integration exemplifies how combining these tools can streamline the development and deployment of intelligent applications.
4. **Develop and Deploy a Custom Python Code Assistant**: Gain hands-on experience in creating a Python-based code assistant using OpenAI's model. Then, actually see it in action as it is used within a Jupyter Notebook environment to give helpful assistance during development.
5. **Improve Code Quality with AI-driven Insights**: Apply AI-powered analysis to review and enhance your code. Learn how an AI assistant can provide real-time feedback on code quality, suggest improvements, and help maintain high coding standards.
6. **Explore Advanced Python Features for Robust Development**: Understand advanced Python features like decorators and functional programming. These are crucial for building efficient, scalable, and maintainable software solutions, especially when integrating AI capabilities.

### Key Concepts Covered[​](#key-concepts-covered "Direct link to Key Concepts Covered")

1. **MLflow's Model Management**: Explore MLflow's features for tracking experiments, packaging code into reproducible runs, and managing and deploying models.
2. **Custom Python Model**: Learn how to use MLflow's built-in customization for defining a generic Python function that will allow you to craft your own processing logic while interfacing with OpenAI to perform alternative handling to the LLM's output.
3. **Python Decorators and Functional Programming**: Learn about advanced Python concepts like decorators and functional programming for efficient code evaluation and enhancement.

### Why Use MLflow for this?[​](#why-use-mlflow-for-this "Direct link to Why Use MLflow for this?")

MLflow emerges as a pivotal element in this tutorial, making our use case not only feasible but also highly efficient. It offers a secure and seamless interface with OpenAI's advanced language models. In this tutorial, we'll explore how MLflow greatly simplifies the process of storing specific instructional prompts for OpenAI, and enhances the user experience by adding readable formatting to the returned text.

The flexibility and scalability of MLflow make it a robust choice for integrating with various tools, particularly in interactive coding environments like Jupyter Notebooks. We'll witness firsthand how MLflow facilitates rapid experimentation and iteration, allowing us to create a functional tool with minimal effort. This tool will not just assist in development but will also elevate the overall coding and model management experience. By leveraging MLflow's comprehensive features, we'll navigate through a seamless end-to-end workflow, from setting up intricate models to executing complex tasks efficiently.

### Important Cost Considerations for GPT-4 Usage[​](#important-cost-considerations-for-gpt-4-usage "Direct link to Important Cost Considerations for GPT-4 Usage")

#### High(er) Cost of GPT-4[​](#higher-cost-of-gpt-4 "Direct link to High(er) Cost of GPT-4")

It's crucial to note that **using GPT-4, as opposed to GPT-4o-mini, can incur higher costs**. GPT-4's advanced capabilities and enhanced performance come with a price premium, making it a more expensive option compared to earlier models like GPT-3.5.

#### Why Choose GPT-4 in This Tutorial[​](#why-choose-gpt-4-in-this-tutorial "Direct link to Why Choose GPT-4 in This Tutorial")

* **Enhanced Capabilities**: We opt for GPT-4 in this tutorial primarily due to its superior capabilities, especially in areas such as code refactoring and detecting issues in code implementations.
* **Demonstration Purposes**: The use of GPT-4 here serves as a demonstration to showcase the cutting-edge advancements in language model technology and its applications in complex tasks.

#### Consider Alternatives for Cost-Effectiveness[​](#consider-alternatives-for-cost-effectiveness "Direct link to Consider Alternatives for Cost-Effectiveness")

For projects where cost is a significant concern, or where the advanced features of GPT-4 are not essential, **consider using GPT-4o-mini or other more cost-effective alternatives**. These models still offer robust performance for a wide range of applications but at a lower cost.

#### Budgeting for GPT-4[​](#budgeting-for-gpt-4 "Direct link to Budgeting for GPT-4")

If you choose to proceed with GPT-4, it is recommended to:

* **Monitor Usage Closely**: Keep track of your API usage to manage costs effectively.
* **Budget Accordingly**: Allocate sufficient resources to cover the higher costs associated with GPT-4.

By being mindful of these cost considerations, you can make informed decisions about which OpenAI model best suits your project's needs and budget.

python

```
import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)
```

python

```
import functools
import inspect
import os
import textwrap

import openai

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

# Run a quick validation that we have an entry for the OPEN_API_KEY within environment variables
assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable must be set"
```

### Initializing the MLflow Client[​](#initializing-the-mlflow-client "Direct link to Initializing the MLflow Client")

Depending on where you are running this notebook, your configuration may vary for how you initialize the MLflow Client. If you are uncertain about how to configure and use an MLflow Tracking server or what options are available, you can see [the guide to running notebooks here](https://www.mlflow.org/docs/latest/ml/getting-started/running-notebooks/) for more information on setting the tracking server uri and configuring access to either managed or self-managed MLflow tracking servers.

### Setting the MLflow Experiment[​](#setting-the-mlflow-experiment "Direct link to Setting the MLflow Experiment")

In this section of the tutorial, we use MLflow's `set_experiment` function to define an experiment named "Code Helper". This step is essential in MLflow's workflow for several reasons:

1. **Unique Identification**: A unique and distinct experiment name like "Code Helper" is crucial for easy identification and segregation of the runs pertaining to this specific project, especially when working on multiple projects or experiments simultaneously.

2. **Simplified Tracking**: Naming the experiment enables effortless tracking of all the runs and models associated with it, maintaining a clear history of model development, parameters, metrics, and results.

3. **Ease of Access in MLflow UI**: A distinct experiment name ensures quick location and access to our experiment's runs and models within the MLflow UI, facilitating analysis, comparison of different runs, and sharing findings.

4. **Facilitates Better Organization**: As projects grow in complexity, having a well-named experiment aids in better organization and management of the machine learning lifecycle, making it easier to navigate through different stages of the experiment.

The use of a unique experiment name like "Code Helper" lays the foundation for efficient model management and tracking, a critical aspect of any machine learning workflow, especially in dynamic and collaborative environments.

python

```
mlflow.set_experiment("Code Helper")
```

```
<Experiment: artifact_location='file:///Users/benjamin.wilson/repos/mlflow-fork/mlflow/docs/source/llms/openai/notebooks/mlruns/703316263508654123', creation_time=1701891935339, experiment_id='703316263508654123', last_update_time=1701891935339, lifecycle_stage='active', name='Code Helper', tags={}>
```

### Defining the Instruction Set for the AI Model[​](#defining-the-instruction-set-for-the-ai-model "Direct link to Defining the Instruction Set for the AI Model")

In this part of the tutorial, we define a specific set of instructions to guide the behavior of our AI model. This is achieved through the `instruction` array, which outlines the roles and expected interactions between the system (AI model) and the user. Here's a breakdown of its components:

1. **System Role**: The first element of the array defines the role of the AI model as a 'system'. It describes the model as a 'helpful expert Software Engineer' whose purpose is to assist in code analysis and provide educational support. The AI model is expected to:

   * Offer clear explanations of the code's intent.
   * Assess the code's correctness and readability.
   * Suggest improvements while focusing on simplicity, maintainability, and adherence to best coding practices.

2. **User Role**: The second element represents the 'user' role. This part is where the user (in this case, the person learning from the tutorial) interacts with the AI model by submitting code for review. The user is expected to:

   * Provide code snippets for evaluation.
   * Seek feedback and suggestions for code improvement from the AI model.

This instruction set is crucial for creating an interactive learning experience. It guides the AI model in providing targeted, constructive feedback, making it an invaluable tool for understanding coding practices and enhancing coding skills.

python

```
instruction = [
  {
      "role": "system",
      "content": (
          "As an AI specializing in code review, your task is to analyze and critique the submitted code. For each code snippet, provide a detailed review that includes: "
          "1. Identification of any errors or bugs. "
          "2. Suggestions for optimizing code efficiency and structure. "
          "3. Recommendations for enhancing code readability and maintainability. "
          "4. Best practice advice relevant to the code's language and functionality. "
          "Your feedback should help the user improve their coding skills and understand best practices in software development."
      ),
  },
  {"role": "user", "content": "Review my code and suggest improvements: {code}"},
]
```

### Defining and Utilizing the Model Signature in MLflow[​](#defining-and-utilizing-the-model-signature-in-mlflow "Direct link to Defining and Utilizing the Model Signature in MLflow")

In this part of the tutorial, we define a `ModelSignature` for our OpenAI model, which is a crucial step in both saving the base model and later in our custom Python Model implementation. Here's an overview of the process:

1. **Model Signature Definition**:

   <!-- -->

   * We create a `ModelSignature` object that specifies the input, output, and parameters of our model.
   * The `inputs` and `outputs` are defined as schemas with a single string column, indicating that our model will be processing string type data.
   * The `params` schema includes two parameters: `max_tokens` and `temperature`, each with a default value and data type defined.

> **Note** We're explicitly defining the model signature here for purposes of demonstration. The schema will be automatically inferred if you do not specify one and will be set based on the `task` that is defined when logging or saving the model.

2. **Logging the Base OpenAI Model**:

   <!-- -->

   * Using `mlflow.openai.log_model`, we log the base OpenAI model (`gpt-4`) along with the `instruction` set we defined earlier.
   * The `signature` we defined is also passed in this step, ensuring that the model is saved with the correct specifications for inputs, outputs, and parameters.

This dual-purpose signature is vital as it ensures consistency in how the model processes data both in its base form and when it's later wrapped in a custom Python Model. This approach streamlines the workflow and maintains uniformity across different stages of model implementation and deployment.

python

```
# Define the model signature that will be used for both the base model and the eventual custom pyfunc implementation later.
signature = ModelSignature(
  inputs=Schema([ColSpec(type="string", name=None)]),
  outputs=Schema([ColSpec(type="string", name=None)]),
  params=ParamSchema(
      [
          ParamSpec(name="max_tokens", default=500, dtype="long"),
          ParamSpec(name="temperature", default=0, dtype="float"),
      ]
  ),
)

# Log the base OpenAI model with the included instruction set (prompt)
with mlflow.start_run():
  model_info = mlflow.openai.log_model(
      model="gpt-4",
      task=openai.chat.completions,
      name="base_model",
      messages=instruction,
      signature=signature,
  )
```

### Our logged model in the MLflow UI[​](#our-logged-model-in-the-mlflow-ui "Direct link to Our logged model in the MLflow UI")

After logging the model, you can open up the MLflow UI and see the components that have been logged. Notice that the configuration for our model, including the model type (gpt-4), the endpoint API type (task) is recorded (chat.completions), and the prompt have all been logged.

![openai-ui](https://i.imgur.com/72EGEG8.png)

### Enhancing User Experience with Custom Pyfunc Implementation[​](#enhancing-user-experience-with-custom-pyfunc-implementation "Direct link to Enhancing User Experience with Custom Pyfunc Implementation")

In this section, we introduce a custom Python Model, `CodeHelper`, which significantly improves the user experience when interacting with the OpenAI model in an interactive development environment like Jupyter Notebook. The `CodeHelper` class is designed to format the output from the OpenAI model, making it more readable and visually appealing, similar to a chat interface. Here's how it works:

1. **Initialization and Model Loading**:

   * The `CodeHelper` class inherits from `PythonModel`.
   * The `load_context` method is used to load the OpenAI model, which is saved as `self.model`. This model is loaded from the `context.artifacts`, ensuring that the appropriate model is used for predictions.

2. **Response Formatting**:

   * The `_format_response` method is crucial for enhancing the output format.
   * It processes each item in the response, handling text and code blocks differently.
   * Text lines outside of code blocks are wrapped to a width of 80 characters for better readability.
   * Lines within code blocks (marked by ` ``` `) are not wrapped, preserving the code structure.
   * This formatting creates an output that resembles a chat interface, making the interaction more intuitive and user-friendly.

3. **Making Predictions**:

   * The `predict` method is where the model's prediction occurs.
   * It calls the loaded OpenAI model to get the raw response for the given input.
   * The raw response is then passed to the `_format_response` method for formatting.
   * The formatted response is returned, providing a clear and easy-to-read output.

By implementing this custom `pyfunc`, we enhance the user's interaction with the AI code helper. It not only makes the output easier to understand but also presents it in a familiar format, akin to messaging, which is especially beneficial in interactive coding environments.

python

````
# Custom pyfunc implementation that applies text and code formatting to the output results from the OpenAI model
class CodeHelper(PythonModel):
  def __init__(self):
      self.model = None

  def load_context(self, context):
      self.model = mlflow.pyfunc.load_model(context.artifacts["model_path"])

  @staticmethod
  def _format_response(response):
      formatted_output = ""
      in_code_block = False

      for item in response:
          lines = item.split("
")
          for line in lines:
              # Check for the start/end of a code block
              if line.strip().startswith("```"):
                  in_code_block = not in_code_block
                  formatted_output += line + "
"
                  continue

              if in_code_block:
                  # Don't wrap lines inside code blocks
                  formatted_output += line + "
"
              else:
                  # Wrap lines outside of code blocks
                  wrapped_lines = textwrap.fill(line, width=80)
                  formatted_output += wrapped_lines + "
"

      return formatted_output

  def predict(self, context, model_input, params):
      # Call the loaded OpenAI model instance to get the raw response
      raw_response = self.model.predict(model_input, params=params)

      # Return the formatted response so that it is easier to read
      return self._format_response(raw_response)
````

### Saving the Custom Python Model with MLflow[​](#saving-the-custom-python-model-with-mlflow "Direct link to Saving the Custom Python Model with MLflow")

This part of the tutorial demonstrates how to save the custom Python model, `CodeHelper`, using MLflow. The process involves specifying the model's location and additional information to ensure it is properly stored and can be retrieved for future use. Here's an overview:

1. **Defining Artifacts**:

   * An `artifacts` dictionary is created with a key `"model_path"` pointing to the location of the base OpenAI model. This step is important to link our custom model with the necessary base model files. We retrieve the location of the logged openai model from earlier by accessing the `model_uri` property from the return of the `log_model()` function.

2. **Saving the Model**:

   * The `mlflow.pyfunc.save_model` function is used to save the `CodeHelper` model.
   * `path`: Specifies the location (`final_model_path`) where the model will be saved.
   * `python_model`: An instance of the `CodeHelper` class is provided, indicating the model to be saved.
   * `input_example`: An example input (`["x = 1"]`) is given, which is useful for understanding the model's expected input format.
   * `signature`: The previously defined `ModelSignature` is passed, ensuring consistency in how the model processes data.
   * `artifacts`: The `artifacts` dictionary is included to associate the base OpenAI model with our custom model.

This step is crucial for encapsulating the entire functionality of our `CodeHelper` model in a format that MLflow can manage and track. It allows for easy deployment and versioning of the model, facilitating its use in various applications and environments.

python

```
# Define the location of the base model that we'll be using within our custom pyfunc implementation
artifacts = {"model_path": model_info.model_uri}

with mlflow.start_run():
  helper_model = mlflow.pyfunc.log_model(
      name="code_helper",
      python_model=CodeHelper(),
      input_example=["x = 1"],
      signature=signature,
      artifacts=artifacts,
  )
```

```
Downloading artifacts:   0%|          | 0/5 [00:00<?, ?it/s]
```

### Load our saved Custom Python Model[​](#load-our-saved-custom-python-model "Direct link to Load our saved Custom Python Model")

In this next section, we load the model that we just saved so that we can use it!

python

```
loaded_helper = mlflow.pyfunc.load_model(helper_model.model_uri)
```

### Comparing Two Approaches for Code Review with MLflow Models[​](#comparing-two-approaches-for-code-review-with-mlflow-models "Direct link to Comparing Two Approaches for Code Review with MLflow Models")

In this tutorial, we'll explore two different approaches to utilizing MLflow models for reviewing and providing feedback on code. These approaches offer varying levels of complexity and integration, catering to different use cases and preferences.

#### Approach 1: The Simple `review` Function[​](#approach-1-the-simple-review-function "Direct link to approach-1-the-simple-review-function")

Our first approach is a straightforward `review` function. This method is less intrusive and does not modify the original function's behavior. It's ideal for scenarios where you want to manually trigger a review of the function's code and don't need to see the output result of the function to have context of the LLM's analysis.

* **How it works**: The `review` function takes a function and an MLflow model as arguments. It then uses the model to evaluate the source code of the given function.
* **Manual Invocation**: You need to explicitly call `review(my_func)` to review `my_func`. This approach is manual and does not automatically integrate with function calls.
* **Simplicity**: This method is simpler and more direct, making it suitable for one-off evaluations or for use cases where automatic review is not required.

#### Approach 2: The Advanced `code_inspector` Decorator[​](#approach-2-the-advanced-code_inspector-decorator "Direct link to approach-2-the-advanced-code_inspector-decorator")

The second approach is an advanced decorator, `code_inspector`, which integrates more deeply by automatically reviewing the function and allowing the function's evaluation to execute. This can be helpful for more complex functions where the output result, in conjunction with the evaluation from the code helper, can allow for a deeper understanding of any observed logical flaws.

* **Automatic Evaluation**: When applied as a decorator, `code_inspector` evaluates the function's code automatically on each call.
* **Error Handling**: Includes robust error handling within the evaluation process.
* **Function Modification**: This method modifies the function's behavior, incorporating an automatic review process.

#### Introduction to the `review` Function[​](#introduction-to-the-review-function "Direct link to introduction-to-the-review-function")

We'll start by examining the `review` function. This function will be defined in the next cell of our Jupyter notebook. Here's a quick overview of what the `review` function does:

* **Inputs**: It takes a function and an MLflow model as inputs.
* **Functionality**: Extracts the source code of the input function and uses the MLflow model to provide feedback on it.
* **Error Handling**: Enhanced with error handling to manage exceptions gracefully.

In the following Jupyter notebook cell, you'll see the implementation of the `review` function, demonstrating its simplicity and effectiveness in evaluating code.

***

After exploring the `review` function, we will delve into the more complex `code_inspector` decorator to understand its automatic evaluation process and error handling mechanisms.

python

```
def review(func, model):
  """
  Function to review the source code of a given function using a specified MLflow model.

  Args:
  func (function): The function to review.
  model (MLflow pyfunc model): The MLflow pyfunc model used for evaluation.

  Returns:
  The model's prediction or an error message.
  """
  try:
      # Extracting the source code of the function
      source_code = inspect.getsource(func)

      # Using the model to predict/evaluate the source code
      prediction = model.predict([source_code])
      print(prediction)
  except Exception as e:
      # Handling any exceptions that occur and returning an error message
      return f"Error during model prediction or source code inspection: {e}"
```

### Explanation and Review of `process_data` Function[​](#explanation-and-review-of-process_data-function "Direct link to explanation-and-review-of-process_data-function")

#### Function Overview[​](#function-overview "Direct link to Function Overview")

The `process_data` function aims to process a list by identifying unique elements and counting duplicates. However, the implementation has several inefficiencies and readability issues.

#### Suggested Revised Code[​](#suggested-revised-code "Direct link to Suggested Revised Code")

The output from GPT-4's analysis provides clear and concise feedback, precisely as the prompt instructed it to. With the MLflow integration of this application, the simplicity of using the tool is evident, allowing us to get high-quality guidance during the development process with as little as a single, simple function call.

python

```
def process_data(lst):
  s = 0
  q = []
  for i in range(len(lst)):
      a = lst[i]
      for j in range(i + 1, len(lst)):
          b = lst[j]
          if a == b:
              s += 1
          else:
              q.append(b)
  rslt = [x for x in lst if x not in q]
  k = []
  for i in rslt:
      if i not in k:
          k.append(i)
  final_data = sorted(k, reverse=True)
  return final_data, s


review(process_data, loaded_helper)
```

````
Your code seems to be trying to find the count of duplicate elements in a list
and return a sorted list of unique elements in descending order along with the
count of duplicates. Here are some suggestions to improve your code:

1. **Errors or Bugs**: There are no syntax errors in your code, but the logic is
flawed. The variable `s` is supposed to count the number of duplicate elements,
but it only counts the number of times an element is equal to another element in
the list, which is not the same thing. Also, the way you're trying to get unique
elements is inefficient and can lead to incorrect results.

2. **Optimizing Code Efficiency and Structure**: You can use Python's built-in
`set` and `list` data structures to simplify your code and make it more
efficient. A `set` in Python is an unordered collection of unique elements. You
can convert your list to a set to remove duplicates, and then convert it back to
a list. The length of the original list minus the length of the list with
duplicates removed will give you the number of duplicate elements.

3. **Enhancing Code Readability and Maintainability**: Use meaningful variable
names to make your code easier to understand. Also, add comments to explain what
each part of your code does.

4. **Best Practice Advice**: It's a good practice to write a docstring at the
beginning of your function to explain what it does.

Here's a revised version of your code incorporating these suggestions:

```python
def process_data(lst):
  """
  This function takes a list as input, removes duplicate elements, sorts the remaining elements in descending order,
  and counts the number of duplicate elements in the original list.
  It returns a tuple containing the sorted list of unique elements and the count of duplicate elements.
  """
  # Convert the list to a set to remove duplicates, then convert it back to a list
  unique_elements = list(set(lst))
  
  # Sort the list of unique elements in descending order
  sorted_unique_elements = sorted(unique_elements, reverse=True)
  
  # Count the number of duplicate elements
  duplicate_count = len(lst) - len(unique_elements)
  
  return sorted_unique_elements, duplicate_count
```
This version of the code is simpler, more efficient, and easier to understand.
It also correctly counts the number of duplicate elements in the list.
````

### The `code_inspector` Decorator Function[​](#the-code_inspector-decorator-function "Direct link to the-code_inspector-decorator-function")

The `code_inspector` function is a Python decorator designed to augment functions with automatic code review capabilities using an MLflow pyfunc model. This decorator enhances the functionality of functions, allowing them to be automatically reviewed for code quality and correctness using an MLflow pyfunc model, thereby enriching the development and learning experience. As compared to the above implementation for the `review()` function, this approach will allow the function to be executed when called, enhancing the contextual information when paired with the automated code review.

python

```
import functools
import inspect


def code_inspector(model):
  """
  Decorator for automatic code review using an MLflow pyfunc model.

  Args:
      model: The MLflow pyfunc model for code evaluation.
  """

  def decorator_check_my_function(func):
      # Decorator that wraps around the given function
      @functools.wraps(func)
      def wrapper(*args, **kwargs):
          try:
              # Extracting the source code of the decorated function
              parsed_func = inspect.getsource(func)

              # Using the MLflow model to evaluate the extracted source code
              response = model.predict([parsed_func])

              # Printing the response for code review feedback
              print(response)

          except Exception as e:
              # Handling exceptions during model prediction or source code extraction
              print("Error during model prediction or formatting:", e)

          # Executing and returning the original function's output
          return func(*args, **kwargs)

      return wrapper

  return decorator_check_my_function
```

### First Usage Trial: The `summing_function` with `code_inspector`[​](#first-usage-trial-the-summing_function-with-code_inspector "Direct link to first-usage-trial-the-summing_function-with-code_inspector")

We apply the `code_inspector` decorator to a function named `summing_function`. This function is designed to calculate the sum of sums for a given range. Here's an insight into its functionality and the enhancement brought by `code_inspector`:

1. **Function Overview**:

   * `summing_function` calculates the cumulative sum of numbers up to `n`. It does so by iterating over a range and summing the intermediate sums at each step.
   * A dictionary, `intermediate_sums`, is used to store these sums, which are then aggregated to find the final sum.

2. **Using `code_inspector`**:

   * The function is decorated with `code_inspector(loaded_helper)`. This means that each time `summing_function` is called, the MLflow model loaded as `loaded_helper` analyzes its code.
   * The decorator provides real-time feedback on the code, assessing aspects like quality, efficiency, and best practices.

3. **Educational Benefit**:

   * This setup is ideal for learning, allowing users to receive instant, actionable feedback on their code.
   * It offers a practical way to understand the logic behind the function and learn coding optimizations and improvements.

By integrating `code_inspector` with `summing_function`, the tutorial demonstrates an interactive approach to enhancing coding skills, with immediate feedback aiding in understanding and improvement.

Before proceeding to see the response from GPT-4, can you identify all of the issues in this code (there are more than a few)?

python

```
@code_inspector(loaded_helper)
def summing_function(n):
  sum_result = 0

  intermediate_sums = {}

  for i in range(1, n + 1):
      intermediate_sums[str(i)] = sum(x for x in range(1, i + 1))
      for key in intermediate_sums:
          if key == str(i):
              sum_result = intermediate_sums[key]  # noqa: F841

  final_sum = sum([intermediate_sums[key] for key in intermediate_sums if int(key) == n])

  return int(str(final_sum))
```

### Execution and Analysis of `summing_function(1000)`[​](#execution-and-analysis-of-summing_function1000 "Direct link to execution-and-analysis-of-summing_function1000")

When we execute `summing_function(1000)`, several key processes take place, utilizing our custom MLflow model through the `code_inspector` decorator. Here's what happens:

1. **Decorator Activation**:

   * On calling `summing_function(1000)`, the `code_inspector` decorator is the first to activate. This decorator is designed to use the `loaded_helper` model to analyze the decorated function.

2. **Model Analyzes the Function Code**:

   * `code_inspector` retrieves the source code of `summing_function` using the `inspect` module.
   * This source code is then passed to the `loaded_helper` model, which performs an analysis based on its training and provided instructions. The model predicts feedback on code quality, efficiency, and best practices.

3. **Feedback Presentation**:

   * The feedback generated by the model is printed out. This feedback might include suggestions for code optimization, identification of potential errors, or general advice on coding practices.
   * This step provides an educational insight into the code quality before the function executes its logic.

4. **Function Execution**:

   * After the feedback is displayed, the `summing_function` proceeds to execute with the input `1000`.
   * The function calculates the cumulative sum of numbers up to 1000, but due to its inefficient implementation, this process may be slower and more resource-intensive than necessary.

5. **Return of Result**:

   * The function returns the final computed sum, which is the result of the summing logic implemented within it.

This demonstration highlights how the `code_inspector` decorator, combined with our custom MLflow model, provides a unique, real-time code analysis and feedback mechanism, enhancing the learning and development experience in an interactive environment.

python

```
summing_function(1000)
```

````
Here's a detailed review of your code:

1. Errors or bugs: There are no syntax errors in your code, but there is a
logical error. The summing_function is supposed to calculate the sum of numbers
from 1 to n, but it's doing more than that. It's calculating the sum of numbers
from 1 to i for each i in the range 1 to n, storing these sums in a dictionary,
and then summing these sums again. This is unnecessary and inefficient.

2. Optimizing code efficiency and structure: The function can be simplified
significantly. The sum of numbers from 1 to n can be calculated directly using
the formula n*(n+1)/2. This eliminates the need for the loop and the dictionary,
making the function much more efficient.

3. Enhancing code readability and maintainability: The code can be made more
readable by simplifying it and removing unnecessary parts. The use of the
dictionary and the conversion of numbers to strings and back to numbers is
confusing and unnecessary.

4. Best practice advice: In Python, it's best to keep things simple and
readable. Avoid unnecessary complexity and use built-in functions and operators
where possible. Also, avoid unnecessary type conversions.

Here's a simplified version of your function:

```python
def summing_function(n):
  return n * (n + 1) // 2
```

This function does exactly the same thing as your original function, but it's
much simpler, more efficient, and more readable.
````

```
500500
```

### Analysis of `one_liner` Function[​](#analysis-of-one_liner-function "Direct link to analysis-of-one_liner-function")

The `one_liner` function, decorated with `code_inspector`, demonstrates an interesting approach but has several issues:

1. **Complexity**: The function uses nested lambda expressions to calculate the factorial of `n`. While compact, this approach is overly complex and hard to read, making the code less maintainable and understandable.

2. **Readability**: Good coding practice emphasizes readability, which is compromised here due to the one-liner approach. Such code can be challenging to debug and understand, especially for those unfamiliar with the specific coding style.

3. **Best Practices**: While demonstrating Python's capabilities for writing concise code, this example strays from common best practices, particularly in terms of clarity and simplicity.

When reviewed by the `code_inspector` model, these issues are likely to be highlighted, emphasizing the importance of balancing clever coding with readability and maintainability.

python

```
@code_inspector(loaded_helper)
def one_liner(n):
  return (
      (lambda f, n: f(f, n))(lambda f, n: n * f(f, n - 1) if n > 1 else 1, n)
      if isinstance(n, int) and n >= 0
      else "Invalid input"
  )
```

python

```
one_liner(10)
```

````
The code you've provided is a one-liner function that calculates the factorial
of a given number `n`. It uses a lambda function to recursively calculate the
factorial. Here's a review of your code:

1. Errors or bugs: There are no syntax errors or bugs in your code. It correctly
checks if the input is a non-negative integer and calculates the factorial. If
the input is not a non-negative integer, it returns "Invalid input".

2. Optimizing code efficiency and structure: The code is already quite efficient
as it uses recursion to calculate the factorial. However, the structure of the
code is quite complex due to the use of a lambda function for recursion. This
can make the code difficult to understand and maintain.

3. Enhancing code readability and maintainability: The code could be made more
readable by breaking it down into multiple lines and adding comments to explain
what each part of the code does. The use of a lambda function for recursion
makes the code more difficult to understand than necessary. A more
straightforward recursive function could be used instead.

4. Best practice advice: In Python, it's generally recommended to use clear and
simple code over complex one-liners. This is because clear code is easier to
read, understand, and maintain. While one-liners can be fun and clever, they can
also be difficult to understand and debug.

Here's a revised version of your code that's easier to understand:

```python
def factorial(n):
  # Check if the input is a non-negative integer
  if not isinstance(n, int) or n < 0:
      return "Invalid input"
  
  # Base case: factorial of 0 is 1
  if n == 0:
      return 1
  
  # Recursive case: n! = n * (n-1)!
  return n * factorial(n - 1)
```

This version of the code does the same thing as your original code, but it's
much easier to understand because it uses a straightforward recursive function
instead of a lambda function.
````

```
3628800
```

### Reviewing `find_phone_numbers` Function[​](#reviewing-find_phone_numbers-function "Direct link to reviewing-find_phone_numbers-function")

The `find_phone_numbers` function, enhanced with the `code_inspector`, is designed to extract phone numbers from a given text but contains a few notable issues and expected behaviors:

1. **Typographical Error**: The function incorrectly uses `re.complie` instead of `re.compile`, leading to a runtime exception.

2. **Pattern Matching Inaccuracy**: The regular expression pattern `"(\d{3})-\d{3}-\d{4}"`, while formatted for typical phone numbers, can result in errors if a phone number does not appear in the string.

3. **Lack of Error Handling**: Directly accessing the first element in `phone_numbers` without checking if the list is empty can lead to an `IndexError`.

4. **Import Statement Position**: The `import re` statement is inside the function, which is unconventional. Imports are typically placed at the top of a script for clarity.

5. **Analysis and Exception Handling**:

   * Due to how we crafted our custom MLflow model in `code_inspector`, the function's issues will be analyzed and feedback will be returned before the function's logic is executed.
   * After this analysis, the execution of the function will likely result in an exception (due to the typographical error), demonstrating the importance of careful code review and testing.

The `code_inspector` model's review will highlight these coding missteps, emphasizing the value of proper syntax, pattern accuracy, and error handling in Python programming.

python

```
import re


@code_inspector(loaded_helper)
def find_phone_numbers(text):
  pattern = r"(d{3})-d{3}-d{4}"

  compiled_pattern = re.complie(pattern)

  phone_numbers = compiled_pattern.findall(text)
  first_number = phone_numbers[0]

  print(f"First found phone number: {first_number}")
  return phone_numbers
```

python

```
find_phone_numbers("Give us a call at 888-867-5309")
```

````
Here's a detailed review of your code:

1. Errors or Bugs:
 - There's a typo in the `re.compile` function. You've written `re.complie`
instead of `re.compile`.

2. Suggestions for Optimizing Code Efficiency and Structure:
 - The import statement `import re` is inside the function. It's a good
practice to keep all import statements at the top of the file. This makes it
easier to see what modules are being used in the script.
 - The function will throw an error if no phone numbers are found in the text
because you're trying to access the first element of `phone_numbers` without
checking if it exists. You should add a check to see if any phone numbers were
found before trying to access the first one.

3. Recommendations for Enhancing Code Readability and Maintainability:
 - The function name `find_phone_numbers` is clear and descriptive, which is
good. However, the variable `pattern` could be more descriptive. Consider
renaming it to `phone_number_pattern` or something similar.
 - You should add docstrings to your function to describe what it does, what
its parameters are, and what it returns.

4. Best Practice Advice:
 - Use exception handling to catch potential errors and make your program more
robust.
 - Avoid using print statements in functions that are meant to return a value.
If you want to debug, consider using logging instead.

Here's how you could improve your code:

```python
import re

def find_phone_numbers(text):
  """
  This function finds all phone numbers in the given text.

  Parameters:
  text (str): The text to search for phone numbers.

  Returns:
  list: A list of all found phone numbers.
  """
  phone_number_pattern = "(d{3})-d{3}-d{4}"
  compiled_pattern = re.compile(phone_number_pattern)

  phone_numbers = compiled_pattern.findall(text)

  if phone_numbers:
      print(f"First found phone number: {phone_numbers[0]}")

  return phone_numbers
```

Remember, the print statement is not recommended in production code. It's there
for the sake of this example.
````

```
---------------------------------------------------------------------------
```

```
AttributeError                            Traceback (most recent call last)
```

```
/var/folders/cd/n8n0rm2x53l_s0xv_j_xklb00000gp/T/ipykernel_38633/78508464.py in <cell line: 1>()
----> 1 find_phone_numbers("Give us a call at 888-867-5309")
```

```
/var/folders/cd/n8n0rm2x53l_s0xv_j_xklb00000gp/T/ipykernel_38633/2021999358.py in wrapper(*args, **kwargs)
   18             except Exception as e:
   19                 print("Error during model prediction or formatting:", e)
---> 20             return func(*args, **kwargs)
   21 
   22         return wrapper
```

```
/var/folders/cd/n8n0rm2x53l_s0xv_j_xklb00000gp/T/ipykernel_38633/773713950.py in find_phone_numbers(text)
    5     import re
    6 
----> 7     compiled_pattern = re.complie(pattern)
    8 
    9     phone_numbers = compiled_pattern.findall(text)
```

```
AttributeError: module 're' has no attribute 'complie'
```

### Conclusion: Harnessing the Power of MLflow in AI-Assisted Development[​](#conclusion-harnessing-the-power-of-mlflow-in-ai-assisted-development "Direct link to Conclusion: Harnessing the Power of MLflow in AI-Assisted Development")

As we conclude this tutorial, we have traversed through the integration of OpenAI's language models with the robust capabilities of MLflow, creating a powerful toolkit for AI-assisted software development. Here's a recap of our journey and the key takeaways:

1. **Integrating OpenAI with MLflow**:

   * We explored how to seamlessly integrate OpenAI's advanced language models within the MLflow framework. This integration highlighted the potential of combining AI intelligence with robust model management.

2. **Implementing a Custom Python Model**:

   * Our journey included creating a custom `CodeHelper` model, which showcased MLflow's flexibility in handling custom Python functions. This model significantly enhanced the user experience by formatting AI responses into a more readable format.

3. **Real-Time Code Analysis and Feedback**:

   * By employing the `code_inspector` decorator, we demonstrated MLflow's utility in providing real-time, insightful feedback on code quality and efficiency, fostering a learning environment that guides towards best coding practices.

4. **Handling Complex Code Analysis**:

   * The tutorial presented complex code examples, revealing how MLflow, combined with OpenAI, can handle intricate code analysis, offering suggestions and identifying potential issues.

5. **Learning from Interactive Feedback**:

   * The interactive feedback loop, enabled by our MLflow model, illustrated a practical approach to learning and improving coding skills, making this toolset particularly valuable for educational and development purposes.

6. **Flexibility and Scalability of MLflow**:

   * Throughout the tutorial, MLflow's flexibility and scalability were evident. Whether it's managing simple Python functions or integrating state-of-the-art AI models, MLflow proved to be an invaluable asset in streamlining the model management process.

In summary, this tutorial not only provided insights into effective coding practices but also underscored the versatility of MLflow in enhancing AI-assisted software development. It stands as a testament to how machine learning tools and models can be innovatively applied to improve code quality, efficiency, and the overall development experience.

### What's Next?[​](#whats-next "Direct link to What's Next?")

To continue your learning journey, see the additional [advanced tutorials for MLflow's OpenAI flavor](https://www.mlflow.org/docs/latest/genai/flavors/openai/index.html#advanced-tutorials).
