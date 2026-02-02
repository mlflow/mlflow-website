"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["2126"],{6073(e,t,i){i.r(t),i.d(t,{metadata:()=>n,default:()=>u,frontMatter:()=>d,contentTitle:()=>c,toc:()=>h,assets:()=>p});var n=JSON.parse('{"id":"flavors/dspy/notebooks/dspy_quickstart-ipynb","title":"DSPy Quickstart","description":"Download this notebook","source":"@site/docs/genai/flavors/dspy/notebooks/dspy_quickstart-ipynb.mdx","sourceDirName":"flavors/dspy/notebooks","slug":"/flavors/dspy/notebooks/dspy_quickstart","permalink":"/mlflow-website/docs/latest/genai/flavors/dspy/notebooks/dspy_quickstart","draft":false,"unlisted":false,"editUrl":"https://github.com/mlflow/mlflow/edit/master/docs/docs/genai/flavors/dspy/notebooks/dspy_quickstart.ipynb","tags":[],"version":"current","frontMatter":{"custom_edit_url":"https://github.com/mlflow/mlflow/edit/master/docs/docs/genai/flavors/dspy/notebooks/dspy_quickstart.ipynb","slug":"dspy_quickstart"},"sidebar":"genAISidebar","previous":{"title":"MLflow DSPy Flavor","permalink":"/mlflow-website/docs/latest/genai/flavors/dspy/"},"next":{"title":"Using DSPy Optimizers","permalink":"/mlflow-website/docs/latest/genai/flavors/dspy/optimizer"}}'),a=i(74848),s=i(28453),r=i(75940),l=i(75453);i(66354);var o=i(42676);let d={custom_edit_url:"https://github.com/mlflow/mlflow/edit/master/docs/docs/genai/flavors/dspy/notebooks/dspy_quickstart.ipynb",slug:"dspy_quickstart"},c="DSPy Quickstart",p={},h=[{value:"How does it work?",id:"how-does-it-work",level:2},{value:"This Demo",id:"this-demo",level:2},{value:"Setup",id:"setup",level:2},{value:"Set Up LLM",id:"set-up-llm",level:3},{value:"Create MLflow Experiment",id:"create-mlflow-experiment",level:3},{value:"Turn on Auto Tracing with MLflow",id:"turn-on-auto-tracing-with-mlflow",level:3},{value:"Set Up Data",id:"set-up-data",level:3},{value:"Set up DSPy Signature and Module",id:"set-up-dspy-signature-and-module",level:3},{value:"Run it!",id:"run-it",level:2},{value:"Hello World",id:"hello-world",level:3},{value:"Review Traces",id:"review-traces",level:3},{value:"Compilation",id:"compilation",level:2},{value:"Training",id:"training",level:3},{value:"Compare Pre/Post Compiled Accuracy",id:"compare-prepost-compiled-accuracy",level:3},{value:"Log and Load the Model with MLflow",id:"log-and-load-the-model-with-mlflow",level:2},{value:"Next Steps",id:"next-steps",level:2},{value:"DSPy",id:"dspy",level:3},{value:"MLflow",id:"mlflow",level:3}];function m(e){let t={a:"a",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",img:"img",li:"li",ol:"ol",p:"p",strong:"strong",ul:"ul",...(0,s.R)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(t.header,{children:(0,a.jsx)(t.h1,{id:"dspy-quickstart",children:"DSPy Quickstart"})}),"\n",(0,a.jsx)(o.O,{href:"https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/flavors/dspy/notebooks/dspy_quickstart.ipynb",children:"Download this notebook"}),"\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.a,{href:"https://dspy-docs.vercel.app/",children:"DSPy"}),' simplifies building language model (LM) pipelines by replacing manual prompt engineering with structured "text transformation graphs." These graphs use flexible, learning modules that automate and optimize LM tasks like reasoning, retrieval, and answering complex questions.']}),"\n",(0,a.jsx)(t.h2,{id:"how-does-it-work",children:"How does it work?"}),"\n",(0,a.jsx)(t.p,{children:"At a high level, DSPy optimizes prompts, selects the best language model, and can even fine-tune the model using training data."}),"\n",(0,a.jsxs)(t.p,{children:["The process follows these three steps, common to most DSPy ",(0,a.jsx)(t.a,{href:"https://dspy.ai/learn/optimization/optimizers/",children:"optimizers"}),":"]}),"\n",(0,a.jsxs)(t.ol,{children:["\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Candidate Generation"}),": DSPy finds all ",(0,a.jsx)(t.code,{children:"Predict"})," modules in the program and generates variations of instructions and demonstrations (e.g., examples for prompts). This step creates a set of possible candidates for the next stage."]}),"\n",(0,a.jsxs)(t.li,{children:[(0,a.jsx)(t.strong,{children:"Parameter Optimization"}),": DSPy then uses methods like random search, TPE, or Optuna to select the best candidate. Fine-tuning models can also be done at this stage."]}),"\n"]}),"\n",(0,a.jsx)(t.h2,{id:"this-demo",children:"This Demo"}),"\n",(0,a.jsx)(t.p,{children:"Below we create a simple program that demonstrates the power of DSPy. We will build a text classifier leveraging OpenAI. By the end of this tutorial, we will..."}),"\n",(0,a.jsxs)(t.ol,{children:["\n",(0,a.jsxs)(t.li,{children:["Define a ",(0,a.jsx)(t.a,{href:"https://dspy.ai/learn/programming/signatures/",children:"dspy.Signature"})," and ",(0,a.jsx)(t.a,{href:"https://dspy.ai/learn/programming/modules/",children:"dspy.Module"})," to perform text classification."]}),"\n",(0,a.jsxs)(t.li,{children:["Leverage ",(0,a.jsx)(t.a,{href:"https://dspy.ai/api/optimizers/SIMBA/",children:"dspy.SIMBA"})," to compile our module so it's better at classifying our text."]}),"\n",(0,a.jsx)(t.li,{children:"Analyze internal steps with MLflow Tracing."}),"\n",(0,a.jsx)(t.li,{children:"Log the compiled model with MLflow."}),"\n",(0,a.jsx)(t.li,{children:"Load the logged model and perform inference."}),"\n"]}),"\n",(0,a.jsx)(r.d,{executionCount:" ",children:'%pip install -U datasets openai "dspy>=3.0.3" "mlflow[genai]>=3.4.0"'}),"\n",(0,a.jsx)(t.h2,{id:"setup",children:"Setup"}),"\n",(0,a.jsx)(t.h3,{id:"set-up-llm",children:"Set Up LLM"}),"\n",(0,a.jsxs)(t.p,{children:["After installing the relevant dependencies, let's set up access to an OpenAI LLM. Here, will leverage OpenAI's ",(0,a.jsx)(t.code,{children:"gpt-4o-mini"})," model."]}),"\n",(0,a.jsx)(r.d,{executionCount:" ",children:`# Set OpenAI API Key to the environment variable. You can also pass the token to dspy.LM()
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI Key:")`}),"\n",(0,a.jsx)(r.d,{executionCount:4,children:`import dspy

# Define your model. We will use OpenAI for simplicity
model_name = "gpt-4o-mini"

# Note that an OPENAI_API_KEY environment must be present. You can also pass the token to dspy.LM()
lm = dspy.LM(
  model=f"openai/{model_name}",
  max_tokens=500,
  temperature=0.1,
)
dspy.settings.configure(lm=lm)`}),"\n",(0,a.jsx)(t.h3,{id:"create-mlflow-experiment",children:"Create MLflow Experiment"}),"\n",(0,a.jsx)(t.p,{children:'Create a new MLflow Experiment to track your DSPy models, metrics, parameters, and traces in one place. Although there is already a "default" experiment created in your workspace, it is highly recommended to create one for different tasks to organize experiment artifacts.'}),"\n",(0,a.jsx)(r.d,{executionCount:" ",children:`import mlflow

mlflow.set_experiment("DSPy Quickstart")`}),"\n",(0,a.jsx)(t.h3,{id:"turn-on-auto-tracing-with-mlflow",children:"Turn on Auto Tracing with MLflow"}),"\n",(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.a,{href:"https://mlflow.org/docs/latest/llms/tracing/index.html",children:"MLflow Tracing"})," is a powerful observability tool for monitoring and debugging what happens inside your DSPy modules, helping you identify potential bottlenecks or issues quickly. To enable DSPy tracing, you just need to call ",(0,a.jsx)(t.code,{children:"mlflow.dspy.autolog"})," and that's it!"]}),"\n",(0,a.jsx)(r.d,{executionCount:7,children:"mlflow.dspy.autolog()"}),"\n",(0,a.jsx)(t.h3,{id:"set-up-data",children:"Set Up Data"}),"\n",(0,a.jsxs)(t.p,{children:["Next, we will download the ",(0,a.jsx)(t.a,{href:"https://huggingface.co/datasets/yangwang825/reuters-21578",children:"Reuters 21578"})," dataset from Huggingface. We also write a utility to ensure that our train/test split has the same labels."]}),"\n",(0,a.jsx)(r.d,{executionCount:25,children:`import numpy as np
import pandas as pd
from datasets import load_dataset
from dspy.datasets.dataset import Dataset


def read_data_and_subset_to_categories() -> tuple[pd.DataFrame]:
  """
  Read the reuters-21578 dataset. Docs can be found in the url below:
  https://huggingface.co/datasets/yangwang825/reuters-21578
  """

  # Read train/test split
  dataset = load_dataset("yangwang825/reuters-21578")
  train = pd.DataFrame(dataset["train"])
  test = pd.DataFrame(dataset["test"])

  # Clean the labels
  label_map = {
      0: "acq",
      1: "crude",
      2: "earn",
      3: "grain",
      4: "interest",
      5: "money-fx",
      6: "ship",
      7: "trade",
  }

  train["label"] = train["label"].map(label_map)
  test["label"] = test["label"].map(label_map)

  return train, test


class CSVDataset(Dataset):
  def __init__(
      self, n_train_per_label: int = 20, n_test_per_label: int = 10, *args, **kwargs
  ) -> None:
      super().__init__(*args, **kwargs)
      self.n_train_per_label = n_train_per_label
      self.n_test_per_label = n_test_per_label

      self._create_train_test_split_and_ensure_labels()

  def _create_train_test_split_and_ensure_labels(self) -> None:
      """Perform a train/test split that ensure labels in \`dev\` are also in \`train\`."""
      # Read the data
      train_df, test_df = read_data_and_subset_to_categories()

      # Sample for each label
      train_samples_df = pd.concat(
          [group.sample(n=self.n_train_per_label) for _, group in train_df.groupby("label")]
      )
      test_samples_df = pd.concat(
          [group.sample(n=self.n_test_per_label) for _, group in test_df.groupby("label")]
      )

      # Set DSPy class variables
      self._train = train_samples_df.to_dict(orient="records")
      self._dev = test_samples_df.to_dict(orient="records")


# Limit to a small dataset to showcase the value of bootstrapping
dataset = CSVDataset(n_train_per_label=3, n_test_per_label=1)

# Create train and test sets containing DSPy
# Note that we must specify the expected input value name
train_dataset = [example.with_inputs("text") for example in dataset.train]
test_dataset = [example.with_inputs("text") for example in dataset.dev]
unique_train_labels = {example.label for example in dataset.train}

print(len(train_dataset), len(test_dataset))
print(f"Train labels: {unique_train_labels}")
print(train_dataset[0])`}),"\n",(0,a.jsx)(l.p,{children:`24 8
Train labels: {'interest', 'earn', 'money-fx', 'trade', 'ship', 'grain', 'acq', 'crude'}
Example({'text': 'bankamerica bacp raises prime rate to pct bankamerica corp following moves by other major banks said it has raised its prime rate to pct from pct effective today reuter', 'label': 'interest'}) (input_keys={'text'})`}),"\n",(0,a.jsx)(t.h3,{id:"set-up-dspy-signature-and-module",children:"Set up DSPy Signature and Module"}),"\n",(0,a.jsx)(t.p,{children:"Finally, we will define our task: text classification."}),"\n",(0,a.jsx)(t.p,{children:"There are a variety of ways you can provide guidelines to DSPy signature behavior. Currently, DSPy allows users to specify:"}),"\n",(0,a.jsxs)(t.ol,{children:["\n",(0,a.jsx)(t.li,{children:"A high-level goal via the class docstring."}),"\n",(0,a.jsx)(t.li,{children:"A set of input fields, with optional metadata."}),"\n",(0,a.jsx)(t.li,{children:"A set of output fields with optional metadata."}),"\n"]}),"\n",(0,a.jsx)(t.p,{children:"DSPy will then leverage this information to inform optimization."}),"\n",(0,a.jsxs)(t.p,{children:["In the below example, note that we simply provide the expected labels to ",(0,a.jsx)(t.code,{children:"output"})," field in the ",(0,a.jsx)(t.code,{children:"TextClassificationSignature"})," class. From this initial state, we'll look to use DSPy to learn to improve our classifier accuracy."]}),"\n",(0,a.jsx)(r.d,{executionCount:10,children:`class TextClassificationSignature(dspy.Signature):
  text = dspy.InputField()
  label = dspy.OutputField(
      desc=f"Label of predicted class. Possible labels are {unique_train_labels}"
  )


class TextClassifier(dspy.Module):
  def __init__(self):
      super().__init__()
      self.generate_classification = dspy.Predict(TextClassificationSignature)

  def forward(self, text: str):
      return self.generate_classification(text=text)`}),"\n",(0,a.jsx)(t.h2,{id:"run-it",children:"Run it!"}),"\n",(0,a.jsx)(t.h3,{id:"hello-world",children:"Hello World"}),"\n",(0,a.jsxs)(t.p,{children:["Let's demonstrate predicting via the DSPy module and associated signature. The program has correctly learned our labels from the signature ",(0,a.jsx)(t.code,{children:"desc"})," field and generates reasonable predictions."]}),"\n",(0,a.jsx)(r.d,{executionCount:21,children:`# Initilize our impact_improvement class
text_classifier = TextClassifier()

message = "I am interested in space"
print(text_classifier(text=message))

message = "I enjoy ice skating"
print(text_classifier(text=message))`}),"\n",(0,a.jsx)(l.p,{children:`Prediction(
  label='interest'
)
Prediction(
  label='interest'
)`}),"\n",(0,a.jsx)(t.h3,{id:"review-traces",children:"Review Traces"}),"\n",(0,a.jsxs)(t.ol,{children:["\n",(0,a.jsxs)(t.li,{children:["Open the MLflow UI and select the ",(0,a.jsx)(t.code,{children:'"DSPy Quickstart"'})," experiment."]}),"\n",(0,a.jsxs)(t.li,{children:["Go to the ",(0,a.jsx)(t.code,{children:'"Traces"'})," tab to view the generated traces."]}),"\n"]}),"\n",(0,a.jsx)(t.p,{children:"Now, you can observe how DSPy translates your query and interacts with the LLM. This feature is extremely valuable for debugging, iteratively refining components within your system, and monitoring models in production. While the module in this tutorial is relatively simple, the tracing feature becomes even more powerful as your model grows in complexity."}),"\n",(0,a.jsx)(t.p,{children:(0,a.jsx)(t.img,{alt:"MLflow DSPy Trace",src:i(23049).A+"",width:"3104",height:"1710"})}),"\n",(0,a.jsx)(t.h2,{id:"compilation",children:"Compilation"}),"\n",(0,a.jsx)(t.h3,{id:"training",children:"Training"}),"\n",(0,a.jsxs)(t.p,{children:["To train, we will leverage ",(0,a.jsx)(t.a,{href:"https://dspy.ai/api/optimizers/SIMBA/",children:"SIMBA"}),", an optimizer that will take bootstrap samples from our training set and leverage a random search strategy to optimize our predictive accuracy."]}),"\n",(0,a.jsxs)(t.p,{children:["Note that in the below example, we leverage a simple metric definition of exact match, as defined in ",(0,a.jsx)(t.code,{children:"validate_classification"}),", but ",(0,a.jsx)(t.a,{href:"https://dspy.ai/learn/evaluation/metrics/",children:"dspy.Metrics"})," can contain complex and LM-based logic to properly evaluate our accuracy."]}),"\n",(0,a.jsx)(r.d,{executionCount:" ",children:`from dspy import SIMBA


def validate_classification(example, prediction, trace=None) -> bool:
  return example.label == prediction.label


optimizer = SIMBA(
  metric=validate_classification,
  max_demos=2,
  bsize=12,
  num_threads=1,
)

compiled_pe = optimizer.compile(TextClassifier(), trainset=train_dataset)`}),"\n",(0,a.jsx)(t.h3,{id:"compare-prepost-compiled-accuracy",children:"Compare Pre/Post Compiled Accuracy"}),"\n",(0,a.jsx)(t.p,{children:"Finally, let's explore how well our trained model can predict on unseen test data."}),"\n",(0,a.jsx)(r.d,{executionCount:27,children:`def check_accuracy(classifier, test_data: pd.DataFrame = test_dataset) -> float:
  residuals = []
  predictions = []
  for example in test_data:
      prediction = classifier(text=example["text"])
      residuals.append(int(validate_classification(example, prediction)))
      predictions.append(prediction)
  return residuals, predictions


uncompiled_residuals, uncompiled_predictions = check_accuracy(TextClassifier())
print(f"Uncompiled accuracy: {np.mean(uncompiled_residuals)}")

compiled_residuals, compiled_predictions = check_accuracy(compiled_pe)
print(f"Compiled accuracy: {np.mean(compiled_residuals)}")`}),"\n",(0,a.jsx)(l.p,{children:`Uncompiled accuracy: 0.875
Compiled accuracy: 1.0`}),"\n",(0,a.jsx)(t.p,{children:"As shown above, our compiled accuracy is non-zero - our base LLM inferred meaning of the classification labels simply via our initial prompt. However, with DSPy training, the prompts, demonstrations, and input/output signatures have been updated to give our model to 100% accuracy on unseen data. That's a gain of 12 percentage points!"}),"\n",(0,a.jsx)(t.p,{children:"Let's take a look at each prediction in our test set."}),"\n",(0,a.jsx)(r.d,{executionCount:19,children:`for uncompiled_residual, uncompiled_prediction in zip(uncompiled_residuals, uncompiled_predictions):
  is_correct = "Correct" if bool(uncompiled_residual) else "Incorrect"
  prediction = uncompiled_prediction.label
  print(f"{is_correct} prediction: {' ' * (12 - len(is_correct))}{prediction}")`}),"\n",(0,a.jsx)(l.p,{children:`Incorrect prediction:    money-fx
Correct prediction:      crude
Correct prediction:      money-fx
Correct prediction:      earn
Incorrect prediction:    interest
Correct prediction:      grain
Correct prediction:      trade
Incorrect prediction:    trade`}),"\n",(0,a.jsx)(r.d,{executionCount:28,children:`for compiled_residual, compiled_prediction in zip(compiled_residuals, compiled_predictions):
  is_correct = "Correct" if bool(compiled_residual) else "Incorrect"
  prediction = compiled_prediction.label
  print(f"{is_correct} prediction: {' ' * (12 - len(is_correct))}{prediction}")`}),"\n",(0,a.jsx)(l.p,{children:`Correct prediction:      interest
Correct prediction:      crude
Correct prediction:      money-fx
Correct prediction:      earn
Correct prediction:      acq
Correct prediction:      grain
Correct prediction:      trade
Correct prediction:      ship`}),"\n",(0,a.jsx)(t.h2,{id:"log-and-load-the-model-with-mlflow",children:"Log and Load the Model with MLflow"}),"\n",(0,a.jsx)(t.p,{children:"Now that we have a compiled model with higher classification accuracy, let's leverage MLflow to log this model and load it for inference."}),"\n",(0,a.jsx)(r.d,{executionCount:21,children:`import mlflow

with mlflow.start_run():
  model_info = mlflow.dspy.log_model(
      compiled_pe,
      name="model",
      input_example="what is 2 + 2?",
  )`}),"\n",(0,a.jsx)(l.p,{children:"Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]"}),"\n",(0,a.jsxs)(t.p,{children:["Open the MLflow UI again and check the complied model is recorded to a new MLflow Run. Now you can load the model back for inference using ",(0,a.jsx)(t.code,{children:"mlflow.dspy.load_model"})," or ",(0,a.jsx)(t.code,{children:"mlflow.pyfunc.load_model"}),"."]}),"\n",(0,a.jsxs)(t.p,{children:["\u{1F4A1} MLflow will remember the environment configuration stored in ",(0,a.jsx)(t.code,{children:"dspy.settings"}),", such as the language model (LM) used during the experiment. This ensures excellent reproducibility for your experiment."]}),"\n",(0,a.jsx)(r.d,{executionCount:22,children:`# Define input text
print("
==============Input Text============")
text = test_dataset[0]["text"]
print(f"Text: {text}")

# Inference with original DSPy object
print("
--------------Original DSPy Prediction------------")
print(compiled_pe(text=text).label)

# Inference with loaded DSPy object
print("
--------------Loaded DSPy Prediction------------")
loaded_model_dspy = mlflow.dspy.load_model(model_info.model_uri)
print(loaded_model_dspy(text=text).label)

# Inference with MLflow PyFunc API
loaded_model_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
print("
--------------PyFunc Prediction------------")
print(loaded_model_pyfunc.predict(text)["label"])`}),"\n",(0,a.jsx)(l.p,{children:`
==============Input Text============
Text: top discount rate at u k bill tender rises to pct

--------------Original DSPy Prediction------------
interest

--------------Loaded DSPy Prediction------------
interest

--------------PyFunc Prediction------------
interest`}),"\n",(0,a.jsx)(t.h2,{id:"next-steps",children:"Next Steps"}),"\n",(0,a.jsx)(t.p,{children:"This example demonstrates how DSPy works. Below are some potential extensions for improving this project, both with DSPy and MLflow."}),"\n",(0,a.jsx)(t.h3,{id:"dspy",children:"DSPy"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsx)(t.li,{children:"Use real-world data for the classifier."}),"\n",(0,a.jsx)(t.li,{children:"Experiment with different optimizers."}),"\n",(0,a.jsxs)(t.li,{children:["For more in-depth examples, check out the ",(0,a.jsx)(t.a,{href:"https://dspy.ai/tutorials/",children:"tutorials"})," and ",(0,a.jsx)(t.a,{href:"https://dspy.ai/learn/",children:"documentation"}),"."]}),"\n"]}),"\n",(0,a.jsx)(t.h3,{id:"mlflow",children:"MLflow"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsx)(t.li,{children:"Deploy the model using MLflow serving."}),"\n",(0,a.jsx)(t.li,{children:"Use MLflow to experiment with various optimization strategies."}),"\n",(0,a.jsxs)(t.li,{children:["Track your DSPy experiments using ",(0,a.jsx)(t.a,{href:"https://mlflow.org/docs/latest/genai/flavors/dspy/optimizer/",children:"DSPy Optimizer Autologging"}),"."]}),"\n"]}),"\n",(0,a.jsx)(t.p,{children:"Happy coding!"})]})}function u(e={}){let{wrapper:t}={...(0,s.R)(),...e.components};return t?(0,a.jsx)(t,{...e,children:(0,a.jsx)(m,{...e})}):m(e)}},23049(e,t,i){i.d(t,{A:()=>n});let n=i.p+"assets/images/dspy-trace-bd339ce15bda9cbb5f88a48a24c2bbf4.png"},75453(e,t,i){i.d(t,{p:()=>a});var n=i(74848);let a=({children:e,isStderr:t})=>(0,n.jsx)("pre",{style:{margin:0,borderRadius:0,background:"none",fontSize:"0.85rem",flexGrow:1,padding:"var(--padding-sm)"},children:e})},75940(e,t,i){i.d(t,{d:()=>s});var n=i(74848),a=i(37449);let s=({children:e,executionCount:t})=>(0,n.jsx)("div",{style:{flexGrow:1,minWidth:0,marginTop:"var(--padding-md)",width:"100%"},children:(0,n.jsx)(a.A,{className:"codeBlock_oJcR",language:"python",children:e})})},42676(e,t,i){i.d(t,{O:()=>r});var n=i(74848),a=i(96540);let s="3.9.1.dev0";function r({children:e,href:t}){let i=(0,a.useCallback)(async e=>{if(e.preventDefault(),window.gtag)try{window.gtag("event","notebook-download",{href:t})}catch{}s.includes("dev")||(t=t.replace(/\/master\//,`/v${s}/`));let i=await fetch(t),n=await i.blob(),a=window.URL.createObjectURL(n),r=document.createElement("a");r.style.display="none",r.href=a,r.download=t.split("/").pop(),document.body.appendChild(r),r.click(),window.URL.revokeObjectURL(a),document.body.removeChild(r)},[t]);return(0,n.jsx)("a",{className:"button button--primary",style:{marginBottom:"1rem",display:"block",width:"min-content"},href:t,download:!0,onClick:i,children:e})}},66354(e,t,i){i.d(t,{Q:()=>a});var n=i(74848);let a=({children:e})=>(0,n.jsx)("div",{style:{flexGrow:1,minWidth:0,fontSize:"0.8rem",width:"100%"},children:e})},52915(e,t,i){i.d(t,{A:()=>p});var n=i(74848);i(96540);var a=i(34164),s=i(71643),r=i(66697),l=i(92949),o=i(64560),d=i(47819);function c({language:e}){return(0,n.jsxs)("div",{className:(0,a.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,n.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,n.jsx)(d.A,{})]})}function p({className:e}){let{metadata:t}=(0,s.Ph)(),i=t.language||"text";return(0,n.jsxs)(r.A,{as:"div",className:(0,a.A)(e,t.className),children:[t.title&&(0,n.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,n.jsx)(l.A,{children:t.title})}),(0,n.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,n.jsx)(c,{language:i}),(0,n.jsx)(o.A,{})]})]})}}}]);