"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["5108"],{57956(e,t,n){n.r(t),n.d(t,{metadata:()=>i,default:()=>p,frontMatter:()=>l,contentTitle:()=>c,toc:()=>u,assets:()=>d});var i=JSON.parse('{"id":"eval-monitor/notebooks/quickstart-eval-ipynb","title":"GenAI Evaluation Quickstart","description":"Download this notebook","source":"@site/docs/genai/eval-monitor/notebooks/quickstart-eval-ipynb.mdx","sourceDirName":"eval-monitor/notebooks","slug":"/eval-monitor/notebooks/quickstart-eval","permalink":"/mlflow-website/docs/latest/genai/eval-monitor/notebooks/quickstart-eval","draft":false,"unlisted":false,"editUrl":"https://github.com/mlflow/mlflow/edit/master/docs/docs/genai/eval-monitor/notebooks/quickstart-eval.ipynb","tags":[],"version":"current","frontMatter":{"custom_edit_url":"https://github.com/mlflow/mlflow/edit/master/docs/docs/genai/eval-monitor/notebooks/quickstart-eval.ipynb","slug":"quickstart-eval"}}'),s=n(74848),o=n(28453),r=n(75940);n(75453),n(66354);var a=n(42676);let l={custom_edit_url:"https://github.com/mlflow/mlflow/edit/master/docs/docs/genai/eval-monitor/notebooks/quickstart-eval.ipynb",slug:"quickstart-eval"},c="GenAI Evaluation Quickstart",d={},u=[{value:"Prerequisites",id:"prerequisites",level:2},{value:"Step 1: Set up your environment",id:"step-1-set-up-your-environment",level:2},{value:"Connect to MLflow",id:"connect-to-mlflow",level:3},{value:"Step 2: Define your mock agent&#39;s prediction function",id:"step-2-define-your-mock-agents-prediction-function",level:2},{value:"Step 3: Prepare an evaluation dataset",id:"step-3-prepare-an-evaluation-dataset",level:2},{value:"Step 4: Define evaluation criteria using Scorers",id:"step-4-define-evaluation-criteria-using-scorers",level:2},{value:"Step 5: Run the evaluation",id:"step-5-run-the-evaluation",level:2},{value:"View Results",id:"view-results",level:2},{value:"Summary",id:"summary",level:2}];function h(e){let t={a:"a",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,o.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(t.header,{children:(0,s.jsx)(t.h1,{id:"genai-evaluation-quickstart",children:"GenAI Evaluation Quickstart"})}),"\n",(0,s.jsx)(a.O,{href:"https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/eval-monitor/notebooks/quickstart-eval.ipynb",children:"Download this notebook"}),"\n",(0,s.jsx)(t.p,{children:"This notebook will walk you through evaluating your GenAI applications with MLflow's comprehensive evaluation framework. In less than 5 minutes, you'll learn how to evaluate LLM outputs, use built-in and custom evaluation criteria, and analyze results in the MLflow UI."}),"\n",(0,s.jsx)(t.h2,{id:"prerequisites",children:"Prerequisites"}),"\n",(0,s.jsx)(t.p,{children:"Install the required packages by running:"}),"\n",(0,s.jsx)(r.d,{executionCount:" ",children:"pip install 'mlflow[genai]' openai"}),"\n",(0,s.jsx)(t.h2,{id:"step-1-set-up-your-environment",children:"Step 1: Set up your environment"}),"\n",(0,s.jsx)(t.h3,{id:"connect-to-mlflow",children:"Connect to MLflow"}),"\n",(0,s.jsx)(t.p,{children:"Before running evaluation, start the MLflow tracking server:"}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-bash",children:"mlflow server\n"})}),"\n",(0,s.jsxs)(t.p,{children:["This starts MLflow at ",(0,s.jsx)(t.a,{href:"http://localhost:5000",children:"http://localhost:5000"})," with a SQLite backend (default)."]}),"\n",(0,s.jsx)(t.p,{children:"Then configure your environment in the notebook:"}),"\n",(0,s.jsx)(r.d,{executionCount:" ",children:`import os

import mlflow

# Configure environment
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your API key
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("GenAI Evaluation Quickstart")`}),"\n",(0,s.jsx)(t.h2,{id:"step-2-define-your-mock-agents-prediction-function",children:"Step 2: Define your mock agent's prediction function"}),"\n",(0,s.jsx)(t.p,{children:"Create a prediction function that takes a question and returns an answer using OpenAI's gpt-4o-mini model."}),"\n",(0,s.jsx)(r.d,{executionCount:" ",children:`from openai import OpenAI

client = OpenAI()


def my_agent(question: str) -> str:
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {
              "role": "system",
              "content": "You are a helpful assistant. Answer questions concisely.",
          },
          {"role": "user", "content": question},
      ],
  )
  return response.choices[0].message.content


# Wrapper function for evaluation
def qa_predict_fn(question: str) -> str:
  return my_agent(question)`}),"\n",(0,s.jsx)(t.h2,{id:"step-3-prepare-an-evaluation-dataset",children:"Step 3: Prepare an evaluation dataset"}),"\n",(0,s.jsxs)(t.p,{children:["The evaluation dataset is a list of samples, each with an ",(0,s.jsx)(t.code,{children:"inputs"})," and ",(0,s.jsx)(t.code,{children:"expectations"})," field."]}),"\n",(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.code,{children:"inputs"}),": The input to the ",(0,s.jsx)(t.code,{children:"predict_fn"})," function. ",(0,s.jsxs)(t.strong,{children:["The key(s) must match the parameter name of the ",(0,s.jsx)(t.code,{children:"predict_fn"})," function"]}),"."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.code,{children:"expectations"}),": The expected output from the ",(0,s.jsx)(t.code,{children:"predict_fn"})," function, namely, ground truth for the answer."]}),"\n"]}),"\n",(0,s.jsx)(r.d,{executionCount:" ",children:`# Define a simple Q&A dataset with questions and expected answers
eval_dataset = [
  {
      "inputs": {"question": "What is the capital of France?"},
      "expectations": {"expected_response": "Paris"},
  },
  {
      "inputs": {"question": "Who was the first person to build an airplane?"},
      "expectations": {"expected_response": "Wright Brothers"},
  },
  {
      "inputs": {"question": "Who wrote Romeo and Juliet?"},
      "expectations": {"expected_response": "William Shakespeare"},
  },
]`}),"\n",(0,s.jsx)(t.h2,{id:"step-4-define-evaluation-criteria-using-scorers",children:"Step 4: Define evaluation criteria using Scorers"}),"\n",(0,s.jsxs)(t.p,{children:[(0,s.jsx)(t.strong,{children:"Scorer"})," is a function that computes a score for a given input-output pair against various evaluation criteria.\nYou can use built-in scorers provided by MLflow for common evaluation criteria, as well as create your own custom scorers."]}),"\n",(0,s.jsx)(t.p,{children:"Here we use three scorers:"}),"\n",(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Correctness"}),': Evaluates if the answer is factually correct, using the "expected_response" field in the dataset.']}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Guidelines"}),": Evaluates if the answer meets the given guidelines."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"is_concise"}),": A custom scorer to judge if the answer is concise (less than 5 words)."]}),"\n"]}),"\n",(0,s.jsxs)(t.p,{children:["The first two scorers use LLMs to evaluate the response, so-called ",(0,s.jsx)(t.strong,{children:"LLM-as-a-Judge"}),"."]}),"\n",(0,s.jsx)(r.d,{executionCount:" ",children:`from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines


@scorer
def is_concise(outputs: str) -> bool:
  """Evaluate if the answer is concise (less than 5 words)"""
  return len(outputs.split()) <= 5


scorers = [
  Correctness(),
  Guidelines(name="is_english", guidelines="The answer must be in English"),
  is_concise,
]`}),"\n",(0,s.jsx)(t.h2,{id:"step-5-run-the-evaluation",children:"Step 5: Run the evaluation"}),"\n",(0,s.jsx)(t.p,{children:"Now we have all three components of the evaluation: dataset, prediction function, and scorers. Let's run the evaluation!"}),"\n",(0,s.jsx)(r.d,{executionCount:" ",children:`# Run evaluation
results = mlflow.genai.evaluate(
  data=eval_dataset,
  predict_fn=qa_predict_fn,
  scorers=scorers,
)`}),"\n",(0,s.jsx)(t.h2,{id:"view-results",children:"View Results"}),"\n",(0,s.jsx)(t.p,{children:"After running the evaluation, go to the MLflow UI and navigate to your experiment. You'll see the evaluation results with detailed metrics for each scorer."}),"\n",(0,s.jsx)(t.p,{children:"By clicking on each row in the table, you can see the detailed rationale behind the score and the trace of the prediction."}),"\n",(0,s.jsx)(t.h2,{id:"summary",children:"Summary"}),"\n",(0,s.jsx)(t.p,{children:"Congratulations! You've successfully:"}),"\n",(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsx)(t.li,{children:"\u2705 Set up MLflow GenAI Evaluation for your applications"}),"\n",(0,s.jsx)(t.li,{children:"\u2705 Evaluated a Q&A application with built-in scorers"}),"\n",(0,s.jsx)(t.li,{children:"\u2705 Created custom evaluation guidelines"}),"\n",(0,s.jsx)(t.li,{children:"\u2705 Learned to analyze results in the MLflow UI"}),"\n"]}),"\n",(0,s.jsx)(t.p,{children:"MLflow's evaluation framework provides comprehensive tools for assessing GenAI application quality, helping you build more reliable and effective AI systems."})]})}function p(e={}){let{wrapper:t}={...(0,o.R)(),...e.components};return t?(0,s.jsx)(t,{...e,children:(0,s.jsx)(h,{...e})}):h(e)}},75453(e,t,n){n.d(t,{p:()=>s});var i=n(74848);let s=({children:e,isStderr:t})=>(0,i.jsx)("pre",{style:{margin:0,borderRadius:0,background:"none",fontSize:"0.85rem",flexGrow:1,padding:"var(--padding-sm)"},children:e})},75940(e,t,n){n.d(t,{d:()=>o});var i=n(74848),s=n(37449);let o=({children:e,executionCount:t})=>(0,i.jsx)("div",{style:{flexGrow:1,minWidth:0,marginTop:"var(--padding-md)",width:"100%"},children:(0,i.jsx)(s.A,{className:"codeBlock_oJcR",language:"python",children:e})})},42676(e,t,n){n.d(t,{O:()=>r});var i=n(74848),s=n(96540);let o="3.9.1.dev0";function r({children:e,href:t}){let n=(0,s.useCallback)(async e=>{if(e.preventDefault(),window.gtag)try{window.gtag("event","notebook-download",{href:t})}catch{}o.includes("dev")||(t=t.replace(/\/master\//,`/v${o}/`));let n=await fetch(t),i=await n.blob(),s=window.URL.createObjectURL(i),r=document.createElement("a");r.style.display="none",r.href=s,r.download=t.split("/").pop(),document.body.appendChild(r),r.click(),window.URL.revokeObjectURL(s),document.body.removeChild(r)},[t]);return(0,i.jsx)("a",{className:"button button--primary",style:{marginBottom:"1rem",display:"block",width:"min-content"},href:t,download:!0,onClick:n,children:e})}},66354(e,t,n){n.d(t,{Q:()=>s});var i=n(74848);let s=({children:e})=>(0,i.jsx)("div",{style:{flexGrow:1,minWidth:0,fontSize:"0.8rem",width:"100%"},children:e})},52915(e,t,n){n.d(t,{A:()=>u});var i=n(74848);n(96540);var s=n(34164),o=n(71643),r=n(66697),a=n(92949),l=n(64560),c=n(47819);function d({language:e}){return(0,i.jsxs)("div",{className:(0,s.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,i.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,i.jsx)(c.A,{})]})}function u({className:e}){let{metadata:t}=(0,o.Ph)(),n=t.language||"text";return(0,i.jsxs)(r.A,{as:"div",className:(0,s.A)(e,t.className),children:[t.title&&(0,i.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,i.jsx)(a.A,{children:t.title})}),(0,i.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,i.jsx)(d,{language:n}),(0,i.jsx)(l.A,{})]})]})}}}]);