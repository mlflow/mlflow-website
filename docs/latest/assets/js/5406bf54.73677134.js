"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["9086"],{54772(e,t,o){o.r(t),o.d(t,{metadata:()=>n,default:()=>f,frontMatter:()=>c,contentTitle:()=>m,toc:()=>h,assets:()=>u});var n=JSON.parse('{"id":"traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction-ipynb","title":"Creating a Custom Model: \\"Add N\\" Model","description":"Download this notebook","source":"@site/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction-ipynb.mdx","sourceDirName":"traditional-ml/tutorials/creating-custom-pyfunc/notebooks","slug":"/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction","draft":false,"unlisted":false,"editUrl":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction.ipynb","tags":[],"version":"current","frontMatter":{"custom_edit_url":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction.ipynb","slug":"introduction"},"sidebar":"classicMLSidebar","previous":{"title":"Introduction to PythonModel","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/basic-pyfunc"},"next":{"title":"Customizing the `predict` method","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict"}}'),d=o(74848),i=o(28453),l=o(75940),a=o(75453),s=o(66354),r=o(42676);let c={custom_edit_url:"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction.ipynb",slug:"introduction"},m='Creating a Custom Model: "Add N" Model',u={},h=[{value:"Step 1: Define the Model Class",id:"step-1-define-the-model-class",level:4},{value:"Step 2: Save the Model",id:"step-2-save-the-model",level:4},{value:"Step 3: Load the Model",id:"step-3-load-the-model",level:4},{value:"Step 4: Evaluate the Model",id:"step-4-evaluate-the-model",level:4},{value:"Conclusion",id:"conclusion",level:4}];function p(e){let t={h1:"h1",h4:"h4",header:"header",p:"p",...(0,i.R)(),...e.components};return(0,d.jsxs)(d.Fragment,{children:[(0,d.jsx)(t.header,{children:(0,d.jsx)(t.h1,{id:"creating-a-custom-model-add-n-model",children:'Creating a Custom Model: "Add N" Model'})}),"\n",(0,d.jsxs)(t.p,{children:[(0,d.jsx)(r.O,{href:"https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction.ipynb",children:"Download this notebook"}),"\nOur first example is simple yet illustrative. We'll create a model that adds a specified numeric value, n, to all columns of a Pandas DataFrame input. This will demonstrate the process of defining a custom model, saving it, loading it back, and performing predictions."]}),"\n",(0,d.jsx)(t.h4,{id:"step-1-define-the-model-class",children:"Step 1: Define the Model Class"}),"\n",(0,d.jsx)(t.p,{children:"We begin by defining a Python class for our model. This class should inherit from mlflow.pyfunc.PythonModel and implement the necessary methods."}),"\n",(0,d.jsx)(l.d,{executionCount:1,children:`import mlflow.pyfunc


class AddN(mlflow.pyfunc.PythonModel):
  """
  A custom model that adds a specified value \`n\` to all columns of the input DataFrame.

  Attributes:
  -----------
  n : int
      The value to add to input columns.
  """

  def __init__(self, n):
      """
      Constructor method. Initializes the model with the specified value \`n\`.

      Parameters:
      -----------
      n : int
          The value to add to input columns.
      """
      self.n = n

  def predict(self, context, model_input, params=None):
      """
      Prediction method for the custom model.

      Parameters:
      -----------
      context : Any
          Ignored in this example. It's a placeholder for additional data or utility methods.

      model_input : pd.DataFrame
          The input DataFrame to which \`n\` should be added.

      params : dict, optional
          Additional prediction parameters. Ignored in this example.

      Returns:
      --------
      pd.DataFrame
          The input DataFrame with \`n\` added to all columns.
      """
      return model_input.apply(lambda column: column + self.n)`}),"\n",(0,d.jsx)(t.h4,{id:"step-2-save-the-model",children:"Step 2: Save the Model"}),"\n",(0,d.jsx)(t.p,{children:"Now that our model class is defined, we can instantiate it and save it using MLflow."}),"\n",(0,d.jsx)(l.d,{executionCount:2,children:`# Define the path to save the model
model_path = "/tmp/add_n_model"

# Create an instance of the model with \`n=5\`
add5_model = AddN(n=5)

# Save the model using MLflow
mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)`}),"\n",(0,d.jsx)(a.p,{isStderr:!0,children:`/Users/benjamin.wilson/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils.
warnings.warn("Setuptools is replacing distutils.")`}),"\n",(0,d.jsx)(t.h4,{id:"step-3-load-the-model",children:"Step 3: Load the Model"}),"\n",(0,d.jsx)(t.p,{children:"With our model saved, we can load it back using MLflow and then use it for predictions."}),"\n",(0,d.jsx)(l.d,{executionCount:3,children:`# Load the saved model
loaded_model = mlflow.pyfunc.load_model(model_path)`}),"\n",(0,d.jsx)(t.h4,{id:"step-4-evaluate-the-model",children:"Step 4: Evaluate the Model"}),"\n",(0,d.jsx)(t.p,{children:"Let's now use our loaded model to perform predictions on a sample input and verify its correctness."}),"\n",(0,d.jsx)(l.d,{executionCount:4,children:`import pandas as pd

# Define a sample input DataFrame
model_input = pd.DataFrame([range(10)])

# Use the loaded model to make predictions
model_output = loaded_model.predict(model_input)`}),"\n",(0,d.jsx)(l.d,{executionCount:5,children:"model_output"}),"\n",(0,d.jsx)(s.Q,{children:(0,d.jsx)("div",{dangerouslySetInnerHTML:{__html:`<div>
<style scoped>
  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: right;
  }
</style>
<table border="1" class="dataframe">
<thead>
  <tr style="text-align: right;">
    <th></th>
    <th>0</th>
    <th>1</th>
    <th>2</th>
    <th>3</th>
    <th>4</th>
    <th>5</th>
    <th>6</th>
    <th>7</th>
    <th>8</th>
    <th>9</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>0</th>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
    <td>11</td>
    <td>12</td>
    <td>13</td>
    <td>14</td>
  </tr>
</tbody>
</table>
</div>`}})}),"\n",(0,d.jsx)(t.h4,{id:"conclusion",children:"Conclusion"}),"\n",(0,d.jsx)(t.p,{children:"This simple example demonstrates the power and flexibility of MLflow's custom pyfunc. By encapsulating arbitrary Python code and its dependencies, custom pyfunc models ensure a consistent and unified interface for a wide range of use cases. Whether you're working with a niche machine learning framework, need custom preprocessing steps, or want to integrate unique prediction logic, pyfunc is the tool for the job."})]})}function f(e={}){let{wrapper:t}={...(0,i.R)(),...e.components};return t?(0,d.jsx)(t,{...e,children:(0,d.jsx)(p,{...e})}):p(e)}},75453(e,t,o){o.d(t,{p:()=>d});var n=o(74848);let d=({children:e,isStderr:t})=>(0,n.jsx)("pre",{style:{margin:0,borderRadius:0,background:"none",fontSize:"0.85rem",flexGrow:1,padding:"var(--padding-sm)"},children:e})},75940(e,t,o){o.d(t,{d:()=>i});var n=o(74848),d=o(37449);let i=({children:e,executionCount:t})=>(0,n.jsx)("div",{style:{flexGrow:1,minWidth:0,marginTop:"var(--padding-md)",width:"100%"},children:(0,n.jsx)(d.A,{className:"codeBlock_oJcR",language:"python",children:e})})},42676(e,t,o){o.d(t,{O:()=>l});var n=o(74848),d=o(96540);let i="3.9.1.dev0";function l({children:e,href:t}){let o=(0,d.useCallback)(async e=>{if(e.preventDefault(),window.gtag)try{window.gtag("event","notebook-download",{href:t})}catch{}i.includes("dev")||(t=t.replace(/\/master\//,`/v${i}/`));let o=await fetch(t),n=await o.blob(),d=window.URL.createObjectURL(n),l=document.createElement("a");l.style.display="none",l.href=d,l.download=t.split("/").pop(),document.body.appendChild(l),l.click(),window.URL.revokeObjectURL(d),document.body.removeChild(l)},[t]);return(0,n.jsx)("a",{className:"button button--primary",style:{marginBottom:"1rem",display:"block",width:"min-content"},href:t,download:!0,onClick:o,children:e})}},66354(e,t,o){o.d(t,{Q:()=>d});var n=o(74848);let d=({children:e})=>(0,n.jsx)("div",{style:{flexGrow:1,minWidth:0,fontSize:"0.8rem",width:"100%"},children:e})},52915(e,t,o){o.d(t,{A:()=>m});var n=o(74848);o(96540);var d=o(34164),i=o(71643),l=o(66697),a=o(92949),s=o(64560),r=o(47819);function c({language:e}){return(0,n.jsxs)("div",{className:(0,d.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,n.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,n.jsx)(r.A,{})]})}function m({className:e}){let{metadata:t}=(0,i.Ph)(),o=t.language||"text";return(0,n.jsxs)(l.A,{as:"div",className:(0,d.A)(e,t.className),children:[t.title&&(0,n.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,n.jsx)(a.A,{children:t.title})}),(0,n.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,n.jsx)(c,{language:o}),(0,n.jsx)(s.A,{})]})]})}}}]);