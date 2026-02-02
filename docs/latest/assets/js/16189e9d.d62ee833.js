"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["2905"],{11899(e,t,o){o.r(t),o.d(t,{metadata:()=>d,default:()=>_,frontMatter:()=>m,contentTitle:()=>u,toc:()=>h,assets:()=>c});var d=JSON.parse('{"id":"traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial-ipynb","title":"Deploy an MLflow PyFunc model with Model Serving","description":"Download this notebook","source":"@site/docs/classic-ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial-ipynb.mdx","sourceDirName":"traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks","slug":"/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial","draft":false,"unlisted":false,"editUrl":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial.ipynb","tags":[],"version":"current","frontMatter":{"custom_edit_url":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial.ipynb","slug":"MME_Tutorial"},"sidebar":"classicMLSidebar","previous":{"title":"Serving Multiple Models on a Single Endpoint with a Custom PyFunc Model","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/"},"next":{"title":"Scikit-learn","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/sklearn/"}}'),l=o(74848),r=o(28453),n=o(75940),i=o(75453),s=o(66354),a=o(42676);let m={custom_edit_url:"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial.ipynb",slug:"MME_Tutorial"},u="Deploy an MLflow PyFunc model with Model Serving",c={},h=[{value:"Install and import libraries",id:"install-and-import-libraries",level:2},{value:"1 - Create Some Sample Models",id:"1---create-some-sample-models",level:2},{value:"1.1 - Create Dummy Data",id:"11---create-dummy-data",level:4},{value:"1.2 - Train Models for Each Day of the Week",id:"12---train-models-for-each-day-of-the-week",level:4},{value:"1.3 - Test inference on our DOW models",id:"13---test-inference-on-our-dow-models",level:4},{value:"2 - Create an MME Custom PyFunc Model",id:"2---create-an-mme-custom-pyfunc-model",level:2},{value:"2.1 - Create a Child Implementation of <code>mlflow.pyfunc.PythonModel</code>",id:"21---create-a-child-implementation-of-mlflowpyfuncpythonmodel",level:4},{value:"2.2 - Test our Implementation",id:"22---test-our-implementation",level:4},{value:"2.3 - Register our Custom PyFunc Model",id:"23---register-our-custom-pyfunc-model",level:4},{value:"3 - Serve our Model",id:"3---serve-our-model",level:2},{value:"4 - Query our Served Model",id:"4---query-our-served-model",level:2}];function p(e){let t={code:"code",h1:"h1",h2:"h2",h4:"h4",header:"header",li:"li",ol:"ol",p:"p",...(0,r.R)(),...e.components};return(0,l.jsxs)(l.Fragment,{children:[(0,l.jsx)(t.header,{children:(0,l.jsxs)(t.h1,{id:"deploy-an-mlflow-pyfunc-model-with-model-serving",children:["Deploy an MLflow ",(0,l.jsx)(t.code,{children:"PyFunc"})," model with Model Serving"]})}),"\n",(0,l.jsx)(a.O,{href:"https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial.ipynb",children:"Download this notebook"}),"\n",(0,l.jsx)(t.p,{children:"In this notebook, learn how to deploy a custom MLflow PyFunc model to a serving endpoint. MLflow pyfunc offers greater flexibility and customization to your deployment. You can run any custom model, add preprocessing or post-processing logic, or execute any arbitrary Python code. While using the MLflow built-in flavor is recommended for optimal performance, you can use MLflow PyFunc models where more customization is required."}),"\n",(0,l.jsx)(t.h2,{id:"install-and-import-libraries",children:"Install and import libraries"}),"\n",(0,l.jsx)(n.d,{executionCount:13,children:"%pip install --upgrade mlflow scikit-learn -q"}),"\n",(0,l.jsx)(i.p,{isStderr:!0,children:"213.32s - pydevd: Sending message related to process being replaced timed-out after 5 seconds"}),"\n",(0,l.jsx)(n.d,{executionCount:2,children:`import json
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor

import mlflow
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")`}),"\n",(0,l.jsx)(n.d,{executionCount:3,children:`DOW_MODEL_NAME_PREFIX = "DOW_model_"
MME_MODEL_NAME = "MME_DOW_model"`}),"\n",(0,l.jsx)(t.h2,{id:"1---create-some-sample-models",children:"1 - Create Some Sample Models"}),"\n",(0,l.jsx)(t.h4,{id:"11---create-dummy-data",children:"1.1 - Create Dummy Data"}),"\n",(0,l.jsx)(n.d,{executionCount:4,children:`def create_weekly_dataset(n_dates, n_observations_per_date):
  rng = pd.date_range(start="today", periods=n_dates, freq="D")
  df = pd.DataFrame(
      np.random.randn(n_dates * n_observations_per_date, 4),
      columns=["x1", "x2", "x3", "y"],
      index=np.tile(rng, n_observations_per_date),
  )
  df["dow"] = df.index.dayofweek
  return df


df = create_weekly_dataset(n_dates=30, n_observations_per_date=500)
print(df.shape)
df.head()`}),"\n",(0,l.jsx)(i.p,{children:"(15000, 5)"}),"\n",(0,l.jsx)(s.Q,{children:(0,l.jsx)("div",{dangerouslySetInnerHTML:{__html:`<div>
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
    <th>x1</th>
    <th>x2</th>
    <th>x3</th>
    <th>y</th>
    <th>dow</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>2024-01-26 18:30:42.810981</th>
    <td>-1.137854</td>
    <td>0.165915</td>
    <td>0.711107</td>
    <td>0.046467</td>
    <td>4</td>
  </tr>
  <tr>
    <th>2024-01-27 18:30:42.810981</th>
    <td>0.475331</td>
    <td>-0.749121</td>
    <td>0.318395</td>
    <td>0.520535</td>
    <td>5</td>
  </tr>
  <tr>
    <th>2024-01-28 18:30:42.810981</th>
    <td>2.525948</td>
    <td>1.019708</td>
    <td>0.038251</td>
    <td>-0.270675</td>
    <td>6</td>
  </tr>
  <tr>
    <th>2024-01-29 18:30:42.810981</th>
    <td>1.113931</td>
    <td>0.376434</td>
    <td>-1.464181</td>
    <td>-0.069208</td>
    <td>0</td>
  </tr>
  <tr>
    <th>2024-01-30 18:30:42.810981</th>
    <td>-0.304569</td>
    <td>1.389245</td>
    <td>-1.152598</td>
    <td>-1.137589</td>
    <td>1</td>
  </tr>
</tbody>
</table>
</div>`}})}),"\n",(0,l.jsx)(t.h4,{id:"12---train-models-for-each-day-of-the-week",children:"1.2 - Train Models for Each Day of the Week"}),"\n",(0,l.jsx)(n.d,{executionCount:5,children:`for dow in df["dow"].unique():
  # Create dataset corresponding to a single day of the week
  X = df.loc[df["dow"] == dow]
  X.pop("dow")  # Remove DOW as a predictor column
  y = X.pop("y")

  # Fit our DOW model
  model = RandomForestRegressor().fit(X, y)

  # Infer signature of the model
  signature = infer_signature(X, model.predict(X))

  with mlflow.start_run():
      model_path = f"model_{dow}"

      # Log and register our DOW model with signature
      mlflow.sklearn.log_model(
          model,
          name=model_path,
          signature=signature,
          registered_model_name=f"{DOW_MODEL_NAME_PREFIX}{dow}",
      )
      mlflow.set_tag("dow", dow)`}),"\n",(0,l.jsx)(i.p,{isStderr:!0,children:`Successfully registered model 'DOW_model_4'.
Created version '1' of model 'DOW_model_4'.
Successfully registered model 'DOW_model_5'.
Created version '1' of model 'DOW_model_5'.
Successfully registered model 'DOW_model_6'.
Created version '1' of model 'DOW_model_6'.
Successfully registered model 'DOW_model_0'.
Created version '1' of model 'DOW_model_0'.
Successfully registered model 'DOW_model_1'.
Created version '1' of model 'DOW_model_1'.
Successfully registered model 'DOW_model_2'.
Created version '1' of model 'DOW_model_2'.
Successfully registered model 'DOW_model_3'.
Created version '1' of model 'DOW_model_3'.`}),"\n",(0,l.jsx)(t.h4,{id:"13---test-inference-on-our-dow-models",children:"1.3 - Test inference on our DOW models"}),"\n",(0,l.jsx)(n.d,{executionCount:6,children:`# Load Tuesday's model
tuesday_dow = 1
model_name = f"{DOW_MODEL_NAME_PREFIX}{tuesday_dow}"
model_uri = f"models:/{model_name}/latest"
model = mlflow.sklearn.load_model(model_uri)

# Perform inference using our training data for Tuesday
predictor_columns = [column for column in df.columns if column not in {"y", "dow"}]
head_of_training_data = df.loc[df["dow"] == tuesday_dow, predictor_columns].head()
tuesday_fitted_values = model.predict(head_of_training_data)
print(tuesday_fitted_values)`}),"\n",(0,l.jsx)(i.p,{children:"[-0.8571552   0.61833952  0.61625155  0.28999143  0.49778144]"}),"\n",(0,l.jsx)(t.h2,{id:"2---create-an-mme-custom-pyfunc-model",children:"2 - Create an MME Custom PyFunc Model"}),"\n",(0,l.jsxs)(t.h4,{id:"21---create-a-child-implementation-of-mlflowpyfuncpythonmodel",children:["2.1 - Create a Child Implementation of ",(0,l.jsx)(t.code,{children:"mlflow.pyfunc.PythonModel"})]}),"\n",(0,l.jsx)(n.d,{executionCount:7,children:`class DOWModel(mlflow.pyfunc.PythonModel):
  def __init__(self, model_uris):
      self.model_uris = model_uris
      self.models = {}

  @staticmethod
  def _model_uri_to_dow(model_uri: str) -> int:
      return int(model_uri.split("/")[-2].split("_")[-1])

  def load_context(self, context):
      self.models = {
          self._model_uri_to_dow(model_uri): mlflow.sklearn.load_model(model_uri)
          for model_uri in self.model_uris
      }

  def predict(self, context, model_input, params):
      # Parse the dow parameter
      dow = params.get("dow")
      if dow is None:
          raise ValueError("DOW param is not passed.")

      # Get the model associated with the dow parameter
      model = self.models.get(dow)
      if model is None:
          raise ValueError(f"Model {dow} version was not found: {self.models.keys()}.")

      # Perform inference
      return model.predict(model_input)`}),"\n",(0,l.jsx)(t.h4,{id:"22---test-our-implementation",children:"2.2 - Test our Implementation"}),"\n",(0,l.jsx)(n.d,{executionCount:8,children:"head_of_training_data"}),"\n",(0,l.jsx)(s.Q,{children:(0,l.jsx)("div",{dangerouslySetInnerHTML:{__html:`<div>
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
    <th>x1</th>
    <th>x2</th>
    <th>x3</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>2024-01-30 18:30:42.810981</th>
    <td>-0.304569</td>
    <td>1.389245</td>
    <td>-1.152598</td>
  </tr>
  <tr>
    <th>2024-02-06 18:30:42.810981</th>
    <td>0.521323</td>
    <td>0.814452</td>
    <td>0.115571</td>
  </tr>
  <tr>
    <th>2024-02-13 18:30:42.810981</th>
    <td>0.229761</td>
    <td>-1.936210</td>
    <td>0.139201</td>
  </tr>
  <tr>
    <th>2024-02-20 18:30:42.810981</th>
    <td>-0.865488</td>
    <td>1.024857</td>
    <td>-0.857649</td>
  </tr>
  <tr>
    <th>2024-01-30 18:30:42.810981</th>
    <td>-1.454631</td>
    <td>0.462055</td>
    <td>0.703858</td>
  </tr>
</tbody>
</table>
</div>`}})}),"\n",(0,l.jsx)(n.d,{executionCount:9,children:`# Instantiate our DOW MME
model_uris = [f"models:/{DOW_MODEL_NAME_PREFIX}{i}/latest" for i in df["dow"].unique()]
dow_model = DOWModel(model_uris)
dow_model.load_context(None)
print("Model URIs:")
print(model_uris)

# Perform inference using our training data for Tuesday
params = {"dow": 1}
mme_tuesday_fitted_values = dow_model.predict(None, head_of_training_data, params=params)
assert all(tuesday_fitted_values == mme_tuesday_fitted_values)

print("
Tuesday fitted values:")
print(mme_tuesday_fitted_values)`}),"\n",(0,l.jsx)(i.p,{children:`Model URIs:
['models:/DOW_model_4/latest', 'models:/DOW_model_5/latest', 'models:/DOW_model_6/latest', 'models:/DOW_model_0/latest', 'models:/DOW_model_1/latest', 'models:/DOW_model_2/latest', 'models:/DOW_model_3/latest']

Tuesday fitted values:
[-0.8571552   0.61833952  0.61625155  0.28999143  0.49778144]`}),"\n",(0,l.jsx)(t.h4,{id:"23---register-our-custom-pyfunc-model",children:"2.3 - Register our Custom PyFunc Model"}),"\n",(0,l.jsx)(n.d,{executionCount:10,children:`with mlflow.start_run():
  # Instantiate the custom pyfunc model
  model = DOWModel(model_uris)
  model.load_context(None)
  model_path = "MME_model_path"

  signature = infer_signature(
      model_input=head_of_training_data,
      model_output=tuesday_fitted_values,
      params=params,
  )
  print(signature)

  # Log the model to the experiment
  mlflow.pyfunc.log_model(
      name=model_path,
      python_model=model,
      signature=signature,
      pip_requirements=["scikit-learn=1.3.2"],
      registered_model_name=MME_MODEL_NAME,  # also register the model for easy access
  )

  # Set some relevant information about our model
  # (Assuming model has a property 'models' that can be counted)
  mlflow.log_param("num_models", len(model.models))`}),"\n",(0,l.jsx)(i.p,{children:`inputs: 
['x1': double (required), 'x2': double (required), 'x3': double (required)]
outputs: 
[Tensor('float64', (-1,))]
params: 
['dow': long (default: 1)]`}),"\n",(0,l.jsx)(i.p,{isStderr:!0,children:`Successfully registered model 'MME_DOW_model'.
Created version '1' of model 'MME_DOW_model'.`}),"\n",(0,l.jsx)(t.h2,{id:"3---serve-our-model",children:"3 - Serve our Model"}),"\n",(0,l.jsx)(t.p,{children:"To test our endpoint, let's serve our model on our local machine."}),"\n",(0,l.jsxs)(t.ol,{children:["\n",(0,l.jsxs)(t.li,{children:["Open a new shell window in the root containing ",(0,l.jsx)(t.code,{children:"mlruns"})," directory e.g. the same directory you ran this notebook."]}),"\n",(0,l.jsxs)(t.li,{children:["Ensure mlflow is installed: ",(0,l.jsx)(t.code,{children:"pip install --upgrade mlflow scikit-learn"})]}),"\n",(0,l.jsx)(t.li,{children:"Run the bash command printed below."}),"\n"]}),"\n",(0,l.jsx)(n.d,{executionCount:11,children:`PORT = 1234
print(
  f"""Run the below command in a new window. You must be in the same repo as your mlruns directory and have mlflow installed...
  mlflow models serve -m "models:/{MME_MODEL_NAME}/latest" --env-manager local -p {PORT}"""
)`}),"\n",(0,l.jsx)(i.p,{children:`Run the below command in a new window. You must be in the same repo as your mlruns directory and have mlflow installed...
  mlflow models serve -m "models:/MME_DOW_model/latest" --env-manager local -p 1234`}),"\n",(0,l.jsx)(t.h2,{id:"4---query-our-served-model",children:"4 - Query our Served Model"}),"\n",(0,l.jsx)(n.d,{executionCount:12,children:`def score_model(pdf, params):
  headers = {"Content-Type": "application/json"}
  url = f"http://127.0.0.1:{PORT}/invocations"
  ds_dict = {"dataframe_split": pdf, "params": params}
  data_json = json.dumps(ds_dict, allow_nan=True)

  response = requests.request(method="POST", headers=headers, url=url, data=data_json)
  response.raise_for_status()

  return response.json()


print("Inference on dow model 1 (Tuesday):")
inference_df = head_of_training_data.reset_index(drop=True).to_dict(orient="split")
print(score_model(inference_df, params={"dow": 1}))`}),"\n",(0,l.jsx)(i.p,{children:`Inference on dow model 1 (Tuesday):
{'predictions': [-0.8571551951905747, 0.618339524354309, 0.6162515496343108, 0.2899914313294642, 0.4977814353066934]}`})]})}function _(e={}){let{wrapper:t}={...(0,r.R)(),...e.components};return t?(0,l.jsx)(t,{...e,children:(0,l.jsx)(p,{...e})}):p(e)}},75453(e,t,o){o.d(t,{p:()=>l});var d=o(74848);let l=({children:e,isStderr:t})=>(0,d.jsx)("pre",{style:{margin:0,borderRadius:0,background:"none",fontSize:"0.85rem",flexGrow:1,padding:"var(--padding-sm)"},children:e})},75940(e,t,o){o.d(t,{d:()=>r});var d=o(74848),l=o(37449);let r=({children:e,executionCount:t})=>(0,d.jsx)("div",{style:{flexGrow:1,minWidth:0,marginTop:"var(--padding-md)",width:"100%"},children:(0,d.jsx)(l.A,{className:"codeBlock_oJcR",language:"python",children:e})})},42676(e,t,o){o.d(t,{O:()=>n});var d=o(74848),l=o(96540);let r="3.9.1.dev0";function n({children:e,href:t}){let o=(0,l.useCallback)(async e=>{if(e.preventDefault(),window.gtag)try{window.gtag("event","notebook-download",{href:t})}catch{}r.includes("dev")||(t=t.replace(/\/master\//,`/v${r}/`));let o=await fetch(t),d=await o.blob(),l=window.URL.createObjectURL(d),n=document.createElement("a");n.style.display="none",n.href=l,n.download=t.split("/").pop(),document.body.appendChild(n),n.click(),window.URL.revokeObjectURL(l),document.body.removeChild(n)},[t]);return(0,d.jsx)("a",{className:"button button--primary",style:{marginBottom:"1rem",display:"block",width:"min-content"},href:t,download:!0,onClick:o,children:e})}},66354(e,t,o){o.d(t,{Q:()=>l});var d=o(74848);let l=({children:e})=>(0,d.jsx)("div",{style:{flexGrow:1,minWidth:0,fontSize:"0.8rem",width:"100%"},children:e})},52915(e,t,o){o.d(t,{A:()=>u});var d=o(74848);o(96540);var l=o(34164),r=o(71643),n=o(66697),i=o(92949),s=o(64560),a=o(47819);function m({language:e}){return(0,d.jsxs)("div",{className:(0,l.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,d.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,d.jsx)(a.A,{})]})}function u({className:e}){let{metadata:t}=(0,r.Ph)(),o=t.language||"text";return(0,d.jsxs)(n.A,{as:"div",className:(0,l.A)(e,t.className),children:[t.title&&(0,d.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,d.jsx)(i.A,{children:t.title})}),(0,d.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,d.jsx)(m,{language:o}),(0,d.jsx)(s.A,{})]})]})}}}]);