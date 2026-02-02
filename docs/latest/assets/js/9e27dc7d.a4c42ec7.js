"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["2571"],{20239(e,t,i){i.r(t),i.d(t,{metadata:()=>o,default:()=>u,frontMatter:()=>a,contentTitle:()=>c,toc:()=>p,assets:()=>h});var o=JSON.parse('{"id":"traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict-ipynb","title":"Customizing a Model\'s predict method","description":"Download this notebook","source":"@site/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict-ipynb.mdx","sourceDirName":"traditional-ml/tutorials/creating-custom-pyfunc/notebooks","slug":"/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict","draft":false,"unlisted":false,"editUrl":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict.ipynb","tags":[],"version":"current","frontMatter":{"custom_edit_url":"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict.ipynb","slug":"override-predict"},"sidebar":"classicMLSidebar","previous":{"title":"Custom Model Basics","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/introduction"},"next":{"title":"Serving Multiple Models on a Single Endpoint with a Custom PyFunc Model","permalink":"/mlflow-website/docs/latest/ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/"}}'),n=i(74848),r=i(28453),l=i(75940),s=i(75453);i(66354);var d=i(42676);let a={custom_edit_url:"https://github.com/mlflow/mlflow/edit/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict.ipynb",slug:"override-predict"},c="Customizing a Model's predict method",h={},p=[{value:"Configure the tracking server uri",id:"configure-the-tracking-server-uri",level:4},{value:"What can we do to support the original behavior of the model when deployed?",id:"what-can-we-do-to-support-the-original-behavior-of-the-model-when-deployed",level:3},{value:"Defining our custom <code>PythonModel</code>",id:"defining-our-custom-pythonmodel",level:4}];function m(e){let t={code:"code",h1:"h1",h3:"h3",h4:"h4",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)(t.header,{children:(0,n.jsx)(t.h1,{id:"customizing-a-models-predict-method",children:"Customizing a Model's predict method"})}),"\n",(0,n.jsx)(d.O,{href:"https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict.ipynb",children:"Download this notebook"}),"\n",(0,n.jsx)(t.p,{children:"In this tutorial, we will explore the process of customizing the predict method of a model in the context of MLflow's PyFunc flavor. This is particularly useful when you want to have more flexibility in how your model behaves after you've deployed it using MLflow."}),"\n",(0,n.jsx)(t.p,{children:"To illustrate this, we'll use the famous Iris dataset and build a basic Logistic Regression model with scikit-learn."}),"\n",(0,n.jsx)(l.d,{executionCount:1,children:`from joblib import dump
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel`}),"\n",(0,n.jsx)(t.h4,{id:"configure-the-tracking-server-uri",children:"Configure the tracking server uri"}),"\n",(0,n.jsx)(t.p,{children:"This step is important to ensure that all of the calls to MLflow that we're going to be doing within this notebook will actually be logged to our tracking server that is running locally."}),"\n",(0,n.jsx)(t.p,{children:"If you are following along with this notebook in a different environment and wish to execute the remainder of this notebook to a remote tracking server, change the following cell."}),"\n",(0,n.jsxs)(t.p,{children:["Databricks: ",(0,n.jsx)(t.code,{children:'mlflow.set_tracking_uri("databricks")'})]}),"\n",(0,n.jsxs)(t.p,{children:["Your hosted MLflow: ",(0,n.jsx)(t.code,{children:'mlflow.set_tracking_uri("http://my.company.mlflow.tracking.server:<port>)'})]}),"\n",(0,n.jsx)(t.p,{children:"Your local tracking server\nAs in the introductory tutorial, we can start a local tracking server via command line as follows:"}),"\n",(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-bash",children:"mlflow server --host 127.0.0.1 --port 8080\n"})}),"\n",(0,n.jsx)(t.p,{children:"And the MLflow UI server can be started locally via:"}),"\n",(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-bash",children:"mlflow server --host 127.0.0.1 --port 8090\n"})}),"\n",(0,n.jsx)(l.d,{executionCount:2,children:'mlflow.set_tracking_uri("http://localhost:8080")'}),"\n",(0,n.jsx)(t.p,{children:"Let's begin by loading the Iris dataset and splitting it into training and testing sets. We'll then train a simple Logistic Regression model on the training data."}),"\n",(0,n.jsx)(l.d,{executionCount:3,children:`iris = load_iris()
x = iris.data[:, 2:]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9001)

model = LogisticRegression(random_state=0, max_iter=5_000, solver="newton-cg").fit(x_train, y_train)`}),"\n",(0,n.jsx)(t.p,{children:"This is a common scenario in machine learning. We have a trained model, and we want to use it to make predictions. With scikit-learn, the model provides a few methods to do this:"}),"\n",(0,n.jsxs)(t.ul,{children:["\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.code,{children:"predict"})," - to predict class labels"]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.code,{children:"predict_proba"})," - to get class membership probabilities"]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.code,{children:"predict_log_proba"})," - to get logarithmic probabilities for each class"]}),"\n"]}),"\n",(0,n.jsx)(t.p,{children:"We can predict the class labels, as shown below."}),"\n",(0,n.jsx)(l.d,{executionCount:4,children:"model.predict(x_test)[:5]"}),"\n",(0,n.jsx)(s.p,{children:"array([1, 2, 2, 1, 0])"}),"\n",(0,n.jsx)(t.p,{children:"We can also get the class membership probability."}),"\n",(0,n.jsx)(l.d,{executionCount:5,children:"model.predict_proba(x_test)[:5]"}),"\n",(0,n.jsx)(s.p,{children:`array([[2.64002987e-03, 6.62306827e-01, 3.35053144e-01],
     [1.24429110e-04, 8.35485037e-02, 9.16327067e-01],
     [1.30646549e-04, 1.37480519e-01, 8.62388835e-01],
     [3.70944840e-03, 7.13202611e-01, 2.83087941e-01],
     [9.82629868e-01, 1.73700532e-02, 7.88350143e-08]])`}),"\n",(0,n.jsx)(t.p,{children:"As well as generate logarithmic probabilites for each class."}),"\n",(0,n.jsx)(l.d,{executionCount:6,children:"model.predict_log_proba(x_test)[:5]"}),"\n",(0,n.jsx)(s.p,{children:`array([[ -5.93696505,  -0.41202635,  -1.09346612],
     [ -8.99177441,  -2.48232793,  -0.08738192],
     [ -8.94301498,  -1.98427305,  -0.14804903],
     [ -5.59687209,  -0.33798973,  -1.26199768],
     [ -0.01752276,  -4.05300763, -16.35590859]])`}),"\n",(0,n.jsx)(t.p,{children:"While using the model directly within the same Python session is straightforward, what happens when we want to save this model and load it elsewhere, especially when using MLflow's PyFunc flavor? Let's explore this scenario."}),"\n",(0,n.jsx)(l.d,{executionCount:7,children:`mlflow.set_experiment("Overriding Predict Tutorial")

sklearn_path = "/tmp/sklearn_model"

with mlflow.start_run() as run:
  mlflow.sklearn.save_model(
      sk_model=model,
      path=sklearn_path,
      input_example=x_train[:2],
  )`}),"\n",(0,n.jsx)(s.p,{isStderr:!0,children:`/Users/benjamin.wilson/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils.
warnings.warn("Setuptools is replacing distutils.")`}),"\n",(0,n.jsx)(t.p,{children:"Once the model is loaded as a pyfunc, the default behavior only supports the predict method. This is evident when you try to call other methods like predict_proba, leading to an AttributeError. This can be limiting, especially when you want to preserve the full capability of the original model."}),"\n",(0,n.jsx)(l.d,{executionCount:8,children:"loaded_logreg_model = mlflow.pyfunc.load_model(sklearn_path)"}),"\n",(0,n.jsx)(l.d,{executionCount:9,children:"loaded_logreg_model.predict(x_test)"}),"\n",(0,n.jsx)(s.p,{children:`array([1, 2, 2, 1, 0, 1, 2, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 2, 1,
     1, 0, 1, 1, 0, 0, 1, 2])`}),"\n",(0,n.jsx)(t.p,{children:"This works precisely as we expect. The output is the same as the model direct usage prior to saving."}),"\n",(0,n.jsx)(t.p,{children:"Let's try to use the predict_proba method."}),"\n",(0,n.jsx)(t.p,{children:"We're not actually going to run this, as it will raise an Exception. Here is the behavior if we try to execute this:"}),"\n",(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-python",children:"loaded_logreg_model.predict_proba(x_text)\n"})}),"\n",(0,n.jsx)(t.p,{children:"Which will result in this error:"}),"\n",(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-shell",children:"---------------------------------------------------------------------------\nAttributeError                            Traceback (most recent call last)\n/var/folders/cd/n8n0rm2x53l_s0xv_j_xklb00000gp/T/ipykernel_15410/1677830262.py in <cell line: 1>()\n----\x3e 1 loaded_logreg_model.predict_proba(x_text)\n\nAttributeError: 'PyFuncModel' object has no attribute 'predict_proba'\n"})}),"\n",(0,n.jsx)(t.h3,{id:"what-can-we-do-to-support-the-original-behavior-of-the-model-when-deployed",children:"What can we do to support the original behavior of the model when deployed?"}),"\n",(0,n.jsxs)(t.p,{children:["We can create a custom pyfunc that overrides the behavior of the ",(0,n.jsx)(t.code,{children:"predict"})," method."]}),"\n",(0,n.jsx)(t.p,{children:"For the example below, we're going to be showing two features of pyfunc that can be leveraged to handle custom model logging capabilities:"}),"\n",(0,n.jsxs)(t.ul,{children:["\n",(0,n.jsx)(t.li,{children:"override of the predict method"}),"\n",(0,n.jsx)(t.li,{children:"custom loading of an artifact"}),"\n"]}),"\n",(0,n.jsx)(t.p,{children:"A key thing to note is the use of joblib for serialization. While pickle has been historically used for serializing scikit-learn models, joblib is now recommended as it provides better performance and support, especially for large numpy arrays."}),"\n",(0,n.jsxs)(t.p,{children:["We'll be using ",(0,n.jsx)(t.code,{children:"joblib"})," and it's ",(0,n.jsx)(t.code,{children:"dump"})," and ",(0,n.jsx)(t.code,{children:"load"})," APIs to handle loading of our model object into our custom pyfunc implementation. This process of using the load_context method to handle loading files when instantiating the pyfunc object is particularly useful for models that have very large or numerous artifact dependencies (such as LLMs) and can help to dramatically lessen the total memory footprint of a pyfunc that is being loaded in a distributed system (such as Apache Spark or Ray)."]}),"\n",(0,n.jsx)(l.d,{executionCount:10,children:`from joblib import dump

from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel`}),"\n",(0,n.jsxs)(t.p,{children:["To see how we can leverage the ",(0,n.jsx)(t.code,{children:"load_context"})," functionality within a custom Python Model, we'll first serialize our model locally using ",(0,n.jsx)(t.code,{children:"joblib"}),".\nThe usage of ",(0,n.jsx)(t.code,{children:"joblib"})," here is purely to demonstrate a non-standard method (one that is not natively supported in MLflow) to illustrate the flexibility of the Python Model implementation. Provided that we import this library within the ",(0,n.jsx)(t.code,{children:"load_context"})," and have it available in the environment where we will be loading this model, the model artifact will be deserialized properly."]}),"\n",(0,n.jsx)(l.d,{executionCount:11,children:`model_directory = "/tmp/sklearn_model.joblib"
dump(model, model_directory)`}),"\n",(0,n.jsx)(s.p,{children:"['/tmp/sklearn_model.joblib']"}),"\n",(0,n.jsxs)(t.h4,{id:"defining-our-custom-pythonmodel",children:["Defining our custom ",(0,n.jsx)(t.code,{children:"PythonModel"})]}),"\n",(0,n.jsxs)(t.p,{children:["The ",(0,n.jsx)(t.code,{children:"ModelWrapper"})," class below is an example of a custom ",(0,n.jsx)(t.code,{children:"pyfunc"})," that extends MLflow's ",(0,n.jsx)(t.code,{children:"PythonModel"}),". It provides flexibility in the prediction method by using the ",(0,n.jsx)(t.code,{children:"params"})," argument of the ",(0,n.jsx)(t.code,{children:"predict method"}),". This way, we can specify if we want the regular ",(0,n.jsx)(t.code,{children:"predict"}),", ",(0,n.jsx)(t.code,{children:"predict_proba"}),", or ",(0,n.jsx)(t.code,{children:"predict_log_proba"})," behavior when we call the ",(0,n.jsx)(t.code,{children:"predict"})," method on the loaded ",(0,n.jsx)(t.code,{children:"pyfunc"})," instance."]}),"\n",(0,n.jsx)(l.d,{executionCount:12,children:`class ModelWrapper(PythonModel):
  def __init__(self):
      self.model = None

  def load_context(self, context):
      from joblib import load

      self.model = load(context.artifacts["model_path"])

  def predict(self, context, model_input, params=None):
      params = params or {"predict_method": "predict"}
      predict_method = params.get("predict_method")

      if predict_method == "predict":
          return self.model.predict(model_input)
      elif predict_method == "predict_proba":
          return self.model.predict_proba(model_input)
      elif predict_method == "predict_log_proba":
          return self.model.predict_log_proba(model_input)
      else:
          raise ValueError(f"The prediction method '{predict_method}' is not supported.")`}),"\n",(0,n.jsxs)(t.p,{children:["After defining the custom ",(0,n.jsx)(t.code,{children:"pyfunc"}),", the next steps involve saving the model with MLflow and then loading it back. The loaded model will retain the flexibility we built into the custom ",(0,n.jsx)(t.code,{children:"pyfunc"}),", allowing us to choose the prediction method dynamically."]}),"\n",(0,n.jsxs)(t.p,{children:[(0,n.jsx)(t.strong,{children:"NOTE"}),": The ",(0,n.jsx)(t.code,{children:"artifacts"})," reference below is incredibly important. In order for the ",(0,n.jsx)(t.code,{children:"load_context"})," to have access to the path that we are specifying as the location of our saved model, this must be provided as a dictionary that maps the appropriate access key to the relevant value. Failing to provide this dictionary as part of the ",(0,n.jsx)(t.code,{children:"mlflow.save_model()"})," or ",(0,n.jsx)(t.code,{children:"mlflow.log_model()"})," will render this custom ",(0,n.jsx)(t.code,{children:"pyfunc"})," model unable to be properly loaded."]}),"\n",(0,n.jsx)(l.d,{executionCount:13,children:`# Define the required artifacts associated with the saved custom pyfunc
artifacts = {"model_path": model_directory}

# Define the signature associated with the model
signature = infer_signature(x_train, params={"predict_method": "predict_proba"})`}),"\n",(0,n.jsxs)(t.p,{children:["We can see how the defined params are used within the signature definition. As is shown below, the params receive a slight alteration when logged. We have a param key that is defined (",(0,n.jsx)(t.code,{children:"predict_method"}),"), and expected type (",(0,n.jsx)(t.code,{children:"string"}),"), and a default value. What this ends up meaning for this ",(0,n.jsx)(t.code,{children:"params"})," definition is:"]}),"\n",(0,n.jsxs)(t.ul,{children:["\n",(0,n.jsxs)(t.li,{children:["\n",(0,n.jsxs)(t.p,{children:["We can only provide a ",(0,n.jsx)(t.code,{children:"params"})," override for the key ",(0,n.jsx)(t.code,{children:"predict_method"}),". Anything apart from this will be ignored and a warning will be shown indicating that the unknown parameter will not be passed to the underlying model."]}),"\n"]}),"\n",(0,n.jsxs)(t.li,{children:["\n",(0,n.jsxs)(t.p,{children:["The value associated with ",(0,n.jsx)(t.code,{children:"predict_method"})," must be a string. Any other type will not be permitted and will raise an Exception for an unexpected type."]}),"\n"]}),"\n",(0,n.jsxs)(t.li,{children:["\n",(0,n.jsxs)(t.p,{children:["If no value for the ",(0,n.jsx)(t.code,{children:"predict_method"})," is provided when calling ",(0,n.jsx)(t.code,{children:"predict"}),", the default value of ",(0,n.jsx)(t.code,{children:"predict_proba"})," will be used by the model."]}),"\n"]}),"\n"]}),"\n",(0,n.jsx)(l.d,{executionCount:14,children:"signature"}),"\n",(0,n.jsx)(s.p,{children:`inputs: 
[Tensor('float64', (-1, 2))]
outputs: 
None
params: 
['predict_method': string (default: predict_proba)]`}),"\n",(0,n.jsxs)(t.p,{children:["We can now save our custom model. We're providing a path to save it to, as well as the ",(0,n.jsx)(t.code,{children:"artifacts"})," definition that contains the location of the manually serialized instance that we stored via ",(0,n.jsx)(t.code,{children:"joblib"}),". Also included is the ",(0,n.jsx)(t.code,{children:"signature"}),", which is a ",(0,n.jsx)(t.strong,{children:"key component"})," to making this example work; without the paramater defined within the signature, we wouldn't be able to override the method of prediction that the ",(0,n.jsx)(t.code,{children:"predict"})," method will use."]}),"\n",(0,n.jsxs)(t.p,{children:[(0,n.jsx)(t.strong,{children:"Note"})," that we're overriding the ",(0,n.jsx)(t.code,{children:"pip_requirements"})," here to ensure that we specify the requirements for our two dependent libraries: ",(0,n.jsx)(t.code,{children:"joblib"})," and ",(0,n.jsx)(t.code,{children:"sklearn"}),". This helps to ensure that whatever environment that we deploy this model to will pre-load both of these dependencies prior to loading this saved model."]}),"\n",(0,n.jsx)(l.d,{executionCount:15,children:`pyfunc_path = "/tmp/dynamic_regressor"

with mlflow.start_run() as run:
  mlflow.pyfunc.save_model(
      path=pyfunc_path,
      python_model=ModelWrapper(),
      input_example=x_train,
      signature=signature,
      artifacts=artifacts,
      pip_requirements=["joblib", "sklearn"],
  )`}),"\n",(0,n.jsx)(s.p,{children:"Downloading artifacts:   0%|          | 0/1 [00:00<?, ?it/s]"}),"\n",(0,n.jsxs)(t.p,{children:["We can now load our model back by using the ",(0,n.jsx)(t.code,{children:"mlflow.pyfunc.load_model"})," API."]}),"\n",(0,n.jsx)(l.d,{executionCount:16,children:"loaded_dynamic = mlflow.pyfunc.load_model(pyfunc_path)"}),"\n",(0,n.jsxs)(t.p,{children:["Let's see what the pyfunc model will produce with no overrides to the ",(0,n.jsx)(t.code,{children:"params"})," argument."]}),"\n",(0,n.jsx)(l.d,{executionCount:17,children:"loaded_dynamic.predict(x_test)"}),"\n",(0,n.jsx)(s.p,{children:`array([[2.64002987e-03, 6.62306827e-01, 3.35053144e-01],
     [1.24429110e-04, 8.35485037e-02, 9.16327067e-01],
     [1.30646549e-04, 1.37480519e-01, 8.62388835e-01],
     [3.70944840e-03, 7.13202611e-01, 2.83087941e-01],
     [9.82629868e-01, 1.73700532e-02, 7.88350143e-08],
     [6.54171552e-03, 7.54211950e-01, 2.39246334e-01],
     [2.29127680e-06, 1.29261337e-02, 9.87071575e-01],
     [9.71364952e-01, 2.86348857e-02, 1.62618524e-07],
     [3.36988442e-01, 6.61070371e-01, 1.94118691e-03],
     [9.81908726e-01, 1.80911360e-02, 1.38374097e-07],
     [9.70783357e-01, 2.92164276e-02, 2.15395762e-07],
     [6.54171552e-03, 7.54211950e-01, 2.39246334e-01],
     [1.06968794e-02, 8.88253152e-01, 1.01049969e-01],
     [3.35084116e-03, 6.57732340e-01, 3.38916818e-01],
     [9.82272901e-01, 1.77269948e-02, 1.04445227e-07],
     [9.82629868e-01, 1.73700532e-02, 7.88350143e-08],
     [1.62626101e-03, 5.43474542e-01, 4.54899197e-01],
     [9.82629868e-01, 1.73700532e-02, 7.88350143e-08],
     [5.55685308e-03, 8.02036140e-01, 1.92407007e-01],
     [1.01733783e-02, 8.62455340e-01, 1.27371282e-01],
     [1.43317140e-08, 1.15653085e-03, 9.98843455e-01],
     [4.33536629e-02, 9.32351526e-01, 2.42948113e-02],
     [3.97007654e-02, 9.08506559e-01, 5.17926758e-02],
     [9.19762712e-01, 8.02357267e-02, 1.56085268e-06],
     [4.21970838e-02, 9.26463030e-01, 3.13398863e-02],
     [3.13635521e-02, 9.17295925e-01, 5.13405229e-02],
     [9.77454643e-01, 2.25452265e-02, 1.30412321e-07],
     [9.71364952e-01, 2.86348857e-02, 1.62618524e-07],
     [3.23802803e-02, 9.27626313e-01, 3.99934070e-02],
     [1.21876019e-06, 1.79695714e-02, 9.82029210e-01]])`}),"\n",(0,n.jsxs)(t.p,{children:["As expected, it returned the default value of ",(0,n.jsx)(t.code,{children:"params"})," ",(0,n.jsx)(t.code,{children:"predict_method"}),", that of ",(0,n.jsx)(t.code,{children:"predict_proba"}),". We can now attempt to override that functionality to return the class predictions."]}),"\n",(0,n.jsx)(l.d,{executionCount:18,children:'loaded_dynamic.predict(x_test, params={"predict_method": "predict"})'}),"\n",(0,n.jsx)(s.p,{children:`array([1, 2, 2, 1, 0, 1, 2, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 2, 1,
     1, 0, 1, 1, 0, 0, 1, 2])`}),"\n",(0,n.jsxs)(t.p,{children:["We can also override it to return the ",(0,n.jsx)(t.code,{children:"predict_log_proba"})," logarithmic probailities of class membership."]}),"\n",(0,n.jsx)(l.d,{executionCount:19,children:'loaded_dynamic.predict(x_test, params={"predict_method": "predict_log_proba"})'}),"\n",(0,n.jsx)(s.p,{children:`array([[-5.93696505e+00, -4.12026346e-01, -1.09346612e+00],
     [-8.99177441e+00, -2.48232793e+00, -8.73819177e-02],
     [-8.94301498e+00, -1.98427305e+00, -1.48049026e-01],
     [-5.59687209e+00, -3.37989732e-01, -1.26199768e+00],
     [-1.75227629e-02, -4.05300763e+00, -1.63559086e+01],
     [-5.02955584e+00, -2.82081850e-01, -1.43026157e+00],
     [-1.29864013e+01, -4.34850415e+00, -1.30127244e-02],
     [-2.90530299e-02, -3.55312953e+00, -1.56318587e+01],
     [-1.08770665e+00, -4.13894984e-01, -6.24445569e+00],
     [-1.82569224e-02, -4.01233318e+00, -1.57933050e+01],
     [-2.96519488e-02, -3.53302414e+00, -1.53507887e+01],
     [-5.02955584e+00, -2.82081850e-01, -1.43026157e+00],
     [-4.53780322e+00, -1.18498496e-01, -2.29214015e+00],
     [-5.69854387e+00, -4.18957208e-01, -1.08200058e+00],
     [-1.78861062e-02, -4.03266667e+00, -1.60746030e+01],
     [-1.75227629e-02, -4.05300763e+00, -1.63559086e+01],
     [-6.42147176e+00, -6.09772414e-01, -7.87679430e-01],
     [-1.75227629e-02, -4.05300763e+00, -1.63559086e+01],
     [-5.19272332e+00, -2.20601610e-01, -1.64814232e+00],
     [-4.58798095e+00, -1.47971911e-01, -2.06064898e+00],
     [-1.80607910e+01, -6.76233040e+00, -1.15721450e-03],
     [-3.13836408e+00, -7.00453618e-02, -3.71749248e+00],
     [-3.22638481e+00, -9.59531718e-02, -2.96050653e+00],
     [-8.36395634e-02, -2.52278639e+00, -1.33702783e+01],
     [-3.16540417e+00, -7.63811370e-02, -3.46286367e+00],
     [-3.46210882e+00, -8.63251488e-02, -2.96927492e+00],
     [-2.28033892e-02, -3.79223192e+00, -1.58525647e+01],
     [-2.90530299e-02, -3.55312953e+00, -1.56318587e+01],
     [-3.43020568e+00, -7.51263075e-02, -3.21904066e+00],
     [-1.36176765e+01, -4.01907543e+00, -1.81342258e-02]])`}),"\n",(0,n.jsx)(t.p,{children:"We've successfully created a pyfunc model that retains the full capabilities of the original scikit-learn model, while simultaneously using a custom loader methodology that eschews the standard pickle methodology."}),"\n",(0,n.jsx)(t.p,{children:"This tutorial highlights the power and flexibility of MLflow's PyFunc flavor, demonstrating how you can tailor it to fit your specific needs. As you continue building and deploying models, consider how custom pyfuncs can be used to enhance your model's capabilities and adapt to various scenarios."})]})}function u(e={}){let{wrapper:t}={...(0,r.R)(),...e.components};return t?(0,n.jsx)(t,{...e,children:(0,n.jsx)(m,{...e})}):m(e)}},75453(e,t,i){i.d(t,{p:()=>n});var o=i(74848);let n=({children:e,isStderr:t})=>(0,o.jsx)("pre",{style:{margin:0,borderRadius:0,background:"none",fontSize:"0.85rem",flexGrow:1,padding:"var(--padding-sm)"},children:e})},75940(e,t,i){i.d(t,{d:()=>r});var o=i(74848),n=i(37449);let r=({children:e,executionCount:t})=>(0,o.jsx)("div",{style:{flexGrow:1,minWidth:0,marginTop:"var(--padding-md)",width:"100%"},children:(0,o.jsx)(n.A,{className:"codeBlock_oJcR",language:"python",children:e})})},42676(e,t,i){i.d(t,{O:()=>l});var o=i(74848),n=i(96540);let r="3.9.1.dev0";function l({children:e,href:t}){let i=(0,n.useCallback)(async e=>{if(e.preventDefault(),window.gtag)try{window.gtag("event","notebook-download",{href:t})}catch{}r.includes("dev")||(t=t.replace(/\/master\//,`/v${r}/`));let i=await fetch(t),o=await i.blob(),n=window.URL.createObjectURL(o),l=document.createElement("a");l.style.display="none",l.href=n,l.download=t.split("/").pop(),document.body.appendChild(l),l.click(),window.URL.revokeObjectURL(n),document.body.removeChild(l)},[t]);return(0,o.jsx)("a",{className:"button button--primary",style:{marginBottom:"1rem",display:"block",width:"min-content"},href:t,download:!0,onClick:i,children:e})}},66354(e,t,i){i.d(t,{Q:()=>n});var o=i(74848);let n=({children:e})=>(0,o.jsx)("div",{style:{flexGrow:1,minWidth:0,fontSize:"0.8rem",width:"100%"},children:e})},52915(e,t,i){i.d(t,{A:()=>h});var o=i(74848);i(96540);var n=i(34164),r=i(71643),l=i(66697),s=i(92949),d=i(64560),a=i(47819);function c({language:e}){return(0,o.jsxs)("div",{className:(0,n.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,o.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,o.jsx)(a.A,{})]})}function h({className:e}){let{metadata:t}=(0,r.Ph)(),i=t.language||"text";return(0,o.jsxs)(l.A,{as:"div",className:(0,n.A)(e,t.className),children:[t.title&&(0,o.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,o.jsx)(s.A,{children:t.title})}),(0,o.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,o.jsx)(c,{language:i}),(0,o.jsx)(d.A,{})]})]})}}}]);