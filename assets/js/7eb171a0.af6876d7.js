"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[2078],{6002:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>r,contentTitle:()=>s,default:()=>d,frontMatter:()=>o,metadata:()=>h,toc:()=>a});var l=t(5893),i=t(1151);const o={title:"MLflow 2.8.1",slug:"2.8.1",authors:["mlflow-maintainers"]},s=void 0,h={permalink:"/mlflow-website/releases/2.8.1",source:"@site/releases/2023-11-15-2.8.1-release.md",title:"MLflow 2.8.1",description:"MLflow 2.8.1 is a patch release, containing some critical bug fixes and an update to our continued work on reworking our docs.",date:"2023-11-15T00:00:00.000Z",formattedDate:"November 15, 2023",tags:[],readingTime:1.54,hasTruncateMarker:!1,authors:[{name:"MLflow maintainers",title:"MLflow maintainers",url:"https://github.com/mlflow/mlflow.git",imageURL:"https://github.com/mlflow-automation.png",key:"mlflow-maintainers"}],frontMatter:{title:"MLflow 2.8.1",slug:"2.8.1",authors:["mlflow-maintainers"]},unlisted:!1,prevItem:{title:"MLflow 2.9.0",permalink:"/mlflow-website/releases/2.9.0"},nextItem:{title:"MLflow 2.8.0",permalink:"/mlflow-website/releases/2.8.0"}},r={authorsImageUrls:[void 0]},a=[];function c(e){const n={a:"a",code:"code",li:"li",p:"p",ul:"ul",...(0,i.a)(),...e.components};return(0,l.jsxs)(l.Fragment,{children:[(0,l.jsx)(n.p,{children:"MLflow 2.8.1 is a patch release, containing some critical bug fixes and an update to our continued work on reworking our docs."}),"\n",(0,l.jsx)(n.p,{children:"Notable details:"}),"\n",(0,l.jsxs)(n.ul,{children:["\n",(0,l.jsxs)(n.li,{children:["The API ",(0,l.jsx)(n.code,{children:"mlflow.llm.log_predictions"})," is being marked as deprecated, as its functionality has been incorporated into ",(0,l.jsx)(n.code,{children:"mlflow.log_table"}),". This API will be removed in the 2.9.0 release. (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10414",children:"#10414"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/dbczumar",children:"@dbczumar"}),")"]}),"\n"]}),"\n",(0,l.jsx)(n.p,{children:"Bug fixes:"}),"\n",(0,l.jsxs)(n.ul,{children:["\n",(0,l.jsxs)(n.li,{children:["[Artifacts] Fix a regression in 2.8.0 where downloading a single file from a registered model would fail (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10362",children:"#10362"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/BenWilson2",children:"@BenWilson2"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Evaluate] Fix the ",(0,l.jsx)(n.code,{children:"Azure OpenAI"})," integration for ",(0,l.jsx)(n.code,{children:"mlflow.evaluate"})," when using LLM ",(0,l.jsx)(n.code,{children:"judge"})," metrics (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10291",children:"#10291"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/prithvikannan",children:"@prithvikannan"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Evaluate] Change ",(0,l.jsx)(n.code,{children:"Examples"})," to optional for the ",(0,l.jsx)(n.code,{children:"make_genai_metric"})," API (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10353",children:"#10353"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/prithvikannan",children:"@prithvikannan"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Evaluate] Remove the ",(0,l.jsx)(n.code,{children:"fastapi"})," dependency when using ",(0,l.jsx)(n.code,{children:"mlflow.evaluate"})," for LLM results (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10354",children:"#10354"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/prithvikannan",children:"@prithvikannan"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Evaluate] Fix syntax issues and improve the formatting for generated prompt templates (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10402",children:"#10402"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/annzhang-db",children:"@annzhang-db"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Gateway] Fix the Gateway configuration validator pre-check for OpenAI to perform instance type validation (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10379",children:"#10379"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/BenWilson2",children:"@BenWilson2"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Tracking] Fix an intermittent issue with hanging threads when using asynchronous logging (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10374",children:"#10374"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/chenmoneygithub",children:"@chenmoneygithub"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Tracking] Add a timeout for the ",(0,l.jsx)(n.code,{children:"mlflow.login()"})," API to catch invalid hostname configuration input errors (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10239",children:"#10239"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/chenmoneygithub",children:"@chenmoneygithub"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Tracking] Add a ",(0,l.jsx)(n.code,{children:"flush"})," operation at the conclusion of logging system metrics (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10320",children:"#10320"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/chenmoneygithub",children:"@chenmoneygithub"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Models] Correct the prompt template generation logic within the Prompt Engineering UI so that the prompts can be used in the Python API (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10341",children:"#10341"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/daniellok-db",children:"@daniellok-db"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Models] Fix an issue in the ",(0,l.jsx)(n.code,{children:"SHAP"})," model explainability functionality within ",(0,l.jsx)(n.code,{children:"mlflow.shap.log_explanation"})," so that duplicate or conflicting dependencies are not registered when logging (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10305",children:"#10305"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/BenWilson2",children:"@BenWilson2"}),")"]}),"\n"]}),"\n",(0,l.jsx)(n.p,{children:"Documentation updates:"}),"\n",(0,l.jsxs)(n.ul,{children:["\n",(0,l.jsxs)(n.li,{children:["[Docs] Add MLflow Tracking Quickstart (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10285",children:"#10285"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/BenWilson2",children:"@BenWilson2"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Docs] Add tracking server configuration guide (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10241",children:"#10241"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/chenmoneygithub",children:"@chenmoneygithub"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Docs] Refactor and improve the model deployment quickstart guide (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10322",children:"#10322"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/prithvikannan",children:"@prithvikannan"}),")"]}),"\n",(0,l.jsxs)(n.li,{children:["[Docs] Add documentation for system metrics logging (",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/pull/10261",children:"#10261"}),", ",(0,l.jsx)(n.a,{href:"https://github.com/chenmoneygithub",children:"@chenmoneygithub"}),")"]}),"\n"]}),"\n",(0,l.jsxs)(n.p,{children:["For a comprehensive list of changes, see the ",(0,l.jsx)(n.a,{href:"https://github.com/mlflow/mlflow/releases/tag/v2.8.1",children:"release change log"}),", and check out the latest documentation on ",(0,l.jsx)(n.a,{href:"http://mlflow.org/",children:"mlflow.org"}),"."]})]})}function d(e={}){const{wrapper:n}={...(0,i.a)(),...e.components};return n?(0,l.jsx)(n,{...e,children:(0,l.jsx)(c,{...e})}):c(e)}},1151:(e,n,t)=>{t.d(n,{Z:()=>h,a:()=>s});var l=t(7294);const i={},o=l.createContext(i);function s(e){const n=l.useContext(o);return l.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function h(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:s(e.components),l.createElement(o.Provider,{value:n},e.children)}}}]);