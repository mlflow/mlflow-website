"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([["8762"],{16822(e,t,a){a.r(t),a.d(t,{metadata:()=>n,default:()=>h,frontMatter:()=>s,contentTitle:()=>l,toc:()=>p,assets:()=>c});var n=JSON.parse('{"id":"tracing/integrations/listing/portkey","title":"Tracing Portkey","description":"","source":"@site/docs/genai/tracing/integrations/listing/portkey.mdx","sourceDirName":"tracing/integrations/listing","slug":"/tracing/integrations/listing/portkey","permalink":"/mlflow-website/docs/latest/genai/tracing/integrations/listing/portkey","draft":false,"unlisted":false,"tags":[],"version":"current","sidebarPosition":106,"frontMatter":{"sidebar_position":106,"sidebar_label":"Portkey"},"sidebar":"genAISidebar","previous":{"title":"OpenRouter","permalink":"/mlflow-website/docs/latest/genai/tracing/integrations/listing/openrouter"},"next":{"title":"Pydantic AI Gateway","permalink":"/mlflow-website/docs/latest/genai/tracing/integrations/listing/pydantic-ai-gateway"}}'),i=a(74848),r=a(28453),o=a(87721);let s={sidebar_position:106,sidebar_label:"Portkey"},l="Tracing Portkey",c={},p=[];function d(e){let t={h1:"h1",header:"header",...(0,r.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(t.header,{children:(0,i.jsx)(t.h1,{id:"tracing-portkey",children:"Tracing Portkey"})}),"\n",(0,i.jsx)(o.h,{gatewayId:"portkey"})]})}function h(e={}){let{wrapper:t}={...(0,r.R)(),...e.components};return t?(0,i.jsx)(t,{...e,children:(0,i.jsx)(d,{...e})}):d(e)}},8060(e,t,a){a.d(t,{Ay:()=>p,RM:()=>l});var n=a(74848),i=a(28453),r=a(78010),o=a(57250),s=a(95986);let l=[];function c(e){let t={a:"a",code:"code",p:"p",pre:"pre",...(0,i.R)(),...e.components};return(0,n.jsx)(s.A,{children:(0,n.jsxs)(r.A,{children:[(0,n.jsxs)(o.A,{value:"local",label:"Local (pip)",default:!0,children:[(0,n.jsxs)(t.p,{children:["If you have a local Python environment >= 3.10, you can start the MLflow server locally using the ",(0,n.jsx)(t.code,{children:"mlflow"})," CLI command."]}),(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-bash",children:"mlflow server\n"})})]}),(0,n.jsxs)(o.A,{value:"docker",label:"Local (docker)",children:[(0,n.jsx)(t.p,{children:"MLflow also provides a Docker Compose file to start a local MLflow server with a postgres database and a minio server."}),(0,n.jsx)(t.pre,{children:(0,n.jsx)(t.code,{className:"language-bash",children:"git clone --depth 1 --filter=blob:none --sparse https://github.com/mlflow/mlflow.git\ncd mlflow\ngit sparse-checkout set docker-compose\ncd docker-compose\ncp .env.dev.example .env\ndocker compose up -d\n"})}),(0,n.jsxs)(t.p,{children:["Refer to the ",(0,n.jsx)(t.a,{href:"https://github.com/mlflow/mlflow/tree/master/docker-compose/README.md",children:"instruction"})," for more details, e.g., overriding the default environment variables."]})]})]})})}function p(e={}){let{wrapper:t}={...(0,i.R)(),...e.components};return t?(0,n.jsx)(t,{...e,children:(0,n.jsx)(c,{...e})}):c(e)}},46077(e,t,a){a.d(t,{A:()=>r});var n=a(74848);a(96540);var i=a(66497);function r({src:e,alt:t,width:a,caption:r,className:o}){return(0,n.jsxs)("div",{className:`container_JwLF ${o||""}`,children:[(0,n.jsx)("div",{className:"imageWrapper_RfGN",style:a?{width:a}:{},children:(0,n.jsx)("img",{src:(0,i.default)(e),alt:t,className:"image_bwOA"})}),r&&(0,n.jsx)("p",{className:"caption_jo2G",children:r})]})}},87721(e,t,a){a.d(t,{h:()=>x});var n=a(74848);a(96540);var i=a(78010),r=a(57250),o=a(36625),s=a(10440),l=a(77541),c=a(46077),p=a(89001),d=a(8060),h=a(95986),u=a(93893),m=a(60665),g=a(43975),f=a(37449);let y=[{id:"openrouter",name:"OpenRouter",description:'<a href="https://openrouter.ai/">OpenRouter</a> is a unified API gateway that provides access to 280+ LLMs from providers like OpenAI, Anthropic, Google, Meta, and many others through a single OpenAI-compatible API. This allows developers to easily switch between models without changing their code.',baseUrl:"https://openrouter.ai/api/v1",apiKeyPlaceholder:"<YOUR_OPENROUTER_API_KEY>",sampleModel:"anthropic/claude-sonnet-4.5",heroImage:"/images/llms/openrouter/openrouter-tracing.png",prerequisite:'Before following the steps below, you need to create an <ins><a href="https://openrouter.ai/">OpenRouter account</a></ins> and generate an API key from the <ins><a href="https://openrouter.ai/keys">Keys page</a></ins>.'},{id:"vercel-ai-gateway",name:"Vercel AI Gateway",description:'<a href="https://vercel.com/docs/ai-gateway">Vercel AI Gateway</a> provides a unified API to access hundreds of LLMs through a single endpoint. Key features include high reliability with automatic fallbacks to other providers, spend monitoring across providers, and zero markup on token costs. It works seamlessly with the OpenAI SDK, Anthropic SDK, and Vercel AI SDK.',baseUrl:"https://ai-gateway.vercel.sh/v1",apiKeyPlaceholder:"<YOUR_VERCEL_AI_GATEWAY_API_KEY>",sampleModel:"anthropic/claude-sonnet-4.5",prerequisite:'Create a <ins><a href="https://vercel.com/">Vercel account</a></ins> and enable <ins><a href="https://vercel.com/docs/ai-gateway">AI Gateway</a></ins> for your project. You can find your API key in the project settings.'},{id:"truefoundry",name:"TrueFoundry",displayName:"TrueFoundry AI Gateway",description:'<a href="https://www.truefoundry.com/ai-gateway">TrueFoundry AI Gateway</a> is an enterprise-grade LLM gateway that provides access to 1000+ LLMs through a unified OpenAI-compatible API. It offers built-in governance, observability, rate limiting, and cost controls for production AI applications.',baseUrl:"https://<your-control-plane>.truefoundry.cloud/api/llm/v1",apiKeyPlaceholder:"<YOUR_TRUEFOUNDRY_API_KEY>",sampleModel:"openai/gpt-4o",prerequisite:'Create a <ins><a href="https://www.truefoundry.com/">TrueFoundry account</a></ins> with at least one model provider configured, then generate an API key from the TrueFoundry dashboard.'},{id:"kong",name:"Kong AI Gateway",description:'<a href="https://konghq.com/products/kong-ai-gateway">Kong AI Gateway</a> is an enterprise-grade API gateway that provides a unified OpenAI-compatible API to access multiple LLM providers including OpenAI, Anthropic, Azure, AWS Bedrock, Google Gemini, and more. It offers built-in rate limiting, caching, load balancing, and observability.',baseUrl:"http://<your-kong-gateway>:8000/v1",apiKeyPlaceholder:"<YOUR_API_KEY>",sampleModel:"gpt-4o",prerequisite:'Set up <ins><a href="https://konghq.com/products/kong-ai-gateway">Kong AI Gateway</a></ins> by following the installation guide and configure your LLM provider credentials.'},{id:"helicone",name:"Helicone",displayName:"Helicone AI Gateway",description:'<a href="https://www.helicone.ai/">Helicone AI Gateway</a> is an open-source LLM gateway that provides unified access to 100+ AI models through an OpenAI-compatible API. It offers built-in caching, rate limiting, automatic failover, and comprehensive analytics with minimal latency overhead.',baseUrl:"http://localhost:8080/ai",apiKeyPlaceholder:"placeholder-api-key",sampleModel:"anthropic/claude-4-5-sonnet",prerequisite:'Before following the steps below, you need to set up Helicone AI Gateway server.<ol><li>Set up your <code>.env</code> file with your LLM provider API keys (e.g., <code>OPENAI_API_KEY</code>, <code>ANTHROPIC_API_KEY</code>).</li><li>Run the gateway locally with <code>npx @helicone/ai-gateway@latest</code>.</li></ol>See the <ins><a href="https://docs.helicone.ai/gateway/overview#ai-gateway-overview">Helicone AI Gateway docs</a></ins> for more details.'},{id:"portkey",name:"Portkey",description:'<a href="https://portkey.ai/">Portkey</a> is an enterprise-grade AI gateway that provides unified access to 1600+ LLMs through a single OpenAI-compatible API. It offers built-in guardrails, observability, caching, load balancing, and fallback mechanisms for production AI applications.',baseUrl:"https://api.portkey.ai/v1",apiKeyPlaceholder:"<YOUR_PORTKEY_API_KEY>",sampleModel:"gpt-4o",defaultHeaders:{python:{"x-portkey-provider":"openai"},typescript:{"x-portkey-provider":"openai"}},headerComment:'or "anthropic", "google", etc.',prerequisite:'Create a <ins><a href="https://portkey.ai/">Portkey account</a></ins> and generate an API key from the <ins><a href="https://app.portkey.ai/api-keys">API Keys page</a></ins>. Configure your virtual keys for the LLM providers you want to use.'}];function A(e){let t=Object.entries(e);if(0===t.length)return"";let a=t.map(([e,t])=>`"${e}": "${t}"`).join(", ");return`, default_headers={${a}}`}function w(e){let t=Object.entries(e);if(0===t.length)return"";let a=t.map(([e,t])=>`"${e}": "${t}"`).join(", ");return`, defaultHeaders: { ${a} }`}let x=({gatewayId:e})=>{let t=y.find(t=>t.id===e);if(!t)return(0,n.jsxs)("div",{children:['Gateway "',e,'" not found in configuration.']});let a=t.displayName||t.name,x=t.heroImage||"/images/llms/tracing/basic-openai-trace.png",b=t.defaultHeaders?.python||{},j=t.defaultHeaders?.typescript||{},I=Object.keys(b).length>0,v=Object.keys(j).length>0;return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)("div",{dangerouslySetInnerHTML:{__html:t.description}}),(0,n.jsx)(c.A,{src:x,alt:`${a} Tracing`}),(0,n.jsxs)("p",{children:["Since ",a," exposes an OpenAI-compatible API, you can use MLflow's OpenAI autolog integration to automatically trace all your LLM calls through the gateway."]}),(0,n.jsx)("h2",{children:"Getting Started"}),(0,n.jsx)(o.A,{type:"tip",title:"Prerequisites",children:(0,n.jsx)("div",{dangerouslySetInnerHTML:{__html:t.prerequisite}})}),(0,n.jsx)(p.A,{number:1,title:"Install Dependencies"}),(0,n.jsx)(h.A,{children:(0,n.jsxs)(i.A,{groupId:"programming-language",children:[(0,n.jsx)(r.A,{value:"python",label:"Python",default:!0,children:(0,n.jsx)(f.A,{language:"bash",children:"pip install mlflow openai"})}),(0,n.jsx)(r.A,{value:"typescript",label:"TypeScript",children:(0,n.jsx)(f.A,{language:"bash",children:"npm install mlflow-openai openai"})})]})}),(0,n.jsx)(p.A,{number:2,title:"Start MLflow Server"}),(0,n.jsx)(d.Ay,{}),(0,n.jsx)(p.A,{number:3,title:"Enable Tracing and Make API Calls"}),(0,n.jsx)(h.A,{children:(0,n.jsxs)(i.A,{groupId:"programming-language",children:[(0,n.jsxs)(r.A,{value:"python",label:"Python",default:!0,children:[(0,n.jsxs)("p",{children:["Enable tracing with ",(0,n.jsx)("code",{children:"mlflow.openai.autolog()"})," and configure the OpenAI client to use"," ",a,"'s base URL."]}),(0,n.jsx)(f.A,{language:"python",children:`import mlflow
from openai import OpenAI

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Set tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("${t.name}")

# Create OpenAI client pointing to ${a}
client = OpenAI(
    base_url="${t.baseUrl}",
    api_key="${t.apiKeyPlaceholder}",${I?function(e,t){let a=Object.entries(e);if(0===a.length)return"";let n=a.map(([e,a])=>{let n=t?`  # ${t}`:"";return`        "${e}": "${a}",${n}`});return`
    default_headers={
${n.join("\n")}
    },`}(b,t.headerComment):""}
)

# Make API calls - traces will be captured automatically
response = client.chat.completions.create(
    model="${t.sampleModel}",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
print(response.choices[0].message.content)`})]}),(0,n.jsxs)(r.A,{value:"typescript",label:"TypeScript",children:[(0,n.jsxs)("p",{children:["Initialize MLflow tracing with ",(0,n.jsx)("code",{children:"init()"})," and wrap the OpenAI client with the"," ",(0,n.jsx)("code",{children:"tracedOpenAI"})," function."]}),(0,n.jsx)(f.A,{language:"typescript",children:`import { init } from "mlflow-tracing";
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

// Initialize MLflow tracing
init({
  trackingUri: "http://localhost:5000",
  experimentId: "<experiment-id>",
});

// Wrap the OpenAI client pointing to ${a}
const client = tracedOpenAI(
  new OpenAI({
    baseURL: "${t.baseUrl}",
    apiKey: "${t.apiKeyPlaceholder}",${v?function(e,t){let a=Object.entries(e);if(0===a.length)return"";let n=a.map(([e,a])=>{let n=t?` // ${t}`:"";return`      "${e}": "${a}",${n}`});return`
    defaultHeaders: {
${n.join("\n")}
    },`}(j,t.headerComment):""}
  })
);

// Make API calls - traces will be captured automatically
const response = await client.chat.completions.create({
  model: "${t.sampleModel}",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is the capital of France?" },
  ],
});
console.log(response.choices[0].message.content);`})]})]})}),(0,n.jsx)(p.A,{number:4,title:"View Traces in MLflow UI"}),(0,n.jsxs)("p",{children:["Open the MLflow UI at http://localhost:5000 to see the traces from your ",a," API calls."]}),(0,n.jsx)("h2",{children:"Combining with Manual Tracing"}),(0,n.jsx)("p",{children:"You can combine auto-tracing with MLflow's manual tracing to create comprehensive traces that include your application logic:"}),(0,n.jsx)(h.A,{children:(0,n.jsxs)(i.A,{groupId:"programming-language",children:[(0,n.jsx)(r.A,{value:"python",label:"Python",default:!0,children:(0,n.jsx)(f.A,{language:"python",children:`import mlflow
from mlflow.entities import SpanType
from openai import OpenAI

mlflow.openai.autolog()

client = OpenAI(
    base_url="${t.baseUrl}",
    api_key="${t.apiKeyPlaceholder}"${I?A(b):""},
)


@mlflow.trace(span_type=SpanType.CHAIN)
def ask_question(question: str) -> str:
    """A traced function that calls the LLM through ${a}."""
    response = client.chat.completions.create(
        model="${t.sampleModel}", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


# The entire function call and nested LLM call will be traced
answer = ask_question("What is machine learning?")
print(answer)`})}),(0,n.jsx)(r.A,{value:"typescript",label:"TypeScript",children:(0,n.jsx)(f.A,{language:"typescript",children:`import { init, trace, SpanType } from "mlflow-tracing";
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

init({
  trackingUri: "http://localhost:5000",
  experimentId: "<experiment-id>",
});

const client = tracedOpenAI(
  new OpenAI({
    baseURL: "${t.baseUrl}",
    apiKey: "${t.apiKeyPlaceholder}"${v?w(j):""},
  })
);

// Wrap your function with trace() to create a span
const askQuestion = trace(
  { name: "askQuestion", spanType: SpanType.CHAIN },
  async (question: string): Promise<string> => {
    const response = await client.chat.completions.create({
      model: "${t.sampleModel}",
      messages: [{ role: "user", content: question }],
    });
    return response.choices[0].message.content ?? "";
  }
);

// The entire function call and nested LLM call will be traced
const answer = await askQuestion("What is machine learning?");
console.log(answer);`})})]})}),(0,n.jsx)("h2",{children:"Streaming Support"}),(0,n.jsxs)("p",{children:["MLflow supports tracing streaming responses from ",a,":"]}),(0,n.jsx)(h.A,{children:(0,n.jsxs)(i.A,{groupId:"programming-language",children:[(0,n.jsx)(r.A,{value:"python",label:"Python",default:!0,children:(0,n.jsx)(f.A,{language:"python",children:`import mlflow
from openai import OpenAI

mlflow.openai.autolog()

client = OpenAI(
    base_url="${t.baseUrl}",
    api_key="${t.apiKeyPlaceholder}"${I?A(b):""},
)

stream = client.chat.completions.create(
    model="${t.sampleModel}",
    messages=[{"role": "user", "content": "Write a haiku about machine learning."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")`})}),(0,n.jsx)(r.A,{value:"typescript",label:"TypeScript",children:(0,n.jsx)(f.A,{language:"typescript",children:`import { init } from "mlflow-tracing";
import { tracedOpenAI } from "mlflow-openai";
import { OpenAI } from "openai";

init({
  trackingUri: "http://localhost:5000",
  experimentId: "<experiment-id>",
});

const client = tracedOpenAI(
  new OpenAI({
    baseURL: "${t.baseUrl}",
    apiKey: "${t.apiKeyPlaceholder}"${v?w(j):""},
  })
);

const stream = await client.chat.completions.create({
  model: "${t.sampleModel}",
  messages: [{ role: "user", content: "Write a haiku about machine learning." }],
  stream: true,
});

for await (const chunk of stream) {
  if (chunk.choices[0].delta.content) {
    process.stdout.write(chunk.choices[0].delta.content);
  }
}`})})]})}),(0,n.jsx)("p",{children:"MLflow will automatically capture the complete streamed response in the trace."}),(0,n.jsx)("h2",{children:"Next Steps"}),(0,n.jsxs)(s.A,{children:[(0,n.jsx)(l.A,{icon:u.A,iconSize:48,title:"Track User Feedback",description:"Record user feedback on traces for tracking user satisfaction.",href:"/genai/tracing/collect-user-feedback",linkText:"Learn about feedback \u2192",containerHeight:64}),(0,n.jsx)(l.A,{icon:m.A,iconSize:48,title:"Manage Prompts",description:"Learn how to manage prompts with MLflow's prompt registry.",href:"/genai/prompt-registry",linkText:"Manage prompts \u2192",containerHeight:64}),(0,n.jsx)(l.A,{icon:g.A,iconSize:48,title:"Evaluate Traces",description:"Evaluate traces with LLM judges to understand and improve your AI application's behavior.",href:"/genai/eval-monitor/running-evaluation/traces",linkText:"Evaluate traces \u2192",containerHeight:64})]})]})}},89001(e,t,a){a.d(t,{A:()=>i});var n=a(74848);a(96540);let i=({number:e,title:t})=>(0,n.jsxs)("div",{className:"stepHeader_RqmM",children:[(0,n.jsx)("div",{className:"stepNumber_exmH",children:e}),(0,n.jsx)("h3",{className:"stepTitle_SzBx",children:t})]})},95986(e,t,a){a.d(t,{A:()=>i});var n=a(74848);a(96540);function i({children:e}){return(0,n.jsx)("div",{className:"wrapper_sf5q",children:e})}},77541(e,t,a){a.d(t,{A:()=>c});var n=a(74848);a(96540);var i=a(95310),r=a(34164);let o="tileImage_O4So";var s=a(66497),l=a(92802);function c({icon:e,image:t,imageDark:a,imageWidth:c,imageHeight:p,iconSize:d=32,containerHeight:h,title:u,description:m,href:g,linkText:f="Learn more \u2192",className:y}){if(!e&&!t)throw Error("TileCard requires either an icon or image prop");let A=h?{height:`${h}px`}:{},w={};return c&&(w.width=`${c}px`),p&&(w.height=`${p}px`),(0,n.jsxs)(i.A,{href:g,className:(0,r.A)("tileCard_NHsj",y),children:[(0,n.jsx)("div",{className:"tileIcon_pyoR",style:A,children:e?(0,n.jsx)(e,{size:d}):a?(0,n.jsx)(l.A,{sources:{light:(0,s.default)(t),dark:(0,s.default)(a)},alt:u,className:o,style:w}):(0,n.jsx)("img",{src:(0,s.default)(t),alt:u,className:o,style:w})}),(0,n.jsx)("h3",{children:u}),(0,n.jsx)("p",{children:m}),(0,n.jsx)("div",{className:"tileLink_iUbu",children:f})]})}},10440(e,t,a){a.d(t,{A:()=>r});var n=a(74848);a(96540);var i=a(34164);function r({children:e,className:t}){return(0,n.jsx)("div",{className:(0,i.A)("tilesGrid_hB9N",t),children:e})}},52915(e,t,a){a.d(t,{A:()=>d});var n=a(74848);a(96540);var i=a(34164),r=a(71643),o=a(66697),s=a(92949),l=a(64560),c=a(47819);function p({language:e}){return(0,n.jsxs)("div",{className:(0,i.A)("codeBlockHeader_C_1e"),"aria-label":`Code block header for ${e} code with copy and toggle buttons`,children:[(0,n.jsx)("span",{className:"languageLabel_zr_I",children:e}),(0,n.jsx)(c.A,{})]})}function d({className:e}){let{metadata:t}=(0,r.Ph)(),a=t.language||"text";return(0,n.jsxs)(o.A,{as:"div",className:(0,i.A)(e,t.className),children:[t.title&&(0,n.jsx)("div",{className:"codeBlockTitle_d3dP",children:(0,n.jsx)(s.A,{children:t.title})}),(0,n.jsxs)("div",{className:"codeBlockContent_bxn0",children:[(0,n.jsx)(p,{language:a}),(0,n.jsx)(l.A,{})]})]})}}}]);