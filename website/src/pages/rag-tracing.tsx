import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";

import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import TraceUiImage from "@site/static/img/rag-tracing/mlflow-trace-ui-span-tree.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE = "RAG Tracing: Trace and Debug RAG Pipelines | MLflow";
const SEO_DESCRIPTION =
  "RAG tracing records every retrieval, prompt, and generation step in a RAG pipeline. Debug bad retrieval and ungrounded answers with open-source MLflow Tracing.";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText: string;
}[] = [
  {
    question: "What is RAG tracing?",
    answer:
      "RAG tracing is the practice of recording every step of a retrieval-augmented generation request as a structured trace: the user query, any query rewriting, the retrieval call and the documents it returned, reranking, the assembled prompt, and the LLM generation. Each step is a span with inputs, outputs, latency, and errors, so you can see exactly which context the model was given and why an answer was good or bad.",
    answerText:
      "RAG tracing is the practice of recording every step of a retrieval-augmented generation request as a structured trace: the user query, any query rewriting, the retrieval call and the documents it returned, reranking, the assembled prompt, and the LLM generation. Each step is a span with inputs, outputs, latency, and errors, so you can see exactly which context the model was given and why an answer was good or bad.",
  },
  {
    question: "How do I trace a RAG application with MLflow?",
    answer: (
      <>
        If you use LangChain, LlamaIndex, Haystack, DSPy, or the OpenAI SDK,
        call the framework's <code>autolog()</code> function once, for example{" "}
        <code>mlflow.langchain.autolog()</code>, and MLflow records every step
        automatically. For custom pipelines, decorate your retrieval function
        with <code>@mlflow.trace(span_type="RETRIEVER")</code> and return a list
        of <code>mlflow.entities.Document</code> objects. See the{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
          tracing quickstart
        </Link>{" "}
        for setup.
      </>
    ),
    answerText:
      'If you use LangChain, LlamaIndex, Haystack, DSPy, or the OpenAI SDK, call the framework\'s autolog() function once, for example mlflow.langchain.autolog(), and MLflow records every step automatically. For custom pipelines, decorate your retrieval function with @mlflow.trace(span_type="RETRIEVER") and return a list of mlflow.entities.Document objects. See the tracing quickstart for setup.',
  },
  {
    question: "What should a RAG trace capture?",
    answer:
      "At minimum: the user query, the retrieval span with the returned documents (content, source metadata, and ids), the final prompt sent to the LLM, the model response, latency per step, and token usage. If you rewrite queries, rerank, or filter documents, give each of those steps its own span so you can tell which stage dropped or demoted the right document.",
    answerText:
      "At minimum: the user query, the retrieval span with the returned documents (content, source metadata, and ids), the final prompt sent to the LLM, the model response, latency per step, and token usage. If you rewrite queries, rerank, or filter documents, give each of those steps its own span so you can tell which stage dropped or demoted the right document.",
  },
  {
    question: "How do I debug a RAG pipeline that returns wrong answers?",
    answer:
      "Open the trace for a bad answer and work backwards. First check the retrieval span: were the right documents returned? If not, the problem is retrieval (chunking, embeddings, filters, or top-k). If the right documents were retrieved, check the prompt span to confirm they made it into the context. If the context was correct and the answer is still wrong, the problem is generation. This three-way split tells you which part of the pipeline to fix.",
    answerText:
      "Open the trace for a bad answer and work backwards. First check the retrieval span: were the right documents returned? If not, the problem is retrieval (chunking, embeddings, filters, or top-k). If the right documents were retrieved, check the prompt span to confirm they made it into the context. If the context was correct and the answer is still wrong, the problem is generation. This three-way split tells you which part of the pipeline to fix.",
  },
  {
    question: "What is the difference between RAG tracing and RAG evaluation?",
    answer: (
      <>
        Tracing records what happened on each request. Evaluation scores whether
        it was good. They work together: MLflow's RAG judges{" "}
        <code>RetrievalRelevance</code>, <code>RetrievalGroundedness</code>, and{" "}
        <code>RetrievalSufficiency</code> read the retriever spans in your
        traces and score each request. See{" "}
        <Link href="/llm-evaluation">LLM evaluation</Link> and the{" "}
        <Link
          href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/llm-judge/rag/"}
        >
          RAG judges documentation
        </Link>
        .
      </>
    ),
    answerText:
      "Tracing records what happened on each request. Evaluation scores whether it was good. They work together: MLflow's RAG judges RetrievalRelevance, RetrievalGroundedness, and RetrievalSufficiency read the retriever spans in your traces and score each request.",
  },
  {
    question: "Does MLflow work with my vector database?",
    answer:
      "MLflow does not need a vector database integration. Tracing wraps the function or framework call that queries your vector store, so any store works: Pinecone, Weaviate, Chroma, Qdrant, pgvector, Elasticsearch, or your own. If you use LangChain, LlamaIndex, or Haystack, retriever calls are captured as RETRIEVER spans automatically. Otherwise, decorate your own retrieval function.",
    answerText:
      "MLflow does not need a vector database integration. Tracing wraps the function or framework call that queries your vector store, so any store works: Pinecone, Weaviate, Chroma, Qdrant, pgvector, Elasticsearch, or your own. If you use LangChain, LlamaIndex, or Haystack, retriever calls are captured as RETRIEVER spans automatically. Otherwise, decorate your own retrieval function.",
  },
  {
    question: "Can I trace RAG in production without slowing it down?",
    answer: (
      <>
        Yes. Use the lightweight production tracing SDK, log traces
        asynchronously, and set <code>MLFLOW_TRACE_SAMPLING_RATIO</code> to
        record only a fraction of high-volume traffic. See{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing"}>
          production tracing
        </Link>{" "}
        for the checklist.
      </>
    ),
    answerText:
      "Yes. Use the lightweight production tracing SDK, log traces asynchronously, and set MLFLOW_TRACE_SAMPLING_RATIO to record only a fraction of high-volume traffic. See production tracing for the checklist.",
  },
  {
    question: "Are retrieved documents stored in the trace?",
    answer: (
      <>
        Yes, the retriever span output holds the documents you return. If
        documents are large or sensitive, you can reduce what is stored. The{" "}
        <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/faq"}>tracing FAQ</Link>{" "}
        shows how to register a span processor that trims retriever outputs, and
        MLflow also supports{" "}
        <Link
          href={MLFLOW_GENAI_DOCS_URL + "tracing/observe-with-traces/masking"}
        >
          masking sensitive data
        </Link>
        .
      </>
    ),
    answerText:
      "Yes, the retriever span output holds the documents you return. If documents are large or sensitive, you can reduce what is stored. The tracing FAQ shows how to register a span processor that trims retriever outputs, and MLflow also supports masking sensitive data.",
  },
  {
    question: "Is MLflow RAG tracing free and open source?",
    answer:
      "Yes. MLflow is open source under the Apache 2.0 license and backed by the Linux Foundation. Tracing, the trace UI, search, and the RAG judges are all included at no cost, and you can store traces in your own infrastructure. Managed MLflow is also available on Databricks and other platforms if you prefer a hosted option.",
    answerText:
      "Yes. MLflow is open source under the Apache 2.0 license and backed by the Linux Foundation. Tracing, the trace UI, search, and the RAG judges are all included at no cost, and you can store traces in your own infrastructure. Managed MLflow is also available on Databricks and other platforms if you prefer a hosted option.",
  },
];

const faqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: faqs.map((faq) => ({
    "@type": "Question",
    name: faq.question,
    acceptedAnswer: {
      "@type": "Answer",
      text: faq.answerText,
    },
  })),
};

const softwareJsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "MLflow",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Cross-platform",
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
  },
  description:
    "Open-source platform for tracing, debugging, and evaluating RAG pipelines, agents, and LLM applications.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

const AUTOLOG_CODE = `import mlflow

mlflow.set_experiment("rag-app")

# Enable once, before your app runs. Pick the line that matches your stack.
mlflow.langchain.autolog()     # LangChain and LangGraph retrievers and chains
# mlflow.llama_index.autolog() # LlamaIndex query engines and retrievers
# mlflow.haystack.autolog()    # Haystack pipelines
# mlflow.openai.autolog()      # the LLM call inside any custom pipeline

answer = rag_chain.invoke({"question": "How do I enable tracing?"})`;

const MANUAL_CODE = `import mlflow
from mlflow.entities import Document, SpanType


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(query: str) -> list[Document]:
    hits = vector_store.search(query, top_k=5)  # your vector database
    return [
        Document(
            id=h.id,
            page_content=h.text,
            metadata={"source": h.source, "score": h.score},
        )
        for h in hits
    ]


@mlflow.trace(span_type=SpanType.CHAIN)
def answer(query: str) -> str:
    docs = retrieve(query)
    with mlflow.start_span(name="build_prompt", span_type=SpanType.PARSER) as span:
        context = "\\n\\n".join(d.page_content for d in docs)
        prompt = f"Answer using only this context:\\n{context}\\n\\nQuestion: {query}"
        span.set_inputs({"query": query, "num_docs": len(docs)})
        span.set_outputs(prompt)
    return call_llm(prompt)  # traced by mlflow.openai.autolog() or similar`;

const INSPECT_CODE = `import mlflow
from mlflow.entities import SpanType

traces = mlflow.search_traces(max_results=50, return_type="list")

for trace in traces:
    for span in trace.search_spans(span_type=SpanType.RETRIEVER):
        docs = span.outputs or []
        print(trace.info.trace_id, f"{len(docs)} documents retrieved")`;

const EVAL_CODE = `import mlflow
from mlflow.genai.scorers import RetrievalRelevance, RetrievalGroundedness

# Score the retrieval step and the grounding of the answer on recent traces
traces = mlflow.search_traces(max_results=100)

results = mlflow.genai.evaluate(
    data=traces,
    scorers=[RetrievalRelevance(), RetrievalGroundedness()],
)`;

function CodeBlock({ code }: { code: string }) {
  return (
    <div
      className="rounded-lg border border-white/10 overflow-hidden"
      style={{ backgroundColor: CODE_BG, margin: "8px 0 24px" }}
    >
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-white/5">
        <span className="text-xs text-white/50 font-mono">python</span>
        <CopyButton code={code} />
      </div>
      <div className="p-3 overflow-x-auto">
        <Highlight theme={customNightOwl} code={code} language="python">
          {({ style, tokens, getLineProps, getTokenProps }) => (
            <pre
              className="text-xs font-mono !m-0 !p-0 text-left"
              style={{ ...style, backgroundColor: "transparent" }}
            >
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}

const TH_STYLE = {
  padding: "14px 16px",
  textAlign: "left" as const,
  fontWeight: 600,
  color: "#1a1a1a",
};
const TD_STYLE = {
  padding: "14px 16px",
  verticalAlign: "top" as const,
  color: "#3d3d3d",
};

function SimpleTable({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div
      style={{
        overflowX: "auto",
        border: "1px solid #e2e8f0",
        borderRadius: "8px",
        margin: "24px 0 40px",
      }}
    >
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "14px",
          lineHeight: "1.6",
        }}
      >
        <thead>
          <tr
            style={{
              backgroundColor: "#f8fafc",
              borderBottom: "2px solid #e2e8f0",
            }}
          >
            {head.map((h) => (
              <th key={h} style={TH_STYLE}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #e2e8f0" }}>
              {row.map((cell, j) => (
                <td
                  key={j}
                  style={{ ...TD_STYLE, fontWeight: j === 0 ? 600 : 400 }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RagTracing() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="article:modified_time" content="2026-11-26" />
        <meta property="og:url" content="https://mlflow.org/rag-tracing" />
        <link rel="canonical" href="https://mlflow.org/rag-tracing" />
        <script type="application/ld+json">{JSON.stringify(faqJsonLd)}</script>
        <script type="application/ld+json">
          {JSON.stringify(softwareJsonLd)}
        </script>
        <style>{`
          /* Import MLflow docs fonts */
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

          /* Black header */
          header, nav {
            background: #000000 !important;
          }

          body {
            background: #ffffff;
            margin: 0;
            padding: 0;
            font-family: 'DM Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          }
          .article-page {
            background: #ffffff;
            min-height: 100vh;
          }
          .article-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 60px 24px 100px;
          }
          .article-container h1 {
            font-family: 'DM Sans', sans-serif;
            font-size: 3rem !important;
            font-weight: 700 !important;
            color: #1a1a1a !important;
            margin: 48px 0 32px 0 !important;
            line-height: 1.0 !important;
            letter-spacing: -0.03em !important;
          }
          .article-container h2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #1a1a1a;
            margin: 64px 0 32px 0;
            line-height: 1.2;
            letter-spacing: -0.01em;
          }
          .article-container h3 {
            font-family: 'DM Sans', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a1a1a;
            margin: 32px 0 16px 0;
            line-height: 1.2;
          }
          .article-container p {
            font-family: 'DM Sans', sans-serif;
            font-size: 16px;
            color: #3d3d3d;
            line-height: 1.7;
            margin: 0 0 24px 0;
          }
          .article-container strong {
            font-weight: 600;
            color: #1a1a1a;
          }
          .article-container a {
            color: #0194e2 !important;
            text-decoration: none;
            transition: all 0.2s ease;
          }
          .article-container a:hover {
            color: #0072b0 !important;
            text-decoration: underline;
          }
          .info-box a {
            color: #0194e2 !important;
            font-weight: 500;
          }
          .info-box a:hover {
            color: #0072b0 !important;
          }
          .article-container ul {
            list-style-type: disc;
            margin: 20px 0;
            padding-left: 24px;
            list-style-position: outside;
          }
          .article-container li {
            font-family: 'DM Sans', sans-serif;
            font-size: 16px;
            color: #3d3d3d;
            line-height: 1.7;
            margin-bottom: 20px;
            padding-left: 8px;
            text-wrap: pretty;
          }
          .article-container li a {
            font-weight: 600;
            color: #0194e2 !important;
          }
          .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin: 40px 0 56px 0;
          }
          .card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 32px;
            box-shadow: 0 0 0 1px rgba(50, 50, 93, 0.05), 0 0 14px 5px rgba(50, 50, 93, 0.08), 0 0 10px 3px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
          }
          .card:hover {
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
          }
          .card h3 {
            font-family: 'DM Sans', sans-serif;
            margin-top: 0 !important;
            margin-bottom: 32px !important;
            font-size: 1.3rem;
            font-weight: 600;
            color: #1a1a1a;
            line-height: 1.2;
          }
          .card p {
            font-family: 'DM Sans', sans-serif;
            margin-bottom: 14px;
            font-size: 14px;
            line-height: 1.5;
            color: #505050;
          }
          .card p:last-child {
            margin-bottom: 0;
          }
          .faq-list {
            margin: 32px 0;
          }
          .faq-item {
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            margin-bottom: 16px;
            background: #ffffff;
            transition: all 0.2s ease;
          }
          .faq-item:hover {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
          }
          .faq-question {
            font-family: 'DM Sans', sans-serif;
            padding: 20px 24px;
            font-size: 18px;
            font-weight: 500;
            color: #1a1a1a;
            cursor: pointer;
            background: transparent;
            border: none;
            width: 100%;
            text-align: left;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s ease;
          }
          .faq-question:hover {
            background: #f9fafb;
          }
          .faq-answer {
            font-family: 'DM Sans', sans-serif;
            padding: 0 24px 20px;
            font-size: 16px;
            color: #3d3d3d;
            line-height: 1.6;
          }
          .faq-chevron {
            transition: transform 0.2s ease;
            flex-shrink: 0;
            margin-left: 16px;
            color: #6b7280;
            font-size: 12px;
          }
          .faq-chevron.open {
            transform: rotate(180deg);
          }
          .info-box {
            background: rgb(238, 249, 253);
            border: 1px solid rgb(41, 120, 115);
            padding: 20px 24px;
            margin: 40px 0;
            border-radius: 5.6px;
          }
          .info-box p {
            font-family: 'DM Sans', sans-serif;
            margin: 0;
            font-size: 16px;
            line-height: 1.6;
            color: #1e293b;
          }
          .article-sidebar {
            position: fixed;
            top: 100px;
            left: calc(50% + 900px / 2 + 48px);
            width: 280px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
          }
          .article-sidebar .toc-title {
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
            margin: 0 0 12px 0;
          }
          .article-sidebar ul {
            margin: 0;
            padding: 0;
            list-style: none;
            border-left: 1px solid #e5e7eb;
          }
          .article-sidebar li {
            margin: 0;
            padding: 0;
          }
          .article-sidebar a {
            font-family: 'DM Sans', sans-serif;
            display: block;
            padding: 8px 0 8px 16px;
            font-size: 16px;
            color: #0194e2 !important;
            text-decoration: none !important;
            transition: all 0.15s ease;
            line-height: 1.4;
          }
          .article-sidebar a:hover {
            color: #0072b0 !important;
          }
          .article-sidebar .toc-divider {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 12px 0 12px 0;
          }
          @media (max-width: 1400px) {
            .article-sidebar {
              display: none;
            }
          }
          @media (max-width: 768px) {
            .article-container {
              padding: 40px 20px 80px;
            }
            .article-container h1 {
              font-size: 36px !important;
              margin-bottom: 24px !important;
            }
            .article-container h2 {
              font-size: 28px;
              margin: 40px 0 20px 0;
            }
            .grid-2 {
              grid-template-columns: 1fr;
              gap: 20px;
            }
            .card {
              padding: 24px;
            }
            .faq-question {
              padding: 18px 20px;
              font-size: 17px;
            }
            .faq-answer {
              padding: 0 20px 20px;
              font-size: 16px;
            }
          }
        `}</style>
      </Head>

      <div className="article-page">
        <Header />

        <div className="article-container">
          <h1>RAG Tracing</h1>

          <p>
            <strong>RAG tracing</strong> records every step of a
            retrieval-augmented generation request, from the user query and the
            documents retrieved to the final prompt and the model's answer, as a
            single trace you can inspect. It shows you what context the LLM was
            actually given, which is the first thing you need to know when a RAG
            application gives a wrong or unsupported answer.
          </p>

          <p>
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>
              MLflow Tracing
            </Link>{" "}
            is an open-source, OpenTelemetry-compatible way to do this. It
            captures each retrieval, rerank, prompt, and generation step as a
            span with inputs, outputs, latency, and token usage. You can turn it
            on with one line for popular frameworks, or add it to a custom
            pipeline with a decorator.
          </p>

          <p style={{ fontSize: "14px", color: "#6b7280" }}>
            Last updated: 2026-11-26
          </p>

          <figure style={{ margin: "40px 0" }}>
            <img
              src={TraceUiImage}
              alt="MLflow trace UI showing the span tree of a single agent request, with per-span latency, token counts, estimated cost, and the request's input and output"
              style={{
                width: "100%",
                borderRadius: "8px",
                border: "1px solid #e5e7eb",
              }}
            />
            <figcaption
              style={{
                fontSize: "14px",
                color: "#6b7280",
                marginTop: "12px",
                textAlign: "center",
              }}
            >
              A trace in the MLflow UI: the span tree on the left with latency,
              tokens, and cost per step, and the request's input and output on
              the right. Demo data.
            </figcaption>
          </figure>

          <div
            style={{
              background: "#f9fafb",
              border: "1px solid #e5e7eb",
              borderRadius: "8px",
              padding: "20px 24px",
              margin: "40px 0",
            }}
          >
            <p style={{ marginBottom: "12px" }}>
              <strong>Quick Navigation:</strong>
            </p>
            <ul style={{ margin: 0, paddingLeft: "24px" }}>
              <li style={{ marginBottom: "8px" }}>
                <a href="#why-trace-rag">Why trace RAG</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#what-a-rag-trace-contains">
                  What a RAG trace contains
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#how-to-trace-rag">How to trace RAG with MLflow</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#debug-retrieval">Debug bad retrieval</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#score-retrieval">Score retrieval quality</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#production">RAG tracing in production</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2 id="why-trace-rag" data-toc="Why trace a RAG pipeline">
            Why Should You Trace a RAG Pipeline?
          </h2>

          <p>
            {" "}
            You trace a RAG pipeline to find out where a bad answer came from:
            retrieval, prompt assembly, or generation. From the outside, all
            three failures look the same, a wrong answer. A trace shows each
            step, so you can fix the right one.{" "}
          </p>

          <div className="grid-2">
            <div className="card">
              <h3>Irrelevant retrieval</h3>
              <p>
                <strong>Symptom:</strong> The answer is off-topic or generic.
              </p>
              <p>
                <strong>What the trace shows:</strong> The retrieval span
                returned documents that do not match the query, pointing at
                chunking, embeddings, filters, or top-k.
              </p>
            </div>
            <div className="card">
              <h3>Missing context</h3>
              <p>
                <strong>Symptom:</strong> The model says it does not know, or
                guesses.
              </p>
              <p>
                <strong>What the trace shows:</strong> The needed document was
                never retrieved, or a rerank or filter step dropped it.
              </p>
            </div>
            <div className="card">
              <h3>Ungrounded answers</h3>
              <p>
                <strong>Symptom:</strong> The answer sounds right but cites
                facts that are not in your data.
              </p>
              <p>
                <strong>What the trace shows:</strong> The right context was in
                the prompt, but the generation span's output contradicts or goes
                beyond it.
              </p>
            </div>
            <div className="card">
              <h3>Slow or costly requests</h3>
              <p>
                <strong>Symptom:</strong> High latency or token spend on some
                queries.
              </p>
              <p>
                <strong>What the trace shows:</strong> Per-step latency and
                token usage, such as an oversized context window or a slow
                vector-store call.
              </p>
            </div>
          </div>

          <h2
            id="what-a-rag-trace-contains"
            data-toc="What a RAG trace contains"
          >
            What Does a RAG Trace Contain?
          </h2>

          <p>
            {" "}
            A RAG trace is a tree of spans, one for each pipeline step: query
            handling, retrieval, reranking, prompt assembly, and generation.
            Each span has a type, so you can find the retrieval step in any
            request without reading logs. A typical trace looks like this:{" "}
          </p>

          <SimpleTable
            head={["Step", "Span type", "What to look at"]}
            rows={[
              [
                "Query handling",
                "CHAIN or PARSER",
                "The raw user query and any rewritten or expanded query",
              ],
              [
                "Retrieval",
                "RETRIEVER",
                "The documents returned: content, source metadata, ids, and scores",
              ],
              [
                "Reranking or filtering",
                "RERANKER or TOOL",
                "Which documents were kept, reordered, or dropped",
              ],
              [
                "Prompt assembly",
                "PARSER or CHAIN",
                "The final prompt, including the context actually passed to the model",
              ],
              [
                "Generation",
                "LLM or CHAT_MODEL",
                "The model, parameters, response, latency, and token usage",
              ],
            ]}
          />

          <p>
            The <code>RETRIEVER</code> span type matters most. MLflow's RAG
            judges use it to find your retrieval step, and the LangChain,
            LlamaIndex, and Haystack integrations set it for you.
          </p>

          <h2 id="how-to-trace-rag" data-toc="How to trace RAG with MLflow">
            How Do You Trace RAG with MLflow?
          </h2>

          <p>
            {" "}
            Call <code>autolog()</code> for a supported framework, or decorate
            your retrieval function with{" "}
            <code>@mlflow.trace(span_type="RETRIEVER")</code> for a custom
            pipeline. Both produce the same kind of trace, and you can combine
            them.{" "}
          </p>

          <h3>Option 1: Use autolog for popular frameworks</h3>

          <p>
            MLflow provides{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/app-instrumentation/automatic"
              }
            >
              automatic tracing
            </Link>{" "}
            for the frameworks most RAG applications are built on, including{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/langchain"
              }
            >
              LangChain
            </Link>
            ,{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/integrations/listing/llama_index"
              }
            >
              LlamaIndex
            </Link>
            ,{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/haystack"
              }
            >
              Haystack
            </Link>
            ,{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/dspy"}
            >
              DSPy
            </Link>
            , and the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/integrations/listing/openai"
              }
            >
              OpenAI SDK
            </Link>
            . Call <code>autolog()</code> once and every retriever, chain, and
            LLM call is recorded.
          </p>

          <CodeBlock code={AUTOLOG_CODE} />

          <h3>Option 2: Decorate your own retrieval code</h3>

          <p>
            For a custom pipeline, use{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/app-instrumentation/manual-tracing"
              }
            >
              manual tracing
            </Link>
            . Mark the retrieval function as a <code>RETRIEVER</code> span and
            return <code>Document</code> objects. Use{" "}
            <code>mlflow.start_span</code> for any extra step you want to see,
            such as prompt assembly.
          </p>

          <CodeBlock code={MANUAL_CODE} />

          <p>
            The two options combine well. You can autolog the LLM call and
            decorate the retrieval function, and both appear in the same trace.
          </p>

          <h2 id="debug-retrieval" data-toc="Debug bad retrieval">
            How Do You Debug Bad Retrieval?
          </h2>

          <p>
            {" "}
            Open the trace of a bad answer and ask three questions in order:
            were the right documents retrieved, did they reach the prompt, and
            did the model use them? The first "no" tells you which stage to fix.{" "}
          </p>

          <ol>
            <li>
              <strong>Were the right documents retrieved?</strong> Open the
              retrieval span. If the answer was in your corpus but not in the
              output, look at chunking, embeddings, metadata filters, and top-k.
            </li>
            <li>
              <strong>Did they reach the prompt?</strong> Open the prompt span.
              A rerank, truncation, or formatting bug can drop good documents
              after retrieval.
            </li>
            <li>
              <strong>Did the model use them?</strong> If the context was right
              and the answer is still wrong, the problem is in generation, such
              as the prompt wording or the model choice.
            </li>
          </ol>

          <p>
            In the MLflow UI, filter the Traces tab to find failing requests,
            open one, and click through the span tree. To find patterns across
            many requests, search traces from code and inspect the retriever
            spans programmatically. See{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/search-traces"}>
              searching traces
            </Link>{" "}
            for the filter syntax.
          </p>

          <CodeBlock code={INSPECT_CODE} />

          <h2 id="score-retrieval" data-toc="Score retrieval quality">
            How Do You Score Retrieval Quality on Your Traces?
          </h2>

          <p>
            {" "}
            Use MLflow's built-in RAG judges. They read the retriever spans in
            your traces and score each request, so you can find weak retrieval
            across thousands of requests instead of reading traces one at a
            time:{" "}
          </p>

          <SimpleTable
            head={["Judge", "Question it answers", "Needs ground truth"]}
            rows={[
              [
                "RetrievalRelevance",
                "Is each retrieved document relevant to the user's request?",
                "No",
              ],
              [
                "RetrievalGroundedness",
                "Is the response supported by the retrieved context?",
                "No",
              ],
              [
                "RetrievalSufficiency",
                "Do the retrieved documents contain everything needed for the expected answer?",
                "Yes",
              ],
            ]}
          />

          <CodeBlock code={EVAL_CODE} />

          <p>
            Results are logged to MLflow, so you can compare retrieval changes,
            such as a new chunk size or embedding model, on the same set of
            requests. Judge scores are model-generated and can be wrong, so
            spot-check a sample of low and high scores against your own judgment
            before you rely on them. For the full workflow, see{" "}
            <Link href="/llm-evaluation">LLM evaluation</Link> and the{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/llm-judge/rag/"
              }
            >
              RAG judges guide
            </Link>
            .
          </p>

          <h2 id="production" data-toc="RAG tracing in production">
            Can You Trace RAG in Production?
          </h2>

          <p>
            {" "}
            Yes. The same tracing works in production. For high-volume services,
            MLflow recommends the lightweight production tracing SDK,
            asynchronous trace logging, and sampling through the{" "}
            <code>MLFLOW_TRACE_SAMPLING_RATIO</code> environment variable to
            control overhead and storage. Tracing is{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "tracing/app-instrumentation/opentelemetry"
              }
            >
              OpenTelemetry compatible
            </Link>
            , so it fits alongside existing observability tools.
          </p>

          <p>
            Retrieved documents can be large or sensitive. You can trim what a
            retriever span stores and{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL + "tracing/observe-with-traces/masking"
              }
            >
              mask sensitive data
            </Link>{" "}
            before it is logged. See{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/prod-tracing"}>
              production tracing
            </Link>{" "}
            for the full checklist.
          </p>

          <div className="info-box">
            <p>
              <strong>Getting started:</strong> Install MLflow with{" "}
              <code>pip install mlflow</code>, start the UI with{" "}
              <code>mlflow server</code>, and follow the{" "}
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
                tracing quickstart
              </Link>
              .
            </p>
          </div>

          <h2 id="faq">Frequently Asked Questions</h2>

          <div className="faq-list">
            {faqs.map((faq, index) => (
              <div key={index} className="faq-item">
                <button
                  className="faq-question"
                  onClick={() =>
                    setOpenFaqIndex(openFaqIndex === index ? null : index)
                  }
                >
                  <span>{faq.question}</span>
                  <span
                    className={`faq-chevron ${openFaqIndex === index ? "open" : ""}`}
                  >
                    ▼
                  </span>
                </button>
                {openFaqIndex === index && (
                  <div className="faq-answer">{faq.answer}</div>
                )}
              </div>
            ))}
          </div>

          <h2>Related Resources</h2>

          <ul>
            <li>
              <Link href="/llm-tracing">LLM Tracing</Link>
            </li>
            <li>
              <Link href="/llm-observability">AI Observability</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">LLM Evaluation</Link>
            </li>
            <li>
              <Link href="/llm-as-a-judge">LLM as a Judge</Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/quickstart/"}>
                Tracing Quickstart
              </Link>
            </li>
            <li>
              <Link
                href={
                  MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/llm-judge/rag/"
                }
              >
                RAG Evaluation with Built-in Judges
              </Link>
            </li>
            <li>
              <Link href="/genai">MLflow for Agents and LLMs Overview</Link>
            </li>
            <li>
              <Link href="/rag-evaluation">RAG Evaluation</Link>
            </li>
            <li>
              <Link href="/agent-tracing">Agent Tracing</Link>
            </li>
            <li>
              <Link href="/trace-analytics">Trace Analytics</Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
