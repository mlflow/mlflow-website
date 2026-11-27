import { useState } from "react";
import Head from "@docusaurus/Head";
import Link from "@docusaurus/Link";
import { Highlight } from "prism-react-renderer";
import { Header } from "../components/Header/Header";
import { SocialLinksFooter } from "../components/SocialLinksFooter/SocialLinksFooter";
import { ArticleSidebar } from "../components/ArticleSidebar/ArticleSidebar";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";
import RagEvalRuns from "@site/static/img/rag-evaluation/mlflow-evaluation-runs-rag.png";
import { CopyButton } from "../components/CodeSnippet/CopyButton";
import { customNightOwl, CODE_BG } from "../components/CodeSnippet/codeTheme";

const SEO_TITLE = "RAG Evaluation: How to Evaluate RAG Applications | MLflow";
const SEO_DESCRIPTION =
  "RAG evaluation measures retrieval relevance, sufficiency, and groundedness. Learn how to evaluate RAG pipelines with MLflow's open-source judges and tracing.";

const DOCS_RAG = MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/llm-judge/rag/";

const faqs: {
  question: string;
  answer: React.ReactNode;
  answerText?: string;
}[] = [
  {
    question: "What is RAG evaluation?",
    answer:
      "RAG evaluation is the process of measuring how well a retrieval-augmented generation application retrieves relevant documents and how faithfully its generated answer uses them. A complete RAG evaluation scores retrieval quality (relevance and sufficiency of the retrieved chunks) and generation quality (groundedness, correctness, and relevance of the answer) separately, so you can tell whether a bad answer came from the retriever or from the language model.",
  },
  {
    question: "What metrics should I use to evaluate a RAG system?",
    answer:
      "Start with three retrieval-focused metrics and two answer-focused metrics. For retrieval: context relevance (are the retrieved chunks about the question?) and context sufficiency (do they contain everything needed to answer?). For generation: groundedness (is every claim supported by the retrieved context?), correctness (does the answer contain the expected facts?), and relevance to the query. In MLflow these map to RetrievalRelevance, RetrievalSufficiency, RetrievalGroundedness, Correctness, and RelevanceToQuery.",
  },
  {
    question: "How do I evaluate a RAG application with MLflow?",
    answer: (
      <>
        Mark your retrieval function with{" "}
        <code>@mlflow.trace(span_type="RETRIEVER")</code>, build a dataset of
        questions with expected facts, and call{" "}
        <code>mlflow.genai.evaluate()</code> with your RAG function as{" "}
        <code>predict_fn</code> and the RAG judges as <code>scorers</code>.
        MLflow traces every call, runs each judge, and stores the scores in the
        MLflow UI. The <Link href={DOCS_RAG}>RAG evaluation documentation</Link>{" "}
        has a complete example.
      </>
    ),
    answerText:
      'Mark your retrieval function with @mlflow.trace(span_type="RETRIEVER"), build a dataset of questions with expected facts, and call mlflow.genai.evaluate() with your RAG function as predict_fn and the RAG judges as scorers. MLflow traces every call, runs each judge, and stores the scores in the MLflow UI. The RAG evaluation documentation has a complete example.',
  },
  {
    question: "What is the difference between groundedness and correctness?",
    answer:
      "Groundedness checks whether the answer is supported by the documents your retriever returned, which detects hallucinations relative to the retrieved context. Correctness checks whether the answer contains the expected facts you wrote in your dataset, which detects wrong answers relative to ground truth. An answer can be grounded but incorrect (the retriever fetched the wrong document and the model faithfully summarized it) or correct but ungrounded (the model answered from memory instead of the context).",
  },
  {
    question: "Do I need ground-truth labels to evaluate RAG?",
    answer:
      "Not for all metrics. RetrievalRelevance and RetrievalGroundedness need no ground truth; they judge the retrieved chunks and the response directly. RetrievalSufficiency and Correctness compare against expected facts or response, so they need an expectations field in your dataset. A practical approach is to start with the label-free judges on production traces, then add expected facts for your most important questions.",
  },
  {
    question: "Why do the RAG judges require traces?",
    answer:
      'The retrieval judges evaluate the documents that your retriever actually returned, and MLflow reads them from the trace. They look for spans with type RETRIEVER and score each top-level retriever span separately; a retriever nested inside another retriever is skipped. Add @mlflow.trace(span_type="RETRIEVER") to your retrieval function, or use MLflow autologging for a supported framework such as LangChain or LlamaIndex, which records retriever spans for you.',
  },
  {
    question: "Can I use RAGAS or DeepEval metrics with MLflow?",
    answer: (
      <>
        Yes. MLflow provides{" "}
        <Link
          href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/third-party/"}
        >
          third-party scorer integrations
        </Link>{" "}
        for frameworks including RAGAS and DeepEval, so you can run their
        metrics through the same <code>mlflow.genai.evaluate()</code> call and
        compare them with MLflow's built-in judges.
      </>
    ),
    answerText:
      "Yes. MLflow provides third-party scorer integrations for frameworks including RAGAS and DeepEval, so you can run their metrics through the same mlflow.genai.evaluate() call and compare them with MLflow's built-in judges.",
  },
  {
    question: "Which LLM judge model should I use for RAG evaluation?",
    answer:
      'Pass the judge model to each scorer with the model argument, for example model="openai:/gpt-4o-mini". The judge can come from a different provider than your application, and you can calibrate it against human feedback. Use a stronger model for judges that must reason over long retrieved context, and check a sample of judge verdicts by hand before trusting aggregate scores.',
  },
  {
    question: "Is MLflow RAG evaluation free?",
    answer:
      "Yes. MLflow is open source under the Apache 2.0 license and backed by the Linux Foundation. The RAG judges, evaluation datasets, tracing, and UI are all included at no cost. You pay only for your own infrastructure and for the LLM calls the judge model makes.",
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
      text: faq.answerText || faq.answer,
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
    "Open-source platform for RAG evaluation, LLM evaluation, tracing, and experiment tracking.",
  url: "https://mlflow.org",
  license: "https://www.apache.org/licenses/LICENSE-2.0",
};

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

function Table({ head, rows }: { head: string[]; rows: React.ReactNode[][] }) {
  const cell = {
    padding: "16px",
    color: "#3d3d3d",
    verticalAlign: "top" as const,
  };
  return (
    <div style={{ overflowX: "auto", margin: "32px 0" }}>
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
              <th
                key={h}
                style={{
                  ...cell,
                  textAlign: "left",
                  fontWeight: "600",
                  color: "#1a1a1a",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #e2e8f0" }}>
              {row.map((c, j) => (
                <td key={j} style={cell}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RagEvaluation() {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  return (
    <>
      <Head>
        <title>{SEO_TITLE}</title>
        <meta name="description" content={SEO_DESCRIPTION} />
        <meta property="og:title" content={SEO_TITLE} />
        <meta property="og:description" content={SEO_DESCRIPTION} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://mlflow.org/rag-evaluation" />
        <link rel="canonical" href="https://mlflow.org/rag-evaluation" />
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
          @media (max-width: 1600px) {
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
          <h1>RAG Evaluation</h1>

          <p>
            <strong>RAG evaluation</strong> measures how well a
            retrieval-augmented generation (RAG) application finds the right
            documents and how faithfully it uses them to answer. A complete RAG
            evaluation scores three things separately: whether the retrieved
            chunks are <strong>relevant</strong>, whether they are{" "}
            <strong>sufficient</strong> to answer the question, and whether the
            final response is <strong>grounded</strong> in them.
          </p>

          <p>
            MLflow provides a built-in LLM judge for each of these checks:{" "}
            <Link href={DOCS_RAG + "relevance/"}>RetrievalRelevance</Link>,{" "}
            <Link href={DOCS_RAG + "context-sufficiency/"}>
              RetrievalSufficiency
            </Link>
            , and{" "}
            <Link href={DOCS_RAG + "groundedness/"}>RetrievalGroundedness</Link>
            . You run them with <code>mlflow.genai.evaluate()</code>, and{" "}
            <Link href="/llm-tracing">MLflow Tracing</Link> records every
            retrieval so you can see exactly which chunks produced each score.
          </p>

          <p style={{ fontSize: "14px", color: "#6b7280" }}>
            Last updated: November 26, 2026
          </p>

          <figure style={{ margin: "40px 0" }}>
            <img
              src={RagEvalRuns}
              alt="MLflow Evaluation runs page showing a trend chart of correctness, groundedness, relevance, and safety scores across three evaluation runs, with a table of runs and their mean scores"
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
              The MLflow Evaluation runs page compares judge scores across runs
              (shown with MLflow's demo data).
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
                <a href="#why-rag-evaluation">
                  Why RAG evaluation is different
                </a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#rag-metrics">RAG evaluation metrics</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#how-to-evaluate-rag">How to evaluate a RAG app</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#evaluation-dataset">Build an evaluation dataset</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#debug-rag">Debug failures with traces</a>
              </li>
              <li style={{ marginBottom: "8px" }}>
                <a href="#custom-metrics">Custom and third-party metrics</a>
              </li>
              <li style={{ marginBottom: "0" }}>
                <a href="#faq">FAQ</a>
              </li>
            </ul>
          </div>

          <h2
            id="why-rag-evaluation"
            data-toc="Why RAG evaluation is different"
          >
            Why Is RAG Evaluation Different from LLM Evaluation?
          </h2>

          <p>
            A RAG application has two stages that can fail independently. The
            retriever can return the wrong documents, and the language model can
            ignore or contradict the documents it was given. A single "is the
            answer good?" score hides which stage broke, so you end up tuning
            prompts when the real problem is chunking, or swapping retrievers
            when the real problem is hallucination.
          </p>

          <p>
            Evaluating each stage separately turns a vague quality problem into
            a specific one. The three most common RAG failure modes each have a
            matching check:
          </p>

          <Table
            head={["Failure mode", "What you see", "Check with"]}
            rows={[
              [
                <strong key="a">Poor retrieval</strong>,
                "Retrieved documents are off-topic for the question",
                "RetrievalRelevance",
              ],
              [
                <strong key="b">Insufficient context</strong>,
                "Documents are on-topic but miss facts needed to answer",
                "RetrievalSufficiency",
              ],
              [
                <strong key="c">Hallucination</strong>,
                "The answer states things the retrieved documents do not support",
                "RetrievalGroundedness",
              ],
            ]}
          />

          <h2 id="rag-metrics" data-toc="RAG evaluation metrics">
            What Metrics Does MLflow Provide for RAG Evaluation?
          </h2>

          <p>
            MLflow's <Link href={DOCS_RAG}>built-in RAG judges</Link> cover
            retrieval quality and grounding. Pair them with general
            response-quality judges to cover the answer itself.
          </p>

          <Table
            head={[
              "Judge",
              "Question it answers",
              "Needs ground truth?",
              "Needs a trace?",
            ]}
            rows={[
              [
                <code key="1">RetrievalRelevance</code>,
                "Is each retrieved chunk relevant to the user's request?",
                "No",
                "Yes",
              ],
              [
                <code key="2">RetrievalSufficiency</code>,
                "Do the retrieved chunks contain everything needed to answer?",
                "Yes (expected facts or response)",
                "Yes",
              ],
              [
                <code key="3">RetrievalGroundedness</code>,
                "Is the response supported by the retrieved chunks?",
                "No",
                "Yes",
              ],
              [
                <code key="4">Correctness</code>,
                "Does the response contain the expected facts?",
                "Yes (expected facts or response)",
                "No",
              ],
              [
                <code key="5">RelevanceToQuery</code>,
                "Does the response address the user's question?",
                "No",
                "No",
              ],
            ]}
          />

          <p>
            RetrievalRelevance scores each retrieved chunk individually, so you
            can see which documents in a result set are noise. All three
            retrieval judges read the retrieved documents from the trace, which
            is why your retriever needs a span of type <code>RETRIEVER</code>.
          </p>

          <h2 id="how-to-evaluate-rag" data-toc="How to evaluate a RAG app">
            How Do You Evaluate a RAG Application with MLflow?
          </h2>

          <p>
            The workflow has three steps: instrument the retriever, define test
            questions with expected facts, and run the judges.
          </p>

          <h3>Step 1: Instrument your retriever</h3>

          <p>
            Decorate the function that fetches documents with{" "}
            <code>@mlflow.trace(span_type="RETRIEVER")</code> and return MLflow{" "}
            <code>Document</code> objects. If you use a framework such as
            LangChain or LlamaIndex, MLflow's{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "tracing/"}>autologging</Link>{" "}
            can record retriever spans for you.
          </p>

          <CodeBlock
            code={`import mlflow
import openai
from mlflow.entities import Document

mlflow.openai.autolog()  # traces the LLM call too
client = openai.OpenAI()


@mlflow.trace(span_type="RETRIEVER")
def retrieve_docs(query: str) -> list[Document]:
    # Replace with a query to your vector database
    return [
        Document(
            id="doc_1",
            page_content="MLflow is an open source AI engineering platform ...",
            metadata={"source": "mlflow_overview.txt"},
        )
    ]


@mlflow.trace
def rag_app(query: str):
    docs = retrieve_docs(query)
    context = "\\n\\n".join(d.page_content for d in docs)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"Answer using only this context:\\n\\n{context}",
            },
            {"role": "user", "content": query},
        ],
    )
    return {"response": response.choices[0].message.content}`}
          />

          <h3>Step 2: Run the RAG judges</h3>

          <p>
            Pass your application as <code>predict_fn</code> and the RAG judges
            as <code>scorers</code>. Each row of the dataset is run through your
            application, traced, and scored.
          </p>

          <CodeBlock
            code={`import mlflow
from mlflow.genai.scorers import (
    RetrievalRelevance,
    RetrievalSufficiency,
    RetrievalGroundedness,
    Correctness,
)

results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=rag_app,
    scorers=[
        RetrievalRelevance(model="openai:/gpt-4o-mini"),
        RetrievalSufficiency(model="openai:/gpt-4o-mini"),
        RetrievalGroundedness(model="openai:/gpt-4o-mini"),
        Correctness(model="openai:/gpt-4o-mini"),
    ],
)`}
          />

          <p>
            Open the run in the MLflow UI to compare scores, filter to
            low-scoring rows, and click through to the trace behind each one.
            Run the same dataset after every change to your chunking, embedding
            model, retriever settings, or prompt, and compare runs side by side.
          </p>

          <h2 id="evaluation-dataset" data-toc="Build an evaluation dataset">
            How Do You Build a RAG Evaluation Dataset?
          </h2>

          <p>
            A good RAG test set mixes the questions users actually ask with the
            hard cases you worry about: multi-document questions, questions the
            knowledge base cannot answer, and near-duplicate topics. Each record
            has <code>inputs</code> (the arguments to your{" "}
            <code>predict_fn</code>) and optional <code>expectations</code>{" "}
            holding the facts a correct answer must contain.
          </p>

          <CodeBlock
            code={`from mlflow.genai.datasets import create_dataset

dataset = create_dataset(name="rag_eval_v1")

dataset.merge_records(
    [
        {
            "inputs": {"query": "What does MLflow offer for agents and LLMs?"},
            "expectations": {
                "expected_facts": [
                    "MLflow is an open source AI engineering platform",
                    "Features include observability and evaluation",
                ]
            },
        },
    ]
)`}
          />

          <p>
            You can also seed a dataset from production traces with{" "}
            <code>mlflow.search_traces()</code>, which keeps your test set close
            to real traffic. See the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "datasets/"}>
              evaluation datasets documentation
            </Link>{" "}
            for versioning and record management.
          </p>

          <h2 id="debug-rag" data-toc="Debug failures with traces">
            How Do You Debug RAG Failures with Traces?
          </h2>

          <p>
            A low score tells you something is wrong; the trace tells you why.
            Because MLflow evaluates your application through tracing, every
            scored row links to a trace showing the query, the retrieved chunks,
            the prompt sent to the model, and the final answer. That makes the
            usual diagnoses quick:
          </p>

          <ul>
            <li>
              <strong>Low RetrievalRelevance:</strong> inspect the retrieved
              chunks. Check embedding model choice, chunk size, top-k, and
              metadata filters.
            </li>
            <li>
              <strong>Low RetrievalSufficiency with good relevance:</strong> the
              right documents are retrieved but the needed fact is in a chunk
              you did not return. Increase top-k, adjust chunk overlap, or add
              reranking.
            </li>
            <li>
              <strong>Low RetrievalGroundedness:</strong> the model is adding
              unsupported claims. Tighten the system prompt, lower temperature,
              or require citations.
            </li>
            <li>
              <strong>Low Correctness with high groundedness:</strong> the
              answer faithfully reflects wrong or outdated documents. Fix the
              knowledge base, not the prompt.
            </li>
          </ul>

          <p>
            Learn more about capturing these spans in{" "}
            <Link href="/llm-tracing">LLM tracing with MLflow</Link>.
          </p>

          <h2 id="custom-metrics" data-toc="Custom and third-party metrics">
            Can You Add Custom or Third-Party RAG Metrics?
          </h2>

          <p>
            Built-in judges are a starting point. For domain rules, such as "the
            answer must cite a source" or "never recommend a competitor", create
            a{" "}
            <Link
              href={
                MLFLOW_GENAI_DOCS_URL +
                "eval-monitor/scorers/llm-judge/custom-judges/"
              }
            >
              custom LLM judge
            </Link>{" "}
            with <code>make_judge()</code>, or write a code-based scorer with
            the{" "}
            <Link href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/custom/"}>
              <code>@scorer</code> decorator
            </Link>
            . To use metrics from other libraries, MLflow offers{" "}
            <Link
              href={MLFLOW_GENAI_DOCS_URL + "eval-monitor/scorers/third-party/"}
            >
              integrations for RAGAS, DeepEval, and others
            </Link>{" "}
            that run through the same <code>mlflow.genai.evaluate()</code> call.
          </p>

          <p>
            LLM judges are themselves models, so validate them. Spot-check
            verdicts against your own judgment and use{" "}
            <Link href="/llm-as-a-judge">human feedback to align a judge</Link>{" "}
            before relying on its scores in a release gate.
          </p>

          <div className="info-box">
            <p>
              <strong>
                MLflow is the open source AI engineering platform for agents,
                LLMs, and ML models.
              </strong>{" "}
              It is licensed under Apache 2.0 and backed by the Linux
              Foundation, so RAG evaluation, tracing, and datasets run on your
              own infrastructure with no per-evaluation fees.{" "}
              <Link href={DOCS_RAG}>Read the RAG evaluation guide →</Link>
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
              <Link href={DOCS_RAG}>RAG Evaluation Documentation</Link>
            </li>
            <li>
              <Link href="/llm-evaluation">
                LLM Evaluation and Agent Evaluation
              </Link>
            </li>
            <li>
              <Link href="/llm-as-a-judge">LLM as a Judge</Link>
            </li>
            <li>
              <Link href="/llm-tracing">LLM Tracing</Link>
            </li>
            <li>
              <Link href={MLFLOW_GENAI_DOCS_URL + "datasets/"}>
                Evaluation Datasets Documentation
              </Link>
            </li>
            <li>
              <Link href="/genai">MLflow for Agents and LLMs Overview</Link>
            </li>
          </ul>
        </div>

        <ArticleSidebar />
        <SocialLinksFooter />
      </div>
    </>
  );
}
