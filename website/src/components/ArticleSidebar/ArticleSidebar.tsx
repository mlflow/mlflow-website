import { useEffect, useState } from "react";
import { MLFLOW_GENAI_DOCS_URL } from "@site/src/constants";

type TocItem = {
  id: string;
  text: string;
};

export function ArticleSidebar() {
  const [tocItems, setTocItems] = useState<TocItem[]>([]);

  useEffect(() => {
    const container = document.querySelector(".article-container");
    if (!container) return;
    const h2s = container.querySelectorAll("h2[id]");
    const items: TocItem[] = [];
    h2s.forEach((h2) => {
      items.push({
        id: h2.id,
        text: h2.textContent || "",
      });
    });
    setTocItems(items);
  }, []);

  if (tocItems.length === 0) return null;

  return (
    <aside className="article-sidebar">
      <p className="toc-title">On this page</p>
      <ul>
        {tocItems.map((item) => (
          <li key={item.id}>
            <a href={`#${item.id}`}>{item.text}</a>
          </li>
        ))}
      </ul>
      <hr className="toc-divider" />
      <p className="toc-title">Resources</p>
      <ul>
        <li>
          <a href={MLFLOW_GENAI_DOCS_URL}>Documentation</a>
        </li>
        <li>
          <a href="https://go.mlflow.org/slack">Slack</a>
        </li>
        <li>
          <a href="https://github.com/mlflow/mlflow">GitHub</a>
        </li>
      </ul>
    </aside>
  );
}
