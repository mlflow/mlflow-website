import { useState, useEffect } from "react";

const CACHE_KEY = "mlflow_github_stars";
const CACHE_TTL = 1000 * 60 * 60; // 1 hour

function formatStars(count: number): string {
  if (count >= 1000) {
    return `${Math.floor(count / 1000)}K`;
  }
  return String(count);
}

export function useGitHubStars(): string | null {
  const [stars, setStars] = useState<string | null>(null);

  useEffect(() => {
    const cached = localStorage.getItem(CACHE_KEY);
    if (cached) {
      const { value, timestamp } = JSON.parse(cached);
      if (Date.now() - timestamp < CACHE_TTL) {
        setStars(value);
        return;
      }
    }

    fetch("https://api.github.com/repos/mlflow/mlflow")
      .then((res) => res.json())
      .then((data) => {
        if (data.stargazers_count) {
          const formatted = formatStars(data.stargazers_count);
          setStars(formatted);
          localStorage.setItem(
            CACHE_KEY,
            JSON.stringify({ value: formatted, timestamp: Date.now() }),
          );
        }
      })
      .catch(() => {
        // Fallback silently - stars will remain null
      });
  }, []);

  return stars;
}
