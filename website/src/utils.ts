import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { MLFLOW_DOCS_URL, MLFLOW_GENAI_DOCS_URL } from "./constants";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Helper function to determine the appropriate "Get started" link based on current page
export function getStartedLinkForPage(
  pathname: string,
  classicalMLPath: string,
  genAIPath: string,
): string {
  if (pathname.startsWith(classicalMLPath)) {
    return "/classical-ml#get-started";
  }
  if (pathname.startsWith(genAIPath)) {
    // Map each GenAI page to its corresponding documentation link
    // to match the hero section's "Get Started" button
    const genAIRoutes: Record<string, string> = {
      [`${genAIPath}`]: MLFLOW_GENAI_DOCS_URL,
      [`${genAIPath}/`]: MLFLOW_GENAI_DOCS_URL,
      [`${genAIPath}/observability`]: `${MLFLOW_GENAI_DOCS_URL}tracing/quickstart/`,
      [`${genAIPath}/evaluations`]: `${MLFLOW_GENAI_DOCS_URL}eval-monitor/quickstart/`,
      [`${genAIPath}/prompt-registry`]: `${MLFLOW_GENAI_DOCS_URL}prompt-registry/create-and-edit-prompts/`,
      [`${genAIPath}/app-versioning`]: `${MLFLOW_GENAI_DOCS_URL}version-tracking/quickstart/`,
      [`${genAIPath}/ai-gateway`]: `${MLFLOW_GENAI_DOCS_URL}governance/ai-gateway/setup/`,
      [`${genAIPath}/governance`]: MLFLOW_DOCS_URL,
      [`${genAIPath}/human-feedback`]: MLFLOW_DOCS_URL,
    };
    
    return genAIRoutes[pathname] || MLFLOW_GENAI_DOCS_URL;
  }
  return "/#get-started";
}

// Helper function to determine if current page is classical ML
export function isClassicalMLPage(
  pathname: string,
  classicalMLPath: string,
): boolean {
  return pathname.startsWith(classicalMLPath);
}

// Helper function to determine if current page is GenAI
export function isGenAIPage(pathname: string, genAIPath: string): boolean {
  return pathname.startsWith(genAIPath);
}
