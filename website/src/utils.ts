import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

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
      [`${genAIPath}`]: "https://mlflow.org/docs/latest/genai/",
      [`${genAIPath}/`]: "https://mlflow.org/docs/latest/genai/",
      [`${genAIPath}/observability`]: "https://mlflow.org/docs/latest/genai/tracing/quickstart/python-openai/",
      [`${genAIPath}/evaluations`]: "https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/",
      [`${genAIPath}/prompt-registry`]: "https://mlflow.org/docs/latest/genai/prompt-registry/create-and-edit-prompts/",
      [`${genAIPath}/app-versioning`]: "https://mlflow.org/docs/latest/genai/version-tracking/quickstart/",
      [`${genAIPath}/ai-gateway`]: "https://mlflow.org/docs/latest/genai/governance/ai-gateway/setup/",
      [`${genAIPath}/governance`]: "https://mlflow.org/docs/latest/",
      [`${genAIPath}/human-feedback`]: "https://mlflow.org/docs/latest/",
    };
    
    return genAIRoutes[pathname] || "https://mlflow.org/docs/latest/genai/";
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
