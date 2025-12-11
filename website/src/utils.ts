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
      [`${genAIPath}`]: "/docs/latest/genai/",
      [`${genAIPath}/`]: "/docs/latest/genai/",
      [`${genAIPath}/observability`]: "/docs/latest/genai/tracing/quickstart/",
      [`${genAIPath}/evaluations`]: "/docs/latest/genai/eval-monitor/quickstart/",
      [`${genAIPath}/prompt-registry`]: "/docs/latest/genai/prompt-registry/create-and-edit-prompts/",
      [`${genAIPath}/app-versioning`]: "/docs/latest/genai/version-tracking/quickstart/",
      [`${genAIPath}/ai-gateway`]: "/docs/latest/genai/governance/ai-gateway/setup/",
      [`${genAIPath}/governance`]: "/docs/latest/",
      [`${genAIPath}/human-feedback`]: "/docs/latest/",
    };
    
    return genAIRoutes[pathname] || "/docs/latest/genai/";
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
