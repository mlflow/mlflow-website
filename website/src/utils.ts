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
    // For the main GenAI landing page, link to documentation
    if (pathname === genAIPath || pathname === `${genAIPath}/`) {
      return "/docs/latest/genai/";
    }
    // For GenAI subpages, link to the Get Started section
    return "/genai#get-started";
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
