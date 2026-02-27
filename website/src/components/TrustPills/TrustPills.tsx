import { Star, Download } from "lucide-react";
import LinuxFoundationLogo from "@site/static/img/linux-foundation.svg";
import { useGitHubStars } from "../../hooks/useGitHubStars";

const GitHubIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 48 48" fill="none" className={className}>
    <path
      d="M24 3.9C35.05 3.9 44 12.85 44 23.9c0 4.19-1.315 8.275-3.759 11.68-2.444 3.404-5.894 5.955-9.864 7.296-.996.2-1.371-.425-1.371-.95 0-.675.025-2.825.025-5.5 0-1.875-.625-3.075-1.35-3.7 4.45-.5 9.125-2.2 9.125-9.875 0-2.2-.775-3.975-2.05-5.375.2-.5.9-2.55-.2-5.3 0 0-1.675-.55-5.5 2.05-1.6-.45-3.3-.675-5-.675s-3.7.225-5.3.675c-3.825-2.575-5.5-2.05-5.5-2.05-1.1 2.75-.4 4.8-.2 5.3-1.275 1.4-2.05 3.2-2.05 5.375 0 7.65 4.65 9.4 9.1 9.9-.575.5-1.1 1.375-1.275 2.675-1.15.525-4.025 1.375-5.825-1.65-.375-.6-1.5-2.075-3.075-2.05-1.675.025-.675.95.025 1.325.85.475 1.825 2.25 2.05 2.825.4 1.125 1.7 3.275 6.725 2.35 0 1.675.025 3.25.025 3.725 0 .525-.375 1.125-1.375.95-3.983-1.326-7.447-3.872-9.902-7.278C5.318 32.192 3.998 28.1 4 23.901 4 12.851 12.95 3.9 24 3.9Z"
      fill="currentColor"
    />
  </svg>
);

export function TrustPills() {
  const stars = useGitHubStars();
  return (
    <div className="flex flex-wrap justify-center items-center gap-4 mt-2">
      <div className="flex items-center gap-2 rounded-full bg-white/5 border border-white/10 px-5 py-2.5 text-sm text-white/70">
        <LinuxFoundationLogo className="h-6 w-auto text-white/70" />
      </div>
      <a
        href="https://github.com/mlflow/mlflow"
        target="_blank"
        rel="noreferrer noopener"
        className="flex items-center gap-2 rounded-full bg-white/5 border border-white/10 px-5 py-2.5 text-sm text-white/70 hover:bg-white/10 hover:border-white/20 transition-all"
      >
        <GitHubIcon className="w-5 h-5" />
        <Star className="w-4 h-4" fill="currentColor" />
        {stars && <span>{stars}+ Stars</span>}
      </a>
      <div className="flex items-center gap-2 rounded-full bg-white/5 border border-white/10 px-5 py-2.5 text-sm text-white/70">
        <Download className="w-4 h-4" />
        <span>30M+ Downloads/mo</span>
      </div>
    </div>
  );
}
