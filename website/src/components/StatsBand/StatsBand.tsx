import LinuxFoundationLogo from "@site/static/img/linux-foundation.svg";
import { Section } from "../Section/Section";
import { LogosCarousel } from "../LogosCarousel/LogosCarousel";

export const StatsBand = () => {
  return (
    <Section
      title="Most Trusted Open-Source MLOps Platform"
      body="Backed by Linux Foundation, MLflow has been fully committed to open-source for 5+ years. Now trusted by thousands of organizations and research teams worldwide."
      align="center"
    >
      <div className="flex w-full flex-col items-center gap-16 relative">
        <div className="grid w-full max-w-4xl grid-cols-1 gap-6 text-center sm:grid-cols-3 relative z-10">
          <div>
            <LinuxFoundationLogo className="h-20 w-auto text-white" />
          </div>

          <a
            href="https://github.com/mlflow/mlflow"
            className="github-stats-card flex items-center gap-3 rounded-2xl bg-white/5 px-5 py-3 text-white border border-white/10 hover:border-white/20 hover:bg-white/10 transition-all"
            target="_blank"
            rel="noreferrer noopener"
          >
            <div className="github-stats-icon">
              <div className="github-stats-icon-inner">
                <div className="github-stats-face github-stats-face-front">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-white/10">
                    <svg
                      width="22"
                      height="22"
                      viewBox="0 0 48 48"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                      className="text-white"
                    >
                      <path
                        d="M24 3.9C35.05 3.9 44 12.85 44 23.9c0 4.19-1.315 8.275-3.759 11.68-2.444 3.404-5.894 5.955-9.864 7.296-.996.2-1.371-.425-1.371-.95 0-.675.025-2.825.025-5.5 0-1.875-.625-3.075-1.35-3.7 4.45-.5 9.125-2.2 9.125-9.875 0-2.2-.775-3.975-2.05-5.375.2-.5.9-2.55-.2-5.3 0 0-1.675-.55-5.5 2.05-1.6-.45-3.3-.675-5-.675s-3.7.225-5.3.675c-3.825-2.575-5.5-2.05-5.5-2.05-1.1 2.75-.4 4.8-.2 5.3-1.275 1.4-2.05 3.2-2.05 5.375 0 7.65 4.65 9.4 9.1 9.9-.575.5-1.1 1.375-1.275 2.675-1.15.525-4.025 1.375-5.825-1.65-.375-.6-1.5-2.075-3.075-2.05-1.675.025-.675.95.025 1.325.85.475 1.825 2.25 2.05 2.825.4 1.125 1.7 3.275 6.725 2.35 0 1.675.025 3.25.025 3.725 0 .525-.375 1.125-1.375.95-3.983-1.326-7.447-3.872-9.902-7.278C5.318 32.192 3.998 28.1 4 23.901 4 12.851 12.95 3.9 24 3.9Z"
                        fill="currentColor"
                      />
                    </svg>
                  </span>
                </div>
                <div
                  className="github-stats-face github-stats-face-back"
                  aria-hidden
                >
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-white/20">
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                      xmlns="http://www.w3.org/2000/svg"
                      className="text-white"
                    >
                      <path d="M12 2l2.89 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l7.11-1.01L12 2z" />
                    </svg>
                  </span>
                </div>
              </div>
            </div>
            <span className="text-sm font-semibold">mlflow/mlflow</span>
            <div className="flex items-center gap-1.5 text-sm font-semibold text-white/80">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M12 2l2.89 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l7.11-1.01L12 2z" />
              </svg>
              23K
            </div>
          </a>

          <div className="flex flex-col items-center gap-1.5">
            <span className="text-3xl font-semibold leading-tight text-white sm:text-4xl">
              25 Million+
            </span>
            <span className="text-xs text-white/70 sm:text-sm">
              Package Downloads / Month
            </span>
          </div>
        </div>
      </div>
      <LogosCarousel />
    </Section>
  );
};
