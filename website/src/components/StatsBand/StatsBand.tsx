import LinuxFoundationLogo from "@site/static/img/linux-foundation.svg";
import { Section } from "../Section/Section";

type Stat = {
  value: string;
  label: string;
  type?: "github";
  repo?: string;
};

export const StatsBand = () => {
  return (
    <Section
      title="The Most Used Open-Source MLOps Platform"
      body="Backed by Linux Foundation, MLflow is fully committed to open-source for 6 years."
    >
      <div className="flex w-full flex-col items-center gap-16">
        <a
          href="https://github.com/mlflow/mlflow"
          className="github-stats-card relative flex items-center gap-4 rounded-2xl bg-[linear-gradient(135deg,#1f2f63,#1b2342)] px-6 py-4 text-white shadow-xl overflow-hidden"
          target="_blank"
          rel="noreferrer noopener"
        >
          <div className="github-stats-icon">
            <div className="github-stats-icon-inner">
              <div className="github-stats-face github-stats-face-front">
                <span className="inline-flex h-11 w-11 items-center justify-center rounded-full bg-white/10">
                  <svg
                    width="24"
                    height="24"
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
              <div className="github-stats-face github-stats-face-back" aria-hidden>
                <span className="github-stats-back-icon">
                  <svg
                    width="22"
                    height="22"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M14 5h5m0 0v5m0-5L10 14"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M12 5H8a3 3 0 0 0-3 3v8a3 3 0 0 0 3 3h8a3 3 3 0 0 0 3-3v-4"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      opacity="0.78"
                    />
                  </svg>
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-base font-semibold">mlflow/mlflow</span>
            <div className="flex items-center gap-2 text-base font-semibold">
              23K
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
            </div>
          </div>
        </a>
        <div className="grid w-full max-w-4xl grid-cols-1 gap-6 text-center sm:grid-cols-3">
          <LinuxFoundationLogo className="h-20 w-auto text-white" />
            <div className="flex flex-col items-center gap-1.5">
              <span className="text-3xl font-semibold leading-tight text-white sm:text-4xl">
                25 Million+
              </span>
              <span className="text-xs text-white/70 sm:text-sm">Package Downloads / Month</span>
            </div>
          <div className="flex items-center gap-6 rounded-xl px-6 py-8 h-20 border border-gray-200 text-[22px] font-black font-weight-black uppercase text-white leading-tight">
            <span className="flex h-8 w-8 items-center justify-center text-white/80">
              <svg
                width="26"
                height="24"
                viewBox="0 0 16 16"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path d="M8.75.75V2h.985c.304 0 .603.08.867.231l1.29.736c.038.022.08.033.124.033h2.234a.75.75 0 0 1 0 1.5h-.427l2.111 4.692a.75.75 0 0 1-.154.838l-.53-.53.529.531-.001.002-.002.002-.006.006-.006.005-.01.01-.045.04c-.21.176-.441.327-.686.45C14.556 10.78 13.88 11 13 11a4.498 4.498 0 0 1-2.023-.454 3.544 3.544 0 0 1-.686-.45l-.045-.04-.016-.015-.006-.006-.004-.004v-.001a.75.75 0 0 1-.154-.838L12.178 4.5h-.162c-.305 0-.604-.079-.868-.231l-1.29-.736a.245.245 0 0 0-.124-.033H8.75V13h2.5a.75.75 0 0 1 0 1.5h-6.5a.75.75 0 0 1 0-1.5h2.5V3.5h-.984a.245.245 0 0 0-.124.033l-1.289.737c-.265.15-.564.23-.869.23h-.162l2.112 4.692a.75.75 0 0 1-.154.838l-.53-.53.529.531-.001.002-.002.002-.006.006-.016.015-.045.04c-.21.176-.441.327-.686.45C4.556 10.78 3.88 11 3 11a4.498 4.498 0 0 1-2.023-.454 3.544 3.544 0 0 1-.686-.45l-.045-.04-.016-.015-.006-.006-.004-.004v-.001a.75.75 0 0 1-.154-.838L2.178 4.5H1.75a.75.75 0 0 1 0-1.5h2.234a.249.249 0 0 0 .125-.033l1.288-.737c.265-.15.564-.23.869-.23h.984V.75a.75.75 0 0 1 1.5 0Zm2.945 8.477c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L13 6.327Zm-10 0c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L3 6.327Z" />
              </svg>
            </span>
            <div className="flex flex-col items-start gap-1">
              <span>Apache-2.0</span>
              <span>License</span>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
};
