import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

import {
  Button,
  SectionLabel,
  CopyCommand,
  LogosCarousel,
} from "../components";

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-[#0E1416] gap-8">
      <h1 className="text-4xl font-bold">New homepage</h1>
      <div className="flex flex-row gap-12 w-full justify-center">
        <div className="flex flex-col gap-4 w-[300px]">
          <p>Primary</p>
          <Button variant="primary" size="small">
            Small Primary Button
          </Button>
          <Button variant="primary" size="medium">
            Medium Primary Button
          </Button>
          <Button variant="primary" size="large">
            Large Primary Button
          </Button>
          <Button variant="primary" size="large" width="full">
            Full Width Primary Button
          </Button>
        </div>
        <div className="flex flex-col gap-4 w-[300px]">
          <p>Secondary</p>
          <Button variant="secondary" size="small">
            Small Secondary Button
          </Button>
          <Button variant="secondary" size="medium">
            Medium Secondary Button
          </Button>
          <Button variant="secondary" size="large">
            Large Secondary Button
          </Button>
          <Button variant="secondary" size="large" width="full">
            Full Width Secondary Button
          </Button>
        </div>
        <div className="flex flex-col gap-4 w-[300px]">
          <p>Outline</p>
          <Button variant="outline" size="small">
            Small Outline Button
          </Button>
          <Button variant="outline" size="medium">
            Medium Outline Button
          </Button>
          <Button variant="outline" size="large">
            Large Outline Button
          </Button>
          <Button variant="outline" size="large" width="full">
            Full Width Outline Button
          </Button>
        </div>
        <div className="flex flex-col gap-5 min-w-[300px]">
          <SectionLabel label="Core Features" color="red" />
          <SectionLabel label="Core Features" color="green" />
        </div>
      </div>
      <CopyCommand code="pip install mlflow" />
      <LogosCarousel>
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/transistor-logo-white.svg"
          alt="Transistor"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/reform-logo-white.svg"
          alt="Reform"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/tuple-logo-white.svg"
          alt="Tuple"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/savvycal-logo-white.svg"
          alt="SavvyCal"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/statamic-logo-white.svg"
          alt="Statamic"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/reform-logo-white.svg"
          alt="Reform"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/tuple-logo-white.svg"
          alt="Tuple"
        />
        <img
          className="mx-4 inline h-16"
          src="https://tailwindcss.com/plus-assets/img/logos/158x48/savvycal-logo-white.svg"
          alt="SavvyCal"
        />
      </LogosCarousel>
    </div>
  );
}
