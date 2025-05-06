import { Button } from "../Button/Button";

export const GetStartedTagline = () => {
  return (
    <div className="flex flex-col md:flex-row justify-center items-center gap-6 md:gap-16">
      <span className="text-lg text-white/60 text-center">
        Join the industry leading companies building with MLflow
      </span>
      <Button variant="primary">Get Started &gt;</Button>
    </div>
  );
};
