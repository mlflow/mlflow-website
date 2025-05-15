import Link from "@docusaurus/Link";
import { Button } from "../Button/Button";
import { MLFLOW_SIGNUP_URL } from "@site/src/constants";

interface Props {
  variant?: "blue" | "primary" | "dark";
  width?: "default" | "full";
  size?: "small" | "medium" | "large";
  link?: string;
}

export const GetStartedButton = ({
  link = MLFLOW_SIGNUP_URL,
  size = "medium",
  width = "default",
  variant = "primary",
}: Props) => {
  return (
    <Link to={link}>
      <Button size={size} width={width} variant={variant}>
        Get started
      </Button>
    </Link>
  );
};
