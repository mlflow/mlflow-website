import Link from "@docusaurus/Link";
import { Button } from "../Button/Button";

interface Props {
  variant?: "blue" | "primary" | "dark";
  width?: "default" | "full";
  size?: "small" | "medium" | "large";
  link?: string;
}

export const GetStartedButton = ({
  link = "https://login.databricks.com/?intent=SIGN_UP",
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
