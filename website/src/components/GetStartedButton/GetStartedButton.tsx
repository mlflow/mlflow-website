import { Button } from "../Button/Button";

interface Props {
  variant?: "blue" | "primary" | "dark";
  width?: "default" | "full";
  size?: "small" | "medium" | "large";
}

export const GetStartedButton = ({
  size = "medium",
  width = "default",
  variant = "primary",
}: Props) => {
  return (
    <Button size={size} width={width} variant={variant}>
      Get started
    </Button>
  );
};
