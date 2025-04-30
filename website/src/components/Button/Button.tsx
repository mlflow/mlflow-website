import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "../../utils"

const buttonVariants = cva(
  "rounded-xl inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 hover:cursor-pointer",
  {
    variants: {
      variant: {
        primary:
          "bg-white text-black hover:bg-white/90",
        outline:
          "box-border border border-white bg-transparent text-white hover:text-white/90",
        secondary:
          "bg-black text-white hover:bg-black/80",
      },
      size: {
        small: "text-[15px] px-4 py-3",
        medium: "text-base px-4 py-4",
        large: "text-base px-5 py-4",
      },
      width: {
        default: "w-fit",
        full: "w-full",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "medium",
      width: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, width, asChild = false, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, width, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
