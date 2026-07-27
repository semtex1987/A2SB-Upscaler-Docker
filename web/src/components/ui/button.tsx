import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  // min-h-11 keeps every button at the 44px touch target, including on the
  // transport bar where controls sit close together.
  "inline-flex min-h-11 cursor-pointer items-center justify-center gap-2 rounded-lg " +
    "border font-medium transition-colors duration-150 select-none " +
    "disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-45",
  {
    variants: {
      variant: {
        primary:
          "border-accent bg-accent text-canvas hover:bg-accent-hover hover:border-accent-hover",
        secondary:
          "border-stroke bg-surface text-ink hover:bg-surface-raised hover:border-stroke-strong",
        ghost: "border-transparent bg-transparent text-ink-muted hover:bg-surface hover:text-ink",
        danger: "border-fault/50 bg-fault/10 text-fault hover:bg-fault/20 hover:border-fault",
      },
      size: {
        md: "px-4 text-sm",
        lg: "px-6 text-base",
        icon: "w-11 px-0",
      },
    },
    defaultVariants: { variant: "secondary", size: "md" },
  },
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  children?: ReactNode;
}

export function Button({ className, variant, size, type = "button", ...props }: ButtonProps) {
  return (
    <button type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  );
}
