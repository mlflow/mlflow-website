"use client";

import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";

import { cn } from "../../utils";
import "./VerticalTabs.module.css";
const VerticalTabs = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Root>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Root
    ref={ref}
    orientation="vertical"
    className={cn("flex w-full", className)}
    {...props}
  />
));
VerticalTabs.displayName = TabsPrimitive.Root.displayName;

const VerticalTabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn("flex flex-col w-[440px]", className)}
    {...props}
  />
));
VerticalTabsList.displayName = TabsPrimitive.List.displayName;

type TabsPrimitiveTriggerProps = React.ComponentPropsWithoutRef<
  typeof TabsPrimitive.Trigger
>;
type VerticalTabsTriggerProps = Omit<TabsPrimitiveTriggerProps, "children"> & {
  label: string;
  description: string;
};

const VerticalTabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  VerticalTabsTriggerProps
>(({ className, label, description, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "vertical-tabs-trigger w-full p-5 inline-flex whitespace-nowrap rounded-lg data-[state=active]:bg-[#fff]/4 data-[state=active]:text-foreground data-[state=active]:border data-[state=active]:border-[#fff]/8 cursor-pointer",
      className,
    )}
    {...props}
  >
    <div className="flex flex-col gap-3 w-max">
      <span className="text-lg font-medium text-white text-left text-wrap">
        {label}
      </span>

      <span className="vertical-tabs-trigger-description text-md text-white/50 text-left text-wrap h-0 overflow-hidden">
        {description}
      </span>
    </div>
  </TabsPrimitive.Trigger>
));
VerticalTabsTrigger.displayName = TabsPrimitive.Trigger.displayName;

const VerticalTabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      "ring-offset-background ml-10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 w-full h-full",
      className,
    )}
    {...props}
  />
));
VerticalTabsContent.displayName = TabsPrimitive.Content.displayName;

export {
  VerticalTabs,
  VerticalTabsList,
  VerticalTabsTrigger,
  VerticalTabsContent,
};
