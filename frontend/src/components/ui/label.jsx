import * as React from "react"
import * as LabelPrimitive from "@radix-ui/react-label" // Usually dependent on this, but lets check if I installed it.
// I did NOT install @radix-ui/react-label. I will implement a plain version to avoid install overhead if possible, 
// OR I should just auto-run the install.
// I will create a simple functional version.

import { cn } from "@/lib/utils"

const Label = React.forwardRef(({ className, ...props }, ref) => (
    <label
        ref={ref}
        className={cn(
            "text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70",
            className
        )}
        {...props}
    />
))
Label.displayName = "Label"

export { Label }
