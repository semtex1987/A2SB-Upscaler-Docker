
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-26 - Contextual Instruction Pattern
**Learning:** Users tend to ignore "walls of text" at the top of Gradio applications. Moving global instructions into localized component `info` properties places guidance exactly where users make decisions, significantly improving task completion and reducing confusion.
**Action:** Always distribute global instructions into local component `info` properties rather than piling them in a single block at the top of the interface.
## 2026-11-05 - Added aria-labels and size="icon" to refresh buttons

**Learning:** The TrainView component uses shadcn-like `Button` components for refresh icons (`RefreshCw`), but previously omitted `aria-label` attributes and the `size="icon"` property. This made the buttons inaccessible to screen readers and visually improperly proportioned.

**Action:** When adding icon-only buttons using this codebase's `<Button>` component (found at `web/src/components/ui/button.tsx`), always ensure `size="icon"` is set to ensure square dimensions (width and height match for a 44px touch target) and include a descriptive `aria-label` attribute for screen readers.
