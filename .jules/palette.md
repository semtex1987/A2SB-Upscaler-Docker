
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-26 - Contextual Instruction Pattern
**Learning:** Users tend to ignore "walls of text" at the top of Gradio applications. Moving global instructions into localized component `info` properties places guidance exactly where users make decisions, significantly improving task completion and reducing confusion.
**Action:** Always distribute global instructions into local component `info` properties rather than piling them in a single block at the top of the interface.

## 2026-11-07 - Native Button Component Sizing for Icon-Only Buttons
**Learning:** The custom `<Button>` component in this repository uses a default size prop (`md`) which adds horizontal padding (`px-4`). For icon-only buttons, this default padding causes the button to appear too wide and breaks layout consistency.
**Action:** Always explicitly include the `size="icon"` prop and an `aria-label` on the custom `<Button>` component when it only contains an icon to ensure proper sizing and screen reader accessibility.
