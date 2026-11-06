
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-26 - Contextual Instruction Pattern
**Learning:** Users tend to ignore "walls of text" at the top of Gradio applications. Moving global instructions into localized component `info` properties places guidance exactly where users make decisions, significantly improving task completion and reducing confusion.
**Action:** Always distribute global instructions into local component `info` properties rather than piling them in a single block at the top of the interface.

## 2026-11-06 - Add size="icon" and aria-label to icon-only buttons
**Learning:** In this design system, icon-only `<Button>` components without the `size="icon"` prop receive the default `md` size padding, making them too wide and violating Fitts's Law. Furthermore, missing `aria-label` on icon buttons severely impacts screen reader accessibility.
**Action:** When adding or auditing icon-only buttons using the custom `<Button>` component, always verify the presence of `size="icon"` to ensure correct touch target dimensions without extra width, and explicitly provide an `aria-label`.
