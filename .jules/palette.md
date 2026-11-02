
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-26 - Contextual Instruction Pattern
**Learning:** Users tend to ignore "walls of text" at the top of Gradio applications. Moving global instructions into localized component `info` properties places guidance exactly where users make decisions, significantly improving task completion and reducing confusion.
**Action:** Always distribute global instructions into local component `info` properties rather than piling them in a single block at the top of the interface.
## 2026-11-02 - Expand Checkbox Touch Area
**Learning:** In complex rows, users often click the row text expecting it to toggle the checkbox. Wrapping the text in a <label> expands the hit area. However, wrapping block elements like <div> or <p> inside a <label> creates semantically invalid HTML.
**Action:** When applying Fitts's Law to expand checkbox touch areas, replace block elements with <span> elements and use utility classes like block or flex to visually retain the layout without breaking HTML semantics.
