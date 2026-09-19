
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-26 - Contextual Instruction Pattern
**Learning:** Users tend to ignore "walls of text" at the top of Gradio applications. Moving global instructions into localized component `info` properties places guidance exactly where users make decisions, significantly improving task completion and reducing confusion.
**Action:** Always distribute global instructions into local component `info` properties rather than piling them in a single block at the top of the interface.
## 2026-09-19 - Expand checkbox hit areas
**Learning:** In list views where users select items via checkboxes, the default 16x16px click target violates Fitts's Law, making it hard to select items quickly. Wrapping the adjacent row content in a <label> (while converting inner blocks to inline elements) expands the hit area significantly.
**Action:** Always wrap adjacent row content in a `<label htmlFor={id}>` with `cursor-pointer` when paired with a checkbox, ensuring valid HTML semantics.
