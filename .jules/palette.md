
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-26 - Contextual Instruction Pattern
**Learning:** Users tend to ignore "walls of text" at the top of Gradio applications. Moving global instructions into localized component `info` properties places guidance exactly where users make decisions, significantly improving task completion and reducing confusion.
**Action:** Always distribute global instructions into local component `info` properties rather than piling them in a single block at the top of the interface.
## 2024-08-09 - Expand hit area on complex file rows
**Learning:** List items with complex layouts and a single small checkbox violate Fitts's Law. Wrapping phrasing content in `<label>` dramatically increases usability without JavaScript.
**Action:** Always wrap adjacent descriptive elements in a `<label htmlFor={id}>` alongside the checkbox input, making sure to avoid block-level elements inside the label to satisfy HTML semantics.
