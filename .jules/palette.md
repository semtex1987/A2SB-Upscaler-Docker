
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-26 - Contextual Instruction Pattern
**Learning:** Users tend to ignore "walls of text" at the top of Gradio applications. Moving global instructions into localized component `info` properties places guidance exactly where users make decisions, significantly improving task completion and reducing confusion.
**Action:** Always distribute global instructions into local component `info` properties rather than piling them in a single block at the top of the interface.
## 2026-09-17 - Expand staged file checkbox hit areas
**Learning:** In list views where users select items via checkboxes, the default 16x16px click target violates Fitts's Law. However, when wrapping adjacent row content in a `<label>`, it is crucial to ensure only phrasing content (e.g., spans) is wrapped. Wrapping block elements (`<div>`, `<p>`) creates semantically invalid HTML.
**Action:** Use `<label htmlFor={id}>` with `cursor-pointer` on the adjacent content to expand the hit area, but convert `<div>` and `<p>` tags to `<span>` with appropriate display properties (like `block` or `flex`) to maintain HTML validity.
