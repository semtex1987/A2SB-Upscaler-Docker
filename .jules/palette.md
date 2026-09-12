
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-26 - Contextual Instruction Pattern
**Learning:** Users tend to ignore "walls of text" at the top of Gradio applications. Moving global instructions into localized component `info` properties places guidance exactly where users make decisions, significantly improving task completion and reducing confusion.
**Action:** Always distribute global instructions into local component `info` properties rather than piling them in a single block at the top of the interface.
## $(date +%Y-%m-%d) - Expand clickable area for staged files
**Learning:** By default, native checkboxes only have a 16x16px click area, which violates Fitts's Law and frustrates users.
**Action:** Wrapped the entire list item content (while correctly converting block elements to spans) inside a <label> with cursor-pointer to dramatically increase the hit area without adding JS.
