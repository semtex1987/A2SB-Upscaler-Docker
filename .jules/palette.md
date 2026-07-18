
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-18 - Contextualizing Instructions
**Learning:** Users often ignore large blocks of instructions ('wall of text') at the top of Gradio applications. Moving global instructions into localized component `info` properties provides contextual help text exactly where decisions are made.
**Action:** Break down instructional walls of text and distribute the guidance into the `info` props of the relevant input components, or as stylized `gr.Markdown` underneath them.
