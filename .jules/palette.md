
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2024-10-24 - Contextual UI Guidance
**Learning:** Users often ignore large blocks of 'wall of text' instructions at the top of applications. Moving global instructions into localized component `info` properties provides contextual help text exactly where decisions are made.
**Action:** Break down large instruction blocks and use native component `info` props or stylized `gr.Markdown` helpers to place guidance close to the relevant UI element.
