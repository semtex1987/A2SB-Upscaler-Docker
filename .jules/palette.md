
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-20 - Contextual UI Instructions
**Learning:** Users tend to ignore large blocks of instruction text ('wall of text') at the top of an application. Moving these global instructions into localized component `info` properties provides contextual help text exactly where decisions are made.
**Action:** Break down global instruction blocks and distribute them as localized `info` tooltips or styled helper components directly next to the relevant inputs.
