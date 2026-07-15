
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-15 - Contextual Help Text Over Wall of Text
**Learning:** Users often ignore large blocks of instruction text ('wall of text') at the top of applications, missing important guidance for specific parameters.
**Action:** Move global instructional text into localized component `info` properties to provide contextual help exactly where the user makes decisions.
