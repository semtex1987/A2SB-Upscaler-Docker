
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-24 - Contextualizing Global Instructions
**Learning:** To prevent users from ignoring large blocks of instructions ('wall of text') at the top of Gradio applications, move global instructions into localized component `info` properties to provide contextual help text exactly where decisions are made.
**Action:** Use the `info` parameter on components like `gr.Slider` and `gr.Textbox` and localized `gr.Markdown` components to attach context directly to the controls.
