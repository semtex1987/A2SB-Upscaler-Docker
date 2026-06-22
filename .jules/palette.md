## 2026-06-22 - Component-Level Contextual Help
**Learning:** Global instruction blocks at the top of a Gradio app cause cognitive overload. Users lose context by the time they reach the actual inputs.
**Action:** Replaced large global gr.Markdown headers with localized component-level `info` attributes. For components that lack `info` support (like gr.File or gr.Markdown), use localized gr.Markdown blocks with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`. This improves visual hierarchy and reduces context switching.
