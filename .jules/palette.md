
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-22 - Localized Instructions via Info Props
**Learning:** Users often ignore large 'wall of text' instructions at the top of the app. Moving this text into component-specific `info` props provides contextual help exactly where decisions are made.
**Action:** Use component `info` props or visually consistent helper `gr.Markdown` for all instructional text instead of global markdown blocks.
