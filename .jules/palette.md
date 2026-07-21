
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-21 - Moved Global Instructions to Contextual Info
**Learning:** Users often ignore 'walls of text' at the top of applications. Distributing instructions directly into component `info` properties provides contextual help right when decisions are made, improving readability without adding visual clutter.
**Action:** Always prefer localized component instructions (`info` or styled helper markdown) over large blocks of text at the beginning of an app.
