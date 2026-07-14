
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2024-07-14 - Localized Contextual Instructions
**Learning:** Users often ignore large blocks of instruction text ('walls of text') at the top of applications. Moving global instructions into localized component `info` properties provides contextual help text exactly where decisions are made.
**Action:** Prioritize adding descriptive `info` text to individual form components rather than listing all instructions globally at the top of the page.
