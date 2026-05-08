## 2024-05-01 - Gradio Contextual Helper Text
**Learning:** Avoid wall-of-text explanations. Use the `info` parameter for supported components (e.g., Textbox, Slider) to improve discoverability. For components like `gr.File` that lack `info` support, use localized `gr.Markdown` blocks directly adjacent to the component instead of cluttering the main layout or the component's `label`.
**Action:** Use inline helper text and specific localized markup to improve UX.
