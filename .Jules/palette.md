## 2024-05-18 - Gradio Info Text Discoverability
**Learning:** UX Learning (Gradio): Avoid wall-of-text explanations. Use the `info` parameter for supported components (e.g., Textbox, Slider) to improve discoverability. For components like `gr.File` that lack `info` support, use localized `gr.Markdown` blocks directly adjacent to the component instead of cluttering the main layout or the component's `label`.
**Action:** Move global instructions into component-specific `info` attributes where possible, and use adjacent Markdown blocks for unsupported components.
