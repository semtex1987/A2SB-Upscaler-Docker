## 2024-04-20 - Localized Explanations in Gradio
**Learning:** Avoid wall-of-text explanations at the top of forms. Using `info` parameter for components like `gr.Textbox` and `gr.Slider` provides better context-sensitive help and improves discoverability. For components lacking `info` support like `gr.File`, using a localized `gr.Markdown` block directly adjacent is better than cluttering the label or putting it at the top of the page.
**Action:** Always prefer component-level `info` parameters or adjacent `gr.Markdown` instead of global instruction blocks when building Gradio interfaces.
