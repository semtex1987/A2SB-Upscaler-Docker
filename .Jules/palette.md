## 2024-05-24 - Gradio Discoverability
**Learning:** Wall-of-text explanations at the top of Gradio interfaces are often ignored. Using the `info` parameter for supported components (like Textbox and Slider) and localized `gr.Markdown` blocks for unsupported components (like `gr.File`) improves discoverability without cluttering the UI.
**Action:** Replace top-level markdown walls with component-specific `info` parameters or localized adjacent markdown blocks.
