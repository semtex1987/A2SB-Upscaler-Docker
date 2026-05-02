## 2024-05-02 - Gradio Component Contextual Info
**Learning:** Avoid wall-of-text explanations at the top of the app. Gradio natively supports contextual `info` properties for many input components (like Textbox, Slider). For components without this property like `gr.File`, use a localized `gr.Markdown` block directly above or adjacent to the component.
**Action:** Replace general top-level instructions with component-specific `info` attributes and localized `gr.Markdown` blocks to improve interface context and discoverability without cluttering the main layout.
