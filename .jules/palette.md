## 2026-04-04 - Gradio Component Contextual Info
**Learning:** Wall-of-text explanations at the top of a UI cause cognitive overload and are often ignored. Distributing these instructions closer to the relevant inputs is more effective.
**Action:** Use Gradio's built-in `info` parameter for supported components (like `Textbox`, `Slider`) to improve discoverability. For components like `gr.File` that lack `info` support, use a localized `gr.Markdown` directly adjacent to the component.
