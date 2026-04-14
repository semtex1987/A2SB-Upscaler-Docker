## 2024-05-24 - Contextual Component Guidance in Gradio
**Learning:** Wall-of-text instructions at the top of a UI are often ignored. When using Gradio, users benefit significantly more from localized context.
**Action:** Use the `info` parameter for supported components (like `Textbox`, `Slider`, `Dropdown`) instead of generic top-level `gr.Markdown`. For components like `gr.File` that lack `info`, place localized `gr.Markdown` directly adjacent to the component.
