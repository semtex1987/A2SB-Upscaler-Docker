
## 2024-05-15 - Gradio UI Info Texts
**Learning:** Using massive wall-of-text paragraphs at the top of a Gradio interface reduces discoverability and clutters the UI. Users frequently miss these instructions.
**Action:** Use the `info` parameter for supported components (like Sliders, Textboxes, and Dropdowns) to provide contextual help text right where it is needed. For components that lack the `info` parameter (like `gr.File`), use localized `gr.Markdown` blocks placed immediately before or after the component to provide relevant instructions.
