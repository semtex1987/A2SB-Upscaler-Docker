## 2024-05-24 - Contextual Help in Gradio
**Learning:** Large "wall-of-text" instructions at the top of a Gradio interface hurt discoverability and readability. Users often miss or forget parameter-specific advice by the time they reach the input controls.
**Action:** Move parameter-specific advice into the `info` attribute of supported components (e.g., Textbox, Slider). For components that lack `info` support (like `gr.File`), use localized, small `gr.Markdown` blocks directly adjacent to the component.
