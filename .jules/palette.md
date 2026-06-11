## 2026-06-11 - Contextual Helper Text in Gradio
**Learning:** Using global `gr.Markdown` blocks for instructions at the top of a UI increases cognitive load, requiring users to look back and forth. Localizing instructions to the relevant components reduces this burden.
**Action:** When adding helper text in Gradio, use the `info` property on components where supported (like `gr.Textbox` or `gr.Slider`), or localized `gr.Markdown` elements directly adjacent to the target component (with appropriate styling) for unsupported components (like `gr.File`).
