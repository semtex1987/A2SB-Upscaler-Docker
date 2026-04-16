## 2024-05-24 - Contextualized UI Instructions
**Learning:** Gradio interfaces become cluttered with wall-of-text global instructions. Users miss critical hints about parameters (like batch size and steps) when they are disconnected from the actual controls.
**Action:** Use the `info` parameter on components (like Textbox and Slider) to localize instructions. For components like `gr.File` that lack `info`, place a small, localized `gr.Markdown` block directly above them.
