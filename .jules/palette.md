
## 2024-05-14 - [Gradio UI Contextual Info]
**Learning:** Avoid wall-of-text explanations at the top of the app. Utilizing Gradio's `info` parameter for supported components (like `Textbox` and `Slider`) and localized `gr.Markdown` for unsupported components (like `File`) improves discoverability and declutters the main layout.
**Action:** Use contextual documentation attributes like `info` instead of global Markdown headers for component-specific instructions.
