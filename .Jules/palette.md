## 2025-03-08 - Use `info` property instead of global Markdown
**Learning:** Replacing large global instructions with contextual `info` props on Gradio components reduces cognitive load and improves layout focus.
**Action:** When adding instructions to UI forms in Gradio, default to the `info` parameter for localized helper text. For components lacking `info` support (like `gr.File`), emulate it using `gr.Markdown` with specific text styling classes.
