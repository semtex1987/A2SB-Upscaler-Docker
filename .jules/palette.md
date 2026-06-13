
## 2026-06-13 - Contextual Helper Text over Global Instructions
**Learning:** In Gradio interfaces, global instruction blocks at the top of the app often clutter the UI and reduce user focus. Moving these instructions into localized `info` attributes on relevant components provides better context exactly when the user needs it.
**Action:** Prefer using the `info` attribute on components or localized `gr.Markdown` with small utility classes instead of large, global instruction blocks.
