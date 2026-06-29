## 2026-06-29 - Contextual Help vs Global Instructions
**Learning:** Global Markdown blocks at the top of complex Gradio apps increase cognitive load and visual clutter.
**Action:** Always prefer using the `info` attribute on specific input components (or a localized `gr.Markdown` with small/gray text classes for components like `gr.File` that lack `info`) to provide contextual help exactly where the user interacts with the UI.
