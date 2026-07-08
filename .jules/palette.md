## 2026-07-08 - Move global instructions to contextual tooltips
**Learning:** Large global instruction blocks (walls of text) at the top of the UI cause cognitive overload and are often ignored by users.
**Action:** Prefer using the `info` attribute on Gradio components for contextual helper text, or a separate `gr.Markdown` with utility classes (like `text-sm text-gray-500`) for components that don't support `info`.
