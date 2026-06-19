## 2026-06-19 - Contextual UI Help via Info Attributes
**Learning:** Global markdown instruction blocks cause visual clutter and split user attention. Gradio's `info` attribute on input components provides contextual, localized help text which improves focus and UI clarity. For components like `gr.File` that lack `info` support, `gr.Markdown` with utility classes like `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, matches the visual hierarchy.
**Action:** Prefer localized `info` attributes over global instruction blocks for Gradio applications.
