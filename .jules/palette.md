## 2026-06-28 - Contextual UI Helpers
**Learning:** Global markdown instructions block user focus. Preferring contextual `info` attributes on inputs and localized `gr.Markdown` using utility classes improves readability and focuses the interface without custom CSS.
**Action:** Use Gradio's built-in `info` parameter where possible, and `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, to emulate native helpers on components lacking info attributes.
