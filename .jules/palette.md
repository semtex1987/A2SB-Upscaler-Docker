## 2026-07-05 - Localize Contextual Helper Text in Gradio
**Learning:** Moving large, global instruction blocks at the top of a Gradio app into the `info` attribute of individual components greatly improves interface focus and clarity by placing context exactly where it's needed. For components without `info` support (like `gr.File`), `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, seamlessly matches native visual hierarchy.
**Action:** Prefer component-level `info` attributes for helper text instead of monolithic instruction headers to enhance UI scanability.
