## 2024-05-24 - Contextual Helper Text over Global Instructions
**Learning:** Global markdown blocks push key interactive elements below the fold and are often ignored. Using native `info` attributes on Gradio inputs places guidance in context, significantly improving discoverability and maintaining a cleaner UI.
**Action:** Prefer component-level `info` properties or adjacent localized `gr.Markdown` styling (e.g., `elem_classes=["text-sm", "text-gray-500", "mb-1"]`) over top-level instructional blocks.
