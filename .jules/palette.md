## 2026-06-08 - Contextual Helper Text over Global Blocks
**Learning:** Moving broad markdown instructions into component-level `info` tooltips reduces cognitive load, clarifies context, and improves overall screen reader accessibility by directly associating hints with interactive components.
**Action:** Use component-level `info` attributes for configuration sliders/inputs and localized `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]` for general helpers instead of global introductory blocks.
