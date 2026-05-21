
## 2024-05-21 - Contextual Help UX Enhancement
**Learning:** Moving large, global instructional text blocks into contextual `info` attributes and localized `gr.Markdown` elements with utility classes (e.g. `elem_classes=["text-sm", "text-gray-500", "mb-1"]`) significantly improves interface clarity and reduces cognitive load by placing instructions exactly where users need them.
**Action:** Always prefer localized tooltips and `info` text over global instruction blocks for complex forms, and utilize Gradio`s `elem_classes` to mimic native component visual hierarchy when native `info` parameters are unavailable.
