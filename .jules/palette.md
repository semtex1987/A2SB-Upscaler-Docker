## 2024-07-03 - Contextual help text over global instruction blocks
**Learning:** Large walls of text at the top of an app are often ignored and increase cognitive load.
**Action:** Move instructions directly into the `info` attribute of specific inputs, and use `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]` to mimic native info text for components like `gr.File` that lack an `info` prop. This reduces visual clutter and keeps guidance exactly where it's needed.
