
## 2024-07-11 - Contextual Helper Text in Gradio
**Learning:** Large global instruction blocks in Gradio interfaces often get ignored by users and clutter the UI. Moving instructions into the `info` attribute of specific components, or using customized `gr.Markdown` helpers for components lacking `info` support, significantly improves user focus and interface clarity.
**Action:** Use the `info` parameter for contextual help on inputs. For components like `gr.File` that do not support it, precede them with a `gr.Markdown` block styled with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`.
