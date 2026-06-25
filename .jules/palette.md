## 2026-06-25 - Contextual UI Help Text
**Learning:** Global markdown blocks at the top of a page can cause clutter and cognitive overload, and users may not read them all before interacting with inputs. It is better to use contextual tooltips or inline text directly tied to the relevant input.
**Action:** Use Gradio's `info` attribute on components where available, or a localized `gr.Markdown` with specific styling like `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, to provide targeted guidance exactly when and where the user needs it.
