## 2026-07-04 - [UX/Gradio pattern: Localized Info Text]
**Learning:** Moving large, global instruction blocks into localized `info` properties on their respective inputs reduces cognitive overload and creates a cleaner UI.
**Action:** When a Gradio component doesn't support the `info` property (like `gr.File`), simulate it using `elem_classes=["text-sm", "text-gray-500", "mb-1"]`.
