## 2024-05-24 - Contextual Instructions in Gradio
**Learning:** Large, global `gr.Markdown` instruction blocks at the top of a Gradio interface can overwhelm users and separate instructions from the controls they refer to, violating the principle of proximity.
**Action:** Use the `info` parameter on components (like `gr.Slider`, `gr.Textbox`, `gr.Number`) and localized `gr.Markdown(..., elem_classes=["text-sm", "text-gray-500", "mb-1"])` above components that lack an `info` property (like `gr.File`) to keep helper text contextual, improving focus and interface clarity.
