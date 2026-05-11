## 2024-05-11 - Add Format Helper Text to Gradio File Upload
**Learning:** Gradio's `gr.File` component does not natively support an `info` parameter to describe allowed formats to users, unlike inputs like `gr.Textbox`. When working with `gr.File` with explicit format filters, the UI lacks discoverability.
**Action:** Always pair restricted `gr.File` components with an adjacent `gr.Markdown` helper block. Apply `elem_classes=["text-sm", "text-gray-500", "mb-1"]` to ensure it visually matches standard Gradio descriptive text.
