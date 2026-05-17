## 2025-02-28 - Global Instructions in Gradio
**Learning:** Large `gr.Markdown` blocks at the top of a Gradio interface can overwhelm users and separate instructions from the controls they describe.
**Action:** Prefer moving instructional text to the `info` parameter of specific inputs (like `gr.Slider` or `gr.Textbox`) or using styled text near the relevant component (`gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`) to improve interface intuitiveness and reduce cognitive load.
