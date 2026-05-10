## $(date +%Y-%m-%d) - Contextual Help Text via Gestalt Proximity
**Learning:** Dense paragraphs of instructional text at the top of a UI cause unnecessary cognitive load. In Gradio, `gr.File` lacks an `info` property, which makes it tricky to add contextual help.
**Action:** Move global instructional text directly into the `info` property of related inputs (e.g., Sliders, Textboxes). For components like `gr.File` that lack `info`, emulate the visual hierarchy by placing a `gr.Markdown` block directly above it using `elem_classes=["text-sm", "text-gray-500", "mb-1"]`.
