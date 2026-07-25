
## 2026-07-12 - Native Info Text Styling for Unsupported Gradio Components
**Learning:** Gradio components like `gr.File` lack an `info` prop. Using a separate `gr.Markdown` with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`, we can exactly replicate the visual hierarchy of native component helper text.
**Action:** Always pair unsupported components with this stylized `gr.Markdown` for consistent accessibility and guidance.

## 2026-07-25 - Contextual Help over Wall of Text
**Learning:** Users often ignore large blocks of instructions at the top of Gradio applications. Moving these into localized `info` properties on relevant components increases the likelihood they are read and acted upon correctly.
**Action:** Always distribute instructional text to the point of decision (e.g., using `info` for sliders/inputs or stylized `gr.Markdown` for others) rather than stacking it at the top of the interface.
