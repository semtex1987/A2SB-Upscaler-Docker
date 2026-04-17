## 2026-04-17 - Contextual Info in Gradio Interfaces
**Learning:** Wall-of-text instructions at the top of a Gradio app cause cognitive overload and are often ignored.
**Action:** Use the `info` property on compatible components (like Textbox, Slider, Dropdown) to provide localized, context-sensitive instructions. For components that do not support the `info` property (like gr.File), place a succinct `gr.Markdown` block immediately adjacent to the component rather than clumping it with other instructions at the top of the interface.
