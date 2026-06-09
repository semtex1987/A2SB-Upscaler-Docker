
## 2026-06-09 - Component-Level Contextual Help
**Learning:** Global instructional text blocks create cognitive overload. Moving help text to the `info` attribute of individual Gradio components improves user focus and clarifies which inputs the instructions apply to.
**Action:** Use component `info` fields instead of top-level `gr.Markdown` blocks where possible.
