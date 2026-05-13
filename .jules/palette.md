## $(date +%Y-%m-%d) - Contextual UI Helpers

**Learning:** Global markdown instruction blocks in Gradio interfaces often force users to read a "wall of text" before interacting with the UI. Moving these instructions to the `info` property of individual components reduces cognitive load by displaying the documentation exactly where the user is making the decision.
**Action:** Always prefer contextual `info` arguments on interactive Gradio components (like `gr.Slider`, `gr.Textbox`, `gr.Dropdown`) over a centralized `gr.Markdown` description at the top of the interface.
