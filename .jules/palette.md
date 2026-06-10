## 2024-05-24 - [UX/a11y Pattern] Gradio Info Text vs Markdown
**Learning:** Using `info="..."` on Gradio components is better for contextual helper text than large `gr.Markdown` blocks at the top of the app. It improves focus and interface clarity.
**Action:** Prefer `info` attribute on `gr.Slider`, `gr.Dropdown`, etc., for specific instructions.
