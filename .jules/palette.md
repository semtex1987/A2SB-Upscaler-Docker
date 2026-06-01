## 2024-06-01 - Migrate Global Markdown to Contextual Info
**Learning:** In Gradio applications, large blocks of `gr.Markdown` instruction text placed at the top of the app create visual clutter and force users to process all information at once, regardless of relevance.
**Action:** Always prefer attaching contextual helper text directly to the relevant UI components using the `info` parameter (for supported components like inputs) or by placing a localized, subtly styled `gr.Markdown` block directly above the component (for components like `gr.File` that don't support `info`).
