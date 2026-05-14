
## 2024-05-24 - Contextual Helpers vs Global Instructions
**Learning:** Users tend to ignore large, global instruction blocks at the top of an app. Gradio's `info` attribute provides a much better UX by offering contextual helper text directly on the relevant components, keeping the interface clean and focused. For components like `gr.File` that lack an `info` attribute, localized `gr.Markdown` with utility classes matching native labels (e.g. `text-sm text-gray-500 mb-1`) provides a consistent alternative.
**Action:** When designing or reviewing Gradio UIs, always break down global instruction blocks and distribute the information into localized `info` properties or adjacent Markdown text.
