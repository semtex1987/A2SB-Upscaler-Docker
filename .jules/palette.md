## 2025-02-18 - Improve Gradio Interface Clarity via Localized Info
**Learning:** Global instruction blocks at the top of a Gradio app cause cognitive overload and interface clutter. Users often skip reading them before interacting with controls.
**Action:** Always prefer applying contextual helper text directly to related components using the `info` attribute. For components like `gr.File` that lack `info` support, mimic native helper text by placing a `gr.Markdown` component directly above it with `elem_classes=["text-sm", "text-gray-500", "mb-1"]`.
