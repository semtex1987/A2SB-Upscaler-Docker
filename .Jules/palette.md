## 2026-06-14 - Localized UI Helper Context in Gradio
**Learning:** Global markdown instruction blocks cause cognitive load and push actual inputs down the page. Native Gradio 'info' properties and localized 'elem_classes' provide better contextual UX and visual hierarchy.
**Action:** Prefer applying 'info' text directly to Gradio inputs; fallback to localized gr.Markdown with muted text classes for components like gr.File that lack 'info' support.
