
## 2026-07-10 - Contextual Help in Gradio
**Learning:** Global instructional blocks in Gradio apps create cognitive overload. Moving instructions into localized `info` attributes on components or styled small `gr.Markdown` blocks (for components without `info`) improves interface clarity and contextual focus without sacrificing functionality.
**Action:** Always prefer using component `info` properties over global headers for explanatory text in Gradio.
