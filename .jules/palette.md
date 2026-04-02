
## 2025-02-28 - Component Instruction Text
**Learning:** Wall-of-text explanations at the top of a Gradio interface increase cognitive load. Placing instructional text directly on components using the `info` parameter improves discoverability and contextual understanding without cluttering the main layout. For components that do not support the `info` parameter (like `File`), placing a small, contextual `Markdown` block directly above the component is the preferred fallback.
**Action:** Always prefer `info` parameters on Gradio inputs over global Markdown instructions for component-specific usage guidelines. If `info` is unsupported, place localized Markdown text immediately before the component instantiation.
