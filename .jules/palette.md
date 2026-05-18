
## 2024-05-18 - Component-Level Context Improves App Usability
**Learning:** Large, global instruction blocks at the top of a Gradio application are often ignored or cause cognitive overload. Using the component-specific `info` attribute, or placing styled local `gr.Markdown` right before components that do not support `info` (like `gr.File`), contextualizes help text directly where users interact, improving interface clarity.
**Action:** When designing or refactoring Gradio apps, prefer putting descriptive helper text directly onto the relevant inputs instead of accumulating all instructions at the top of the interface. Use `elem_classes` (e.g. `text-sm text-gray-500 mb-1`) to style plain Markdown elements so they match the visual hierarchy of native component labels.
