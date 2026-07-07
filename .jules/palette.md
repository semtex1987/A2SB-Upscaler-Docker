
## 2026-07-07 - Localized vs Global Helper Text Pattern
**Learning:** Large global instruction blocks at the top of forms create cognitive overload. Moving help text to the point of interaction (e.g., using Gradio's `info` attribute or localized markdown with `text-sm text-gray-500 mb-1`) significantly improves interface focus and form usability.
**Action:** Always prefer localized component-level `info` text or styled sub-labels over massive instruction blocks for user inputs.
