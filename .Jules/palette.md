## 2024-05-17 - A2SB Gradio App Evaluation
**Learning:** Found a gradio web application that handles processing audio files, and uses Gradio for UI rendering. Many elements are accessible by default, but there's a lack of helpful tooltips for users trying to understand specific properties like batch size and cutoff frequencies in this niche domain.
**Action:** Enhance UX by adding tooltips to the input options to describe their effects on processing. Gradio has an `info` parameter for UI components that is specifically meant for this type of helper text and is natively supported.
