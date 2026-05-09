## 2026-05-09 - Refactored Instructions to Inline Help
**Learning:** Moving large block instructions into contextual inline help ('info' props) improves usability by keeping guidance close to the relevant controls. 'gr.File' lacks 'info' support, so a localized 'gr.Markdown' with subtle utility classes ('text-sm text-gray-500 mb-1') serves as a good workaround.
**Action:** Use 'info' on Gradio components for guidance, and localized 'gr.Markdown' for components that don't support 'info'.
