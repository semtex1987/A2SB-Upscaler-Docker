## 2026-05-12 - Contextual UI Instructions
**Learning:** Global instructional text is often missed; moving it directly to component `info` attributes improves context without cluttering the top of the interface. When native `info` isn't supported (like in `gr.File`), using a small `gr.Markdown` with appropriate classes (`text-sm text-gray-500 mb-1`) effectively mimics the native visual hierarchy.
**Action:** Use contextual component instructions (`info=`) instead of large global text blocks whenever possible.
