# GUI Screenshots

This directory contains screenshots for the GUI documentation.

## Required Screenshots

Capture these screenshots from the running GUI at http://127.0.0.1:8765:

| Filename | Page | What to Capture |
|----------|------|-----------------|
| `run-list-page.png` | RunListPage (`/`) | Multiple run cards with varied statuses (pending, running, completed, failed) |
| `new-run-page.png` | NewRunPage (`/runs/new`) | Form with template dropdown expanded |
| `run-detail-logs.png` | RunDetailPage | Logs tab showing log content |
| `run-detail-artifacts.png` | RunDetailPage | Artifacts tab showing file browser |
| `topology-page.png` | TopologyPage (`/topology`) | Network graph with a node selected |
| `config-editor.png` | ConfigEditorPage (`/config`) | Editor showing INI content |
| `codebase-architecture.png` | CodebaseExplorerPage (`/codebase`) | Architecture view with module cards |
| `codebase-code.png` | CodebaseExplorerPage | Code view with file tree and code panel |
| `settings-page.png` | SettingsPage (`/settings`) | Theme selection options |

## Guidelines

- Use a consistent browser window size (1280x800 recommended)
- Capture only the content area, not browser chrome
- Use light theme for consistent documentation appearance
- Ensure no sensitive or personal data is visible
- Save as PNG format

## After Capturing

Once screenshots are added, update the `[Screenshot: ...]` placeholders in
`docs/getting-started/gui/features.rst` with actual image directives:

```rst
.. image:: /_static/images/gui/run-list-page.png
   :alt: Run list page showing simulation runs
   :width: 100%
```
