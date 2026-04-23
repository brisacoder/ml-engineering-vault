---
type: moc
title: Vault Setup
created: 2026-04-22
---

# Vault Setup

Open this vault in Obsidian and follow these steps once.

## 1. Install Community Plugins

Go to **Settings → Community plugins → Browse** and install:

1. **Templater** by SilentVoid — for smart templates with prompts
2. **Dataview** by Michael Brenan — for live queries in [[Home]]

Enable both after installing.

## 2. Configure Templater

Go to **Settings → Templater**:

- **Template folder location**: `_templates`
- **Trigger Templater on new file creation**: ON
- **Enable folder templates**: ON (optional — map folders to templates below)

Optional folder-to-template mappings:

| Folder           | Template              |
|------------------|-----------------------|
| snippets         | `_templates/snippet`  |
| cheatsheets      | `_templates/cheatsheet` |
| concepts         | `_templates/concept`  |
| recipes          | `_templates/recipe`   |
| troubleshooting  | `_templates/troubleshooting` |

With folder templates enabled, creating a new note inside `snippets/` will auto-apply the snippet template.

## 3. Configure Dataview

Go to **Settings → Dataview**:

- **Enable JavaScript queries**: OFF (not needed)
- **Enable inline queries**: ON (useful for inline metadata)

## 4. Set Home as Startup Note

Optional: install **Homepage** plugin and set `Home` as the startup note.

## 5. Verify

Open [[Home]] — if Dataview is working, you'll see live tables (empty for now). Create a test note in `snippets/` — if Templater is working, you'll get the library/tag suggester popup.

## Folder Structure

```
ml-engineering-vault/
├── _templates/        → Templater templates (hidden from normal browsing)
├── _assets/           → Images, diagrams, plots
├── cheatsheets/       → Quick-reference lookup notes
├── concepts/          → ML/stats theory explanations
├── snippets/          → Reusable code patterns
├── recipes/           → End-to-end workflows
├── troubleshooting/   → Error → cause → fix
├── reference/         → Papers, links, API docs
├── Home.md            → Main dashboard (Dataview)
└── Vault Setup.md     → This file (delete when done)
```

## Tagging Convention

Tags use namespace prefixes so they're filterable:

- `task/*` — what you're doing: `task/classification`, `task/nlp`, `task/data-wrangling`
- `concept/*` — what you're learning: `concept/stats`, `concept/optimization`, `concept/attention`
- `error/*` — what broke: `error/shape`, `error/cuda`, `error/memory`

Libraries are tracked in the `libs` frontmatter field, not as tags — this keeps Dataview queries clean.

## Note Philosophy

Notes are **topic-centric**, not atomic. Each file covers a topic (e.g., "Pandas Plotting", "PyTorch Data Loading") and accumulates `##` sections over time. Link to individual sections via `[[Note#Section]]`.

Example: `[[Pandas Plotting]]` links to the whole topic, `[[Pandas Plotting#Hue-Style Scatter via c and cmap]]` links to a specific pattern.

## Linking Strategy

Every note has two linking mechanisms:

1. **`related` frontmatter** — structured links that Dataview can query (e.g., find all notes related to a pandas plotting note)
2. **Inline `[[wikilinks]]`** — natural links in the body text wherever you reference another topic or pattern

The **Orphan Notes** query on [[Home]] catches notes with no `related` links, so nothing stays isolated. The **Cross-Reference** table shows the link graph at a glance.
