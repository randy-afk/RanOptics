# Export

RanOptics supports several output formats.

---

## HTML (default)

Every run produces an interactive HTML file. This is the primary output.

- Set the output path in the **Output HTML** field
- Open it in any browser — no server required
- Pan, zoom, and hover are all interactive
- Share it with colleagues — it's a single self-contained file

---

## CSV

Click **Export CSV** after a successful run to export the data for each panel as a CSV file. One file per panel, named after the panel title.

---

## PNG / PDF

Static image export requires `kaleido`:

```bash
pip install kaleido
```

Configure in the **Export** tab:

| Setting | Description |
|---------|-------------|
| **Format** | PNG or PDF |
| **DPI** | Resolution (default 96) |
| **Width / Height** | Figure dimensions in pixels |

---

## Presets

Save your current GUI configuration (panels, settings, file paths) as a named preset:

- **Presets → Save** — saves to `~/.ranoptics_presets.json`
- **Presets → Load** — restores a previously saved configuration

Presets are stored locally and persist between sessions.
