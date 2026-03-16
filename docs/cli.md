# CLI Reference

voicetag includes a command-line interface built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/).

## Commands

### `voicetag enroll`

Enroll a speaker from one or more audio files.

```bash
voicetag enroll "Alice" alice1.wav alice2.wav
voicetag enroll "Bob" bob_sample.wav --profiles my_profiles.json
```

| Option | Default | Description |
|---|---|---|
| `--profiles PATH` | `voicetag_profiles.json` | Profile storage file |

### `voicetag identify`

Identify speakers in an audio file.

```bash
voicetag identify meeting.wav
voicetag identify meeting.wav --threshold 0.8 --output results.json
voicetag identify meeting.wav --unknown-only  # diarize without matching
```

| Option | Default | Description |
|---|---|---|
| `--profiles PATH` | `voicetag_profiles.json` | Profile storage file |
| `--output, -o PATH` | None | Save results as JSON |
| `--threshold FLOAT` | 0.75 | Similarity threshold (0.0-1.0) |
| `--hf-token TEXT` | `$HF_TOKEN` | HuggingFace API token |
| `--device TEXT` | `cpu` | Torch device (`cpu`, `cuda`, `mps`) |
| `--unknown-only` | False | Skip speaker matching |

### `voicetag profiles list`

List all enrolled speakers.

```bash
voicetag profiles list
voicetag profiles list --profiles custom_profiles.json
```

### `voicetag profiles remove`

Remove a speaker from the profiles.

```bash
voicetag profiles remove "Alice"
```

### `voicetag version`

Print the voicetag version.

```bash
voicetag version
```
