# API Reference

## `VoiceTag`

The main entry point for the voicetag library. Wraps the full diarization, embedding, and matching pipeline.

```python
from voicetag import VoiceTag, VoiceTagConfig

vt = VoiceTag(config=VoiceTagConfig(...))
```

### Methods

#### `enroll(name, audio_paths)`

Register a speaker from one or more audio files. Computes a mean embedding from all provided samples.

- **name** (`str`) -- Speaker name.
- **audio_paths** (`list[str]`) -- Paths to audio files of the speaker.
- **Returns:** `SpeakerProfile`
- **Raises:** `EnrollmentError` if no valid audio files are provided.

#### `identify(audio_path)`

Run the full identification pipeline on an audio file.

- **audio_path** (`str | Path`) -- Path to the audio file.
- **Returns:** `DiarizationResult`
- **Raises:** `AudioLoadError`, `DiarizationError`

#### `save(path)` / `load(path)`

Persist or restore enrolled speaker profiles to/from disk.

- **path** (`str | Path`) -- File path for the profiles.

#### `remove_speaker(name)`

Remove an enrolled speaker by name.

- **name** (`str`) -- Speaker name to remove.

#### `enrolled_speakers` (property)

Returns a list of currently enrolled speaker names.

---

## Configuration

### `VoiceTagConfig`

Pydantic v2 model controlling all pipeline parameters. See [Configuration](configuration.md) for details.

---

## Result Models

### `DiarizationResult`

Returned by `VoiceTag.identify()`.

| Field | Type | Description |
|---|---|---|
| `segments` | `list[SpeakerSegment \| OverlapSegment]` | Ordered speaker timeline |
| `audio_duration` | `float` | Total audio length in seconds |
| `num_speakers` | `int` | Distinct speakers detected |
| `processing_time` | `float` | Pipeline wall-clock time in seconds |

### `SpeakerSegment`

A single identified speaker turn.

| Field | Type | Description |
|---|---|---|
| `speaker` | `str` | Speaker name or `"UNKNOWN"` |
| `start` | `float` | Start time (seconds) |
| `end` | `float` | End time (seconds) |
| `confidence` | `float` | Cosine similarity (0.0-1.0) |
| `duration` | `float` | Property: `end - start` |

### `OverlapSegment`

A region where multiple speakers talk simultaneously.

| Field | Type | Description |
|---|---|---|
| `speakers` | `list[str]` | Overlapping speaker names |
| `start` | `float` | Start time (seconds) |
| `end` | `float` | End time (seconds) |
| `speaker` | `Literal["OVERLAP"]` | Always `"OVERLAP"` |
| `duration` | `float` | Property: `end - start` |

### `SpeakerProfile`

An enrolled speaker's stored embedding.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Speaker name |
| `embedding` | `list[float]` | 256-dim mean embedding vector |
| `num_samples` | `int` | Audio files used for enrollment |
| `created_at` | `datetime` | UTC enrollment timestamp |

---

## Exceptions

All exceptions inherit from `VoiceTagError`.

| Exception | When |
|---|---|
| `VoiceTagConfigError` | Invalid config or missing HuggingFace token |
| `EnrollmentError` | Enrollment failure |
| `DiarizationError` | Pyannote processing failure |
| `AudioLoadError` | File not found or unsupported format |

```python
from voicetag import VoiceTagError

try:
    result = vt.identify("audio.wav")
except VoiceTagError as e:
    print(e)
```
