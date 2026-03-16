# voicetag Architecture

## 1. System Overview

voicetag is a Python library that combines [pyannote.audio](https://github.com/pyannote/pyannote-audio) (speaker diarization) and [resemblyzer](https://github.com/resemble-ai/Resemblyzer) (speaker embeddings) into a unified speaker identification pipeline. Given an audio file, voicetag answers two questions: *who is speaking?* and *when are they speaking?*

The library handles three core tasks:

- **Speaker enrollment** -- register known speakers by providing audio samples, which are converted to embedding vectors and stored for later matching.
- **Speaker identification** -- process an audio file through diarization, compute embeddings for each speech segment, and match them against enrolled speakers.
- **Overlap detection** -- identify regions where multiple speakers talk simultaneously.

voicetag exposes a single `VoiceTag` class for programmatic use and a Typer-based CLI for command-line workflows.

---

## 2. Module Boundaries

```
voicetag/
  __init__.py       # Public API exports
  pipeline.py       # Core orchestration
  encoder.py        # Resemblyzer wrapper & enrollment store
  diarizer.py       # Pyannote.audio wrapper
  overlap.py        # Overlap detection and merging
  models.py         # Pydantic v2 data models
  cli.py            # Typer CLI
  utils.py          # Audio I/O utilities
  exceptions.py     # Custom exception hierarchy
```

### `voicetag/__init__.py`

Public API surface. Re-exports:

- `VoiceTag` (from `pipeline`)
- All models from `models` (`SpeakerSegment`, `OverlapSegment`, `DiarizationResult`, `VoiceTagConfig`, `SpeakerProfile`)
- All exceptions from `exceptions`
- `__version__`

No logic lives here. This module exists solely to define the public contract.

### `voicetag/pipeline.py`

Core orchestration module. Contains the `VoiceTag` class, which owns instances of `SpeakerEncoder` and `Diarizer` and coordinates the full identification pipeline.

**Responsibilities:**

- Accept a `VoiceTagConfig` (or build a default one).
- Delegate diarization to `Diarizer`.
- Delegate embedding computation to `SpeakerEncoder`.
- Run embedding computation in parallel via `concurrent.futures.ThreadPoolExecutor`, bounded by `config.max_workers`.
- Match computed segment embeddings against enrolled speaker profiles using cosine similarity.
- Delegate overlap detection to the `overlap` module.
- Merge and sort results into a list of `SpeakerSegment` objects.
- Expose enrollment convenience methods that forward to `SpeakerEncoder`.
- Expose `save()` / `load()` for profile persistence.

**Key methods:**

| Method | Description |
|---|---|
| `__init__(config=None)` | Initialize pipeline components |
| `enroll(name, audio_paths)` | Register a speaker (forwards to encoder) |
| `identify(audio_path, **kwargs)` | Run full identification pipeline |
| `save(path)` / `load(path)` | Persist / restore enrolled profiles |
| `enrolled_speakers` (property) | List enrolled speaker names |
| `remove_speaker(name)` | Remove an enrolled speaker |

### `voicetag/encoder.py`

Resemblyzer wrapper. Manages the `VoiceEncoder` from resemblyzer and maintains the enrollment store (a `dict[str, np.ndarray]` mapping speaker names to their mean embedding vectors).

**Responsibilities:**

- Load the resemblyzer pretrained encoder on the configured device.
- Compute speaker embeddings from audio waveforms.
- Maintain the enrollment store: add, remove, list, and look up speaker profiles.
- Compare an embedding against all enrolled speakers, returning similarity scores.
- Serialize and deserialize the enrollment store (numpy arrays + metadata).

**Thread safety:** The enrollment store is protected by a `threading.Lock`. All reads and writes to the internal `_profiles` dict acquire this lock. This allows safe concurrent enrollment from multiple threads while the pipeline is running.

**Key methods:**

| Method | Description |
|---|---|
| `enroll(name, waveforms)` | Compute mean embedding from waveforms, store under `name` |
| `get_embedding(waveform)` | Compute a single embedding vector |
| `compare(embedding)` | Return `dict[str, float]` of cosine similarities against all enrolled speakers |
| `save(path)` / `load(path)` | Persist / restore profiles to/from disk |
| `remove(name)` | Remove a speaker from the store |
| `speakers` (property) | List enrolled speaker names |

### `voicetag/diarizer.py`

Pyannote.audio wrapper. Handles model loading, HuggingFace authentication, and raw diarization.

**Responsibilities:**

- Load the pyannote diarization pipeline (`pyannote/speaker-diarization-3.1` or configured alternative).
- Manage the HuggingFace auth token (from config, environment variable `HF_TOKEN`, or `~/.huggingface/token`).
- Run diarization on an audio file, returning a list of raw segments with speaker labels and timestamps.
- Detect overlapping speech regions from the pyannote output.

**Key methods:**

| Method | Description |
|---|---|
| `__init__(hf_token, device)` | Load pyannote pipeline with auth |
| `diarize(audio_path)` | Return list of `(start, end, speaker_label)` tuples |
| `get_overlaps(diarization)` | Extract overlapping regions from raw output |

### `voicetag/overlap.py`

Overlap detection and merging logic. This is a pure-function module with no state.

**Responsibilities:**

- Take raw diarization segments and identify time regions where two or more speakers are active simultaneously.
- Merge adjacent or near-adjacent overlap regions (within a configurable tolerance).
- Return `OverlapSegment` objects with the involved speakers and the overlap ratio.
- Filter overlaps below `config.overlap_threshold`.

**Key functions:**

| Function | Description |
|---|---|
| `detect_overlaps(segments, threshold)` | Find overlapping regions from diarization segments |
| `merge_overlaps(overlaps, tolerance)` | Merge nearby overlap regions |
| `compute_overlap_ratio(seg_a, seg_b)` | Compute the temporal overlap ratio between two segments |

### `voicetag/models.py`

Pydantic v2 data models used throughout the library. All models use strict validation and are immutable (`model_config = ConfigDict(frozen=True)`).

**Models:**

```python
class VoiceTagConfig(BaseModel):
    hf_token: Optional[str] = None
    similarity_threshold: float = 0.75       # min cosine similarity for match
    overlap_threshold: float = 0.5           # min overlap ratio to flag
    max_workers: int = 4                     # parallel embedding workers
    min_segment_duration: float = 0.5        # ignore segments shorter than this (seconds)
    device: str = "cpu"                      # torch device ("cpu", "cuda", "mps")

class SpeakerSegment(BaseModel):
    speaker: str                             # enrolled name or "UNKNOWN"
    start: float                             # seconds
    end: float                               # seconds
    confidence: float                        # cosine similarity score (0.0-1.0)

class OverlapSegment(BaseModel):
    speakers: list[str]                      # speakers involved
    start: float
    end: float
    overlap_ratio: float                     # 0.0-1.0

class DiarizationResult(BaseModel):
    segments: list[SpeakerSegment]
    overlaps: list[OverlapSegment]
    audio_duration: float                    # total audio length in seconds
    num_speakers_detected: int

class SpeakerProfile(BaseModel):
    name: str
    num_samples: int                         # number of audio files used for enrollment
    created_at: datetime
```

**Validation rules:**

- `start` must be non-negative and less than `end` on all segment models.
- `confidence` and `overlap_ratio` must be in `[0.0, 1.0]`.
- `similarity_threshold` must be in `(0.0, 1.0)`.
- `max_workers` must be >= 1.
- `min_segment_duration` must be > 0.
- `device` must be one of `"cpu"`, `"cuda"`, `"mps"`.

### `voicetag/cli.py`

Typer CLI with Rich output for terminal formatting.

**Commands:**

| Command | Description |
|---|---|
| `voicetag enroll NAME FILE [FILE...]` | Enroll a speaker from one or more audio files |
| `voicetag identify FILE` | Identify speakers in an audio file |
| `voicetag profiles list` | List all enrolled speaker profiles |
| `voicetag profiles show NAME` | Show details for an enrolled speaker |
| `voicetag profiles delete NAME` | Remove an enrolled speaker |

**Options (global):**

- `--config PATH` -- path to a JSON/TOML config file
- `--hf-token TEXT` -- HuggingFace token (overrides config and env)
- `--device TEXT` -- torch device
- `--profiles-dir PATH` -- directory for saved profiles (default: `~/.voicetag/profiles`)

Output uses Rich tables for `profiles list`, Rich panels for `identify` results, and Rich progress bars during diarization and embedding computation.

### `voicetag/utils.py`

Audio I/O utilities. No ML dependencies -- only `soundfile` and `numpy`.

**Responsibilities:**

- Load audio files via `soundfile.read()`, returning `(waveform, sample_rate)`.
- Validate that files are readable WAV/FLAC/OGG/MP3 (via soundfile format support).
- Resample audio to 16 kHz (required by both pyannote and resemblyzer).
- Chunk long audio files into overlapping windows for memory-bounded processing.
- Convert stereo to mono by averaging channels.

**Key functions:**

| Function | Description |
|---|---|
| `load_audio(path)` | Load, validate, resample to 16 kHz, return `np.ndarray` |
| `chunk_audio(waveform, chunk_seconds, overlap_seconds)` | Split into overlapping chunks |
| `validate_audio(path)` | Check file exists, is readable, has valid audio format |
| `resample(waveform, orig_sr, target_sr)` | Resample waveform to target sample rate |

### `voicetag/exceptions.py`

Custom exception hierarchy. All exceptions inherit from `VoiceTagError` so callers can catch broadly or narrowly.

```
VoiceTagError
  +-- VoiceTagConfigError     # invalid config values, missing tokens
  +-- EnrollmentError         # enrollment failures (no audio, bad format, duplicate name)
  +-- DiarizationError        # pyannote failures (auth, model loading, processing)
  +-- AudioLoadError          # file not found, unsupported format, corrupt file
```

---

## 3. Data Flow

```
Audio File
  |
  v
[utils.load_audio]  ---- validate format, resample to 16kHz, mono
  |
  v
[diarizer.diarize]  ---- pyannote segments: [(start, end, "SPEAKER_00"), ...]
  |
  +---> [overlap.detect_overlaps] ---- find simultaneous speech regions
  |                                        |
  v                                        v
[Extract waveform slices per segment]   OverlapSegments
  |
  v
[encoder.get_embedding] x N  ---- concurrent.futures.ThreadPoolExecutor
  |                                  (max_workers from config)
  v
[encoder.compare] x N  ---- cosine similarity against enrolled speakers
  |
  v
[Filter by similarity_threshold]  ---- below threshold -> speaker = "UNKNOWN"
  |
  v
[Filter by min_segment_duration]  ---- discard very short segments
  |
  v
[Merge adjacent same-speaker segments]
  |
  v
[Sort by start time]
  |
  v
DiarizationResult(segments=[SpeakerSegment, ...], overlaps=[OverlapSegment, ...])
```

### Detailed step-by-step

1. **Load and validate** -- `utils.load_audio(path)` reads the file via soundfile, converts to mono, resamples to 16 kHz. Raises `AudioLoadError` on failure.

2. **Diarize** -- `diarizer.diarize(audio_path)` runs pyannote's speaker diarization pipeline. Returns raw segments as `(start, end, speaker_label)` tuples. Speaker labels are anonymous (e.g., `SPEAKER_00`, `SPEAKER_01`).

3. **Extract segments** -- For each diarization segment, slice the corresponding waveform window from the loaded audio array.

4. **Compute embeddings (parallel)** -- Submit each waveform slice to `encoder.get_embedding()` via a `ThreadPoolExecutor`. Resemblyzer computes a 256-dimensional embedding vector per segment. Parallelism is bounded by `config.max_workers`.

5. **Match against enrolled speakers** -- For each embedding, call `encoder.compare()` to compute cosine similarity against all enrolled speaker profiles. If the best match exceeds `config.similarity_threshold`, assign that speaker name. Otherwise, label as `"UNKNOWN"`.

6. **Detect overlaps** -- `overlap.detect_overlaps()` scans the raw diarization segments for temporal intersections. Regions where two or more speakers are active simultaneously produce `OverlapSegment` objects. Overlaps below `config.overlap_threshold` are discarded.

7. **Merge and sort** -- Adjacent segments assigned to the same speaker are merged (gap tolerance: 0.1s). The final segment list is sorted by `start` time.

8. **Return** -- A `DiarizationResult` containing the sorted `SpeakerSegment` list, `OverlapSegment` list, total audio duration, and detected speaker count.

---

## 4. Public API Surface

```python
from voicetag import VoiceTag, VoiceTagConfig

# Initialize with default config
vt = VoiceTag()

# Initialize with custom config
vt = VoiceTag(config=VoiceTagConfig(
    hf_token="hf_...",
    similarity_threshold=0.80,
    max_workers=8,
    device="cuda",
))

# Enroll a speaker from one or more audio files
vt.enroll("alice", ["alice_sample1.wav", "alice_sample2.wav"])
vt.enroll("bob", ["bob_sample.wav"])

# List enrolled speakers
vt.enrolled_speakers  # ["alice", "bob"]

# Identify speakers in an audio file
result = vt.identify("meeting.wav")
for seg in result.segments:
    print(f"{seg.start:.1f}-{seg.end:.1f}s: {seg.speaker} ({seg.confidence:.2f})")
for ov in result.overlaps:
    print(f"{ov.start:.1f}-{ov.end:.1f}s: overlap between {ov.speakers}")

# Persist enrolled profiles to disk
vt.save("profiles.voicetag")

# Load profiles (class method or instance method)
vt.load("profiles.voicetag")

# Remove a speaker
vt.remove_speaker("bob")
```

---

## 5. Configuration

`VoiceTagConfig` is a Pydantic v2 `BaseModel` that controls all tunable parameters.

| Field | Type | Default | Description |
|---|---|---|---|
| `hf_token` | `Optional[str]` | `None` | HuggingFace token for pyannote model access. Falls back to `HF_TOKEN` env var, then `~/.huggingface/token`. |
| `similarity_threshold` | `float` | `0.75` | Minimum cosine similarity to consider a match. Segments below this are labeled `"UNKNOWN"`. Range: `(0.0, 1.0)`. |
| `overlap_threshold` | `float` | `0.5` | Minimum overlap ratio to flag a region as overlapping speech. Range: `[0.0, 1.0]`. |
| `max_workers` | `int` | `4` | Number of threads for parallel embedding computation. |
| `min_segment_duration` | `float` | `0.5` | Segments shorter than this (in seconds) are discarded. Very short segments produce unreliable embeddings. |
| `device` | `str` | `"cpu"` | Torch device for model inference. One of `"cpu"`, `"cuda"`, `"mps"`. |

**Token resolution order:**

1. `config.hf_token` (explicit)
2. `HF_TOKEN` environment variable
3. `~/.huggingface/token` file
4. Raise `VoiceTagConfigError` with a message linking to https://huggingface.co/settings/tokens

---

## 6. Thread Safety

**What is thread-safe:**

- The enrollment store in `encoder.py`. All reads and writes to the internal `_profiles` dictionary are serialized through a `threading.Lock`. Multiple threads can safely call `enroll()`, `compare()`, and `remove()` concurrently.
- Profile save/load operations acquire the same lock to prevent concurrent modification during serialization.

**What is NOT thread-safe:**

- The `VoiceTag` pipeline instance itself. The pyannote diarization pipeline and resemblyzer encoder maintain internal state that is not safe to share across threads.
- **Guideline:** Create one `VoiceTag` instance per thread. The enrollment store can be shared by saving/loading profiles, but do not share the `VoiceTag` object itself.

**Parallelism within the pipeline:**

- Embedding computation for segments within a single `identify()` call is parallelized using `concurrent.futures.ThreadPoolExecutor`. This is internal parallelism -- the caller does not need to manage threads.
- The GIL is released during numpy and torch operations, so the thread pool provides real speedup for embedding computation.

---

## 7. Error Handling Strategy

All errors raised by voicetag inherit from `VoiceTagError`, enabling a single catch-all:

```python
try:
    result = vt.identify("audio.wav")
except VoiceTagError as e:
    print(f"voicetag error: {e}")
```

**Exception mapping:**

| Exception | Raised when | Example message |
|---|---|---|
| `VoiceTagConfigError` | Invalid config values or missing auth | `"HuggingFace token required for pyannote. Set hf_token in config, HF_TOKEN env var, or see https://huggingface.co/settings/tokens"` |
| `EnrollmentError` | Enrollment fails | `"Cannot enroll 'alice': no valid audio files provided"` |
| `DiarizationError` | Pyannote processing fails | `"Diarization failed: model requires authentication (HTTP 401)"` |
| `AudioLoadError` | Audio file cannot be loaded | `"Cannot load 'meeting.mp4': unsupported format. Supported: wav, flac, ogg, mp3"` |

**Design principles:**

- Every exception includes a human-readable message explaining what went wrong and, where applicable, how to fix it.
- Config errors always include a link to relevant documentation (HuggingFace token page, pyannote model page).
- Exceptions from upstream libraries (pyannote, resemblyzer, soundfile) are caught and re-raised as the appropriate voicetag exception, preserving the original exception via `raise ... from e`.
- The CLI catches `VoiceTagError` at the top level and prints a formatted error message via Rich, exiting with a non-zero status code.
