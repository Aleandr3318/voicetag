# Configuration

All pipeline behavior is controlled through `VoiceTagConfig`, a Pydantic v2 model.

```python
from voicetag import VoiceTag, VoiceTagConfig

config = VoiceTagConfig(
    hf_token="hf_...",
    similarity_threshold=0.80,
    max_workers=8,
    device="cuda",
)
vt = VoiceTag(config=config)
```

## Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `hf_token` | `Optional[str]` | `None` | HuggingFace token for pyannote model access. Falls back to `HF_TOKEN` env var. |
| `similarity_threshold` | `float` | `0.75` | Minimum cosine similarity to consider a speaker match. Segments below this threshold are labeled `"UNKNOWN"`. Range: 0.0 to 1.0. |
| `overlap_threshold` | `float` | `0.5` | Minimum overlap ratio to flag a region as overlapping speech. Range: 0.0 to 1.0. |
| `max_workers` | `int` | `4` | Number of threads for parallel embedding computation. Higher values use more memory but process faster. Minimum: 1. |
| `min_segment_duration` | `float` | `0.5` | Segments shorter than this (in seconds) are discarded. Very short segments produce unreliable embeddings. |
| `device` | `str` | `"cpu"` | Torch device for model inference. Options: `"cpu"`, `"cuda"` (NVIDIA GPU), `"mps"` (Apple Silicon). |

## HuggingFace Token Resolution

The `hf_token` is resolved in this order:

1. **Explicit value** passed to `VoiceTagConfig(hf_token="...")`
2. **`HF_TOKEN` environment variable**
3. If neither is set, pyannote will raise an authentication error

## Tuning Tips

**`similarity_threshold`** is the most important parameter to tune:

- **Lower values (0.5-0.7):** More matches, but higher risk of misidentification
- **Default (0.75):** Good balance for most use cases
- **Higher values (0.8-0.9):** Fewer matches, but very high confidence when matched

**`max_workers`** affects processing speed:

- For CPU: 2-4 workers is usually optimal
- For GPU: 1-2 workers is often sufficient since GPU operations are already parallel
- More workers use more memory

**`min_segment_duration`** filters noise:

- The default of 0.5 seconds works well for most speech
- Lower it for rapid conversational exchanges
- Raise it if you are getting spurious short segments
