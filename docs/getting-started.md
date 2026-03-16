# Getting Started

## Installation

```bash
pip install voicetag
```

This installs the core package. The ML backends (pyannote.audio, resemblyzer, torch) are required for actual speaker processing:

```bash
pip install voicetag[ml]
```

## Prerequisites

voicetag uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization, which requires a HuggingFace token:

1. Accept the model license at [hf.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set the token:

```bash
export HF_TOKEN="hf_your_token_here"
```

## Your First Identification

### Step 1: Enroll speakers

Provide one or more audio samples per speaker. More samples produce better accuracy.

```python
from voicetag import VoiceTag

vt = VoiceTag()
vt.enroll("Alice", ["alice1.wav", "alice2.wav"])
vt.enroll("Bob", ["bob1.wav"])
```

### Step 2: Identify speakers in a recording

```python
result = vt.identify("meeting.wav")

for segment in result.segments:
    print(f"{segment.speaker}: {segment.start:.1f}s - {segment.end:.1f}s (confidence: {segment.confidence:.2f})")
```

### Step 3: Save profiles for later

```python
vt.save("my_profiles.voicetag")

# In a later session:
vt2 = VoiceTag()
vt2.load("my_profiles.voicetag")
result = vt2.identify("another_meeting.wav")
```

## Next steps

- [API Reference](api.md) for full method documentation
- [CLI](cli.md) for command-line usage
- [Configuration](configuration.md) for tuning thresholds and performance
