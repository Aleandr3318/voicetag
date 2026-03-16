# voicetag

**Know who said what. Automatically.**

voicetag is a Python library for speaker diarization and named speaker identification. It combines [pyannote.audio](https://github.com/pyannote/pyannote-audio) and [resemblyzer](https://github.com/resemble-ai/Resemblyzer) into a simple, unified API.

## Get started in 30 seconds

```python
from voicetag import VoiceTag

vt = VoiceTag()
vt.enroll("Alice", ["alice1.wav", "alice2.wav"])
vt.enroll("Bob", ["bob1.wav"])

result = vt.identify("meeting.wav")
for segment in result.segments:
    print(f"{segment.speaker}: {segment.start:.1f}s - {segment.end:.1f}s")
```

## Next steps

- [Getting Started](getting-started.md) -- installation, prerequisites, and your first identification
- [API Reference](api.md) -- full documentation of the `VoiceTag` class and result models
- [CLI](cli.md) -- command-line usage for enrollment and identification
- [Configuration](configuration.md) -- tuning thresholds, device selection, and performance
- [Contributing](contributing.md) -- how to set up a development environment and submit PRs
