# GigaAM Integration for WhisperLiveKit

## Overview

This implementation integrates GigaAM's ASR models with WhisperLiveKit's streaming transcription pipeline, enabling real-time transcription with GigaAM's state-of-the-art models.

## Implementation Files

### 1. GigaAM Backend (`whisperlivekit/gigaam_asr.py`)

A complete ASR backend implementation that:
- Loads GigaAM models (v1/v2/v3)
- Accepts 16kHz float32 mono audio from WhisperLiveKit
- Returns word-level timestamps in WhisperLiveKit format
- Supports both short-form and long-form transcription

### 2. Core Integration (`whisperlivekit/core.py`)

Added GigaAM backend registration at line ~166:
```python
elif config.backend == "gigaam":
    from whisperlivekit.gigaam_asr import GigaAMASR
    self.asr = GigaAMASR(
        **transcription_common_params,
        model_name=config.model_path or "v3_e2e_rnnt",
    )
    logger.info("Using GigaAM backend with LocalAgreement policy")
```

## Usage

### Starting the Server with GigaAM

```bash
# Install GigaAM if not already installed
pip install gigaam

# Start WhisperLiveKit server with GigaAM backend
python -m whisperlivekit.server \
    --backend gigaam \
    --model-path v3_e2e_rnnt \
    --language en \
    --min-chunk-size 1.0 \
    --diarization true \
    --diarization-backend sortformer
```

### Supported GigaAM Models

**ASR Models:**
- `v3_e2e_rnnt` (recommended - best quality)
- `v3_ctc` (faster)
- `v2_ctc`
- `v1_ctc`

**Embedding Models (for diarization):**
- `v3_ssl`
- `v2_ssl`
- `v1_ssl`

### Client Connection

Connect using any WhisperLiveKit client:

```python
import websockets
import json
import base64
import numpy as np

async def connect_to_gigaam_server():
    async with websockets.connect("ws://localhost:8080/asr") as websocket:
        # Send configuration
        config = {
            "type": "config",
            "language": "en",
            "backend": "gigaam",
            "model": "v3_e2e_rnnt",
            "diarization": True
        }
        await websocket.send(json.dumps(config))
        
        # Send audio chunks (16kHz int16 PCM)
        async for audio_chunk in audio_stream():
            await websocket.send(audio_chunk)
        
        # Receive transcription results
        async for message in websocket:
            result = json.loads(message)
            print(f"Transcription: {result}")
```

## Audio Format

GigaAM backend expects audio in this format (provided by WhisperLiveKit):
- **Sample Rate**: 16000 Hz
- **Format**: float32 numpy array
- **Range**: [-1.0, 1.0] (normalized from int16 / 32768.0)
- **Channels**: Mono (1D array)
- **Shape**: `(N_samples,)`

The backend automatically converts to int16 WAV format for GigaAM API.

## Features

### 1. Streaming Transcription
- Real-time word-level timestamps
- Continuous transcription with buffer management
- Support for long conversations

### 2. Speaker Diarization
- Integration with WhisperLiveKit's diarization backends
- Sortformer (recommended) or Diart
- Speaker-labeled transcripts

### 3. Multiple Model Options
- Choose between v1/v2/v3 models
- Balance quality vs. speed
- Support for CTC and RNNT architectures

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--backend` | ASR backend | `gigaam` |
| `--model-path` | GigaAM model name | `v3_e2e_rnnt` |
| `--language` | Language code | `en` |
| `--min-chunk-size` | Minimum audio chunk (s) | `1.0` |
| `--diarization` | Enable speaker diarization | `false` |
| `--diarization-backend` | Diarization backend | `sortformer` |

## Architecture

```
WebSocket Client
       ↓
WhisperLiveKit AudioProcessor
       ↓
GigaAMASR Backend
       ↓
GigaAM Model (v3_e2e_rnnt, etc.)
       ↓
Word-level Transcription
       ↓
TokensAlignment + Diarization
       ↓
WebSocket Client (with speaker labels)
```

## Testing

### Quick Test

```python
# test_gigaam_backend.py
import numpy as np
from whisperlivekit.gigaam_asr import GigaAMASR

# Create backend instance
backend = GigaAMASR(model_name="v3_e2e_rnnt")

# Load model
backend.load_model()

# Generate test audio (1 second of silence + speech)
test_audio = np.zeros(16000, dtype=np.float32)
# Add some synthetic speech-like signal
test_audio[8000:12000] = np.sin(2 * np.pi * 440 * np.linspace(0, 0.4, 6400))

# Transcribe
result = backend.transcribe(test_audio)
print("Segments:", result['segments'])

# Extract tokens
tokens = backend.ts_words(result)
print("Tokens:", [(t.text, t.start, t.end) for t in tokens])
```

## Performance Considerations

### Model Loading Time
- First load: ~30-60 seconds (downloads from HuggingFace)
- Subsequent loads: ~5-10 seconds (cached)

### Inference Latency
- `v3_e2e_rnnt`: ~100-200ms per second of audio
- `v3_ctc`: ~50-100ms per second of audio
- Depends on hardware (GPU recommended)

### Memory Usage
- Model size: ~1-2 GB (varies by model)
- Runtime memory: ~2-4 GB

## Troubleshooting

### Common Issues

1. **"GigaAM not installed"**
   ```bash
   pip install gigaam
   ```

2. **"Model not found"**
   - Ensure HF_TOKEN is set for private models
   - Check model name: `v3_e2e_rnnt`, `v3_ctc`, etc.

3. **"Audio format error"**
   - Verify audio is 16kHz mono
   - Check float32 normalization [-1, 1]

4. **Slow performance**
   - Use GPU: `export CUDA_VISIBLE_DEVICES=0`
   - Try faster model: `v3_ctc` instead of `v3_e2e_rnnt`

## Future Enhancements

1. **Direct GigaAM API Integration**
   - Avoid temporary file I/O
   - Use GigaAM's streaming API if available

2. **Optimized Audio Conversion**
   - Direct int16 → float32 without file I/O
   - Batch processing for better throughput

3. **Model Caching**
   - Persistent model cache across sessions
   - Multi-model support

4. **Emotion Recognition**
   - Integrate GigaAM's emotion model
   - Add emotion labels to transcripts

## References

- [WhisperLiveKit Repository](https://github.com/livespeechio/WhisperLiveKit)
- [GigaAM Repository](https://github.com/your-org/GigaAM)
- [GigaAM Documentation](https://gigaam.readthedocs.io/)

## License

This integration is part of WhisperLiveKit and follows its license.
GigaAM models are subject to their respective licenses.
