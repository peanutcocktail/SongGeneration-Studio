# SongGeneration Studio

AI Song Generation with Full Style Control using Tencent AI Lab's SongGeneration (LeVo) model.

## Features

- **Full Song Generation**: Generate complete songs with vocals and instrumental tracks
- **Style Control**: Control gender, timbre, genre, emotion, instruments, and BPM
- **Lyrics Support**: Input your own lyrics with section markers (intro, verse, chorus, bridge, outro)
- **Reference Audio**: Use a reference audio file to guide the style of generation
- **Multiple Output Modes**: Generate mixed, vocals only, instrumental only, or separate tracks
- **Web UI**: Easy-to-use web interface for song creation

## Requirements

- **GPU**: NVIDIA GPU with at least 10GB VRAM (24GB+ recommended for full quality)
- **Storage**: ~25GB for models and dependencies
- **OS**: Windows, macOS, or Linux

## Installation

1. Open Pinokio
2. Find "SongGeneration Studio" in the app list
3. Click "Install"
4. Wait for the installation to complete (this may take a while due to large model downloads)

## Usage

1. Click "Start" to launch the server
2. The Web UI will open automatically in your browser
3. Create your song:
   - Enter a song title
   - Add song sections (intro, verse, chorus, etc.) with lyrics
   - Configure style settings (genre, emotion, voice, instruments)
   - Optionally upload a reference audio file
4. Click "Generate Song" and wait for the generation to complete
5. Play and download your generated song

## Song Sections

The following section types are supported:

| Section | Description |
|---------|-------------|
| `intro-short` / `intro` / `intro-long` | Song introduction |
| `verse` / `verse-short` / `verse-long` | Main verses |
| `chorus` / `chorus-short` / `chorus-long` | Song chorus |
| `bridge` / `bridge-short` / `bridge-long` | Bridge section |
| `outro-short` / `outro` / `outro-long` | Song ending |
| `break` | Musical break |
| `solo` | Instrumental solo |

## Style Settings

- **Voice Gender**: Female or Male
- **Timbre**: Bright, Dark, Soft, Powerful, Warm, Clear, Raspy, Smooth
- **Genre**: Pop, Rock, Metal, Jazz, R&B, Folk, Dance/Electronic, Reggae, Chinese Style
- **Emotion**: Happy, Sad, Energetic, Romantic, Angry, Peaceful, Melancholic, Hopeful
- **Instruments**: Specify instruments (e.g., "piano and drums", "electric guitar, bass, drums")
- **BPM**: Tempo (60-200)

## API Documentation

### Generate a Song

```javascript
// JavaScript
const response = await fetch('/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: "My Song",
    sections: [
      { type: "intro-short", lyrics: null },
      { type: "verse", lyrics: "Walking through the city lights" },
      { type: "chorus", lyrics: "We are alive, we are on fire" },
      { type: "outro-short", lyrics: null }
    ],
    gender: "female",
    timbre: "bright",
    genre: "Pop",
    emotion: "happy",
    instruments: "piano and drums",
    bpm: 120,
    output_mode: "mixed"
  })
});
const { generation_id } = await response.json();
```

```python
# Python
import requests

response = requests.post('http://localhost:8000/api/generate', json={
    'title': 'My Song',
    'sections': [
        {'type': 'intro-short', 'lyrics': None},
        {'type': 'verse', 'lyrics': 'Walking through the city lights'},
        {'type': 'chorus', 'lyrics': 'We are alive, we are on fire'},
        {'type': 'outro-short', 'lyrics': None}
    ],
    'gender': 'female',
    'timbre': 'bright',
    'genre': 'Pop',
    'emotion': 'happy',
    'instruments': 'piano and drums',
    'bpm': 120,
    'output_mode': 'mixed'
})
generation_id = response.json()['generation_id']
```

```bash
# cURL
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Song",
    "sections": [
      {"type": "intro-short", "lyrics": null},
      {"type": "verse", "lyrics": "Walking through the city lights"},
      {"type": "chorus", "lyrics": "We are alive, we are on fire"},
      {"type": "outro-short", "lyrics": null}
    ],
    "gender": "female",
    "timbre": "bright",
    "genre": "Pop",
    "emotion": "happy",
    "instruments": "piano and drums",
    "bpm": 120,
    "output_mode": "mixed"
  }'
```

### Check Generation Status

```bash
curl http://localhost:8000/api/generation/{generation_id}
```

### Download Generated Audio

```bash
curl http://localhost:8000/api/audio/{generation_id}/0 --output song.wav
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check and GPU status |
| `/api/models` | GET | List available models |
| `/api/presets` | GET | Get available style presets |
| `/api/generate` | POST | Start a new song generation |
| `/api/generation/{id}` | GET | Get generation status |
| `/api/generations` | GET | List all generations |
| `/api/audio/{id}/{track}` | GET | Download generated audio track |
| `/api/upload-reference` | POST | Upload a reference audio file |

## Credits

- **SongGeneration (LeVo)**: [Tencent AI Lab](https://github.com/tencent-ailab/SongGeneration)
- **Pinokio Launcher**: This installer

## License

This launcher is provided for educational and research purposes. Please refer to the original [SongGeneration repository](https://github.com/tencent-ailab/SongGeneration) for licensing information.
