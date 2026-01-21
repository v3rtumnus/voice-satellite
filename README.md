# Voice Satellite

Voice satellite for Raspberry Pi. Listens for wake word locally, records audio, sends to voice server, and plays back response.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Raspberry Pi Satellite                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Always running locally:                                              │  │
│  │    openWakeWord ("Hey Jarvis")                                        │  │
│  │                                                                       │  │
│  │  On wake word detected:                                               │  │
│  │    1. Play "awake" sound                                              │  │
│  │    2. Record until silence                                            │  │
│  │    3. Send audio to voice server                                      │  │
│  │    4. Receive and play response                                       │  │
│  │    5. Play "done" sound                                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                          WebSocket (only when active)
                                       ▼
                         ┌───────────────────────┐
                         │     Voice Server      │
                         │   (STT / TTS / API)   │
                         └───────────────────────┘
```

## Requirements

- Raspberry Pi 3B+, 4, or Zero 2 W
- USB microphone or audio HAT
- Speaker (USB, 3.5mm, or HAT)
- Raspberry Pi OS Lite (64-bit)

## Quick Start

### 1. Flash Raspberry Pi OS

Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/):
- OS: Raspberry Pi OS Lite (64-bit)
- Configure: WiFi, SSH, hostname

### 2. Copy Satellite Code

```bash
scp -r voice-satellite pi@your-pi-hostname.local:~/
```

### 3. Install

```bash
ssh pi@your-pi-hostname.local
cd voice-satellite
chmod +x install.sh
./install.sh
```

### 4. Configure

```bash
sudo nano /etc/voice-satellite/config.env
```

Set at minimum:
```bash
SERVER_URL=ws://YOUR_SERVER_IP:8765
ROOM=Kitchen
```

### 5. Test Audio

```bash
# List devices
arecord -l
aplay -l

# Test recording
arecord -d 3 test.wav
aplay test.wav
```

### 6. Start Satellite

```bash
sudo systemctl enable voice-satellite
sudo systemctl start voice-satellite

# View logs
journalctl -u voice-satellite -f
```

## Configuration

Edit `/etc/voice-satellite/config.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_URL` | `ws://localhost:8765` | Voice server WebSocket URL |
| `ROOM` | `unknown` | Room name (used in API context) |
| `DEVICE` | hostname | Device identifier |
| `WAKE_WORD_MODEL` | `hey_jarvis` | Wake word model |
| `WAKE_WORD_THRESHOLD` | `0.5` | Detection sensitivity (0.0-1.0) |
| `SILENCE_THRESHOLD` | `500` | Silence detection threshold |
| `SILENCE_DURATION` | `1.5` | Seconds of silence to stop recording |
| `SOUNDS_DIR` | `/opt/voice-satellite/sounds` | Notification sounds |

### Wake Word Models

Available models (openWakeWord):
- `hey_jarvis`
- `alexa`
- `hey_mycroft`
- `ok_nabu`

## Testing

### Test Server Connection

```bash
cd /opt/voice-satellite
source venv/bin/activate
python scripts/test_connection.py --server ws://YOUR_SERVER:8765
```

### Test Wake Word

```bash
python scripts/test_wake_word.py --model hey_jarvis
```

### Test Audio Recording

```bash
python scripts/test_audio.py --duration 5
```

### Full E2E Test

```bash
python scripts/test_e2e.py --server ws://YOUR_SERVER:8765 --room test
```

## Notification Sounds

Create notification sounds (optional):

```bash
sudo apt install sox

# Wake sound (when wake word detected)
sox -n /opt/voice-satellite/sounds/awake.wav synth 0.2 sine 800 vol 0.5

# Done sound (after response plays)
sox -n /opt/voice-satellite/sounds/done.wav synth 0.1 sine 600 vol 0.5
```

## Troubleshooting

### No audio input

```bash
# List recording devices
arecord -l

# Test specific device
arecord -D plughw:1,0 -d 3 test.wav

# Check ALSA config
cat /proc/asound/cards
```

### No audio output

```bash
# List playback devices
aplay -l

# Test specific device
aplay -D plughw:0,0 test.wav

# Set default output
sudo raspi-config  # Advanced Options → Audio
```

### Wake word not detecting

```bash
# Test with lower threshold
python scripts/test_wake_word.py --threshold 0.3

# Check microphone volume
alsamixer
```

### Can't connect to server

```bash
# Test connection
python scripts/test_connection.py --server ws://YOUR_SERVER:8765

# Check network
ping YOUR_SERVER_IP
nc -zv YOUR_SERVER_IP 8765
```

### Service not starting

```bash
# Check status
sudo systemctl status voice-satellite

# View logs
journalctl -u voice-satellite -f

# Check config
cat /etc/voice-satellite/config.env
```

## Project Structure

```
voice-satellite/
├── satellite/
│   ├── __init__.py
│   └── main.py          # Main satellite code
├── scripts/
│   ├── test_wake_word.py
│   ├── test_audio.py
│   ├── test_connection.py
│   └── test_e2e.py
├── sounds/              # Notification sounds (optional)
│   ├── awake.wav
│   └── done.wav
├── install.sh           # Installation script
├── requirements.txt
└── README.md
```

## Manual Installation

If you prefer not to use `install.sh`:

```bash
# Install dependencies
sudo apt-get install python3-pip python3-venv portaudio19-dev

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Run
export SERVER_URL=ws://YOUR_SERVER:8765
export ROOM=Kitchen
python -m satellite.main
```

## Multiple Satellites

For multiple rooms, set unique `ROOM` values:

**Kitchen Pi:**
```bash
ROOM=Kitchen
```

**Living Room Pi:**
```bash
ROOM=Living Room
```

**Bedroom Pi:**
```bash
ROOM=Bedroom
```

Each satellite connects independently to the same voice server.
