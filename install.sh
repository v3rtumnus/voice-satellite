#!/bin/bash
#
# Voice Satellite Installation Script for Raspberry Pi
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# After installation, edit /etc/voice-satellite/config.env and run:
#   sudo systemctl start voice-satellite
#

set -e

echo "========================================"
echo "Voice Satellite Installation"
echo "========================================"

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "Warning: This script is designed for Raspberry Pi"
fi

# Check root
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run as root. Run as your normal user."
    exit 1
fi

# Configuration
INSTALL_DIR="/opt/voice-satellite"
CONFIG_DIR="/etc/voice-satellite"
VENV_DIR="$INSTALL_DIR/venv"
USER=$(whoami)

echo ""
echo "Install directory: $INSTALL_DIR"
echo "Config directory: $CONFIG_DIR"
echo "User: $USER"
echo ""

# Update system
echo "[1/6] Updating system..."
sudo apt-get update

# Install system dependencies
echo "[2/6] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    libopenblas-dev \
    alsa-utils

# Create directories
echo "[3/6] Creating directories..."
sudo mkdir -p $INSTALL_DIR
sudo mkdir -p $CONFIG_DIR
sudo chown $USER:$USER $INSTALL_DIR

# Copy files
echo "[4/6] Copying files..."
cp -r satellite $INSTALL_DIR/
cp requirements.txt $INSTALL_DIR/
mkdir -p $INSTALL_DIR/sounds

# Create virtual environment
echo "[5/6] Creating Python environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install --upgrade pip
pip install -r $INSTALL_DIR/requirements.txt

deactivate

# Create config file
echo "[6/6] Creating configuration..."
if [ ! -f $CONFIG_DIR/config.env ]; then
    sudo tee $CONFIG_DIR/config.env > /dev/null << 'EOF'
# Voice Satellite Configuration
# Edit these values for your setup

# Voice server WebSocket URL
SERVER_URL=ws://192.168.1.100:8765

# Room name (used for context in queries)
ROOM=Living Room

# Device name (optional, defaults to hostname)
# DEVICE=living-room-pi

# Wake word (Porcupine built-in keywords)
# Options: alexa, bumblebee, computer, hey google, hey siri,
#          jarvis, ok google, picovoice, porcupine, terminator
WAKE_WORD_MODEL=jarvis

# Porcupine Access Key (required)
# Get free key at: https://console.picovoice.ai/
PORCUPINE_ACCESS_KEY=your-access-key-here

# Silence detection threshold (lower = more sensitive)
SILENCE_THRESHOLD=500

# Seconds of silence before ending recording
SILENCE_DURATION=1.5

# Sounds directory
SOUNDS_DIR=/opt/voice-satellite/sounds
EOF
    echo "Created $CONFIG_DIR/config.env"
    echo ">>> EDIT THIS FILE with your server URL, room name, and Porcupine access key <<<"
else
    echo "Config file already exists: $CONFIG_DIR/config.env"
fi

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/voice-satellite.service > /dev/null << EOF
[Unit]
Description=Voice Satellite
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$CONFIG_DIR/config.env
ExecStart=$VENV_DIR/bin/python -m satellite.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your settings:"
echo "   sudo nano $CONFIG_DIR/config.env"
echo ""
echo "2. Test audio devices:"
echo "   arecord -l    # List microphones"
echo "   aplay -l      # List speakers"
echo ""
echo "3. Test recording:"
echo "   arecord -d 3 test.wav && aplay test.wav"
echo ""
echo "4. (Optional) Add notification sounds:"
echo "   # Generate simple beeps:"
echo "   sudo apt install sox"
echo "   sox -n $INSTALL_DIR/sounds/awake.wav synth 0.2 sine 800 vol 0.5"
echo "   sox -n $INSTALL_DIR/sounds/done.wav synth 0.1 sine 600 vol 0.5"
echo ""
echo "5. Start the satellite:"
echo "   sudo systemctl enable voice-satellite"
echo "   sudo systemctl start voice-satellite"
echo ""
echo "6. View logs:"
echo "   journalctl -u voice-satellite -f"
echo ""
