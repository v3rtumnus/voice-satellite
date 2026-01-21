#!/usr/bin/env python3
"""
Test audio recording and playback.

Usage:
    python scripts/test_audio.py
    python scripts/test_audio.py --duration 5
"""

import argparse
import asyncio
import logging
import sys
import wave

sys.path.insert(0, "/opt/voice-satellite")

from satellite.main import AudioRecorder, AudioPlayer


async def main():
    parser = argparse.ArgumentParser(description="Test audio recording and playback")
    parser.add_argument("--duration", type=float, default=3.0, help="Recording duration")
    parser.add_argument("--silence-threshold", type=int, default=500)
    parser.add_argument("--output", "-o", default="/tmp/test_audio.wav")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    recorder = AudioRecorder()
    player = AudioPlayer()
    
    print(f"Recording for up to {args.duration} seconds...")
    print("(Recording will stop on silence)")
    print()
    
    audio = await recorder.record_until_silence(
        silence_threshold=args.silence_threshold,
        silence_duration=1.0,
        max_duration=args.duration,
    )
    
    print(f"Recorded {len(audio)} bytes")
    
    # Save
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(audio)
    print(f"Saved to {args.output}")
    
    # Playback
    print("Playing back...")
    player.play(audio, sample_rate=16000)
    
    print("Done!")
    
    recorder.cleanup()
    player.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
