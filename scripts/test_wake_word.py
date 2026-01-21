#!/usr/bin/env python3
"""
Test wake word detection with Porcupine.

Usage:
    python scripts/test_wake_word.py
    python scripts/test_wake_word.py --model alexa
    python scripts/test_wake_word.py --list
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, "/opt/voice-satellite")

BUILTIN_KEYWORDS = [
    "alexa", "bumblebee", "computer", "hey google", "hey siri",
    "jarvis", "ok google", "picovoice", "porcupine", "terminator"
]


async def main():
    parser = argparse.ArgumentParser(description="Test wake word detection")
    parser.add_argument("--model", default="jarvis", help="Wake word keyword")
    parser.add_argument("--list", action="store_true", help="List available keywords")
    parser.add_argument("--access-key", default=os.getenv("PORCUPINE_ACCESS_KEY", ""), help="Porcupine access key")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available Porcupine keywords (free tier):")
        for kw in BUILTIN_KEYWORDS:
            print(f"  - {kw}")
        print("\nGet free access key at: https://console.picovoice.ai/")
        return
    
    logging.basicConfig(level=logging.INFO)
    
    from satellite.main import WakeWordDetector
    
    detector = WakeWordDetector(
        model=args.model,
        access_key=args.access_key,
    )
    
    print(f"Testing wake word: {args.model}")
    print(f"Say '{args.model}' to trigger...")
    print()
    
    try:
        detector.load()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set PORCUPINE_ACCESS_KEY")
        print("Get free key at: https://console.picovoice.ai/")
        sys.exit(1)
    
    detected = await detector.wait_for_wake_word()
    
    if detected:
        print("\n✓ Wake word detected!")
    else:
        print("\n✗ Detection failed")
    
    detector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
