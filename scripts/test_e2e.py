#!/usr/bin/env python3
"""
End-to-end test: Record audio, send to server, play response.

Usage:
    python scripts/test_e2e.py
    python scripts/test_e2e.py --server ws://192.168.1.100:8765 --room kitchen
"""

import argparse
import asyncio
import json
import os
import sys

import aiohttp

sys.path.insert(0, "/opt/voice-satellite")

from satellite.main import AudioRecorder, AudioPlayer


async def main():
    parser = argparse.ArgumentParser(description="End-to-end satellite test")
    parser.add_argument(
        "--server",
        default=os.getenv("SERVER_URL", "ws://localhost:8765"),
    )
    parser.add_argument("--room", default=os.getenv("ROOM", "test"))
    parser.add_argument("--duration", type=float, default=5.0)
    
    args = parser.parse_args()
    
    recorder = AudioRecorder()
    player = AudioPlayer()
    
    print("=" * 50)
    print("Voice Satellite E2E Test")
    print("=" * 50)
    print(f"Server: {args.server}")
    print(f"Room: {args.room}")
    print()
    
    # Record
    print("Recording (speak now)...")
    audio = await recorder.record_until_silence(
        max_duration=args.duration,
        silence_duration=1.5,
    )
    print(f"Recorded {len(audio)} bytes")
    
    if len(audio) < 1000:
        print("Audio too short, aborting")
        return
    
    # Send to server
    print(f"\nConnecting to {args.server}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            ws_url = f"{args.server}/voice?room={args.room}&device=e2e_test"
            
            async with session.ws_connect(ws_url, timeout=60) as ws:
                # Wait for connected
                msg = await ws.receive_json()
                print(f"Connected: {msg.get('satellite_id')}")
                
                # Send config
                await ws.send_json({"type": "config", "sample_rate": 16000})
                
                # Send audio
                print("Sending audio...")
                chunk_size = 4096
                for i in range(0, len(audio), chunk_size):
                    await ws.send_bytes(audio[i:i + chunk_size])
                
                await ws.send_json({"type": "end"})
                print("Waiting for response...")
                
                # Receive
                response_audio = bytearray()
                audio_format = {"sample_rate": 22050, "channels": 1, "sample_width": 2}
                
                while True:
                    msg = await asyncio.wait_for(ws.receive(), timeout=60)
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "processing":
                            print("Processing...")
                        elif msg_type == "transcript":
                            print(f"✓ Transcript: {data.get('text')}")
                        elif msg_type == "response":
                            print(f"✓ Response: {data.get('text')}")
                        elif msg_type == "audio_start":
                            audio_format = {
                                "sample_rate": data.get("sample_rate", 22050),
                                "channels": data.get("channels", 1),
                                "sample_width": data.get("sample_width", 2),
                            }
                        elif msg_type == "audio_end":
                            print(f"✓ Audio: {len(response_audio)} bytes")
                        elif msg_type == "done":
                            print(f"✓ Done in {data.get('duration', 0):.2f}s")
                            break
                        elif msg_type == "error":
                            print(f"✗ Error: {data.get('message')}")
                            break
                    
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        response_audio.extend(msg.data)
                    
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        break
                
                # Play response
                if response_audio:
                    print("\nPlaying response...")
                    player.play(bytes(response_audio), **audio_format)
                    print("Done!")
                else:
                    print("\nNo audio response received")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        recorder.cleanup()
        player.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
