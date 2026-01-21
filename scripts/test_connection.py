#!/usr/bin/env python3
"""
Test connection to voice server.

Usage:
    python scripts/test_connection.py
    python scripts/test_connection.py --server ws://192.168.1.100:8765
"""

import argparse
import asyncio
import json
import os
import sys

import aiohttp


async def main():
    parser = argparse.ArgumentParser(description="Test voice server connection")
    parser.add_argument(
        "--server",
        default=os.getenv("SERVER_URL", "ws://localhost:8765"),
        help="Server WebSocket URL"
    )
    parser.add_argument("--room", default="test")
    
    args = parser.parse_args()
    
    print(f"Testing connection to {args.server}")
    print()
    
    # Test HTTP health endpoint
    http_url = args.server.replace("ws://", "http://").replace("wss://", "https://")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Health check
            print("1. Testing HTTP health endpoint...")
            async with session.get(f"{http_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"   ✓ Server healthy: {data}")
                else:
                    print(f"   ✗ Health check failed: {resp.status}")
                    return
            
            # WebSocket
            print("\n2. Testing WebSocket connection...")
            ws_url = f"{args.server}/voice?room={args.room}&device=test"
            
            async with session.ws_connect(ws_url, timeout=10) as ws:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                if msg.get("type") == "connected":
                    print(f"   ✓ WebSocket connected: {msg}")
                else:
                    print(f"   ✗ Unexpected message: {msg}")
                    return
                
                # Test ping
                print("\n3. Testing ping/pong...")
                await ws.send_json({"type": "ping"})
                msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                if msg.get("type") == "pong":
                    print("   ✓ Ping/pong working")
                else:
                    print(f"   ✗ Unexpected response: {msg}")
            
            print("\n✓ All tests passed!")
            print("\nServer is ready for satellite connections.")
    
    except aiohttp.ClientError as e:
        print(f"✗ Connection error: {e}")
        sys.exit(1)
    except asyncio.TimeoutError:
        print("✗ Connection timeout")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
