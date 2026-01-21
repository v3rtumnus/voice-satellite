#!/usr/bin/env python3
"""
Voice Satellite - Wake word detection and audio streaming.

Listens for wake word locally, then streams audio to voice server
via WebSocket and plays back the response.
"""

import asyncio
import json
import logging
import os
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Satellite configuration."""
    
    # Server connection
    server_url: str = "ws://localhost:8765"
    room: str = "unknown"
    device: str = "satellite"
    
    # Audio
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    
    # Wake word
    wake_word_model: str = "jarvis"
    wake_word_threshold: float = 0.5
    wake_word_disabled: bool = False
    
    # Silence detection
    silence_threshold: int = 500
    silence_duration: float = 1.5  # seconds
    max_record_duration: float = 30.0  # seconds
    
    # Sounds
    sounds_dir: str = "sounds"
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            server_url=os.getenv("SERVER_URL", "ws://localhost:8765"),
            room=os.getenv("ROOM", "unknown"),
            device=os.getenv("DEVICE", os.getenv("HOSTNAME", "satellite")),
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            wake_word_model=os.getenv("WAKE_WORD_MODEL", "jarvis"),
            wake_word_threshold=float(os.getenv("WAKE_WORD_THRESHOLD", "0.5")),
            wake_word_disabled=os.getenv("WAKE_WORD_DISABLED", "").lower() in ("true", "1", "yes"),
            silence_threshold=int(os.getenv("SILENCE_THRESHOLD", "500")),
            silence_duration=float(os.getenv("SILENCE_DURATION", "1.5")),
            sounds_dir=os.getenv("SOUNDS_DIR", "sounds"),
        )


class AudioPlayer:
    """Simple audio playback with resampling support."""
    
    # Force output rate - most USB speakers only truly support 48kHz
    OUTPUT_RATE = 48000
    
    def __init__(self, device_index: Optional[int] = None):
        self.device_index = device_index
        self._pyaudio = None
        self._stream = None
        self._warmed_up = False
    
    def _get_pyaudio(self):
        if self._pyaudio is None:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio
    
    def warm_up(self):
        """Pre-warm the speaker by playing a short silence."""
        import pyaudio
        
        if self._warmed_up:
            return
        
        p = self._get_pyaudio()
        
        # Play 300ms of silence to wake up USB speaker
        silence_samples = int(self.OUTPUT_RATE * 0.3)
        silence = b'\x00' * (silence_samples * 2)
        
        logger.debug("Warming up speaker...")
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.OUTPUT_RATE,
            output=True,
            output_device_index=self.device_index,
        )
        stream.write(silence)
        stream.stop_stream()
        stream.close()
        
        self._warmed_up = True
        logger.debug("Speaker warmed up")
    
    def play(self, audio: bytes, sample_rate: int = 22050, channels: int = 1, sample_width: int = 2):
        """Play raw audio, resampling to OUTPUT_RATE."""
        import pyaudio
        import numpy as np
        
        p = self._get_pyaudio()
        
        # Warm up speaker on first play
        if not self._warmed_up:
            self.warm_up()
        
        # Always resample to OUTPUT_RATE
        if sample_rate != self.OUTPUT_RATE:
            logger.info(f"Resampling audio from {sample_rate}Hz to {self.OUTPUT_RATE}Hz")
            audio_np = np.frombuffer(audio, dtype=np.int16)
            
            # Calculate new length
            new_length = int(len(audio_np) * self.OUTPUT_RATE / sample_rate)
            
            # Linear interpolation resampling
            indices = np.linspace(0, len(audio_np) - 1, new_length)
            resampled = np.interp(indices, np.arange(len(audio_np)), audio_np).astype(np.int16)
            audio = resampled.tobytes()
        
        stream = p.open(
            format=p.get_format_from_width(sample_width),
            channels=channels,
            rate=self.OUTPUT_RATE,
            output=True,
            output_device_index=self.device_index,
        )
        
        try:
            stream.write(audio)
        finally:
            stream.stop_stream()
            stream.close()
    
    def play_wav(self, path: str):
        """Play WAV file."""
        with wave.open(path, "rb") as w:
            audio = w.readframes(w.getnframes())
            self.play(
                audio,
                sample_rate=w.getframerate(),
                channels=w.getnchannels(),
                sample_width=w.getsampwidth(),
            )
    
    def cleanup(self):
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        self._warmed_up = False


class AudioRecorder:
    """Audio recording with silence detection."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        self.target_sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self._pyaudio = None
        self._input_rate = None
        self._need_resample = False
        self._available = None  # None = not checked yet
    
    def _get_pyaudio(self):
        if self._pyaudio is None:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio
    
    def _detect_sample_rate(self) -> bool:
        """Detect supported sample rate. Returns True if mic available."""
        import pyaudio
        
        try:
            p = self._get_pyaudio()
        except Exception as e:
            logger.warning(f"PyAudio initialization failed: {e}")
            self._available = False
            return False
        
        # Try target rate first (16kHz)
        for rate in [self.target_sample_rate, 48000, 44100]:
            try:
                test_stream = p.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk_size,
                )
                test_stream.close()
                self._input_rate = rate
                self._need_resample = (rate != self.target_sample_rate)
                if self._need_resample:
                    logger.info(f"Using {rate}Hz input, will resample to {self.target_sample_rate}Hz")
                self._available = True
                return True
            except Exception as e:
                logger.debug(f"Sample rate {rate}Hz not supported: {e}")
                continue
        
        logger.warning("No microphone available")
        self._available = False
        return False
    
    @property
    def is_available(self) -> bool:
        """Check if microphone is available."""
        if self._available is None:
            self._detect_sample_rate()
        return self._available
    
    async def record_until_silence(
        self,
        silence_threshold: int = 500,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
    ) -> bytes:
        """Record audio until silence is detected."""
        import pyaudio
        import numpy as np
        
        if self._available is None:
            self._detect_sample_rate()
        
        if not self._available:
            raise RuntimeError("No microphone available")
        
        p = self._get_pyaudio()
        
        # Adjust chunk size for input rate
        if self._need_resample:
            input_chunk_size = int(self.chunk_size * self._input_rate / self.target_sample_rate)
        else:
            input_chunk_size = self.chunk_size
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self._input_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=input_chunk_size,
        )
        
        audio_buffer = bytearray()
        silence_chunks = 0
        chunks_for_silence = int(silence_duration * self.target_sample_rate / self.chunk_size)
        max_chunks = int(max_duration * self.target_sample_rate / self.chunk_size)
        chunks_recorded = 0
        
        logger.debug("Recording...")
        
        try:
            while chunks_recorded < max_chunks:
                data = stream.read(input_chunk_size, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                
                # Resample if needed
                if self._need_resample:
                    ratio = self._input_rate // self.target_sample_rate
                    audio_np = audio_np[::ratio]
                    data = audio_np.tobytes()
                
                audio_buffer.extend(data)
                chunks_recorded += 1
                
                # Check for silence
                volume = np.abs(audio_np).mean()
                
                if volume < silence_threshold:
                    silence_chunks += 1
                    if silence_chunks >= chunks_for_silence:
                        logger.debug(f"Silence detected after {chunks_recorded} chunks")
                        break
                else:
                    silence_chunks = 0
                
                # Yield control
                await asyncio.sleep(0)
        
        finally:
            stream.stop_stream()
            stream.close()
        
        logger.debug(f"Recorded {len(audio_buffer)} bytes")
        return bytes(audio_buffer)
    
    def cleanup(self):
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None


class WakeWordDetector:
    """Wake word detection using Porcupine."""
    
    BUILTIN_KEYWORDS = [
        "alexa", "bumblebee", "computer", "hey google", "hey siri",
        "jarvis", "ok google", "picovoice", "porcupine", "terminator"
    ]
    
    def __init__(
        self,
        model: str = "jarvis",
        threshold: float = 0.5,  # Not used by Porcupine, kept for compatibility
        sample_rate: int = 16000,
        device_index: Optional[int] = None,
        access_key: Optional[str] = None,
    ):
        self.keyword = model.lower().replace("_", " ").replace("hey jarvis", "jarvis")
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.access_key = access_key or os.getenv("PORCUPINE_ACCESS_KEY", "")
        self._porcupine = None
        self._pyaudio = None
    
    def load(self):
        """Load Porcupine wake word engine."""
        import pvporcupine
        
        # Validate keyword
        if self.keyword not in self.BUILTIN_KEYWORDS:
            logger.warning(
                f"Unknown keyword '{self.keyword}'. "
                f"Available: {', '.join(self.BUILTIN_KEYWORDS)}. "
                f"Defaulting to 'jarvis'."
            )
            self.keyword = "jarvis"
        
        logger.info(f"Loading Porcupine with keyword: {self.keyword}")
        
        if self.access_key:
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[self.keyword],
            )
        else:
            # Try without access key (limited free usage)
            try:
                self._porcupine = pvporcupine.create(
                    keywords=[self.keyword],
                )
            except pvporcupine.PorcupineActivationError:
                logger.error(
                    "Porcupine requires an access key. "
                    "Get a free key at https://console.picovoice.ai/ "
                    "and set PORCUPINE_ACCESS_KEY environment variable."
                )
                raise
        
        logger.info(f"Porcupine loaded (sample_rate={self._porcupine.sample_rate}, frame_length={self._porcupine.frame_length})")
    
    def _get_pyaudio(self):
        if self._pyaudio is None:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio
    
    async def wait_for_wake_word(self) -> bool:
        """
        Listen continuously until wake word is detected.
        
        Returns:
            True if wake word detected, False on error
        """
        import pyaudio
        import numpy as np
        
        if self._porcupine is None:
            self.load()
        
        p = self._get_pyaudio()
        
        # Porcupine requires 16kHz, but mic might not support it
        # Try 16kHz first, fall back to 48kHz with resampling
        target_rate = self._porcupine.sample_rate  # 16000
        input_rate = target_rate
        need_resample = False
        
        # Try to open stream at 16kHz
        try:
            test_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=target_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self._porcupine.frame_length,
            )
            test_stream.close()
        except OSError:
            # 16kHz not supported, use 48kHz and resample
            input_rate = 48000
            need_resample = True
            logger.info(f"Mic doesn't support 16kHz, using {input_rate}Hz with resampling")
        
        # Calculate frame size for input rate
        if need_resample:
            input_frame_length = int(self._porcupine.frame_length * input_rate / target_rate)
        else:
            input_frame_length = self._porcupine.frame_length
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=input_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=input_frame_length,
        )
        
        logger.info(f"Listening for wake word '{self.keyword}'...")
        
        try:
            while True:
                pcm = stream.read(input_frame_length, exception_on_overflow=False)
                pcm_unpacked = np.frombuffer(pcm, dtype=np.int16)
                
                # Resample if needed
                if need_resample:
                    # Simple decimation (48kHz -> 16kHz = take every 3rd sample)
                    ratio = input_rate // target_rate
                    pcm_unpacked = pcm_unpacked[::ratio]
                    # Ensure correct length
                    if len(pcm_unpacked) > self._porcupine.frame_length:
                        pcm_unpacked = pcm_unpacked[:self._porcupine.frame_length]
                    elif len(pcm_unpacked) < self._porcupine.frame_length:
                        pcm_unpacked = np.pad(pcm_unpacked, (0, self._porcupine.frame_length - len(pcm_unpacked)))
                
                keyword_index = self._porcupine.process(pcm_unpacked)
                
                if keyword_index >= 0:
                    logger.info(f"Wake word detected: {self.keyword}")
                    return True
                
                # Yield control
                await asyncio.sleep(0)
        
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False
        
        finally:
            stream.stop_stream()
            stream.close()
    
    def cleanup(self):
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None


class VoiceSatellite:
    """Voice satellite client."""
    
    def __init__(self, config: Config):
        self.config = config
        
        self.wake_word = None
        if not config.wake_word_disabled:
            self.wake_word = WakeWordDetector(
                model=config.wake_word_model,
                threshold=config.wake_word_threshold,
                sample_rate=config.sample_rate,
            )
        
        self.recorder = AudioRecorder(
            sample_rate=config.sample_rate,
            channels=config.channels,
            chunk_size=config.chunk_size,
        )
        
        self.player = AudioPlayer()
        
        self.sounds_dir = Path(config.sounds_dir)
        self._running = False
    
    def _play_sound(self, name: str):
        """Play notification sound if available."""
        sound_path = self.sounds_dir / f"{name}.wav"
        if sound_path.exists():
            try:
                self.player.play_wav(str(sound_path))
            except Exception as e:
                logger.warning(f"Failed to play sound {name}: {e}")
    
    async def run(self):
        """Main satellite loop."""
        logger.info(f"Starting satellite: room={self.config.room}, device={self.config.device}")
        logger.info(f"Server: {self.config.server_url}")
        
        # Check microphone availability
        mic_available = self.recorder.is_available
        if not mic_available:
            logger.warning("No microphone detected - running in receive-only mode")
            await self._receive_only_mode()
            return
        
        # Mic is available - setup wake word if enabled
        if not self.config.wake_word_disabled:
            logger.info(f"Wake word: {self.config.wake_word_model}")
            try:
                self.wake_word.load()
            except Exception as e:
                logger.error(f"Failed to load wake word: {e}")
                logger.warning("Falling back to receive-only mode")
                await self._receive_only_mode()
                return
        else:
            logger.info("Wake word DISABLED")
        
        self._running = True
        
        # Run with persistent connection
        await self._run_with_connection()
    
    async def _run_with_connection(self):
        """Run main loop with persistent server connection."""
        import aiohttp
        
        ws_url = f"{self.config.server_url}/voice?room={self.config.room}&device={self.config.device}"
        
        # Pre-warm the speaker
        logger.info("Warming up speaker...")
        self.player.warm_up()
        
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    logger.info(f"Connecting to server: {ws_url}")
                    async with session.ws_connect(ws_url, heartbeat=30) as ws:
                        # Wait for connected confirmation
                        msg = await ws.receive_json()
                        logger.info(f"Connected to server: {msg.get('satellite_id')}")
                        
                        # Run both tasks concurrently
                        sender_task = asyncio.create_task(self._sender_loop(ws))
                        receiver_task = asyncio.create_task(self._receiver_loop(ws))
                        
                        done, pending = await asyncio.wait(
                            [sender_task, receiver_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection error: {e}")
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
        
        self.cleanup()
    
    async def _sender_loop(self, ws):
        """Handle wake word detection and audio sending."""
        import aiohttp
        
        while self._running:
            try:
                # Wait for wake word
                detected = await self.wake_word.wait_for_wake_word()
                if not detected:
                    continue
                
                # Play wake sound
                self._play_sound("awake")
                
                # Record audio
                logger.info("Recording...")
                audio = await self.recorder.record_until_silence(
                    silence_threshold=self.config.silence_threshold,
                    silence_duration=self.config.silence_duration,
                    max_duration=self.config.max_record_duration,
                )
                
                if len(audio) < 1000:
                    logger.debug("Audio too short, ignoring")
                    continue
                
                # Send to server via existing connection
                logger.info(f"Sending {len(audio)} bytes to server...")
                await ws.send_json({
                    "type": "config",
                    "sample_rate": self.config.sample_rate,
                })
                
                chunk_size = 4096
                for i in range(0, len(audio), chunk_size):
                    await ws.send_bytes(audio[i:i + chunk_size])
                
                await ws.send_json({"type": "end"})
                logger.debug("Audio sent")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sender error: {e}")
                raise
    
    async def _receiver_loop(self, ws):
        """Handle receiving and playing audio from server."""
        import aiohttp
        
        response_audio = bytearray()
        audio_format = {"sample_rate": 22050, "channels": 1, "sample_width": 2}
        
        while self._running:
            try:
                msg = await ws.receive()
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    
                    if msg_type == "transcript":
                        logger.info(f"Transcript: {data.get('text')}")
                    
                    elif msg_type == "response":
                        logger.info(f"Response: {data.get('text', '')[:100]}...")
                    
                    elif msg_type == "audio_start":
                        response_audio.clear()
                        audio_format = {
                            "sample_rate": data.get("sample_rate", 22050),
                            "channels": data.get("channels", 1),
                            "sample_width": data.get("sample_width", 2),
                        }
                    
                    elif msg_type == "audio_end":
                        if response_audio:
                            logger.info(f"Playing {len(response_audio)} bytes")
                            self.player.play(bytes(response_audio), **audio_format)
                            self._play_sound("done")
                        response_audio.clear()
                    
                    elif msg_type == "done":
                        logger.debug(f"Request completed in {data.get('duration', 0):.2f}s")
                    
                    elif msg_type == "error":
                        logger.error(f"Server error: {data.get('message')}")
                    
                    elif msg_type == "ping":
                        await ws.send_json({"type": "pong"})
                
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    response_audio.extend(msg.data)
                
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.warning("Server closed connection")
                    break
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receiver error: {e}")
                raise

    async def _receive_only_mode(self):
        """Run in receive-only mode - just stay connected and play incoming audio."""
        import aiohttp
        
        logger.info("Running in receive-only mode (no microphone)")
        logger.info("Waiting for audio from server...")
        
        ws_url = f"{self.config.server_url}/voice?room={self.config.room}&device={self.config.device}"
        
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url, timeout=None) as ws:
                        msg = await ws.receive_json()
                        logger.info(f"Connected to server: {msg.get('satellite_id')}")
                        
                        response_audio = bytearray()
                        audio_format = {"sample_rate": 22050, "channels": 1, "sample_width": 2}
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                msg_type = data.get("type")
                                
                                if msg_type == "audio_start":
                                    response_audio.clear()
                                    audio_format = {
                                        "sample_rate": data.get("sample_rate", 22050),
                                        "channels": data.get("channels", 1),
                                        "sample_width": data.get("sample_width", 2),
                                    }
                                    logger.debug(f"Receiving audio: {audio_format}")
                                
                                elif msg_type == "audio_end":
                                    if response_audio:
                                        logger.info(f"Playing {len(response_audio)} bytes")
                                        self.player.play(bytes(response_audio), **audio_format)
                                    response_audio.clear()
                                
                                elif msg_type == "ping":
                                    await ws.send_json({"type": "pong"})
                            
                            elif msg.type == aiohttp.WSMsgType.BINARY:
                                response_audio.extend(msg.data)
                            
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                logger.warning("Connection closed")
                                break
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection error: {e}")
                await asyncio.sleep(5)  # Reconnect delay
        
        self.cleanup()
    
    async def _process_with_server(self, audio: bytes):
        """Send audio to server and play response."""
        import aiohttp
        
        ws_url = f"{self.config.server_url}/voice?room={self.config.room}&device={self.config.device}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url, timeout=60) as ws:
                    # Wait for connected
                    msg = await ws.receive_json()
                    logger.debug(f"Connected: {msg}")
                    
                    # Send config
                    await ws.send_json({
                        "type": "config",
                        "sample_rate": self.config.sample_rate,
                    })
                    
                    # Send audio
                    chunk_size = 4096
                    for i in range(0, len(audio), chunk_size):
                        await ws.send_bytes(audio[i:i + chunk_size])
                    
                    # Signal end
                    await ws.send_json({"type": "end"})
                    logger.debug("Audio sent, waiting for response...")
                    
                    # Receive response
                    response_audio = bytearray()
                    audio_format = {"sample_rate": 22050, "channels": 1, "sample_width": 2}
                    
                    while True:
                        msg = await asyncio.wait_for(ws.receive(), timeout=60)
                        
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            msg_type = data.get("type")
                            
                            if msg_type == "transcript":
                                logger.info(f"Transcript: {data.get('text')}")
                            
                            elif msg_type == "response":
                                logger.info(f"Response: {data.get('text', '')[:100]}...")
                            
                            elif msg_type == "audio_start":
                                audio_format = {
                                    "sample_rate": data.get("sample_rate", 22050),
                                    "channels": data.get("channels", 1),
                                    "sample_width": data.get("sample_width", 2),
                                }
                            
                            elif msg_type == "audio_end":
                                # Play response
                                if response_audio:
                                    logger.debug(f"Playing {len(response_audio)} bytes")
                                    self.player.play(
                                        bytes(response_audio),
                                        **audio_format,
                                    )
                            
                            elif msg_type == "done":
                                logger.debug(f"Done in {data.get('duration', 0):.2f}s")
                                break
                            
                            elif msg_type == "error":
                                logger.error(f"Server error: {data.get('message')}")
                                break
                        
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            response_audio.extend(msg.data)
                        
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break
        
        except asyncio.TimeoutError:
            logger.error("Server timeout")
        except Exception as e:
            logger.error(f"Server communication error: {e}")
    
    def stop(self):
        """Stop the satellite."""
        self._running = False
    
    def cleanup(self):
        """Cleanup resources."""
        if self.wake_word:
            self.wake_word.cleanup()
        self.recorder.cleanup()
        self.player.cleanup()


def main():
    """Entry point."""
    import signal
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    config = Config.from_env()
    satellite = VoiceSatellite(config)
    
    # Handle signals
    loop = asyncio.new_event_loop()
    
    def signal_handler():
        logger.info("Shutting down...")
        satellite.stop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        loop.run_until_complete(satellite.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
