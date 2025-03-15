#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import uuid
from typing import AsyncGenerator, Literal, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import TTSService
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language

try:
    import ormsgpack
    import websockets
    import asyncio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Fish Audio, you need to `pip install pipecat-ai[fish]`. Also, set `FISH_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

# FishAudio supports various output formats
FishAudioOutputFormat = Literal["opus", "mp3", "pcm", "wav"]


class FishAudioTTSService(TTSService, WebsocketService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        latency: Optional[str] = "normal"  # "normal" or "balanced"
        prosody_speed: Optional[float] = 1.0  # Speech speed (0.5-2.0)
        prosody_volume: Optional[int] = 0  # Volume adjustment in dB

    def __init__(
        self,
        *,
        api_key: str,
        model: str,  # This is the reference_id
        output_format: FishAudioOutputFormat = "pcm",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._base_url = "wss://api.fish.audio/v1/tts/live"
        self._websocket = None
        self._receive_task = None
        self._request_id = None
        self._started = False
        self._connection_state = "initializing"  # Track connection state: "initializing", "connected", "disconnected", "failed"
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        
        # Create an audio queue like Azure uses
        self._audio_queue = asyncio.Queue()
        
        self._settings = {
            "sample_rate": sample_rate,
            "latency": params.latency,
            "format": output_format,
            "prosody": {
                "speed": params.prosody_speed,
                "volume": params.prosody_volume,
            },
            "reference_id": model,
        }

        self.set_model_name(model)
        
        # Start connection during initialization
        # This follows Azure's pattern of setting up resources during initialization
        asyncio.create_task(self._initialize_connection())
        
    async def _initialize_connection(self):
        """Initialize the connection during startup (follows Azure pattern)"""
        try:
            await self._connect_websocket()
            if self._websocket and not self._websocket.closed:
                self._connection_state = "connected"
                self._receive_task = self.create_task(self._receive_task_handler(self.push_error))
                logger.info("Fish Audio service initialized successfully")
            else:
                self._connection_state = "failed"
                logger.error("Failed to initialize Fish Audio service")
        except Exception as e:
            logger.error(f"Error during Fish Audio initialization: {e}")
            self._connection_state = "failed"

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        self._settings["reference_id"] = model
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Only connect if not already connected
        if self._connection_state != "connected":
            await self._ensure_connection()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        # Don't disconnect on stop - maintain connection like Azure does
        # Just reset state variables
        self._request_id = None
        self._started = False

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        # Don't disconnect on cancel - maintain connection like Azure does
        # Just reset state variables
        self._request_id = None
        self._started = False
        
    async def _ensure_connection(self):
        """Ensure we have a valid connection, following Azure's pattern of maintaining connections"""
        if self._connection_state == "connected" and self._websocket and not self._websocket.closed:
            return True
            
        if self._connection_state == "initializing":
            # Wait for initialization to complete
            for _ in range(10):  # Wait up to 1 second
                await asyncio.sleep(0.1)
                if self._connection_state in ["connected", "failed"]:
                    break
        
        # If still not connected, try to connect
        if self._connection_state != "connected" or not self._websocket or self._websocket.closed:
            self._connection_state = "initializing"
            await self._connect_websocket()
            if self._websocket and not self._websocket.closed:
                self._connection_state = "connected"
                if not self._receive_task:
                    self._receive_task = self.create_task(self._receive_task_handler(self.push_error))
                return True
            else:
                self._connection_state = "failed"
                return False
        
        return True

    async def _connect_websocket(self):
        try:
            # Use a semaphore to prevent too many simultaneous connection attempts
            if self._reconnect_attempts > 0:
                backoff = min(1.0 * self._reconnect_attempts, 5.0)
                await asyncio.sleep(backoff)
                
            self._websocket = await websockets.connect(
                self._base_url, 
                extra_headers={"Authorization": f"Bearer {self._api_key}"},
                close_timeout=10.0,
            )

            # Send initial start message with ormsgpack
            start_message = {"event": "start", "request": {"text": "", **self._settings}}
            await self._websocket.send(ormsgpack.packb(start_message))
            self._reconnect_attempts = 0  # Reset on successful connection
        except Exception as e:
            logger.error(f"Fish Audio initialization error: {e}")
            self._websocket = None
            self._reconnect_attempts += 1
            if "429" in str(e):
                logger.warning(f"Rate limit (429) detected, will use longer backoff on next attempt")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket and not self._websocket.closed:
                try:
                    stop_message = {"event": "stop"}
                    await self._websocket.send(ormsgpack.packb(stop_message))
                    await self._websocket.close()
                except Exception as e:
                    # Just log critical closure errors
                    if not "1000 (OK)" in str(e):
                        logger.warning(f"Error during websocket closure: {e}")
                
            self._websocket = None
            self._request_id = None
            self._started = False
            self._connection_state = "disconnected"
        except Exception as e:
            if not "1000 (OK)" in str(e):
                logger.error(f"Error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket and not self._websocket.closed:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        try:
            async for message in self._get_websocket():
                try:
                    if isinstance(message, bytes):
                        msg = ormsgpack.unpackb(message)
                        if isinstance(msg, dict):
                            event = msg.get("event")
                            if event == "audio":
                                audio_data = msg.get("audio")
                                if audio_data and len(audio_data) > 1024:
                                    await self._audio_queue.put(audio_data)
                                    await self.stop_ttfb_metrics()
                            elif event == "end":
                                await self._audio_queue.put(None)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"WebSocket connection closed: {e}")
            self._connection_state = "disconnected"
            await self._audio_queue.put(None)
        except Exception as e:
            logger.error(f"Error in receive messages loop: {e}")
            self._connection_state = "failed"
            await self._audio_queue.put(None)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSSpeakFrame):
            await self.pause_processing_frames()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._request_id:
            await self.pause_processing_frames()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.resume_processing_frames()

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._request_id = None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.info(f"Generating TTS for text length: {len(text)}")
        try:
            connected = await self._ensure_connection()
            if not connected:
                yield ErrorFrame(f"Unable to connect to Fish Audio after {self._reconnect_attempts} attempts")
                return

            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()
            
            self._request_id = str(uuid.uuid4())

            try:
                # Send text message
                text_message = {"event": "text", "text": text}
                await self._get_websocket().send(ormsgpack.packb(text_message))
                
                # Send flush event
                flush_message = {"event": "flush"}
                await self._get_websocket().send(ormsgpack.packb(flush_message))
                
                audio_complete = False
                chunks_received = 0
                wait_time = 5.0
                
                while not audio_complete:
                    try:
                        audio_data = await asyncio.wait_for(self._audio_queue.get(), timeout=wait_time)
                        
                        if audio_data is None:
                            audio_complete = True
                        else:
                            chunks_received += 1
                            wait_time = 2.0
                            
                            await self.stop_ttfb_metrics()
                            yield TTSAudioRawFrame(
                                audio=audio_data,
                                sample_rate=self._settings["sample_rate"],
                                num_channels=1,
                            )
                    except asyncio.TimeoutError:
                        if chunks_received == 0:
                            logger.warning("Timeout waiting for first audio chunk")
                            try:
                                await self._get_websocket().send(ormsgpack.packb(flush_message))
                                wait_time = 8.0
                            except Exception as e:
                                logger.error(f"Error resending flush message: {e}")
                                audio_complete = True
                        else:
                            audio_complete = True
                
            except Exception as e:
                logger.error(f"Error in TTS processing: {e}")
                if "429" in str(e):
                    logger.warning("Rate limit (429) detected during TTS request")
                    
                if self._websocket and self._websocket.closed:
                    self._connection_state = "disconnected"
                    connected = await self._ensure_connection()
                    if not connected:
                        yield ErrorFrame(f"Connection lost and reconnection failed")

            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"Error generating TTS: {e}")
            yield ErrorFrame(f"Error: {str(e)}")
            yield TTSStoppedFrame()

    # Like Azure, provide a method to keep the connection alive
    async def keep_alive(self):
        """Send a ping to keep the connection alive without full reconnection"""
        if not self._websocket or self._websocket.closed:
            return await self._ensure_connection()
            
        try:
            ping_message = {"event": "ping"}
            await self._websocket.send(ormsgpack.packb(ping_message))
            return True
        except Exception as e:
            logger.warning(f"Keep-alive ping failed: {e}")
            self._connection_state = "disconnected"
            return False
