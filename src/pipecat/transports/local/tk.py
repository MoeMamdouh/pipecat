#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import InputAudioRawFrame, OutputImageRawFrame, StartFrame
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    import pyaudio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use local audio, you need to `pip install pipecat-ai[local]`. On MacOS, you also need to `brew install portaudio`."
    )
    raise Exception(f"Missing module: {e}")

try:
    import tkinter as tk
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("tkinter missing. Try `apt install python3-tk` or `brew install python-tk@3.10`.")
    raise Exception(f"Missing module: {e}")


class TkTransportParams(TransportParams):
    audio_input_device_index: Optional[int] = None
    audio_output_device_index: Optional[int] = None


class TkInputTransport(BaseInputTransport):
    _params: TkTransportParams

    def __init__(self, py_audio: pyaudio.PyAudio, params: TkTransportParams):
        super().__init__(params)
        self._py_audio = py_audio
        self._in_stream = None
        self._sample_rate = 0

    async def start(self, frame: StartFrame):
        await super().start(frame)

        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        num_frames = int(self._sample_rate / 100) * 2  # 20ms of audio

        self._in_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_in_channels,
            rate=self._sample_rate,
            frames_per_buffer=num_frames,
            stream_callback=self._audio_in_callback,
            input=True,
            input_device_index=self._params.audio_input_device_index,
        )
        self._in_stream.start_stream()

    async def cleanup(self):
        await super().cleanup()
        if self._in_stream:
            self._in_stream.stop_stream()
            self._in_stream.close()

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        frame = InputAudioRawFrame(
            audio=in_data,
            sample_rate=self._sample_rate,
            num_channels=self._params.audio_in_channels,
        )

        asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), self.get_event_loop())

        return (None, pyaudio.paContinue)


class TkOutputTransport(BaseOutputTransport):
    _params: TkTransportParams

    def __init__(self, tk_root: tk.Tk, py_audio: pyaudio.PyAudio, params: TransportParams):
        super().__init__(params)
        self._py_audio = py_audio
        self._out_stream = None
        self._sample_rate = 0

        # We only write audio frames from a single task, so only one thread
        # should be necessary.
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Start with a neutral gray background.
        array = np.ones((1024, 1024, 3)) * 128
        data = f"P5 {1024} {1024} 255 ".encode() + array.astype(np.uint8).tobytes()
        photo = tk.PhotoImage(width=1024, height=1024, data=data, format="PPM")
        self._image_label = tk.Label(tk_root, image=photo)
        self._image_label.pack()

    async def start(self, frame: StartFrame):
        await super().start(frame)

        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate

        self._out_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=self._sample_rate,
            output=True,
            output_device_index=self._params.audio_output_device_index,
        )
        self._out_stream.start_stream()

    async def cleanup(self):
        await super().cleanup()
        if self._out_stream:
            self._out_stream.stop_stream()
            self._out_stream.close()

    async def write_raw_audio_frames(self, frames: bytes):
        if self._out_stream:
            await self.get_event_loop().run_in_executor(
                self._executor, self._out_stream.write, frames
            )

    async def write_frame_to_camera(self, frame: OutputImageRawFrame):
        self.get_event_loop().call_soon(self._write_frame_to_tk, frame)

    def _write_frame_to_tk(self, frame: OutputImageRawFrame):
        width = frame.size[0]
        height = frame.size[1]
        data = f"P6 {width} {height} 255 ".encode() + frame.image
        photo = tk.PhotoImage(width=width, height=height, data=data, format="PPM")
        self._image_label.config(image=photo)

        # This holds a reference to the photo, preventing it from being garbage
        # collected.
        self._image_label.image = photo


class TkLocalTransport(BaseTransport):
    def __init__(self, tk_root: tk.Tk, params: TransportParams):
        super().__init__()
        self._tk_root = tk_root
        self._params = params
        self._pyaudio = pyaudio.PyAudio()

        self._input: Optional[TkInputTransport] = None
        self._output: Optional[TkOutputTransport] = None

    #
    # BaseTransport
    #

    def input(self) -> TkInputTransport:
        if not self._input:
            self._input = TkInputTransport(self._pyaudio, self._params)
        return self._input

    def output(self) -> TkOutputTransport:
        if not self._output:
            self._output = TkOutputTransport(self._tk_root, self._pyaudio, self._params)
        return self._output
