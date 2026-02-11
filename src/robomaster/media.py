# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from . import conn
from . import logger
import threading
import queue
import subprocess
import shutil
import select
import numpy
import cv2

def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg executable not found on PATH. "
            "Install it with: brew install ffmpeg (macOS) / "
            "sudo apt install ffmpeg (Linux) / choco install ffmpeg (Windows)"
        )


class FFmpegH264Decoder:

    def __init__(self, width=1280, height=720):
        _check_ffmpeg()
        self._width = width
        self._height = height
        self._process = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel", "error",
                "-f", "h264",
                "-i", "pipe:0",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-an",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 * 1024 * 1024,
        )

    def decode(self, data):
        frames = []
        frame_size = self._width * self._height * 3

        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()

            try:
                self._process.stdin.write(data)
                self._process.stdin.flush()
            except BrokenPipeError:
                logger.warning("FFmpegH264Decoder: broken pipe, restarting")
                self.close()
                self._start()
                return frames

        while True:
            raw = self._read_exact(frame_size, timeout=0.03)
            if raw is None:
                break
            frame = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(
                (self._height, self._width, 3)
            )
            frames.append(frame)

        return frames

    def _read_exact(self, n, timeout=0.03):
        try:
            ready, _, _ = select.select([self._process.stdout], [], [], timeout)
            if not ready:
                return None
            data = self._process.stdout.read(n)
            if len(data) != n:
                return None
            return data
        except Exception:
            return None

    def close(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                try:
                    self._process.stdin.close()
                    self._process.terminate()
                    self._process.wait(timeout=3)
                except Exception:
                    self._process.kill()
                finally:
                    self._process = None


class LiveView(object):

    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720

    def __init__(self, robot, width=None, height=None):
        self._robot = robot

        self._width = width or self.DEFAULT_WIDTH
        self._height = height or self.DEFAULT_HEIGHT

        self._video_stream_conn = conn.StreamConnection()
        self._video_decoder = None
        self._video_decoder_thread = None
        self._video_display_thread = None
        self._video_frame_queue = queue.Queue(64)
        self._video_streaming = False
        self._displaying = False
        self._video_frame_count = 0

        self._audio_stream_conn = conn.StreamConnection()
        self._audio_decoder = None
        self._audio_decoder_thread = None
        self._audio_playing_thread = None
        self._audio_frame_queue = queue.Queue(32)
        self._audio_streaming = False
        self._playing = False
        self._audio_frame_count = 0

    def __del__(self):
        self.stop()

    def stop(self):
        if self._video_streaming:
            self.stop_video_stream()
        if self._audio_streaming:
            self.stop_audio_stream()

    def start_video_stream(self, display=True, addr=None, ip_proto="tcp"):
        try:
            logger.info(
                "Liveview: try to connect addr {0}, proto={1}".format(addr, ip_proto)
            )
            self._video_stream_conn.connect(addr, ip_proto)
            self._video_decoder = FFmpegH264Decoder(
                width=self._width, height=self._height
            )
            self._video_streaming = True
            self._video_decoder_thread = threading.Thread(
                target=self._video_decoder_task
            )
            self._video_decoder_thread.start()
            if display:
                self._video_display_thread = threading.Thread(
                    target=self._video_display_task
                )
                self._video_display_thread.start()
        except Exception as e:
            logger.error(
                "Liveview: start_video_stream, exception {0}".format(e)
            )
            return False
        return True

    def stop_video_stream(self):
        try:
            self._video_streaming = False
            self._displaying = False
            if self._video_stream_conn:
                self._video_stream_conn.disconnect()
            if self._video_display_thread:
                self._video_frame_queue.put(None)
                self._video_display_thread.join()
            if self._video_decoder_thread:
                self._video_decoder_thread.join()
            if self._video_decoder:
                self._video_decoder.close()
                self._video_decoder = None
            self._video_frame_queue.queue.clear()
        except Exception as e:
            logger.error("LiveView: disconnect exception {0}".format(e))
            return False
        logger.info("LiveView: stop_video_stream stopped.")
        return True

    def read_video_frame(self, timeout=3, strategy="pipeline"):
        if strategy == "pipeline":
            return self._video_frame_queue.get(timeout=timeout)
        elif strategy == "newest":
            while self._video_frame_queue.qsize() > 1:
                self._video_frame_queue.get(timeout=timeout)
            return self._video_frame_queue.get(timeout=timeout)
        else:
            logger.warning(
                "LiveView: read_video_frame, unsupported strategy:{0}".format(strategy)
            )
            return None

    def _video_decoder_task(self):
        self._video_streaming = True
        logger.info("Liveview: _video_decoder_task, started!")
        while self._video_streaming:
            buf = self._video_stream_conn.read_buf()
            if not self._video_streaming:
                break
            if buf:
                frames = self._video_decoder.decode(buf)
                for frame in frames:
                    try:
                        self._video_frame_count += 1
                        if self._video_frame_count % 30 == 1:
                            logger.info(
                                "LiveView: video_decoder_task, get frame {0}.".format(
                                    self._video_frame_count
                                )
                            )
                        self._video_frame_queue.put(frame, timeout=2)
                    except Exception as e:
                        logger.warning(
                            "LiveView: _video_decoder_task, decoder queue is full, e {}.".format(
                                e
                            )
                        )
                        continue
        logger.info("LiveView: _video_decoder_task, quit.")

    def _video_display_task(self, name="RoboMaster LiveView"):
        self._displaying = True
        logger.info("Liveview: _video_display_task, started!")
        while self._displaying and self._video_streaming:
            try:
                frame = self._video_frame_queue.get()
                if frame is None:
                    break
            except Exception as e:
                logger.warning(
                    "LiveView: display_task, video_frame_queue is empty, e {0}".format(
                        e
                    )
                )
                continue
            cv2.imshow(name, frame)
            cv2.waitKey(1)
        logger.info("LiveView: _video_display_task, quit.")

    def _ensure_audio_decoder(self):
        if self._audio_decoder is None:
            import opuslib
            self._audio_decoder = opuslib.Decoder(48000, 1)

    def read_audio_frame(self, timeout=1):
        return self._audio_frame_queue.get(timeout=timeout)

    def start_audio_stream(self, addr=None, ip_proto="tcp"):
        try:
            logger.info(
                "LiveView: try to connect addr:{0}, ip_proto:{1}".format(
                    addr, ip_proto
                )
            )
            self._audio_stream_conn.connect(addr, ip_proto)
            self._ensure_audio_decoder()
            self._audio_decoder_thread = threading.Thread(
                target=self._audio_decoder_task
            )
            self._audio_decoder_thread.start()
        except Exception as e:
            logger.error(
                "LiveView: start_audio_stream, exception {0}".format(e)
            )
            return False
        return True

    def stop_audio_stream(self):
        try:
            logger.info("LiveView: stop_audio_stream stopping...")
            self._audio_streaming = False
            if self._audio_decoder_thread:
                self._audio_decoder_thread.join()
            self._audio_stream_conn.disconnect()
            self._audio_frame_queue.queue.clear()
        except Exception as e:
            logger.error("LiveView: disconnect exception {0}".format(e))
            return False
        logger.info("LiveView: stop_audio_stream stopped.")
        return True

    def _audio_decoder_task(self):
        self._audio_streaming = True
        while self._audio_streaming:
            buf = self._audio_stream_conn.read_buf()
            if buf:
                frame = self._audio_decoder.decode(buf, 960)
                if frame:
                    try:
                        self._audio_frame_count += 1
                        if self._audio_frame_count % 100 == 1:
                            logger.info(
                                "LiveView: audio_decoder_task, get frame {0}.".format(
                                    self._audio_frame_count
                                )
                            )
                        self._audio_frame_queue.put(frame, timeout=1)
                    except Exception as e:
                        if not self._audio_streaming:
                            break
                        logger.warning(
                            "LiveView: _audio_decoder_task, audio_frame_queue full, e {0}!".format(
                                e
                            )
                        )
                        continue
        logger.info("LiveView: _audio_decoder_task, quit.")