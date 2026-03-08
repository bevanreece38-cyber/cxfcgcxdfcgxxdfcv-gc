"""
GStreamerOutput — H.264/RTP/UDP выход для QGroundControl / VLC.

Приоритет энкодеров:
  1. mpph264enc  — Rockchip MPP (RK3588, аппаратный)
  2. nvh264enc   — NVIDIA NVENC
  3. vaapih264enc — Intel/AMD VA-API
  4. x264enc     — программный fallback

Параметры из config.py:
  GSTREAMER_HOST, GSTREAMER_PORT, GSTREAMER_WIDTH, GSTREAMER_HEIGHT
  GSTREAMER_FPS, GSTREAMER_BITRATE, GSTREAMER_ENABLE_HW

Приём на земной станции:
  VLC:    rtp://@:5600
  FFplay: ffplay rtp://@:5600
  QGC:    UDP порт 5600 (автообнаружение)
"""

import cv2
import numpy as np
import queue
import shutil
import subprocess
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GStreamerOutput:

    def __init__(
        self,
        host:           str  = "192.168.1.100",
        port:           int  = 5600,
        width:          int  = 480,
        height:         int  = 360,
        fps:            int  = 25,
        bitrate:        int  = 1000,
        allow_hardware: bool = True,
    ):
        self.host    = host
        self.port    = port
        self.width   = width
        self.height  = height
        self.fps     = fps
        self.bitrate = bitrate

        self._encoder    = self._detect_encoder(allow_hardware)
        self._pipeline   = self._build_pipeline()
        self._out: Optional[cv2.VideoWriter] = None
        self._queue: queue.Queue = queue.Queue(maxsize=1)  # maxsize=1: всегда отправляем свежий кадр
        self._stop_event = threading.Event()
        self._thread:    Optional[threading.Thread] = None
        self._drops      = 0

    # ------------------------------------------------------------------

    def start(self) -> bool:
        try:
            logger.info(
                f"GStreamer output: encoder={self._encoder} "
                f"target={self.host}:{self.port} "
                f"{self.width}x{self.height}@{self.fps}fps {self.bitrate}kbps"
            )
            self._out = cv2.VideoWriter(
                self._pipeline,
                cv2.CAP_GSTREAMER, 0,
                self.fps, (self.width, self.height), True,
            )
            if not self._out.isOpened():
                if self._encoder != "x264enc":
                    logger.warning(
                        f"{self._encoder} не запустился → fallback x264enc"
                    )
                    self._encoder  = "x264enc"
                    self._pipeline = self._build_pipeline()
                    self._out = cv2.VideoWriter(
                        self._pipeline,
                        cv2.CAP_GSTREAMER, 0,
                        self.fps, (self.width, self.height), True,
                    )
                if not self._out.isOpened():
                    logger.error("GStreamer output: не удалось открыть pipeline")
                    self._out = None
                    return False
        except Exception as e:
            logger.error(f"GStreamer output start error: {e}")
            self._out = None
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="gst-output",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"GStreamer output started ({self._encoder})")
        return True

    def send_frame(self, frame: np.ndarray):
        """Поставить кадр в очередь (нон-блокирующий). Дроп при переполнении."""
        if self._out is None:
            return
        try:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
        except Exception as e:
            logger.error(f"GStreamer frame prep error: {e}")
            return

        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._drops += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._out:
            self._out.release()
            self._out = None
        logger.info(f"GStreamer output stopped (drops={self._drops})")

    @property
    def is_active(self) -> bool:
        return self._out is not None and self._out.isOpened()

    # ------------------------------------------------------------------
    #  Внутренние
    # ------------------------------------------------------------------

    def _detect_encoder(self, allow_hardware: bool) -> str:
        if not allow_hardware:
            return "x264enc"
        gst_inspect = shutil.which("gst-inspect-1.0")
        if gst_inspect is None:
            return "x264enc"
        for enc in ("mpph264enc", "nvh264enc", "vaapih264enc"):
            try:
                r = subprocess.run(
                    [gst_inspect, enc],
                    capture_output=True, timeout=3,
                )
                if r.returncode == 0:
                    logger.info(f"GStreamer encoder: {enc} (hardware)")
                    return enc
            except Exception:
                continue
        logger.info("GStreamer encoder: x264enc (software fallback)")
        return "x264enc"

    def _build_pipeline(self) -> str:
        src = (
            f"appsrc ! "
            f"video/x-raw,format=BGR,width={self.width},height={self.height},"
            f"framerate={self.fps}/1 ! videoconvert"
        )
        enc = self._encoder
        if enc == "mpph264enc":
            encode = (
                f" ! mpph264enc bitrate={self.bitrate * 1000} header-mode=1"
                f" ! h264parse"
            )
        elif enc == "nvh264enc":
            encode = f" ! nvh264enc bitrate={self.bitrate} ! h264parse"
        elif enc == "vaapih264enc":
            encode = f" ! vaapih264enc bitrate={self.bitrate} ! h264parse"
        else:
            encode = (
                f" ! x264enc tune=zerolatency bitrate={self.bitrate}"
                f" key-int-max=20 speed-preset=ultrafast"
            )
        sink = (
            f" ! rtph264pay config-interval=1 pt=96"
            f" ! udpsink host={self.host} port={self.port} buffer-size=200000"
        )
        pipeline = src + encode + sink
        logger.debug(f"GStreamer pipeline: {pipeline}")
        return pipeline

    def _writer_loop(self):
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._out and self._out.isOpened():
                    self._out.write(frame)
            except Exception as e:
                logger.error(f"GStreamer write error: {e}")