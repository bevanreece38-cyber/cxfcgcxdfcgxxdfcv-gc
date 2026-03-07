"""
VideoStream — захват видео с цифровой камеры (Radxa Rock 5B).

Поддерживаемые режимы:
  1. GStreamer pipeline (VIDEO_USE_GSTREAMER=True):
       v4l2src → videoscale → MJPEG декодирование → BGR
       Рекомендуется для низкой задержки на RK3588.

  2. OpenCV VideoCapture (fallback):
       Прямой захват через V4L2.

Параметры из config.py:
  VIDEO_SOURCE_INDEX  = 1          (/dev/video1)
  VIDEO_PIXEL_FORMAT  = "MJPEG"    (или "YUYV")
  VIDEO_USE_GSTREAMER = True
  FRAME_WIDTH=640, FRAME_HEIGHT=480, MAX_FPS=30
"""

import cv2
import threading
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Нон-блокирующий захват видео в фоновом потоке.

    Использование:
        vs = VideoStream(src=1, width=640, height=480, fps=30).start()
        ret, frame = vs.read()
        vs.stop()
    """

    def __init__(
        self,
        src:            int  = 1,
        width:          int  = 640,
        height:         int  = 480,
        fps:            int  = 30,
        use_gstreamer:  bool = True,
        pixel_format:   str  = "MJPEG",
        device_path:    str  = "/dev/video1",
    ):
        self.src           = src
        self.width         = width
        self.height        = height
        self.fps           = fps
        self.use_gstreamer = use_gstreamer
        self.pixel_format  = pixel_format
        self.device_path   = device_path

        self._cap:    Optional[cv2.VideoCapture] = None
        self._frame:  Optional[object]            = None
        self._ret:    bool                        = False
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> 'VideoStream':
        self._cap = self._open_capture()
        if self._cap is None or not self._cap.isOpened():
            logger.error("VideoStream: не удалось открыть камеру!")
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="video-capture",
            daemon=True,
        )
        self._thread.start()
        # Дать потоку захватить первый кадр
        time.sleep(0.2)
        return self

    def read(self) -> Tuple[bool, Optional[object]]:
        """Вернуть последний кадр (нон-блокирующий)."""
        with self._lock:
            return self._ret, self._frame

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("VideoStream stopped")

    # ------------------------------------------------------------------
    #  Внутренние
    # ------------------------------------------------------------------

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Попытаться открыть через GStreamer, потом fallback OpenCV."""
        if self.use_gstreamer:
            cap = self._try_gstreamer()
            if cap is not None and cap.isOpened():
                logger.info("VideoStream: GStreamer pipeline OK")
                return cap
            logger.warning("VideoStream: GStreamer failed → OpenCV fallback")

        return self._try_opencv()

    def _try_gstreamer(self) -> Optional[cv2.VideoCapture]:
        """GStreamer pipeline для RK3588."""
        fmt = self.pixel_format.upper()
        if fmt == "MJPEG":
            pipeline = (
                f"v4l2src device={self.device_path} "
                f"! image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 "
                f"! jpegdec "
                f"! videoconvert "
                f"! video/x-raw,format=BGR "
                f"! appsink drop=1"
            )
        else:
            # YUYV
            pipeline = (
                f"v4l2src device={self.device_path} "
                f"! video/x-raw,format=YUY2,width={self.width},"
                f"height={self.height},framerate={self.fps}/1 "
                f"! videoconvert "
                f"! video/x-raw,format=BGR "
                f"! appsink drop=1"
            )
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            return cap if cap.isOpened() else None
        except Exception as e:
            logger.warning(f"GStreamer pipeline error: {e}")
            return None

    def _try_opencv(self) -> Optional[cv2.VideoCapture]:
        """Прямой захват через V4L2."""
        try:
            cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS,          self.fps)
            if self.pixel_format.upper() == "MJPEG":
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*'MJPG'))
            if cap.isOpened():
                logger.info("VideoStream: OpenCV V4L2 OK")
                return cap
        except Exception as e:
            logger.error(f"OpenCV capture error: {e}")
        return None

    def _reader_loop(self):
        """Фоновый поток захвата кадров."""
        logger.debug("VideoStream reader started")
        while not self._stop.is_set():
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                with self._lock:
                    self._ret   = ret
                    self._frame = frame
            else:
                time.sleep(0.1)
        logger.debug("VideoStream reader stopped")