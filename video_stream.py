"""
VideoStream — DroneEngage-стиль захват видео.

Принципы DroneEngage:
  - UVC MJPEG: v4l2src io-mode=2 (mmap) → jpegdec → BGR appsink
  - sync=false, drop=true, max-buffers=1 → нулевая задержка
  - Фоновый поток непрерывно перезаписывает последний кадр
  - Автовыбор pipeline: GStreamer MJPEG → GStreamer raw → OpenCV V4L2 MJPG → OpenCV fallback

https://github.com/DroneEngage/droneengage_camera
"""

import cv2
import threading
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Имена pipeline для логирования
PIPELINE_GST_MJPEG   = "GStreamer UVC MJPEG (io-mode=2, drop=true)"
PIPELINE_GST_RAW     = "GStreamer RAW fallback"
PIPELINE_CV_V4L2     = "OpenCV CAP_V4L2 MJPG"
PIPELINE_CV_FALLBACK = "OpenCV fallback"


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

        self._cap:           Optional[cv2.VideoCapture] = None
        self._pipeline_name: str                        = ""
        self._frame:         Optional[object]           = None
        self._ret:           bool                       = False
        self._lock           = threading.Lock()
        self._stop           = threading.Event()
        self._thread:        Optional[threading.Thread] = None

    def start(self) -> 'VideoStream':
        self._cap, self._pipeline_name = self._open_capture()
        logger.info(f"VideoStream: {self._pipeline_name} → {self.device_path}")
        if self._cap is None or not self._cap.isOpened():
            logger.error("VideoStream: камера не открылась!")
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="video-capture",
            daemon=True,
        )
        self._thread.start()
        # Дать потоку захватить первый кадр
        time.sleep(0.3)
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

    def _open_capture(self) -> Tuple[Optional[cv2.VideoCapture], str]:
        """Попробовать pipelines по приоритету DroneEngage."""
        if self.use_gstreamer:
            # 1. GStreamer UVC MJPEG (DroneEngage primary pipeline)
            pipe1 = (
                f"v4l2src device={self.device_path} io-mode=2 "
                f"! image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 "
                f"! jpegdec ! videoconvert ! video/x-raw,format=BGR "
                f"! appsink name=sink emit-signals=true sync=false drop=true max-buffers=1"
            )
            try:
                cap = cv2.VideoCapture(pipe1, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap, PIPELINE_GST_MJPEG
            except Exception as e:
                logger.warning(f"GStreamer MJPEG pipeline error: {e}")

            # 2. GStreamer без io-mode (fallback для некоторых камер)
            pipe2 = (
                f"v4l2src device={self.device_path} "
                f"! image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 "
                f"! queue leaky=downstream max-size-buffers=1 "
                f"! jpegdec ! videoconvert ! video/x-raw,format=BGR "
                f"! appsink sync=false drop=true max-buffers=1"
            )
            try:
                cap = cv2.VideoCapture(pipe2, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap, PIPELINE_GST_RAW
            except Exception as e:
                logger.warning(f"GStreamer RAW pipeline error: {e}")

            logger.warning("VideoStream: GStreamer failed → OpenCV fallback")

        # 3. OpenCV CAP_V4L2 с MJPG + минимальный буфер
        try:
            cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS,          self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # КРИТИЧНО: минимум задержки
                return cap, PIPELINE_CV_V4L2
        except Exception as e:
            logger.warning(f"OpenCV CAP_V4L2 error: {e}")

        # 4. Последний fallback
        cap = cv2.VideoCapture(self.src)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            logger.warning(f"OpenCV fallback BUFFERSIZE error: {e}")
        return cap, PIPELINE_CV_FALLBACK

    def _reader_loop(self):
        """Непрерывно читать последний кадр (DroneEngage стиль)."""
        logger.debug("VideoStream reader started")
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._ret   = ret
                    self._frame = frame  # всегда только последний кадр
            else:
                time.sleep(0.005)
        logger.debug("VideoStream reader stopped")