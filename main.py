"""
Interceptor System — Radxa Rock 5B + SpeedyBee F405 + ArduPilot

Аппаратура:
  RadioMaster TX → ELRS 2.4GHz → Приёмник → SpeedyBee F405 (ArduPilot)
  SpeedyBee F405 ↔ Radxa Rock 5B (UART MAVLink /dev/ttyS2 115200 baud)
  Radxa Rock 5B ← Цифровая камера USB/MIPI → /dev/video1
  Radxa Rock 5B → Wi-Fi → Оператор (MJPEG :5000 / H.264 RTP :5600)

Логика:
  CH10 OFF: AltHold, оператор управляет пультом
            YOLO детекция → зелёные прямоугольники на видео для оператора
  CH10 ON:  take_control() → TrackerEngine.engage()
            Kalman predictive lead → PID yaw+throttle → pitch рампа → удар
            Потеря цели → release_control() (все каналы = 65535 passthrough)
  CH10 OFF: release_control() → управление возвращается ELRS пульту
  SAFETY:   release_control() + MAV_CMD_NAV_LAND

ИСПРАВЛЕНО:
  - RC_RELEASE = 65535 (passthrough), НЕ 0
  - TrackerEngine.step(yolo_outputs, frame) — правильный порядок аргументов
  - Плавная рампа throttle (нет рывков при захвате)
"""

import cv2
import numpy as np
import time
import sys
import signal
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

from config import (
    FRAME_WIDTH, FRAME_HEIGHT, MODEL_PATH,
    MAX_FPS, CONF_THRESHOLD, NMS_THRESHOLD, TARGET_CLASS_ID,
    HEADLESS_MODE,
    STREAM_PORT, STREAM_QUALITY, STREAM_WIDTH, STREAM_HEIGHT,
    VIDEO_SOURCE_INDEX, VIDEO_DEVICE_PATH,
    VIDEO_PIXEL_FORMAT, VIDEO_USE_GSTREAMER,
    GSTREAMER_ENABLED, GSTREAMER_HOST, GSTREAMER_PORT,
    GSTREAMER_WIDTH, GSTREAMER_HEIGHT, GSTREAMER_FPS,
    GSTREAMER_BITRATE, GSTREAMER_ENABLE_HW,
    RC_RELEASE,
)
from types_enum import SafetyStatus, TrackerState
from utils import setup_logger
from video_stream import VideoStream
from gstreamer_output import GStreamerOutput
from handler import MAVLinkHandler
from state import StateEstimator
from npu import NPUHandler
from flight_logger import FlightLogger
from safety import SafetyManager
from control_manager import ControlManager
from tracker_engine import TrackerEngine, TrackResult, _idle_result

try:
    from drone_model import post_process
except ImportError:
    post_process = None

logger = setup_logger(__name__)
MAX_MAVLINK_MSGS_PER_TICK = 50

# ========================
#  MJPEG СТРИМИНГ
# ========================
LATEST_JPEG: Optional[bytes] = None
LATEST_JPEG_LOCK = threading.Lock()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # не спамим в консоль

    def do_GET(self):
        if self.path in ('/', '/stream'):
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            while True:
                try:
                    with LATEST_JPEG_LOCK:
                        jpeg = LATEST_JPEG
                    if jpeg is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(
                            f'Content-Length: {len(jpeg)}\r\n'.encode())
                        self.wfile.write(b'\r\n')
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                    else:
                        time.sleep(0.05)
                        continue
                    time.sleep(1.0 / MAX_FPS)
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception:
                    break
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ========================
#  ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ========================
class InterceptorApp:

    def __init__(self):
        logger.info("=" * 60)
        logger.info("  INTERCEPTOR SYSTEM")
        logger.info("  Radxa Rock 5B | SpeedyBee F405 | ArduPilot")
        logger.info("  RadioMaster TX → ELRS 2.4GHz")
        logger.info("=" * 60)

        # MAVLink
        self.mav       = MAVLinkHandler()
        self.state_est = StateEstimator()
        self.safety    = SafetyManager(self.mav)

        # DroneEngage-стиль: управление RC через MAVLink override
        self.ctrl = ControlManager(self.mav)

        # PixEagle-стиль: трекинг (YOLO + CSRT + Kalman + PID)
        self.tracker = TrackerEngine()

        # NPU (RK3588 — все 3 ядра)
        self.npu = NPUHandler(MODEL_PATH)

        # Видеозахват (GStreamer MJPEG pipeline, низкая задержка)
        logger.info("Starting video capture (GStreamer MJPEG)...")
        self.video = VideoStream(
            src=VIDEO_SOURCE_INDEX,
            width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=MAX_FPS,
            use_gstreamer=VIDEO_USE_GSTREAMER,
            pixel_format=VIDEO_PIXEL_FORMAT,
            device_path=VIDEO_DEVICE_PATH,
        ).start()

        # GStreamer H.264/RTP/UDP выход (QGroundControl / VLC)
        self.gst_output: Optional[GStreamerOutput] = None
        if GSTREAMER_ENABLED:
            self.gst_output = GStreamerOutput(
                host=GSTREAMER_HOST, port=GSTREAMER_PORT,
                width=GSTREAMER_WIDTH, height=GSTREAMER_HEIGHT,
                fps=GSTREAMER_FPS, bitrate=GSTREAMER_BITRATE,
                allow_hardware=GSTREAMER_ENABLE_HW,
            )
            if self.gst_output.start():
                logger.info(f"H.264 RTP → udp://{GSTREAMER_HOST}:{GSTREAMER_PORT}")
            else:
                logger.warning("GStreamer output не запустился — только MJPEG")
                self.gst_output = None

        # Лог полётных данных
        self.flight_log = FlightLogger()

        # Последний результат тр��кера (для CSV лога)
        self._last_result: TrackResult = _idle_result()

        # MJPEG HTTP сервер (браузер / FPV монитор)
        self.server = ThreadedHTTPServer(('0.0.0.0', STREAM_PORT), StreamHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

        logger.info(f"MJPEG  → http://0.0.0.0:{STREAM_PORT}/stream")
        logger.info(f"Health → http://0.0.0.0:{STREAM_PORT}/health")
        logger.info("READY — CH10 (1800-2000) = ATTACK MODE")
        logger.info("=" * 60)

    # ================================================================
    #  ОСНОВНОЙ ЦИКЛ
    # ================================================================
    def run(self):
        try:
            while True:
                t0 = time.monotonic()

                # 1. Видеокадр (нон-блокирующий — фоновый поток)
                ret, frame = self.video.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                # 2. MAVLink телеметрия (с лимитом пакетов в тик)
                self.mav.ensure_connection()
                for _ in range(MAX_MAVLINK_MSGS_PER_TICK):
                    msg = self.mav.receive_message()
                    if msg is None:
                        break
                    self.state_est.update_from_message(msg)

                state = self.state_est.get_state()

                # 3. Безопасность — приоритет над всем
                safety_status = self.safety.check(
                    state, self.state_est.heartbeat_age
                )
                if safety_status not in (SafetyStatus.OK, SafetyStatus.WARNING):
                    if self.ctrl.is_controlling:
                        logger.critical("SAFETY → принудительный release_control()")
                        self.tracker.disengage()
                        self.ctrl.release_control()  # все каналы = 65535 passthrough
                    self.safety.execute_safety_action(safety_status, self.ctrl)
                    self._last_result = _idle_result()
                    self._push_frame(frame, "SAFETY", (0, 0, 255))
                    self._log(state, safety_status)
                    self._limit_fps(t0)
                    continue
                elif safety_status == SafetyStatus.WARNING:
                    # Предупреждение (нет FC или нет heartbeat но не армирован) — показываем видео с меткой
                    cv2.putText(frame, "NO FC / WARNING", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                    # НЕ делаем continue — продолжаем показывать видео

                # 4. NPU YOLO инференс (RK3588, ~15 мс)
                outputs_pp = self._run_inference(frame)

                # 5. Логика CH10 — атака или пассив
                if state.attack_switch:
                    self._attack_mode(frame, outputs_pp)
                else:
                    self._passive_mode(frame, outputs_pp)

                # 6. Лог
                self._log(state, safety_status)

                # 7. FPS лимитер
                self._limit_fps(t0)

        except KeyboardInterrupt:
            logger.info("Остановка по Ctrl+C")
        except Exception as e:
            logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: {e}", exc_info=True)
        finally:
            self._cleanup()

    # ================================================================
    #  РЕЖИМ АТАКИ (CH10 ON)
    # ================================================================
    def _attack_mode(self, frame: np.ndarray, outputs_pp):
        # DroneEngage: take_control при первом кадре атаки
        if not self.ctrl.is_controlling:
            if not self.ctrl.take_control():
                self._push_frame(frame, "NO MAVLINK", (0, 0, 255))
                return
            self.tracker.engage()

        # PixEagle: один шаг трекинга
        # ИСПРАВЛЕНО: передаём frame вторым аргументом
        result = self.tracker.step(outputs_pp, frame)
        self._last_result = result

        if result.state in (TrackerState.TRACKING,
                            TrackerState.DEAD_RECKON,
                            TrackerState.STRIKING):
            # Отправляем RC значения через ControlManager
            # rc_roll = RC_RELEASE = 65535 → Roll управляет оператор
            self.ctrl.set_channels(
                roll=result.rc_roll,          # 65535 = passthrough к ELRS
                pitch=result.rc_pitch,
                throttle=result.rc_throttle,
                yaw=result.rc_yaw,
            )
            self._draw_hud(frame, result)

        elif result.state == TrackerState.ACQUIRING:
            # Цель ищем: override активен, все каналы = passthrough
            self.ctrl.set_channels()  # все = RC_RELEASE = 65535
            self._push_frame(frame, "ACQUIRING...", (255, 165, 0))

        else:
            # LOST: цель потеряна → немедленно отдать управление оператору
            logger.warning("Цель LOST → release_control() → оператор")
            self.tracker.disengage()
            self.ctrl.release_control()  # all channels = 65535 passthrough
            self._push_frame(frame, "LOST — MANUAL", (0, 255, 255))

    # ================================================================
    #  ПАССИВНЫЙ РЕЖИМ (CH10 OFF)
    # ================================================================
    def _passive_mode(self, frame: np.ndarray, outputs_pp):
        # При переходе из атаки — немедленно release
        if self.ctrl.is_controlling:
            logger.info("CH10 OFF → release_control() → ELRS пульт")
            self.tracker.disengage()
            self.ctrl.release_control()  # все каналы = 65535 passthrough

        self._last_result = _idle_result()

        # Пассив: только YOLO детекция + визуализация для оператора
        if outputs_pp:
            boxes, classes, scores = outputs_pp
            for i, cls in enumerate(classes):
                if int(cls) == TARGET_CLASS_ID and float(scores[i]) > CONF_THRESHOLD:
                    x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{float(scores[i]):.2f}",
                        (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
            self._push_frame(frame, "SEARCH", (0, 255, 0))
        else:
            self._push_frame(frame, "IDLE", (128, 128, 128))

    # ================================================================
    #  HUD ВИЗУАЛИЗАЦИЯ (режим атаки)
    # ================================================================
    def _draw_hud(self, frame: np.ndarray, r: TrackResult):
        tx, ty   = int(r.target_x), int(r.target_y)
        lx, ly   = int(r.lead_x),   int(r.lead_y)    # точка упреждения
        cx, cy   = FRAME_WIDTH // 2, FRAME_HEIGHT // 2

        # Перекрестие центра кадра
        cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (200, 200, 200), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (200, 200, 200), 1)

        # Цвет по состоянию
        color = {
            TrackerState.TRACKING:    (0, 0, 255),
            TrackerState.STRIKING:    (0, 0, 255),
            TrackerState.DEAD_RECKON: (0, 255, 255),
        }.get(r.state, (255, 255, 0))

        # Текущая позиция цели (кружок)
        fill = -1 if r.state == TrackerState.STRIKING else 2
        cv2.circle(frame, (tx, ty), 14, color, fill)
        cv2.rectangle(frame, (tx - 22, ty - 22), (tx + 22, ty + 22), color, 1)

        # Точка упреждения (крестик) — куда наводимся
        cv2.drawMarker(frame, (lx, ly), (255, 100, 0),
                       cv2.MARKER_CROSS, 20, 2)

        # Линия от центра к точке упреждения
        cv2.line(frame, (cx, cy), (lx, ly), (255, 100, 0), 1)

        # Состояние
        cv2.putText(frame, r.state.value, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Прогресс-бар pitch рампы (DIVE)
        bar_w = int(r.ramp_progress * (FRAME_WIDTH - 20))
        cv2.rectangle(frame, (10, FRAME_HEIGHT - 18),
                      (10 + bar_w, FRAME_HEIGHT - 8), (0, 0, 255), -1)
        cv2.putText(frame, f"DIVE {r.ramp_progress * 100:.0f}%",
                    (10, FRAME_HEIGHT - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Прогресс-бар throttle рампы (ACCEL)
        bar_w2 = int(r.throttle_ramp * (FRAME_WIDTH - 20))
        cv2.rectangle(frame, (10, FRAME_HEIGHT - 32),
                      (10 + bar_w2, FRAME_HEIGHT - 22), (0, 255, 100), -1)

        # Метаданные внизу
        cv2.putText(
            frame,
            f"conf:{r.confidence:.2f}  "
            f"err:({r.err_x:+.0f},{r.err_y:+.0f})  "
            f"yaw:{r.rc_yaw}  thr:{r.rc_throttle}  pitch:{r.rc_pitch}",
            (10, FRAME_HEIGHT - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
        )

        self._push_frame_raw(frame)

    # ================================================================
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ================================================================
    def _run_inference(self, frame: np.ndarray):
        """NPU YOLO инференс. Возвращает post_process() результат или None."""
        img = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw = self.npu.inference(rgb)
        if raw is not None and post_process is not None:
            try:
                return post_process(
                    raw, [FRAME_WIDTH, FRAME_HEIGHT],
                    CONF_THRESHOLD, NMS_THRESHOLD,
                )
            except Exception as e:
                logger.warning(f"post_process error: {e}")
        return None

    def _push_frame(self, frame: np.ndarray, label: str, color: tuple):
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        self._push_frame_raw(frame)

    def _push_frame_raw(self, frame: np.ndarray):
        """Отправить кадр в MJPEG и GStreamer потоки."""
        global LATEST_JPEG
        # Масштабируем для стрима (экономия CPU + Wi-Fi)
        # frame.copy() защищает от гонки с VideoStream reader потоком,
        # который может перезаписать буфер кадра через cap.read()
        sf = cv2.resize(frame.copy(), (STREAM_WIDTH, STREAM_HEIGHT))
        ok, jpeg = cv2.imencode(
            '.jpg', sf, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY])
        if ok:
            with LATEST_JPEG_LOCK:
                LATEST_JPEG = jpeg.tobytes()

        # H.264 RTP для QGroundControl / VLC
        if self.gst_output and self.gst_output.is_active:
            self.gst_output.send_frame(sf)

        if not HEADLESS_MODE:
            cv2.imshow('INTERCEPTOR', cv2.resize(frame, (640, 480)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    def _limit_fps(self, t0: float):
        dt = time.monotonic() - t0
        wait = (1.0 / MAX_FPS) - dt
        if wait > 0:
            time.sleep(wait)

    def _log(self, state, safety_status):
        r = self._last_result
        try:
            self.flight_log.write(
                ts=time.time(),
                tx=r.target_x, ty=r.target_y, conf=r.confidence,
                mode=r.state.value, ramp=r.ramp_progress,
                err_x=r.err_x, err_y=r.err_y,
                rc_r=r.rc_roll, rc_p=r.rc_pitch,
                rc_t=r.rc_throttle, rc_y=r.rc_yaw,
                alt=state.altitude, batt=state.battery_voltage,
                armed=state.is_armed, mode_str=r.state.value,
                safety=safety_status.name,
                climb=state.climb_rate,
                temp=state.temperature, fps=MAX_FPS,
                lead_x=r.lead_x, lead_y=r.lead_y,
            )
        except Exception:
            pass

    def _cleanup(self):
        logger.info("Shutdown...")
        if self.ctrl.is_controlling:
            self.tracker.disengage()
            self.ctrl.release_control()  # возвращаем управление ELRS
        self.video.stop()
        if self.gst_output:
            self.gst_output.stop()
        self.mav.release()
        try:
            self.npu.release()
        except Exception:
            pass
        self.flight_log.close()
        try:
            self.server.server_close()
        except Exception:
            pass
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


def main():
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    InterceptorApp().run()


if __name__ == "__main__":
    main()