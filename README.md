# Interceptor System

Система автономного перехвата воздушных целей на базе Radxa Rock 5B + SpeedyBee F405 + ArduPilot.

## Аппаратура

```
RadioMaster TX → ELRS 2.4GHz → Приёмник → SpeedyBee F405 (ArduPilot AltHold)
SpeedyBee F405 ↔ Radxa Rock 5B (UART /dev/ttyS2, 115200 baud, MAVLink)
Radxa Rock 5B  ← Цифровая камера USB (/dev/video1)
Radxa Rock 5B  → Wi-Fi → Оператор (MJPEG :5000 / H.264 RTP :5600)
```

## Датчики (без GPS)

Система работает **полностью без GPS**. Навигация и стабилизация через встроенные датчики полётного контроллера:

| Датчик | Роль в AltHold |
|--------|---------------|
| **ИМУ** (акселерометр + гироскоп) | Стабилизация крена/тангажа, угловые скорости |
| **Барометр** | Удержание высоты (режим AltHold) |
| **Компас** (магнитометр) | Удержание курса (Heading Hold) |

**GUIDED_NO_GPS режим НЕ используется** — он требует источник скорости (optical flow / SLAM / визуальная одометрия), которого в данной конфигурации нет. Система управляет дроном исключительно через `RC_CHANNELS_OVERRIDE` в режиме **AltHold**: компьютер посылает PWM-значения каналов (как джойстик пульта), ArduPilot сам замыкает контуры стабилизации.

## Тактика

| Параметр | Значение |
|---|---|
| Скорость нашего дрона | ~150 км/ч (41.7 м/с) |
| Скорость цели | 160–180 км/ч (44–50 м/с) |
| Лобовое сближение | до 330 км/ч (91 м/с) |
| Упреждение (LEAD_TIME_SEC) | 0.12 сек (~11 м при 91 м/с) |

## Логика управления

- **CH10 OFF** → AltHold, оператор управляет с пульта; YOLO рисует зелёные рамки на видео
- **CH10 ON** → `take_control()` → трекинг (YOLO + CSRT + Kalman + PID) + кинетический удар
- **Потеря цели** → `release_control()` → CH1-8 = 0 (мгновенный сброс override → ELRS пульт)
- **SAFETY** → `release_control()` + `MAV_CMD_NAV_LAND`

## Архитектура (PixEagle + DroneEngage)

```
VideoStream (GStreamer MJPEG, appsink max-buffers=1 sync=false)
    │ нон-блокирующий фоновый поток
    ▼
NPUHandler (RK3588 NPU_CORE_0_1_2, YOLO каждые 2 кадра в атаке)
    │
    ▼
VisionTracker
  ├─ YOLO детекция (NPU, каждые YOLO_EVERY_N_FRAMES кадров)
  ├─ CSRT трекинг (CPU, каждый кадр между YOLO)
  └─ KalmanTargetTracker (сглаживание + скорость → упреждение)
    │
    ▼
TrackerEngine
  ├─ Predictive intercept: lead_x/y = pos + velocity * LEAD_FACTOR * LEAD_TIME_SEC
  ├─ PID yaw   (err_x → rc_yaw)
  ├─ PID alt   (err_y → rc_throttle, с рампой THROTTLE_RAMP_SEC)
  └─ Pitch рампа (RC_MID → PITCH_DIVE за RAMP_DURATION_SEC)
    │
    ▼
ControlManager (DroneEngage)
  ├─ take_control() / release_control()
  ├─ Keepalive поток 25 Гц (ArduPilot требует >3 Гц)
  └─ release_control() → CH1-8 = 0 (мгновенный сброс override → ELRS пульт)
```

## RC_CHANNELS_OVERRIDE

| Канал | Управляет | Значение при трекинге |
|---|---|---|
| CH1 Roll | Оператор | 65535 (passthrough) |
| CH2 Pitch | Компьютер | 1280–1450 (рампа пикирования) |
| CH3 Throttle | Компьютер | 1350–1650 (рампа ускорения) |
| CH4 Yaw | Компьютер | 1100–1900 (PID наводка) |
| CH5–CH18 | Оператор | 65535 (passthrough) |

**ВАЖНО — RC_RELEASE в keepalive vs release:**
- Во время трекинга keepalive посылает `65535` для Roll/CH5–CH18 → ArduPilot игнорирует поле (UINT16_MAX), оператор держит Roll и режимы
- При `release_control()` посылаем `0` для CH1–8 → `set_override(i,0)` (внутр. C++ ArduPilot) → `has_override()=false` → **мгновенный** переход на аппаратный RC (ELRS пульт), без ожидания таймаута
- Подтверждено исходником ArduPilot `GCS_Common.cpp::handle_rc_channels_override()`

## Установка зависимостей

```bash
pip install -r requirements.txt
```

> **На Radxa Rock 5B** дополнительно требуется `rknnlite` из SDK Rockchip:
> ```bash
> pip install rknnlite
> ```

## Запуск

```bash
python3 main.py
```

- MJPEG видеопоток: `http://<radxa-ip>:5000/stream`
- Health-check: `http://<radxa-ip>:5000/health`
- H.264 RTP (QGroundControl / VLC): `rtp://@:5600`

## Видео — нулевая задержка

Три уровня оптимизации задержки:

| Уровень | Что сделано | Выигрыш |
|---------|-------------|---------|
| **Захват (GStreamer)** | `appsink max-buffers=1 sync=false` | Нет буферизации в драйвере |
| **Захват (OpenCV fallback)** | `CAP_PROP_BUFFERSIZE=1` | Убирает стандартный 4-кадровый буфер (–133 мс) |
| **Кодирование JPEG** | Вынесено в фоновый поток `_encode_worker` | RC-цикл не блокируется (~4 мс) |
| **Стриминг MJPEG** | `threading.Condition.wait_for()` без sleep | Кадр уходит клиенту немедленно после кодирования |
| **GStreamer выход** | Queue `maxsize=1` с drop-oldest | Не накапливаются старые кадры |

**Итоговая задержка MJPEG:**
```
Камера → VideoStream (фоновый поток) → main loop (~20 мс YOLO+CSRT)
→ _RAW_FRAME_Q → _encode_worker (3 мс JPEG) → _STREAM_COND.notify_all()
→ StreamHandler (немедленно, нет sleep) → браузер/FPV монитор

Итого: ~23 мс (1 кадр при 30 FPS)
```



| Файл | Описание |
|---|---|
| `main.py` | Главный цикл приложения |
| `config.py` | Все параметры системы |
| `tracker_engine.py` | 3D алгоритм перехвата (PID + рампы) |
| `vision_tracker.py` | YOLO + CSRT + Kalman (PixEagle) |
| `kalman.py` | Kalman фильтр 4D (x, y, vx, vy) |
| `control_manager.py` | RC override MAVLink (DroneEngage) |
| `pid.py` | PID регулятор |
| `handler.py` | MAVLink соединение |
| `state.py` | Разбор телеметрии ArduPilot |
| `safety.py` | Мониторинг безопасности |
| `npu.py` | RK3588 NPU YOLO инференс |
| `video_stream.py` | Захват видео GStreamer / V4L2 |
| `gstreamer_output.py` | H.264 RTP/UDP выход |
| `flight_logger.py` | CSV лог полётных данных |

## RC_CHANNELS_OVERRIDE: 0 vs 65535

Подтверждено исходным кодом ArduPilot `GCS_Common.cpp`:

| Значение | CH1–8 | CH9–18 |
|----------|-------|--------|
| **0** | `set_override(i,0)` → `has_override()=false` → **мгновенный аппаратный RC** | игнорируется |
| **65535** | игнорируется (UINT16_MAX filter) — старый override остаётся до таймаута | игнорируется |
| **1000–2000** | применяется как PWM override | применяется |

**Вывод:**
- `release_control()` посылает `0` для CH1–8 → мгновенный возврат к ELRS пульту
- Keepalive посылает `65535` для каналов оператора → ArduPilot их не трогает
- `RC_RELEASE = 65535` в коде означает «не override этот канал» во время keepalive

## PID настройки для высокоскоростных целей

Пересчитано для скорости сближения до 91 м/с:

| Параметр | Значение | Назначение |
|----------|----------|-----------|
| `KP_YAW` | 0.8 | Быстрый отклик по рысканию |
| `KI_YAW` | 0.008 | Малый интеграл — цель резко меняет курс |
| `KD_YAW` | 0.15 | Демпфирование при рывках |
| `KP_ALT` | 1.0 | Быстрый отклик по высоте |
| `KI_ALT` | 0.015 | Малый интеграл |
| `KD_ALT` | 0.2 | Демпфирование |
| `RAMP_DURATION_SEC` | 0.6 | Рампа pitch: 91 м/с × 0.6 с = 55 м до удара |
| `THROTTLE_RAMP_SEC` | 0.8 | Рампа throttle: плавный набор без рывков |
| `LEAD_TIME_SEC` | 0.12 | Упреждение 120 мс ≈ 11 м при 91 м/с |
| `LEAD_FACTOR` | 2.5 | Усилитель velocity из Kalman |
| `DEAD_RECKONING_SEC` | 0.4 | Dead reckoning: 91 м/с × 0.4 с = 36 м |

## Запуск тестов

```bash
pip install pytest opencv-contrib-python numpy
pytest -v
```
