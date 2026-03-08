# =====================================================
# КОНФИГУРАЦИЯ СИСТЕМЫ ПЕРЕХВАТА — ФИНАЛЬНАЯ ВЕРСИЯ
# Radxa Rock 5B + SpeedyBee F405 + ArduPilot + RadioMaster ELRS 2.4 GHz
#
# ТАКТИКА:
#   Наш дрон:  ~150 км/ч = 41.7 м/с
#   Цель:      160-180 км/ч = 44-50 м/с
#   Сближение: до 320 км/ч = 89 м/с (лоб-в-лоб)
#   Упреждение: LEAD_TIME_SEC = 0.30 сек = ~27 м при 89 м/с
# =====================================================

# --- MAVLink / SpeedyBee F405 UART ---
SERIAL_PORT         = "/dev/ttyS2"
BAUD_RATE           = 115200
RECONNECT_INTERVAL  = 2.0
HEARTBEAT_TIMEOUT   = 3.0
CONNECT_RETRIES     = 5
CONNECT_RETRY_DELAY = 2.0

# --- NPU YOLO (RK3588 NPU) — YOLOv8n RKNN ---
MODEL_PATH      = "drone_model.rknn"   # YOLOv8n экспортированный в .rknn формат
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
MAX_FPS         = 30
CONF_THRESHOLD  = 0.5
NMS_THRESHOLD   = 0.45
TARGET_CLASS_ID = 0

# --- Двойное машинное зрение: YOLO + CSRT/KCF (PixEagle архитектура) ---
# YOLO каждые N кадров: 30/2 = 15 детекций/сек на NPU
# CSRT/KCF заполняет промежутки: 30 FPS трекинга на CPU
YOLO_EVERY_N_FRAMES    = 2
TRACKER_TYPE           = "CSRT"     # базовый трекер; KCF при высокой скорости цели
TRACKER_REINIT_ON_YOLO = True       # переинициализация трекера при каждой YOLO-детекции

# Переключение CSRT → KCF при высокой скорости (px/frame)
# KCF быстрее при скорости >80 px/frame; CSRT точнее при низкой скорости
HIGH_SPEED_TRACKER_THRESHOLD = 80.0   # px/frame → переключиться на KCF
LOW_SPEED_TRACKER_THRESHOLD  = 50.0   # px/frame → вернуться на CSRT (гистерезис)

# --- RC управление (ArduPilot AltHold MAVLink RC_CHANNELS_OVERRIDE) ---
RC_MID     = 1500   # нейтральное положение стика

# КРИТИЧНО: RC_RELEASE = 65535, а НЕ 0 !
# 0   = ArduPilot применяет минимальный PWM (опасно, дрон упадёт!)
# 65535 = ArduPilot читает этот канал с ELRS пульта оператора (passthrough)
RC_RELEASE = 65535

RC_SAFE_MIN = 1100
RC_SAFE_MAX = 1900

# Throttle в AltHold: 1500 = hover
RC_THROTTLE_MIN      = 1350  # мягкое снижение
RC_THROTTLE_MAX      = 1650  # мягкий подъём при TRACKING/DEAD_RECKON
RC_THROTTLE_STRIKING = 1800  # максимальное ускорение при STRIKING (финальный удар)

# Roll-assist: крен для быстрого разворота при большом горизонтальном отклонении
# Включается при |err_x| > ROLL_ASSIST_THRESHOLD пикселей
ROLL_ASSIST_THRESHOLD = 150   # px: порог включения (150 px ≈ 23% ширины кадра)
ROLL_ASSIST_MIN       = 1350  # макс крен влево  (PWM)
ROLL_ASSIST_MAX       = 1650  # макс крен вправо (PWM)

# CH10 RadioMaster — канал атаки
ATTACK_CHANNEL_MIN = 1800
ATTACK_CHANNEL_MAX = 2000

# Keepalive: ArduPilot снимает override если >0.5 сек нет пакетов
# 25 Гц = запас x8 от требуемого минимума 3 Гц
KEEPALIVE_HZ       = 25   # фактическая частота keepalive потока ControlManager
MIN_RC_UPDATE_RATE = 25   # минимальная допустимая частота обновления RC override (Гц)
# KEEPALIVE_HZ — частота keepalive петли в ControlManager
# MIN_RC_UPDATE_RATE — нижняя граница для внешних компонентов / диагностики
# Оба = 25 Гц: ControlManager гарантированно выдерживает минимальную частоту

# --- Dead reckoning ---
# 89 м/с * 0.25 сек = 22 м — допустимо для перехвата
DEAD_RECKONING_SEC = 0.25

# --- REACQUIRE: продолжение манёвра через Kalman vx,vy после DEAD_RECKON ---
# Дрон продолжает лететь в последнем известном направлении цели
# 89 м/с * 1.5 сек = 134 м — максимальная дальность REACQUIRE поиска
REACQUIRE_TIMEOUT = 1.5   # сек

# Скорость затухания скорости при REACQUIRE (PixEagle ENABLE_VELOCITY_DECAY паттерн)
# 0.85^t: через 1 сек скорость = 85%, через 2 сек = 72%, через 3 сек = 61%
# Предотвращает уход дрона от последней позиции при длительной потере цели
REACQUIRE_VELOCITY_DECAY = 0.85   # коэффициент затухания в секунду

# Подтверждение повторного захвата (PixEagle recovery_confirmation_time паттерн)
# Цель должна отслеживаться стабильно в течение REACQUIRE_CONFIRM_SEC секунд
# прежде чем вернуться из REACQUIRE в TRACKING — исключает ложные возвраты
REACQUIRE_CONFIRM_SEC = 0.3   # сек

# --- Predictive intercept (упреждение точки наводки) ---
# Наводимся не на текущую позицию, а на прогнозную точку через LEAD_TIME_SEC
# При скорости сближения 89 м/с за 0.30 сек = ~27 м упреждения (лобовой курс)
LEAD_TIME_SEC = 0.30    # 300 мс — увеличено с 0.12 для встречного курса
LEAD_FACTOR   = 2.5     # усилитель Kalman velocity (компенсация задержки pipeline)

# --- Кинетический удар (Pitch рампа) ---
# ArduPilot AltHold: нос ВНИЗ = меньший PWM на pitch
# PITCH_NEAR: лёгкий наклон вниз (цель вверху кадра)
# PITCH_DIVE: максимальное пикирование (цель внизу кадра)
PITCH_NEAR       = 1450
PITCH_DIVE       = 1280
MIN_ATTACK_PITCH = PITCH_NEAR   # алиасы для совместимости
MAX_ATTACK_PITCH = PITCH_DIVE

# Рампа пикирования: 0.6 сек
# 89 м/с * 0.6 сек = ~53 м до удара — достаточно
RAMP_DURATION_SEC = 0.6

# Рампа throttle: плавное ускорение без рывков
# Нарастает от 0 до 1.0 за THROTTLE_RAMP_SEC секунд
THROTTLE_RAMP_SEC = 0.8

# --- PID (пересчитан для 150-180 км/ч) ---
KP_YAW = 0.8    # было 0.5: увеличен для быстрой угловой цели
KI_YAW = 0.008  # снижен: меньше накопление при рысканье цели
KD_YAW = 0.15   # увеличен: демпфирование при резком изменении курса

KP_ALT = 1.0    # было 0.8
KI_ALT = 0.015
KD_ALT = 0.2

PID_INTEGRAL_LIMIT = 30.0   # снижен с 50: меньше перерегулирование

# --- Kalman фильтр ---
# Высокая скорость = больше шум процесса (цель может резко менять курс)
KALMAN_PROCESS_NOISE     = 5e-4   # было 1e-4
KALMAN_MEASUREMENT_NOISE = 1e-2
KALMAN_ERROR_COV_INIT    = 0.1

# --- Безопасность ---
TEMP_WARNING     = 75.0
TEMP_CRITICAL    = 85.0
MAX_DESCENT_RATE = 2.5     # м/с
BATTERY_LOW      = 14.8    # В
BATTERY_CRITICAL = 14.0    # В
MIN_ALTITUDE     = 5.0     # м

# --- Логирование ---
LOG_DIR      = "./logs"
MAX_LOG_SIZE = 10 * 1024 * 1024   # 10 МБ → ротация

# --- Режим отображения ---
HEADLESS_MODE = True   # False = cv2.imshow (только с монит��ром)

# --- MJPEG HTTP стриминг (браузер / FPV монитор) ---
STREAM_PORT    = 5000
STREAM_QUALITY = 65
STREAM_WIDTH   = 480
STREAM_HEIGHT  = 360

# --- Захват видео ---
VIDEO_SOURCE_INDEX  = 1
VIDEO_DEVICE_PATH   = "/dev/video1"
VIDEO_PIXEL_FORMAT  = "YUYV"          # YUYV для меньшей задержки (нет CPU декодинга MJPEG)
VIDEO_USE_GSTREAMER = True

# --- GStreamer H.264/RTP/UDP выход (QGroundControl / VLC) ---
GSTREAMER_ENABLED   = True
GSTREAMER_HOST      = "192.168.1.100"
GSTREAMER_PORT      = 5600
GSTREAMER_WIDTH     = 480
GSTREAMER_HEIGHT    = 360
GSTREAMER_FPS       = 25
GSTREAMER_BITRATE   = 1000       # kbps
GSTREAMER_ENABLE_HW = True       # mpph264enc Rockchip MPP RK3588

# --- Режим без FC (тест видео без SpeedyBee) ---
# True  = не блокировать видео если MAVLink не подключён
# False = боевой режим (SAFETY при потере heartbeat)
NO_FC_TEST_MODE = False