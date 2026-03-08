# Copilot Instructions

## About This Repository

This is a Python-based autonomous drone intercept system running on a Radxa Rock 5B (RK3588 SBC) connected to a SpeedyBee F405 flight controller running ArduPilot. It detects, tracks, and intercepts aerial targets using YOLO object detection, CSRT visual tracking, Kalman filtering for predictive interception, and MAVLink RC channel overrides.

## Project Structure

| File | Description |
|---|---|
| `main.py` | Main application loop |
| `config.py` | All system parameters (tune here first) |
| `tracker_engine.py` | 3D intercept algorithm (PID + ramps) |
| `vision_tracker.py` | YOLO + CSRT + Kalman tracking pipeline |
| `kalman.py` | 4D Kalman filter (x, y, vx, vy) |
| `control_manager.py` | MAVLink RC override control |
| `pid.py` | PID controller |
| `handler.py` | MAVLink connection handling |
| `state.py` | ArduPilot telemetry parsing |
| `safety.py` | Safety monitoring and failsafes |
| `npu.py` | RK3588 NPU YOLO inference |
| `video_stream.py` | Video capture (GStreamer / V4L2) |
| `gstreamer_output.py` | H.264 RTP/UDP output |
| `flight_logger.py` | CSV flight data logging |
| `types_enum.py` | Enumerations |
| `utils.py` | Utility functions |

Test files are named `test_<module>.py` and live in the repository root.

## Development Workflow

### Install dependencies
```bash
pip install -r requirements.txt
```

On Radxa Rock 5B hardware, also install the NPU runtime:
```bash
pip install rknnlite
```

### Run tests
```bash
pip install pytest
pytest -v
```

### Run the application
```bash
python main.py
```

## Coding Standards

- **Language**: Python 3 only.
- **Style**: Follow PEP 8 conventions. Keep functions small and focused.
- **Types**: Use type hints where practical.
- **Configuration**: All tunable parameters belong in `config.py`. Do not hardcode magic numbers elsewhere.
- **Safety-critical values**: `RC_RELEASE = 65535` means passthrough to the ELRS transmitter. Never set an RC channel to `0` (this means minimum PWM and will crash the drone).
- **Testing**: Unit tests use `pytest`. Mock hardware dependencies (MAVLink, camera, NPU) in tests. Each new module should have a corresponding `test_<module>.py`.
- **Imports**: Standard library first, then third-party (`pymavlink`, `cv2`, `numpy`), then local modules.

## Key Constraints

- The keepalive loop must run at ≥ 3 Hz (ArduPilot requirement); it is currently set to 25 Hz.
- CH10 is the autonomy switch: OFF = manual AltHold, ON = autonomous intercept.
- On target loss or safety trigger, always call `release_control()` (all channels → 65535 passthrough), never set channels to 0.
- The NPU module (`npu.py`) depends on `rknnlite` which is only available on RK3588 hardware. Code that imports it must handle `ImportError` gracefully.
- Video streaming uses GStreamer on hardware; fall back to V4L2/OpenCV on development machines.
