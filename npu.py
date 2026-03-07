import numpy as np
import logging
from rknnlite.api import RKNNLite
from typing import Optional

logger = logging.getLogger(__name__)


class NPUHandler:
    """
    YOLO инференс на NPU Rockchip RK3588 (Radxa Rock 5B).
    Использует все 3 ядра NPU (NPU_CORE_0_1_2) для максимальной скорости.
    В связке с CSRT трекером даёт 15 FPS детекций + 30 FPS трекинга.
    """

    def __init__(self, model_path: str):
        logger.info(f"Loading RKNN model: {model_path}")
        self._init        = False
        self.inf_count    = 0
        self.error_count  = 0
        try:
            self.rknn = RKNNLite()
            if self.rknn.load_rknn(model_path) != 0:
                raise RuntimeError("load_rknn failed")
            if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                raise RuntimeError("init_runtime failed")
            self._init = True
            logger.info("NPU ready  (RK3588 NPU_CORE_0_1_2, all 3 cores)")
        except Exception as e:
            logger.critical(f"NPU init failed: {e}")
            raise

    def inference(self, frame: np.ndarray) -> Optional[list]:
        if not self._init or frame is None or frame.size == 0:
            return None
        try:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if self.inf_count < 5:
                logger.debug(f"NPU input: shape={frame.shape} dtype={frame.dtype}")
            output = self.rknn.inference(inputs=[frame])
            self.inf_count += 1
            return output
        except MemoryError:
            self.error_count += 1
            self._init = False
            logger.critical("NPU MemoryError — inference disabled")
            return None
        except Exception as e:
            self.error_count += 1
            logger.error(f"NPU inference error: {e}")
            return None

    def is_healthy(self) -> bool:
        return self._init and self.error_count < 10

    def release(self):
        try:
            if self._init:
                self.rknn.release()
                self._init = False
                logger.info("NPU released")
        except Exception:
            pass