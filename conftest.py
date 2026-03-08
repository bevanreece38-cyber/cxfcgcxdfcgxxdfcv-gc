"""
conftest.py — подготовка тестовой среды.

Мокаем rknnlite (Rockchip NPU SDK) — недоступен в CI/dev среде.
Мок позволяет импортировать npu.py и main.py без реального железа.
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock

def _stub_rknnlite():
    if 'rknnlite' in sys.modules:
        return
    rknnlite_pkg  = ModuleType('rknnlite')
    rknnlite_api  = ModuleType('rknnlite.api')

    class _FakeRKNNLite:
        NPU_CORE_0_1_2 = 7
        def load_rknn(self, path):    return 0
        def init_runtime(self, **kw): return 0
        def inference(self, inputs):  return None
        def release(self):            pass

    rknnlite_api.RKNNLite     = _FakeRKNNLite
    rknnlite_pkg.api          = rknnlite_api
    sys.modules['rknnlite']   = rknnlite_pkg
    sys.modules['rknnlite.api'] = rknnlite_api

_stub_rknnlite()
