"""Runtime module — abstract runner interface and backend implementations.

Concrete runners should be imported from their own modules, e.g.::

    from src.runtime.fake_runner import FakeRunner
    from src.runtime.onnx_runner import OnnxRunner
"""

from src.runtime.base_runner import BaseRunner

__all__ = [
    "BaseRunner",
]
