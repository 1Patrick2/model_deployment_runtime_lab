"""Tests that quantised ONNX models actually run with ONNX Runtime."""

from pathlib import Path

import numpy as np
import onnxruntime
import pytest


@pytest.mark.skipif(
    not Path("outputs/onnx/mobilenetv3_small_int8_qdq_dummy.onnx").exists(),
    reason="QDQ INT8 artifact not generated; run quantize_onnx with static_qdq first",
)
class TestQuantizedModelRunsWithOrt:
    """Verify that the quantised model can be loaded and executed."""

    def test_session_can_be_created(self):
        session = onnxruntime.InferenceSession(
            "outputs/onnx/mobilenetv3_small_int8_qdq_dummy.onnx",
            providers=["CPUExecutionProvider"],
        )
        assert session is not None

    def test_dummy_input_produces_output(self):
        session = onnxruntime.InferenceSession(
            "outputs/onnx/mobilenetv3_small_int8_qdq_dummy.onnx",
            providers=["CPUExecutionProvider"],
        )
        input_meta = session.get_inputs()[0]
        x = np.zeros([1, 3, 224, 224], dtype=np.float32)
        outputs = session.run(None, {input_meta.name: x})
        assert len(outputs) > 0
        assert outputs[0].shape == (1, 1000)
