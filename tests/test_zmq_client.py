"""Tests for the ZMQ server runner factory and client payload builder."""

from pathlib import Path

import pytest

from src.server.zmq_client import build_request_payload


class TestBuildRequestPayload:
    """Client payload construction — no ZMQ socket needed."""

    def test_default_input_type_is_image_path(self):
        payload = build_request_payload(backend="fake", input_value="test.jpg")
        assert payload["backend"] == "fake"
        assert payload["input_type"] == "image_path"
        assert payload["input"] == "test.jpg"

    def test_dummy_input_type(self):
        payload = build_request_payload(
            backend="onnx", input_value="dummy", input_type="dummy"
        )
        assert payload["backend"] == "onnx"
        assert payload["input_type"] == "dummy"
        assert payload["input"] == "dummy"

    def test_request_id_is_generated(self):
        payload = build_request_payload(backend="fake", input_value="x.jpg")
        assert "request_id" in payload
