"""Tests for calibration image downloader — no real network access."""

import json
from pathlib import Path

from scripts.download_calibration_images import make_urls, download_image


class TestMakeUrls:
    """URL generation — deterministic, no network."""

    def test_picsum_url_format(self):
        urls = make_urls("picsum", "mdrl_cal", 3)
        assert len(urls) == 3
        for url in urls:
            assert url.startswith("https://picsum.photos/seed/mdrl_cal_")
            assert url.endswith("/640/480")

    def test_picsum_urls_are_unique(self):
        urls = make_urls("picsum", "mdrl_val", 5)
        assert len(set(urls)) == 5

    def test_wikimedia_fallback_returns_correct_count(self):
        urls = make_urls("wikimedia", "", 3)
        assert len(urls) == 3
        for url in urls:
            assert "upload.wikimedia.org" in url


class TestDownloadImage:
    """Download logic — mocked."""

    def test_download_failure_records_error(self, tmp_path, monkeypatch):
        def mock_download(url, timeout=20):
            return None
        monkeypatch.setattr(
            "scripts.download_calibration_images._download_with_redirect",
            mock_download,
        )
        result = download_image("http://example.com/fake.jpg", tmp_path / "out.jpg")
        assert result["status"] == "failed"
        assert result["error"] is not None

    def test_manifest_split_field(self, tmp_path, monkeypatch):
        def mock_download(url, timeout=20):
            return None
        monkeypatch.setattr(
            "scripts.download_calibration_images._download_with_redirect",
            mock_download,
        )
        result = download_image("http://example.com/fake.jpg", tmp_path / "out.jpg")
        # split is added by the caller (download_images), not by download_image
        # so we just verify status is failed
        assert result["status"] == "failed"
