"""Download representative images for local calibration and validation.

Usage
-----
.. code-block:: powershell

    python -m scripts.download_calibration_images --calibration-count 50 --validation-count 30 --source picsum
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import urllib.request
import ssl


def _make_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


SSL_CTX = _make_ssl_context()


def _download_with_redirect(url: str, timeout: int = 20) -> Optional[bytes]:
    """Download *url*, follow redirects, return raw bytes or ``None``."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "image" not in ct:
                return None
            return resp.read()
    except Exception:
        return None


def download_image(url: str, output_path: Path, timeout: int = 20) -> Dict[str, Any]:
    """Download a single image from *url* to *output_path*.

    Returns a dict with keys: filename, source_url, final_url, status, error.
    """
    from PIL import Image as PILImage
    import io

    raw = _download_with_redirect(url, timeout=timeout)
    if raw is None or len(raw) < 1024:
        return {
            "filename": output_path.name,
            "source_url": url,
            "final_url": "",
            "status": "failed",
            "error": "download returned None or too small",
        }

    try:
        PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        return {
            "filename": output_path.name,
            "source_url": url,
            "final_url": "",
            "status": "failed",
            "error": f"Pillow validation failed: {exc}",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    return {
        "filename": output_path.name,
        "source_url": url,
        "final_url": "",
        "status": "ok",
        "error": None,
    }


def _picsum_urls(prefix: str, count: int) -> List[str]:
    """Generate deterministic picsum.photos URLs.

    Using ``seed`` ensures the same image is returned each time.
    """
    return [f"https://picsum.photos/seed/{prefix}_{i:03d}/640/480" for i in range(count)]


def _wikimedia_urls(count: int) -> List[str]:
    """Fallback Wikimedia thumbnail URL list (limited source)."""
    base = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg/320px-American_Eskimo_Dog.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Standard_coffee_mug.jpg/320px-Standard_coffee_mug.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/IBM_ThinkPad_A31p_matte_screen.jpg/320px-IBM_ThinkPad_A31p_matte_screen.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Orange_wireless_mouse.jpg/320px-Orange_wireless_mouse.jpg",
    ]
    # Repeat if count exceeds available
    return [base[i % len(base)] for i in range(count)]


def make_urls(source: str, prefix: str, count: int) -> List[str]:
    """Generate image URLs for the given *source* type."""
    if source == "picsum":
        return _picsum_urls(prefix, count)
    elif source == "wikimedia":
        return _wikimedia_urls(count)
    elif source == "mixed":
        picsum = _picsum_urls(prefix, count // 2 + 1)
        wiki = _wikimedia_urls(count)
        combined = []
        for i in range(count):
            combined.append(picsum[i % len(picsum)] if i % 2 == 0 else wiki[i % len(wiki)])
        return combined[:count]
    else:
        raise ValueError(f"Unknown source: {source}")


def download_images(
    source: str,
    prefix: str,
    count: int,
    output_dir: Path,
    manifest: List[dict],
) -> int:
    """Download *count* images to *output_dir*.

    Returns the number of successful downloads.
    """
    urls = make_urls(source, prefix, count)
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0

    for idx, url in enumerate(urls):
        fname = f"{prefix}_{idx:03d}.jpg"
        dst = output_dir / fname

        if dst.exists():
            manifest.append({
                "filename": fname,
                "split": output_dir.name,
                "source": source,
                "source_url": url,
                "final_url": "",
                "status": "already_exists",
                "error": None,
            })
            continue

        result = download_image(url, dst)
        result["split"] = output_dir.name
        result["source"] = source
        manifest.append(result)
        if result["status"] == "ok":
            success += 1

        if (idx + 1) % 10 == 0:
            print(f"  progress: {idx+1}/{count}")

    return success


def write_manifest(manifest: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download calibration / validation images")
    parser.add_argument("--calibration-count", type=int, default=50, help="Calibration images")
    parser.add_argument("--validation-count", type=int, default=30, help="Validation images")
    parser.add_argument(
        "--output-root", default="samples/images",
        help="Root output directory",
    )
    parser.add_argument(
        "--source", default="picsum", choices=["picsum", "wikimedia", "mixed"],
        help="Image source",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    cal_dir = root / "calibration_real"
    val_dir = root / "validation_real"

    print(f"Downloading calibration images ({args.source}) ...")
    cal_manifest: List[dict] = []
    cal_ok = download_images(
        args.source, "mdrl_cal", args.calibration_count, cal_dir, cal_manifest,
    )

    print(f"Downloading validation images ({args.source}) ...")
    val_manifest: List[dict] = []
    val_ok = download_images(
        args.source, "mdrl_val", args.validation_count, val_dir, val_manifest,
    )

    manifest_path = root / "image_manifest.json"
    write_manifest(cal_manifest + val_manifest, manifest_path)

    print()
    print(f"  calibration: {cal_ok}/{args.calibration_count} OK  → {cal_dir}")
    print(f"  validation:  {val_ok}/{args.validation_count} OK  → {val_dir}")
    total = cal_ok + val_ok
    print(f"  合计: {total} images downloaded")
    if total == 0:
        print("  警告: no images were downloaded.")


if __name__ == "__main__":
    main()
