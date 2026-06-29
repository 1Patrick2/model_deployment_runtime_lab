"""Download representative images for local calibration and validation.

Usage
-----
.. code-block:: powershell

    python -m scripts.download_calibration_images --calibration-count 30 --validation-count 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

# Public-domain / CC0 image URLs sourced from well-known open repositories.
# These are commonly used for ML demos and have permissive licenses.
_SOURCE_URLS: List[str] = [
    # Animals
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg/640px-American_Eskimo_Dog.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Cow_%28Fleckvieh%29.jpg/640px-Cow_%28Fleckvieh%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/White-breasted_Kingfisher.jpg/640px-White-breasted_Kingfisher.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Columba_livia_%282011%29.jpg/640px-Columba_livia_%282011%29.jpg",
    # Objects — household
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Standard_coffee_mug.jpg/640px-Standard_coffee_mug.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Orange_wireless_mouse.jpg/640px-Orange_wireless_mouse.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/IBM_ThinkPad_A31p_matte_screen.jpg/640px-IBM_ThinkPad_A31p_matte_screen.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Water_bottle.jpg/640px-Water_bottle.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Office_book.jpg/640px-Office_book.jpg",
    # More animals
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Lion_waiting_in_Namibia.jpg/640px-Lion_waiting_in_Namibia.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/640px-Grosser_Panda.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/Indian_elephant_%2826870766085%29.jpg/640px-Indian_elephant_%2826870766085%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Frog_sitting_on_leaf.jpg/640px-Frog_sitting_on_leaf.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Horse%2C_July_2010.jpg/640px-Horse%2C_July_2010.jpg",
    # Objects — tech / transport
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Black_sedan.jpg/640px-Black_sedan.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Airbus_A380_blue_sky.jpg/640px-Airbus_A380_blue_sky.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Bicycle_in_Helsinki.jpg/640px-Bicycle_in_Helsinki.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Ship_off_the_coast_of_Malta.jpg/640px-Ship_off_the_coast_of_Malta.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Street_sweeper_machine.jpg/640px-Street_sweeper_machine.jpg",
    # Plants / food
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Red_apple.jpg/640px-Red_apple.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Bananas_%28white_background%29.jpg/640px-Bananas_%28white_background%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Orange_-_whole_%26_half.jpg/640px-Orange_-_whole_%26_half.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Sunflower_%28Helianthus_annuus%29.jpg/640px-Sunflower_%28Helianthus_annuus%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Broccoli_and_cross_section.jpg/640px-Broccoli_and_cross_section.jpg",
    # Indoor / furniture
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Modern_office_desk.jpg/640px-Modern_office_desk.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Chair_%282019%29.jpg/640px-Chair_%282019%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Swatch_clock.png/640px-Swatch_clock.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Potted_plant.jpg/640px-Potted_plant.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Wrist-watch_straps.jpg/640px-Wrist-watch_straps.jpg",
    # Scenery / landscapes (more variety)
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Mountain_panorama.jpg/640px-Mountain_panorama.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Beautiful_sunset_at_the_beach.jpg/640px-Beautiful_sunset_at_the_beach.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Forest_path_in_autumn.jpg/640px-Forest_path_in_autumn.jpg",
    # Instruments / sports
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Acoustic_guitar_PNG.png/640px-Acoustic_guitar_PNG.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Basketball_ball.svg/640px-Basketball_ball.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Soccer_ball.svg/640px-Soccer_ball.svg.png",
    # People-related objects
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Backpack_%28PSF%29.jpg/640px-Backpack_%28PSF%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Sunglasses_%282017%29.jpg/640px-Sunglasses_%282017%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Umbrella_%282019%29.jpg/640px-Umbrella_%282019%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Pair_of_shoes.jpg/640px-Pair_of_shoes.jpg",
    # Additional variety
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Music_keyboard_%282019%29.jpg/640px-Music_keyboard_%282019%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Television_remote_control.jpg/640px-Television_remote_control.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/SET_of_office_supplies.JPG/640px-SET_of_office_supplies.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Vase_with_flowers.jpg/640px-Vase_with_flowers.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Toy_train_%282019%29.jpg/640px-Toy_train_%282019%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Strawberry_%282019%29.jpg/640px-Strawberry_%282019%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Small_pumpkin.jpg/640px-Small_pumpkin.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Bell_pepper_%28white_background%29.jpg/640px-Bell_pepper_%28white_background%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Coffee_beans.jpg/640px-Coffee_beans.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/French_horn.jpg/640px-French_horn.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Microphone_%282019%29.jpg/640px-Microphone_%282019%29.jpg",
]


def download_image(url: str, output_path: Path, timeout: int = 15) -> bool:
    """Download a single image from *url* to *output_path*.

    Returns ``True`` on success.
    """
    import urllib.request
    import ssl

    ctx = ssl._create_unverified_context()
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        urllib.request.urlretrieve(url, output_path)
        return output_path.exists() and output_path.stat().st_size > 1024
    except Exception:
        return False


def _name_from_url(url: str, idx: int) -> str:
    """Derive a short filename from the URL."""
    base = url.rstrip("/").split("/")[-1]
    name, _ = os.path.splitext(base)
    # Remove leading digits/hyphens from Wikipedia thumbnails
    parts = name.split("-", 1)
    stem = parts[-1] if len(parts) > 1 else parts[0]
    stem = stem.replace("-", "_")
    return f"{idx:03d}_{stem[:40]}.jpg"


def download_images(
    urls: List[str],
    output_dir: Path,
    count: int,
    manifest: List[dict],
) -> int:
    """Download *count* images from *urls* to *output_dir*.

    Appends manifest entries for each attempt.
    Returns the number of successful downloads.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    for idx, url in enumerate(urls[:count]):
        fname = _name_from_url(url, idx)
        dst = output_dir / fname
        if dst.exists():
            manifest.append(
                {"filename": fname, "split": output_dir.parent.name,
                 "source_url": url, "status": "already_exists"}
            )
            success += 1
            continue
        ok = download_image(url, dst)
        manifest.append(
            {"filename": fname if ok else None,
             "split": output_dir.parent.name,
             "source_url": url, "status": "ok" if ok else "failed"}
        )
        if ok:
            success += 1
        else:
            # Try alternate URL pattern
            alt_url = url.replace("/640px-", "/320px-")
            alt_ok = download_image(alt_url, dst)
            manifest[-1]["status"] = "ok" if alt_ok else "failed"
            if alt_ok:
                manifest[-1]["filename"] = fname
                success += 1
        if (idx + 1) % 10 == 0:
            print(f"  progress: {idx+1}/{count}")
    return success


def write_manifest(manifest: List[dict], path: Path) -> None:
    """Write the download manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download calibration / validation images")
    parser.add_argument("--calibration-count", type=int, default=30, help="Calibration images")
    parser.add_argument("--validation-count", type=int, default=20, help="Validation images")
    parser.add_argument(
        "--output-root",
        default="samples/images",
        help="Root output directory",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    cal_dir = root / "calibration_real"
    val_dir = root / "validation_real"

    print("Downloading calibration images...")
    cal_manifest: list[dict] = []
    cal_ok = download_images(_SOURCE_URLS, cal_dir, args.calibration_count, cal_manifest)

    val_start = args.calibration_count
    val_end = val_start + args.validation_count
    val_urls = _SOURCE_URLS[val_start:val_end] if val_end <= len(_SOURCE_URLS) else _SOURCE_URLS[-args.validation_count:]
    print("Downloading validation images...")
    val_manifest: list[dict] = []
    val_ok = download_images(val_urls, val_dir, args.validation_count, val_manifest)

    manifest_path = root / "image_manifest.json"
    write_manifest(cal_manifest + val_manifest, manifest_path)

    print()
    print(f"  calibration: {cal_ok}/{args.calibration_count} OK  → {cal_dir}")
    print(f"  validation:  {val_ok}/{args.validation_count} OK  → {val_dir}")
    total = cal_ok + val_ok
    print(f"  total: {total} images downloaded")
    if total == 0:
        print("  WARNING: no images were downloaded. The network may be blocking requests.")


if __name__ == "__main__":
    main()
