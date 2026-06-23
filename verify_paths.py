#!/usr/bin/env python3
"""Model Deployment Runtime Lab - Path Verification.

Verifies that the project structure is sound and all critical paths exist.
Hard-fails only for essential files; warns for planned/future paths.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "[OK]"
WARN = "[..]"
FAIL = "[!!]"
INFO = "    "


def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def warn(msg: str) -> None:
    print(f"  {WARN}  {msg}  (planned)")


def hard_fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")
    errors.append(msg)


# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Walk up from CWD looking for configs/paths.yaml."""
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "configs" / "paths.yaml").exists():
            return parent
    return Path(__file__).resolve().parent


ROOT = find_project_root()
errors: list[str] = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  Model Deployment Runtime Lab - Path Verification")
print("=" * 60)
print()
print(f"  Project Root: {ROOT}")
print(f"  Exists: {'YES' if ROOT.exists() else 'NO'}")
print()

# ---------------------------------------------------------------------------
# Hard requirements
# ---------------------------------------------------------------------------

print("  Critical directories:")

HARD_DIRS = ["configs", "src", "docs"]
for d in HARD_DIRS:
    target = ROOT / d
    if target.is_dir():
        ok(f"{d}")
    else:
        hard_fail(f"{d}  -- missing")

print()
print("  Critical files:")

HARD_FILES = [
    "configs/paths.yaml",
    "README.md",
    "src/utils/path_manager.py",
]
for f in HARD_FILES:
    target = ROOT / f
    if target.is_file():
        ok(f"{f}")
    else:
        hard_fail(f"{f}  -- missing")

# ---------------------------------------------------------------------------
# Warnings (planned paths)
# ---------------------------------------------------------------------------

print()
print("  Planned paths (warnings only):")

WARN_DIRS = [
    "outputs",
    "samples/images",
    "models",
    "models/registry.json",
    "configs/model.yaml",
    "configs/export.yaml",
    "configs/quant.yaml",
    "configs/runtime.yaml",
    "configs/zmq.yaml",
    "configs/benchmark.yaml",
    "configs/rknn.yaml",
]
for d in WARN_DIRS:
    target = ROOT / d
    if not target.exists():
        warn(f"{d}")

# ---------------------------------------------------------------------------
# Final
# ---------------------------------------------------------------------------

print()
if errors:
    print(f"  {FAIL}  {len(errors)} critical error(s) found:")
    for e in errors:
        print(f"      - {e}")
    print()
    print("  Fix the issues above, then re-run this script.")
    print()
    sys.exit(1)
else:
    print(f"  {PASS}  All critical paths verified.")
    print()
    print("  Next steps:")
    print("     1. Run setup_win.ps1")
    print("     2. Activate environment (conda activate mdrl-runtime)")
    print("     3. Start developing -- Stage 1 will add fake runtime + ZMQ")
    print()
