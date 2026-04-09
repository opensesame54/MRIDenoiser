#!/usr/bin/env python3

import argparse
from pathlib import Path

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required for conversion. Install with: pip install pillow"
    ) from exc


SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def convert_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        gray = img.convert("L")
        gray.save(dst, format="PPM")  # Pillow writes grayscale PGM when mode is "L"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert image dataset to PGM for CUDA pipeline input.")
    parser.add_argument("--input-dir", required=True, help="Input dataset directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pgm files")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {in_dir}")

    converted = 0
    for src in in_dir.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in SUPPORTED:
            continue

        rel = src.relative_to(in_dir)
        dst = out_dir / rel.with_suffix(".pgm")
        convert_file(src, dst)
        converted += 1

    print(f"Converted {converted} images to {out_dir}")


if __name__ == "__main__":
    main()
