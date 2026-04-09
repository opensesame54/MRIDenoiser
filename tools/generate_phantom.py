#!/usr/bin/env python3

import argparse
import math


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def write_pgm(path, width, height, data):
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(bytes(data))


def generate_phantom(width, height):
    cx = width / 2.0
    cy = height / 2.0
    out = [0] * (width * height)

    for y in range(height):
        for x in range(width):
            dx = (x - cx) / width
            dy = (y - cy) / height

            base = 25 + 20 * math.sin(8 * dx) * math.cos(8 * dy)

            r1 = math.sqrt((dx * 1.2) ** 2 + (dy * 1.0) ** 2)
            r2 = math.sqrt(((dx + 0.15) * 1.8) ** 2 + ((dy - 0.1) * 1.4) ** 2)
            r3 = math.sqrt(((dx - 0.18) * 1.6) ** 2 + ((dy + 0.12) * 2.0) ** 2)

            v = base
            if r1 < 0.30:
                v += 90
            if r2 < 0.16:
                v += 70
            if r3 < 0.12:
                v += 55

            band = abs(math.sin((x + y) * 0.06))
            v += 25 * (band ** 3)

            out[y * width + x] = int(clamp(round(v), 0, 255))

    return out


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical-style phantom PGM image.")
    parser.add_argument("--output", required=True, help="Output PGM path")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Width and height must be positive integers")

    data = generate_phantom(args.width, args.height)
    write_pgm(args.output, args.width, args.height, data)
    print(f"Wrote phantom image to {args.output} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
