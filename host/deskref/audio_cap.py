import argparse
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def list_devices():
    apis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"{i:3d}  {d['name']}  [{apis[d['hostapi']]['name']}]  "
                  f"default {d['default_samplerate']:.0f} Hz")


def main():
    p = argparse.ArgumentParser(description="Record a listed mic take to data/raw/")
    p.add_argument("name", nargs="?", help="take name, e.g. typing_01")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--device", type=int, help="input device index from --list")
    p.add_argument("--rate", type=int, default=48000)
    p.add_argument("--list", action="store_true", help="list input devices and exit")
    args = p.parse_args()

    if args.list or not args.name:
        list_devices()
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{args.name}.wav"
    if path.exists():
        raise SystemExit(f"{path} already exists, pick a new name")

    for n in (3, 2, 1):
        print(f"Recording in {n}...", end="\r", flush=True)
        time.sleep(1)
    print(f"Recording {args.seconds:.0f} s to {path.name}   ")

    data = sd.rec(int(args.seconds * args.rate), samplerate=args.rate,
                  channels=1, dtype="int16", device=args.device)
    sd.wait()
    sf.write(path, data, args.rate, subtype="PCM_16")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()