#!/usr/bin/env python3
import argparse
import torch

from tts_engine import TTSEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--out", "-o", default="output.wav", help="Output WAV filename"
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu",
    )
    args = parser.parse_args()

    tts = TTSEngine(device=args.device)
    output_path = tts.synthesize(args.text, args.out)
    print(f'Saved synthesized audio to "{output_path}"')


if __name__ == "__main__":
    main()
