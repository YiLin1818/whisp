import argparse, csv, os
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="small.en")
    args = ap.parse_args()

    try:
        import whisperx
    except ImportError:
        raise SystemExit("Install whisperx first: pip install whisperx")

    audio_files = []
    for fn in os.listdir(args.audio_dir):
        if fn.lower().endswith((".wav",".mp3",".flac",".m4a",".ogg")):
            audio_files.append(os.path.join(args.audio_dir, fn))
    audio_files.sort()

    model = whisperx.load_model(args.model, device=args.device, compute_type="float16" if args.device=="cuda" else "int8")
    align_model, metadata = whisperx.load_align_model(language_code="en", device=args.device)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id","audio_path","transcript","words_json"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i, path in enumerate(tqdm(audio_files)):
            audio = whisperx.load_audio(path)
            result = model.transcribe(audio, language="en")

            aligned = whisperx.align(result["segments"], align_model, metadata, audio, args.device)
            # collect word timestamps
            words = []
            for seg in aligned["segments"]:
                for wd in seg.get("words", []):
                    if wd.get("word") is None: 
                        continue
                    words.append({"w": wd["word"], "s": wd.get("start", None), "e": wd.get("end", None)})

            transcript = result.get("text","").strip()
            import json
            w.writerow({
                "id": f"cust_{i}",
                "audio_path": path,
                "transcript": transcript,
                "words_json": json.dumps(words, ensure_ascii=False),
            })

if __name__ == "__main__":
    main()
