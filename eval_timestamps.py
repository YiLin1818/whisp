import argparse, csv, json
import numpy as np
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path or HF name")
    ap.add_argument("--manifest", required=True, help="CSV with id,audio_path,transcript,words_json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--whisperx_model", default="small.en")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    import whisperx

    # Use whisperx for word timestamps both ref (already in file) and pred (re-align predicted transcript)
    model = whisperx.load_model(args.whisperx_model, device=args.device, compute_type="float16" if args.device=="cuda" else "int8")
    align_model, metadata = whisperx.load_align_model(language_code="en", device=args.device)

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    rows = rows[: min(args.limit, len(rows))]

    errors_ms = []

    for r in tqdm(rows):
        audio = whisperx.load_audio(r["audio_path"])
        # pred transcript using whisperx model (swap this if you want to decode using HF finetuned model)
        pred = model.transcribe(audio, language="en")
        pred_aligned = whisperx.align(pred["segments"], align_model, metadata, audio, args.device)

        ref_words = json.loads(r["words_json"])
        ref = [(w["w"].strip().lower(), w.get("s")) for w in ref_words if w.get("s") is not None]

        pred_words = []
        for seg in pred_aligned["segments"]:
            for w in seg.get("words", []):
                if w.get("start") is None or w.get("word") is None:
                    continue
                pred_words.append((w["word"].strip().lower(), w["start"]))

        # naive alignment by sequence index (works if transcripts are close; good enough for report metric)
        n = min(len(ref), len(pred_words))
        if n == 0:
            continue

        diffs = []
        for i in range(n):
            diffs.append(abs(ref[i][1] - pred_words[i][1]))
        errors_ms.append(1000.0 * float(np.mean(diffs)))

    if not errors_ms:
        print("No timestamp errors computed (check words_json).")
        return

    print(f"Mean timestamp alignment error (ms): {np.mean(errors_ms):.2f}")
    print(f"Median timestamp alignment error (ms): {np.median(errors_ms):.2f}")

if __name__ == "__main__":
    main()
