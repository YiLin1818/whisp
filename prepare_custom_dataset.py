import argparse, csv, glob, os, random

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--metadata_csv", required=True, help="CSV with id,audio_path,transcript (optional transcript)")
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # If metadata exists, just split it. If not, create blank transcript rows.
    rows = []
    if os.path.exists(args.metadata_csv):
        with open(args.metadata_csv, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append(r)
    else:
        files = []
        for ext in AUDIO_EXTS:
            files += glob.glob(os.path.join(args.audio_dir, f"*{ext}"))
        for i, p in enumerate(sorted(files)):
            rows.append({"id": f"cust_{i}", "audio_path": p, "transcript": ""})

    random.seed(args.seed)
    random.shuffle(rows)
    n_val = int(len(rows) * args.val_ratio)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    def write(rows_, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "audio_path", "transcript"])
            w.writeheader()
            w.writerows(rows_)

    write(train_rows, args.out_train)
    write(val_rows, args.out_val)
    print(f"Train={len(train_rows)}  Val={len(val_rows)}")

if __name__ == "__main__":
    main()
