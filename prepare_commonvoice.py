import argparse, json, os
from datasets import load_dataset
from tqdm import tqdm

def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_train", type=int, default=30000)
    ap.add_argument("--max_val", type=int, default=5000)
    args = ap.parse_args()

    ds = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train")
    ds_val = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="validation")

    def to_row(ex, idx):
        # HuggingFace CommonVoice provides `audio` with path + array
        return {
            "id": f"cv_{idx}",
            "audio_path": ex["audio"]["path"],
            "transcript": ex["sentence"],
        }

    train_rows = [to_row(ds[i], i) for i in tqdm(range(min(args.max_train, len(ds))))]
    val_rows = [to_row(ds_val[i], i) for i in tqdm(range(min(args.max_val, len(ds_val))))]

    # small subsets for tiny-en experiments if needed
    train_small = train_rows[:5000]
    val_small = val_rows[:1000]

    write_jsonl(train_rows, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(val_rows, os.path.join(args.out_dir, "val.jsonl"))
    write_jsonl(train_small, os.path.join(args.out_dir, "train_small.jsonl"))
    write_jsonl(val_small, os.path.join(args.out_dir, "val_small.jsonl"))

    print("Wrote manifests to", args.out_dir)

if __name__ == "__main__":
    main()
