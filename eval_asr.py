import argparse, csv, json
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch, librosa
from jiwer import wer
from tqdm import tqdm

def read_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y

def decode_batch(model, processor, audios, device, num_beams=5):
    feats = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        pred_ids = model.generate(feats, num_beams=num_beams)
    return processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

def eval_manifest(model_name, manifest_csv, device="cuda", limit=None):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    rows = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    if limit:
        rows = rows[:limit]

    preds, refs = [], []
    for r in tqdm(rows):
        audio = read_audio(r["audio_path"])
        pred = decode_batch(model, processor, [audio], device)[0]
        preds.append(pred.strip().lower())
        refs.append((r["transcript"] or "").strip().lower())

    w = wer(refs, preds)
    return {"wer": w, "acc_proxy": 1.0 - w}

def eval_hf_dataset(model_name, dataset_hf, dataset_config, split="validation", limit=500, device="cuda"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    ds = load_dataset(dataset_hf, dataset_config, split=split)
    ds = ds.select(range(min(limit, len(ds))))

    preds, refs = [], []
    for ex in tqdm(ds):
        audio = ex["audio"]["array"]
        ref = ex.get("sentence") or ex.get("text") or ""
        pred = decode_batch(model, processor, [audio], device)[0]
        preds.append(pred.strip().lower())
        refs.append(ref.strip().lower())

    w = wer(refs, preds)
    return {"wer": w, "acc_proxy": 1.0 - w}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF name or local path (e.g., outputs/.../best)")
    ap.add_argument("--manifest", default=None, help="CSV with id,audio_path,transcript")
    ap.add_argument("--dataset_hf", default=None)
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.manifest:
        res = eval_manifest(args.model, args.manifest, device=args.device, limit=args.limit)
    else:
        res = eval_hf_dataset(args.model, args.dataset_hf, args.dataset_config, split=args.split, limit=args.limit, device=args.device)

    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
