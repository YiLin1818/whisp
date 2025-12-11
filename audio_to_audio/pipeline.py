#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
import whisper
from huggingface_hub import hf_hub_download

# Make repo imports work in Colab/standalone runs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import (  # noqa: E402
    set_whisper_dims,
    get_whisper_tokenizer,
    create_dataloader,
)
from models.lora import LoRAWhisper  # noqa: E402
from TTS_model.tts_engine import TTSEngine  # noqa: E402


def download_checkpoint(repo_id: str, filename: str, local_path: str | None = None) -> str:
    if local_path and Path(local_path).exists():
        return str(local_path)
    return hf_hub_download(repo_id=repo_id, filename=filename)


def build_model(
    base_model: str,
    checkpoint_path: str,
    device: str | torch.device = "cuda",
    block_indices: List[int] | None = None,
    rank: int = 128,
    alpha: int = 256,
    dropout: float = 0.05,
) -> tuple[LoRAWhisper, object]:
    set_whisper_dims(base_model)
    tokenizer = get_whisper_tokenizer()

    base = whisper.load_model(base_model, device=device)
    block_indices = block_indices or list(range(12))
    model = LoRAWhisper(
        base_model=base,
        block_indices=block_indices,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def _decode_tokens(token_ids: List[int], tokenizer) -> str:
    clean = []
    for t in token_ids:
        if t == 0:
            break  # padding
        if t == tokenizer.eot:
            break
        if t == tokenizer.sot:
            continue
        clean.append(t)
    return tokenizer.decode(clean).strip()


def predict_next_text(model: LoRAWhisper, tokenizer, mel: torch.Tensor, tokens: torch.Tensor) -> str:
    decoder_input = tokens[:, :-1]
    logits = model(mel, decoder_input)
    pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
    return _decode_tokens(pred_ids, tokenizer)


def run_speech_to_speech(
    base_model: str = "small.en",
    hf_repo: str = "aranos/whisp",
    hf_filename: str = "best_model.pth",
    checkpoint_path: str | None = None,
    dataset_subset: str = "commonvoice",
    dataset_split: str = "train[:20]",
    batch_size: int = 1,
    device: str | torch.device | None = None,
    output_dir: str = "audio_to_audio_outputs",
) -> List[Dict]:
    """
    Main entry point. Returns a list of dicts with text + audio paths.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    ckpt = download_checkpoint(hf_repo, hf_filename, checkpoint_path)
    model, tokenizer = build_model(base_model, ckpt, device=device)
    tts = TTSEngine(device=device)

    loader = create_dataloader(
        datasets_cfg=[{"name": dataset_subset, "split": dataset_split}],
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    results = []
    with torch.no_grad():
        for idx, (mel, token_batch) in enumerate(loader):
            mel = mel.to(device)
            token_batch = token_batch.to(device)

            predicted_text = predict_next_text(model, tokenizer, mel, token_batch)
            target_text = _decode_tokens(token_batch[0].cpu().tolist(), tokenizer)

            audio_path = os.path.join(output_dir, f"sample_{idx:05d}.wav")
            tts.synthesize(predicted_text, out_path=audio_path)

            result = {
                "idx": idx,
                "predicted_text": predicted_text,
                "target_text": target_text,
                "tts_audio": audio_path,
            }
            results.append(result)

            print(
                f"[{idx}] target='{target_text}' | predicted='{predicted_text}' → {audio_path}"
            )

    return results


if __name__ == "__main__":
    run_speech_to_speech()
