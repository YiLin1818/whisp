#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

TARGET_SR = 16000


def _ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    return wav


def _resample(wav: torch.Tensor, sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return F.resample(wav, orig_freq=sr, new_freq=target_sr)


def load_audio(path: str, target_sr: int = TARGET_SR) -> Tuple[torch.Tensor, int]:
    data, sr = sf.read(path)
    wav = torch.from_numpy(data.astype(np.float32))
    wav = _ensure_mono(wav)
    wav = _resample(wav, sr, target_sr)
    return wav, target_sr


def mfcc(wav: torch.Tensor, sr: int, n_mfcc: int = 13) -> torch.Tensor:
    import torchaudio.transforms as T
    mfcc_t = T.MFCC(
        sample_rate=sr, n_mfcc=n_mfcc, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
    )
    return mfcc_t(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)  # [frames, n_mfcc]


def frame_align(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    T = min(a.shape[0], b.shape[0])
    return a[:T], b[:T]


def mcd(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref, est = frame_align(ref, est)
    return torch.norm(ref - est, dim=1).mean().item()


# Lazy-loaded embedding model
_embedding_model = None
_embedding_processor = None


def _load_embedding_model(device: str = "cpu"):
    global _embedding_model, _embedding_processor
    if _embedding_model is None:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        model_name = "facebook/wav2vec2-base"
        _embedding_processor = Wav2Vec2Processor.from_pretrained(model_name)
        _embedding_model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()
    return _embedding_model, _embedding_processor


def get_embedding(wav: torch.Tensor, sr: int = TARGET_SR, device: str = "cpu") -> torch.Tensor:
    """Extract wav2vec2 embedding (mean-pooled over time)."""
    model, processor = _load_embedding_model(device)
    inputs = processor(wav.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pool over time dimension
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding.cpu()


def embedding_cosine_similarity(wav1: torch.Tensor, wav2: torch.Tensor, sr: int = TARGET_SR, device: str = "cpu") -> float:
    """Compute cosine similarity between wav2vec2 embeddings of two audio clips."""
    emb1 = get_embedding(wav1, sr, device)
    emb2 = get_embedding(wav2, sr, device)
    cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    return cos_sim


def compare_audio(
    idx: int,
    pred_dir: str = "audio_to_audio_outputs",
    target_dir: str = "audio_to_audio_target",
    compute_embedding: bool = True,
    device: str = "cpu",
) -> Tuple[Dict, plt.Figure]:
    pred_path = os.path.join(pred_dir, f"sample_{idx:05d}.wav")
    target_path = os.path.join(target_dir, f"sample_{idx:05d}.wav")

    pred_wav, sr = load_audio(pred_path)
    target_wav, _ = load_audio(target_path)

    T = min(pred_wav.shape[-1], target_wav.shape[-1])
    pred_wav = pred_wav[:T]
    target_wav = target_wav[:T]

    target_mfcc = mfcc(target_wav, sr)
    pred_mfcc = mfcc(pred_wav, sr)
    mcd_val = mcd(target_mfcc, pred_mfcc)

    metrics = {"idx": idx, "mcd": mcd_val}
    
    # Embedding similarity (wav2vec2)
    if compute_embedding:
        cos_sim = embedding_cosine_similarity(target_wav, pred_wav, sr, device)
        metrics["embedding_cosine"] = cos_sim

    # Visualization
    target_mfcc_aligned, pred_mfcc_aligned = frame_align(target_mfcc, pred_mfcc)
    diff = (target_mfcc_aligned - pred_mfcc_aligned).abs()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axes[0].imshow(target_mfcc_aligned.numpy().T, aspect="auto", origin="lower")
    axes[0].set_title("Target MFCC")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pred_mfcc_aligned.numpy().T, aspect="auto", origin="lower")
    axes[1].set_title("Predicted MFCC")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff.numpy().T, aspect="auto", origin="lower")
    axes[2].set_title(f"|Diff| (MCD={mcd_val:.2f})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("Frames")
        ax.set_ylabel("MFCC bins")
    fig.tight_layout()

    return metrics, fig


def compare_all(
    pred_dir: str = "audio_to_audio_outputs",
    target_dir: str = "audio_to_audio_target",
    show_individual: bool = False,
    compute_embedding: bool = True,
    device: str = "cpu",
):
    """
    Compare all samples and return aggregate metrics + summary plot.
    """
    import glob
    files = sorted(glob.glob(os.path.join(pred_dir, "sample_*.wav")))
    
    all_metrics = []
    for f in files:
        idx = int(os.path.basename(f).replace("sample_", "").replace(".wav", ""))
        try:
            m, fig = compare_audio(idx, pred_dir, target_dir, compute_embedding, device)
            all_metrics.append(m)
            emb_str = f", CosSim = {m['embedding_cosine']:.3f}" if "embedding_cosine" in m else ""
            print(f"Sample {idx}: MCD = {m['mcd']:.2f}{emb_str}")
            if show_individual:
                plt.show()
            else:
                plt.close(fig)
        except Exception as e:
            print(f"Sample {idx}: Error - {e}")
    
    if not all_metrics:
        print("No samples found.")
        return all_metrics, None
    
    # Aggregate stats
    mcd_vals = [m["mcd"] for m in all_metrics]
    avg_mcd = sum(mcd_vals) / len(mcd_vals)
    min_mcd = min(mcd_vals)
    max_mcd = max(mcd_vals)
    
    print(f"\n=== Summary ===")
    print(f"Samples: {len(all_metrics)}")
    print(f"Avg MCD: {avg_mcd:.2f} (min={min_mcd:.2f}, max={max_mcd:.2f})")
    
    if compute_embedding and "embedding_cosine" in all_metrics[0]:
        cos_vals = [m["embedding_cosine"] for m in all_metrics]
        avg_cos = sum(cos_vals) / len(cos_vals)
        print(f"Avg Embedding Cosine Sim: {avg_cos:.3f} (min={min(cos_vals):.3f}, max={max(cos_vals):.3f})")
    
    # Summary plot
    has_embedding = compute_embedding and "embedding_cosine" in all_metrics[0]
    
    if has_embedding:
        cos_vals = [m["embedding_cosine"] for m in all_metrics]
        avg_cos = sum(cos_vals) / len(cos_vals)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MCD histogram
        axes[0].hist(mcd_vals, bins=max(5, len(mcd_vals)//2), color="steelblue", edgecolor="white")
        axes[0].axvline(avg_mcd, color="red", linestyle="--", linewidth=2, label=f"Avg={avg_mcd:.2f}")
        axes[0].set_xlabel("MCD")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"MCD Distribution\n(Avg={avg_mcd:.2f}, Min={min_mcd:.2f}, Max={max_mcd:.2f})")
        axes[0].legend()
        
        # Cosine similarity box plot
        bp = axes[1].boxplot(cos_vals, patch_artist=True, vert=True)
        bp['boxes'][0].set_facecolor('seagreen')
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"Embedding Cosine Similarity\n(Avg={avg_cos:.3f}, Min={min(cos_vals):.3f}, Max={max(cos_vals):.3f})")
        axes[1].set_xticks([])
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(mcd_vals, bins=max(5, len(mcd_vals)//2), color="steelblue", edgecolor="white")
        ax.axvline(avg_mcd, color="red", linestyle="--", linewidth=2, label=f"Avg={avg_mcd:.2f}")
        ax.set_xlabel("MCD")
        ax.set_ylabel("Count")
        ax.set_title(f"MCD Distribution\n(Avg={avg_mcd:.2f}, Min={min_mcd:.2f}, Max={max_mcd:.2f})")
        ax.legend()
    
    fig.tight_layout()
    
    return all_metrics, fig


if __name__ == "__main__":
    all_metrics, summary_fig = compare_all(show_individual=False)
    if summary_fig:
        plt.show()
