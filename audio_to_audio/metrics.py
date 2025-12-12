#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import torch
import torchaudio
from datasets import load_dataset

TARGET_SR = 16000


def _ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    return wav


def _resample(wav: torch.Tensor, sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)


def load_word_audio(subset: str, split: str, sample_idx: int, word_idx: int) -> Tuple[torch.Tensor, int]:
    ds = load_dataset("nguyenvulebinh/asr-alignment", subset, split=split)
    item = ds[sample_idx]
    wav = torch.tensor(item["audio"]["array"])
    sr = item["audio"]["sampling_rate"]
    start = item["word_start"][word_idx]
    end = item["word_end"][word_idx]
    seg = wav[int(start * sr):int(end * sr)]
    seg = _ensure_mono(seg)
    seg = _resample(seg, sr, TARGET_SR)
    return seg, TARGET_SR


def load_pred_audio(path: str, target_sr: int = TARGET_SR) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    wav = _ensure_mono(wav)
    wav = _resample(wav, sr, target_sr)
    return wav, target_sr


def log_mel(wav: torch.Tensor, sr: int, n_mels: int = 80) -> torch.Tensor:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=400, hop_length=160, n_mels=n_mels
    )(wav.unsqueeze(0))
    logmel = torch.clamp(mel, min=1e-10).log10()
    return logmel.squeeze(0).transpose(0, 1)  # [frames, mels]


def mfcc(wav: torch.Tensor, sr: int, n_mfcc: int = 13) -> torch.Tensor:
    mfcc_t = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=n_mfcc, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
    )
    return mfcc_t(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)  # [frames, n_mfcc]


def frame_align(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    T = min(a.shape[0], b.shape[0])
    return a[:T], b[:T]


def mcd_mfcc_l2(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref, est = frame_align(ref, est)
    return torch.norm(ref - est, dim=1).mean().item()


def logmel_l2_and_cos(ref: torch.Tensor, est: torch.Tensor) -> Tuple[float, float]:
    ref, est = frame_align(ref, est)
    l2 = torch.norm(ref - est, dim=1).mean().item()
    cos = torch.nn.functional.cosine_similarity(ref, est, dim=1).mean().item()
    return l2, cos


def si_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + 1e-8)
    proj = alpha * ref
    noise = est - proj
    ratio = (proj.pow(2).sum() + 1e-8) / (noise.pow(2).sum() + 1e-8)
    return 10 * torch.log10(ratio).item()


def compare_word_audio_notebook(
    pred_audio_path: str,
    dataset_subset: str,
    dataset_split: str,
    sample_idx: int,
    word_idx: int,
):
    """
    Returns (metrics_dict, matplotlib_figure) for inline display.
    """
    ref_wav, ref_sr = load_word_audio(dataset_subset, dataset_split, sample_idx, word_idx)
    est_wav, est_sr = load_pred_audio(pred_audio_path, target_sr=ref_sr)

    T = min(ref_wav.shape[-1], est_wav.shape[-1])
    ref_wav = ref_wav[:T]
    est_wav = est_wav[:T]

    sisdr = si_sdr(ref_wav, est_wav)
    ref_mfcc = mfcc(ref_wav, ref_sr)
    est_mfcc = mfcc(est_wav, est_sr)
    mcd = mcd_mfcc_l2(ref_mfcc, est_mfcc)
    ref_logmel = log_mel(ref_wav, ref_sr)
    est_logmel = log_mel(est_wav, est_sr)
    logmel_l2, logmel_cos = logmel_l2_and_cos(ref_logmel, est_logmel)

    metrics = {
        "sample_idx": sample_idx,
        "word_idx": word_idx,
        "sisdr": sisdr,
        "mcd_mfcc_l2": mcd,
        "logmel_l2": logmel_l2,
        "logmel_cosine": logmel_cos,
        "ref_len_samples": int(ref_wav.numel()),
        "est_len_samples": int(est_wav.numel()),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(ref_logmel.numpy().T, aspect="auto", origin="lower")
    axes[0].set_title("Ref log-mel")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(est_logmel.numpy().T, aspect="auto", origin="lower")
    axes[1].set_title("Pred log-mel")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    diff = (ref_logmel - est_logmel).abs()
    im2 = axes[2].imshow(diff.numpy().T, aspect="auto", origin="lower")
    axes[2].set_title(f"|Diff| (MCD~{mcd:.2f})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")
    fig.tight_layout()

    return metrics, fig


if __name__ == "__main__":
    compare_word_audio_notebook()
