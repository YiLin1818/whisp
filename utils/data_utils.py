# data_utils.py — FIXED VERSION
import os
import sys
from pathlib import Path
import whisper
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import List, Dict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from my_whisper.dataset_parsers.asr_alignment.parser import load_asr_alignment_split

# Global dims - no need to load full model
# Global dims - no need to load full model
WHISPER_DIMS = {
    "tiny.en": {"n_mels": 80, "n_audio_ctx": 1500},
    "tiny": {"n_mels": 80, "n_audio_ctx": 1500},
    "base.en": {"n_mels": 80, "n_audio_ctx": 1500},
    "base": {"n_mels": 80, "n_audio_ctx": 1500},
    "small.en": {"n_mels": 80, "n_audio_ctx": 1500},
    "small": {"n_mels": 80, "n_audio_ctx": 1500},
    "medium.en": {"n_mels": 80, "n_audio_ctx": 1500},
    "medium": {"n_mels": 80, "n_audio_ctx": 1500},
    "large": {"n_mels": 128, "n_audio_ctx": 1500},
    "large-v1": {"n_mels": 128, "n_audio_ctx": 1500},
    "large-v2": {"n_mels": 128, "n_audio_ctx": 1500},
    "large-v3": {"n_mels": 128, "n_audio_ctx": 1500},
    "turbo": {"n_mels": 128, "n_audio_ctx": 1500},
}

N_MELS = 80
N_AUDIO_CTX = 1500
INPUT_FRAMES = 3000

def set_whisper_dims(model_name):
    """Set dimensions without loading the full model"""
    global N_MELS, N_AUDIO_CTX, INPUT_FRAMES
    if model_name in WHISPER_DIMS:
        dims = WHISPER_DIMS[model_name]
        N_MELS = dims["n_mels"]
        N_AUDIO_CTX = dims["n_audio_ctx"]
        INPUT_FRAMES = N_AUDIO_CTX * 2
        print(f"Set Whisper dims → n_mels={N_MELS}, n_audio_ctx={N_AUDIO_CTX}, input_frames={INPUT_FRAMES}")
    else:
        print(f"Warning: Unknown model {model_name}, using defaults")

def get_whisper_tokenizer():
    return whisper.tokenizer.get_tokenizer(multilingual=False, task="transcribe", language="en")

def audio_to_log_mel(waveform, n_fft=400, hop_length=160):
    """Convert waveform to log mel spectrogram"""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MELS
    )(waveform)
    
    log_mel = torch.clamp(mel_spec, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    
    return log_mel.squeeze(0)

def pad_or_trim_mel(mel, target_frames):
    """Pad or trim mel spectrogram to target length"""
    T = mel.shape[-1]
    if T > target_frames:
        return mel[:, :target_frames]
    elif T < target_frames:
        return F.pad(mel, (0, target_frames - T))
    return mel

class NextWordDataset(Dataset):
    """Memory-efficient dataset that generates examples on-the-fly"""
    
    def __init__(self, dataset_name: str, split: str, tokenizer, max_token_len: int = 50):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.SOT = tokenizer.sot
        self.EOT = tokenizer.eot
        
        # Load raw data
        print(f"Loading {dataset_name} {split}...")
        self.data = load_asr_alignment_split(dataset_name, split)
        
        # Create index of valid examples
        self.examples = []
        for sample_idx, sample in enumerate(self.data):
            audio_len = len(sample["audio"])
            for word_idx, word in enumerate(sample["words"]):
                end_time = word["start"]
                end_sample = int(end_time * 16000)
                
                # Skip if too short or too long
                if end_sample < 160 or end_sample > audio_len:
                    continue
                
                # Check if word is tokenizable
                word_text = word["word"].strip()
                if not word_text:
                    continue
                
                tokens = self.tokenizer.encode(word_text)
                if len(tokens) == 0 or len(tokens) > max_token_len - 2:
                    continue
                
                self.examples.append({
                    "sample_idx": sample_idx,
                    "end_sample": end_sample,
                    "word": word_text,
                    "tokens": tokens
                })
        
        print(f"Created {len(self.examples)} valid examples from {len(self.data)} samples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        sample = self.data[ex["sample_idx"]]
        
        # Extract audio up to word boundary
        audio_segment = sample["audio"][:ex["end_sample"]]
        
        # Convert to mel spectrogram
        mel = audio_to_log_mel(audio_segment)
        mel = pad_or_trim_mel(mel, INPUT_FRAMES)
        
        # Create token sequence: [SOT, word_tokens..., EOT, PAD...]
        tokens = [self.SOT] + ex["tokens"] + [self.EOT]
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Pad to max_token_len
        if len(token_tensor) < self.max_token_len:
            token_tensor = F.pad(token_tensor, (0, self.max_token_len - len(token_tensor)), value=0)
        else:
            token_tensor = token_tensor[:self.max_token_len]
        
        return mel, token_tensor

def create_dataloader(datasets_cfg, batch_size=2, shuffle=True, num_workers=4):
    """Create DataLoader from config"""
    tokenizer = get_whisper_tokenizer()
    datasets = []
    
    for cfg in datasets_cfg:
        if hasattr(cfg, 'name'):
            name = cfg.name
            split = getattr(cfg, 'split', 'train[:10]')
        else:
            name = cfg["name"]
            split = cfg.get("split", "train[:10]")
        
        ds = NextWordDataset(name, split, tokenizer)
        datasets.append(ds)
    
    combined = ConcatDataset(datasets)
    return DataLoader(
        combined, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )