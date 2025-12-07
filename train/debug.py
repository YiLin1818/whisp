#!/usr/bin/env python3
"""
Debug script to see what the model is actually predicting
"""
import sys
import os
from pathlib import Path

# CRITICAL: Set cache directories FIRST, before any other imports
os.environ["HF_DATASETS_CACHE"] = "/scratch/scholar/yang2501/hf_cache/datasets"
os.environ["HF_HUB_CACHE"] = "/scratch/scholar/yang2501/hf_cache/hub"
os.environ["HF_HOME"] = "/scratch/scholar/yang2501/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/scholar/yang2501/hf_cache/transformers"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import whisper
from my_whisper.utils.data_utils import create_dataloader, set_whisper_dims
from my_whisper.models.lora import LoRAWhisper

# Configuration
MODEL_NAME = "small.en"
CHECKPOINT_PATH = "./checkpoints/perfect_resume_test_20251206_203246/best_model.pth"
DEVICE = "cuda"

# Setup
set_whisper_dims(MODEL_NAME)
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, task="transcribe", language="en")

# Load model
print(f"Loading model from {CHECKPOINT_PATH}")
base_model = whisper.load_model(MODEL_NAME, device=DEVICE)
model = LoRAWhisper(
    base_model=base_model,
    block_indices=list(range(12)),  # Adjust to your config
    rank=128,
    alpha=256,
    dropout=0.05,
).to(DEVICE)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

# Create validation dataloader
val_cfg = [{
    "name": "commonvoice",
    "split": "train[10000:10010]"  # Just 10 samples
}]
val_loader = create_dataloader(val_cfg, batch_size=1, shuffle=False)

print("\n" + "="*80)
print("PREDICTION ANALYSIS")
print("="*80)

correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for i, (mel, tokens) in enumerate(val_loader):
        if i >= 10:  # Only show first 10
            break
            
        mel = mel.to(DEVICE)
        tokens = tokens.to(DEVICE)
        
        # Get prediction
        decoder_input = tokens[:, :-1]
        targets = tokens[:, 1:]
        
        logits = model(mel, decoder_input)
        predictions = logits.argmax(dim=-1)
        
        # Decode
        target_tokens = targets[0].cpu().tolist()
        pred_tokens = predictions[0].cpu().tolist()
        
        # Remove padding
        target_tokens = [t for t in target_tokens if t != 0]
        pred_tokens = pred_tokens[:len(target_tokens)]
        
        target_text = tokenizer.decode(target_tokens)
        pred_text = tokenizer.decode(pred_tokens)
        
        is_correct = target_text.strip().lower() == pred_text.strip().lower()
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"\nExample {i+1}:")
        print(f"  Target:     '{target_text}'")
        print(f"  Predicted:  '{pred_text}'")
        print(f"  Match: {'✓' if is_correct else '✗'}")
        print(f"  Target tokens:    {target_tokens[:10]}")
        print(f"  Predicted tokens: {pred_tokens[:10]}")

print("\n" + "="*80)
print(f"Overall Accuracy: {correct_predictions}/{total_predictions} = {correct_predictions/total_predictions*100:.1f}%")
print("="*80)

# Check for common failure patterns
print("\n" + "="*80)
print("DIAGNOSTIC CHECKS")
print("="*80)

# Check if model is just predicting most common tokens
from collections import Counter
all_pred_tokens = []

with torch.no_grad():
    for i, (mel, tokens) in enumerate(val_loader):
        if i >= 100:
            break
        mel = mel.to(DEVICE)
        tokens = tokens.to(DEVICE)
        decoder_input = tokens[:, :-1]
        logits = model(mel, decoder_input)
        predictions = logits.argmax(dim=-1)
        all_pred_tokens.extend(predictions[0].cpu().tolist())

token_counts = Counter(all_pred_tokens)
print("\nMost common predicted tokens:")
for token_id, count in token_counts.most_common(10):
    if token_id != 0:  # Skip padding
        print(f"  Token {token_id}: {count} times - '{tokenizer.decode([token_id])}'")