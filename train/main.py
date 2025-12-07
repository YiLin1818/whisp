#!/usr/bin/env python3
"""
main.py - Simple approach with space-separated overrides
Usage: python main.py --config base.yaml KEY1 VALUE1 KEY2 VALUE2
"""
import sys
import os
from pathlib import Path

# CRITICAL: Set cache directories FIRST, before any other imports
os.environ["HF_DATASETS_CACHE"] = "/scratch/scholar/yang2501/hf_cache/datasets"
os.environ["HF_HUB_CACHE"] = "/scratch/scholar/yang2501/hf_cache/hub"
os.environ["HF_HOME"] = "/scratch/scholar/yang2501/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/scholar/yang2501/hf_cache/transformers"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import argparse
from trainer import Trainer
from my_whisper.utils.data_utils import create_dataloader, set_whisper_dims
import whisper
from my_whisper.models.lora import LoRAWhisper


def parse_value(v):
    """Smart type conversion"""
    v = v.strip()
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    if v.lower() in ('null', 'none'):
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def apply_overrides(cfg, overrides):
    """Apply key-value pairs: ['MODEL.BASE', 'tiny', 'LORA.RANK', '16']"""
    if len(overrides) % 2 != 0:
        print(f"ERROR: Overrides must be key-value pairs (got {len(overrides)} args)")
        print(f"Args: {overrides}")
        sys.exit(1)
    
    i = 0
    while i < len(overrides):
        key = overrides[i]
        val = overrides[i + 1]
        
        # Navigate nested dict
        keys = key.split('.')
        current = cfg
        for k in keys[:-1]:
            if k.isdigit():
                # Handle list indices like TRAIN_DATASETS.0.name
                idx = int(k)
                # Ensure list exists and is long enough
                if not isinstance(current, list):
                    current = []
                while len(current) <= idx:
                    current.append({})
                current = current[idx]
            else:
                if k not in current:
                    current[k] = {}
                current = current[k]
        
        # Set value
        final_key = keys[-1]
        if final_key.isdigit():
            idx = int(final_key)
            while len(current) <= idx:
                current.append({})
            current[idx] = parse_value(val)
        else:
            current[final_key] = parse_value(val)
        
        print(f"  {key} = {val}")
        i += 2


def parse_block_indices(s):
    """Convert '6-11,18-23' → [6,7,8,9,10,11,18,19,20,21,22,23]"""
    if not s or s in ("null", "None", None):
        return list(range(24))
    indices = set()
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return sorted(indices)


class DotDict:
    """Dot-accessible dict"""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DotDict(v))
            elif isinstance(v, list):
                setattr(self, k, [DotDict(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    
    print(f"Config file: {args.config}")
    print(f"Number of override args: {len(args.overrides)}")
    
    # Load config
    try:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        print(f"✓ Loaded config from {args.config}")
    except Exception as e:
        print(f"ERROR: Could not load config file {args.config}")
        print(f"  {e}")
        sys.exit(1)
    
    # Apply overrides
    if args.overrides:
        print(f"Applying {len(args.overrides)//2} overrides:")
        apply_overrides(cfg_dict, args.overrides)
        print()
    
    cfg = DotDict(cfg_dict)
    
    # Validate
    if not hasattr(cfg, 'MODEL') or cfg.MODEL.BASE is None:
        print("\n" + "="*70)
        print("ERROR: MODEL.BASE must be set!")
        print("="*70)
        print("\nConfig after overrides:")
        print(yaml.dump(cfg_dict, default_flow_style=False))
        print("="*70)
        sys.exit(1)
    
    print(f"Model: {cfg.MODEL.BASE}")
    print(f"LoRA: rank={cfg.LORA.RANK}, alpha={cfg.LORA.ALPHA}")
    print(f"Training: {len(cfg.TRAIN_DATASETS)} datasets, batch_size={cfg.TRAINER.BATCH_SIZE}")
    print()
    
    # Set dimensions
    set_whisper_dims(cfg.MODEL.BASE)
    
    # Parse blocks
    block_str = getattr(cfg.LORA, "TARGET_BLOCKS", None) or getattr(cfg.LORA, "BLOCK_INDICES", None)
    cfg.LORA.BLOCK_INDICES = parse_block_indices(block_str)
    
    # Create dataloaders
    train_loader = create_dataloader(cfg.TRAIN_DATASETS, batch_size=cfg.TRAINER.BATCH_SIZE)
    val_loader = create_dataloader(cfg.VALIDATION_DATASETS, batch_size=cfg.TRAINER.BATCH_SIZE)
    
    # Load model
    print(f"Loading Whisper model: {cfg.MODEL.BASE}")
    base_model = whisper.load_model(cfg.MODEL.BASE, device=cfg.TRAINER.DEVICE)
    
    model = LoRAWhisper(
        base_model=base_model,
        block_indices=cfg.LORA.BLOCK_INDICES,
        rank=cfg.LORA.RANK,
        alpha=cfg.LORA.ALPHA,
        dropout=cfg.LORA.DROPOUT,
    ).to(cfg.TRAINER.DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)   # Add L2 regularization
    criterion = torch.nn.CrossEntropyLoss()
    
    # Trainer (checkpoint_dir and tsv_path come from cfg)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device=cfg.TRAINER.DEVICE,
        cfg=cfg
    )
    
    # Resume if possible
    checkpoint_dir = cfg.CHECKPOINT_DIR
    latest_ckpt = Trainer.find_latest_checkpoint(checkpoint_dir)
    start_epoch = 0
    if latest_ckpt:
        print(f"Resuming from {latest_ckpt}")
        start_epoch = trainer.load_checkpoint(latest_ckpt)
    
    # Train
    trainer.train(
        dataloader=train_loader,
        criterion=criterion,
        max_epochs=cfg.TRAINER.MAX_EPOCHS,
        start_epoch=start_epoch,
        val_dataloader=val_loader,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Training failed")
        print(f"{'='*70}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}")
        sys.exit(1)