# trainer.py — FIXED VERSION WITH GRADIENT ACCUMULATION
import torch
import os
import torch.nn.functional as F
import yaml
import whisper
from typing import Optional

class Trainer:
    def __init__(self, model, optimizer, scheduler=None, device="cuda", cfg=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_metric = float("inf")
        self.cfg = cfg
        self.checkpoint_dir = cfg.CHECKPOINT_DIR
        self.tsv_path = cfg.TSV_PATH
        
        # Gradient accumulation steps (can be set in config)
        self.accumulation_steps = getattr(cfg.TRAINER, 'ACCUMULATION_STEPS', 1)
        
        # Early stopping parameters
        self.patience = getattr(cfg.TRAINER, 'EARLY_STOPPING_PATIENCE', 3)
        self.epochs_without_improvement = 0
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save config
        exp_name = os.path.basename(self.checkpoint_dir)
        cfg_path = os.path.join(self.checkpoint_dir, f"{exp_name}_config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg.__dict__, f, default_flow_style=False)

        # Create TSV log
        if not os.path.exists(self.tsv_path):
            with open(self.tsv_path, "w") as f:
                f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\n")
        
        # Get tokenizer
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=False, task="transcribe", language="en"
        )
        
        print(f"Trainer initialized with gradient accumulation steps: {self.accumulation_steps}")
        if self.accumulation_steps > 1:
            print(f"  → Effective batch size: {cfg.TRAINER.BATCH_SIZE * self.accumulation_steps}")

    def compute_loss_and_accuracy(self, mel, tokens):
        """
        Compute loss and accuracy with proper teacher forcing.
        
        Tokens format: [SOT, word_tok1, word_tok2, ..., EOT, PAD, PAD]
        
        We want to predict:
        - Given [SOT], predict word_tok1
        - Given [SOT, word_tok1], predict word_tok2
        - Given [SOT, word_tok1, word_tok2], predict EOT
        """
        batch_size = tokens.size(0)
        
        # Find actual sequence lengths (before padding)
        # PAD token is 0
        non_pad_mask = tokens != 0
        seq_lens = non_pad_mask.sum(dim=1)  # [batch]
        
        # Ensure we have at least 2 tokens (SOT + something)
        valid_batch_mask = seq_lens > 1
        if not valid_batch_mask.any():
            # Return dummy values if entire batch is invalid
            return torch.tensor(0.0, device=mel.device), 0.0
        
        # Teacher forcing: input is all tokens except last, target is all tokens except first
        decoder_input = tokens[:, :-1]  # [batch, seq_len-1]
        targets = tokens[:, 1:]         # [batch, seq_len-1]
        
        # Forward pass
        with torch.amp.autocast("cuda"):
            logits = self.model(mel, decoder_input)  # [batch, seq_len-1, vocab]
            
            # Compute loss (ignore padding)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # [batch*(seq_len-1), vocab]
                targets.reshape(-1),                   # [batch*(seq_len-1)]
                ignore_index=0,  # Ignore padding
                reduction='mean'
            )
        
        # Compute accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)  # [batch, seq_len-1]
            
            # Only count non-padding positions
            target_mask = targets != 0
            correct = (predictions == targets) & target_mask
            
            num_correct = correct.sum().item()
            num_total = target_mask.sum().item()
            accuracy = num_correct / max(num_total, 1)
        
        return loss, accuracy

    def train(self, dataloader, criterion, max_epochs, start_epoch=0, val_dataloader=None):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model.to(self.device)
        scaler = torch.amp.GradScaler()
        
        for epoch in range(start_epoch, max_epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_tokens = 0
            num_batches = 0
            
            # Zero gradients at the start of epoch
            self.optimizer.zero_grad()

            import time
            epoch_start = time.time()
            batch_times = []
            
            for batch_idx, (mel, tokens) in enumerate(dataloader):
                batch_start = time.time()
                
                mel = mel.to(self.device)
                tokens = tokens.to(self.device)
                
                # Compute loss and accuracy
                loss, batch_acc = self.compute_loss_and_accuracy(mel, tokens)
                
                batch_times.append(time.time() - batch_start)
                
                # Check for NaN
                if not torch.isfinite(loss):
                    print(f"\n🔥 NaN detected in batch {batch_idx} 🔥")
                    print(f"  mel shape: {mel.shape}, tokens shape: {tokens.shape}")
                    print(f"  tokens sample: {tokens[0][:10]}")
                    raise ValueError("NaN detected — aborting training.")
                
                # Scale loss by accumulation steps (to maintain correct gradient magnitude)
                loss = loss / self.accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Only step optimizer every accumulation_steps batches
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    # Unscale and clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Clean gradients
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1e2, neginf=-1e2)
                    
                    # Optimizer step
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                
                # Accumulate metrics (use unscaled loss for logging)
                total_loss += loss.detach().item() * self.accumulation_steps
                
                # For accuracy, track actual counts
                non_pad_mask = tokens[:, 1:] != 0  # targets are tokens[:, 1:]
                num_tokens_in_batch = non_pad_mask.sum().item()
                total_correct += batch_acc * num_tokens_in_batch
                total_tokens += num_tokens_in_batch
                num_batches += 1
                
                # Periodic logging with timing info
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = total_loss / num_batches
                    avg_acc = total_correct / max(total_tokens, 1)
                    effective_batch = (batch_idx + 1) // self.accumulation_steps
                    avg_batch_time = sum(batch_times[-100:]) / len(batch_times[-100:])
                    batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
                    
                    # Estimate time remaining
                    batches_left = len(dataloader) - (batch_idx + 1)
                    time_left_sec = batches_left * avg_batch_time
                    time_left_min = time_left_sec / 60
                    
                    print(f"  Batch {batch_idx+1}/{len(dataloader)} "
                          f"(effective: {effective_batch}): "
                          f"loss={avg_loss:.4f}, acc={avg_acc:.4f} | "
                          f"{batches_per_sec:.1f} batch/s | "
                          f"ETA: {time_left_min:.1f}m")

            train_loss = total_loss / num_batches
            train_acc = total_correct / max(total_tokens, 1)
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} "
                  f"(took {epoch_time/60:.1f}m, {epoch_time/num_batches:.2f}s/batch)")

            # Validation
            if val_dataloader:
                val_loss, val_acc = self.evaluate(val_dataloader, criterion)
                print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                current_metric = val_loss
            else:
                val_loss = val_acc = None
                current_metric = train_loss

            # Checkpoint
            ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_metric": self.best_metric,
                "metric": current_metric,
            }, ckpt_path)

            # Best model and early stopping
            if val_loss is not None and val_loss < self.best_metric:
                self.best_metric = val_loss
                best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_path)
                print(f"✓ New best model saved (val_loss={val_loss:.4f})")
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s)")
                
                if self.epochs_without_improvement >= self.patience:
                    print(f"\n🛑 Early stopping triggered after {self.patience} epochs without improvement")
                    print(f"Best val_loss: {self.best_metric:.4f}")
                    break

            # Logging
            with open(self.tsv_path, "a") as f:
                val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "NA"
                val_acc_str = f"{val_acc:.6f}" if val_acc is not None else "NA"
                f.write(f"{epoch}\t{train_loss:.6f}\t{train_acc:.6f}\t{val_loss_str}\t{val_acc_str}\n")

            if self.scheduler:
                self.scheduler.step()
            
            print()  # Blank line between epochs

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for mel, tokens in dataloader:
                mel = mel.to(self.device)
                tokens = tokens.to(self.device)
                
                loss, batch_acc = self.compute_loss_and_accuracy(mel, tokens)
                
                total_loss += loss.item()
                
                # Track accuracy
                non_pad_mask = tokens[:, 1:] != 0
                num_tokens_in_batch = non_pad_mask.sum().item()
                total_correct += batch_acc * num_tokens_in_batch
                total_tokens += num_tokens_in_batch
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_correct / max(total_tokens, 1)
        return avg_loss, avg_acc

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "best_metric" in ckpt:
            self.best_metric = ckpt["best_metric"]
        print(f"✓ Resumed from {path}, epoch {ckpt['epoch']}")
        return ckpt["epoch"] + 1

    @staticmethod
    def find_latest_checkpoint(folder):
        if not os.path.exists(folder):
            return None
        ckpts = [f for f in os.listdir(folder) if f.startswith("epoch_") and f.endswith(".pth")]
        if not ckpts:
            return None
        ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        return os.path.join(folder, ckpts[-1])