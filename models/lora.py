import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import math


class LoRAWhisper(nn.Module):
    """
    LoRA wrapper for OpenAI Whisper
    """
    def __init__(self, base_model: whisper, block_indices: list[int], rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.model = base_model
        self.block_indices = block_indices
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Inject LoRA layers into specified encoder & decoder blocks
        for module in self._get_attention_modules():
            self._inject_lora(module)

    def _get_attention_modules(self):
        modules = []
        # Encoder attention blocks
        for i, block in enumerate(self.model.encoder.blocks):
            if i in self.block_indices:
                modules.append(block.attn)
        # Decoder self-attention blocks
        for i, block in enumerate(self.model.decoder.blocks):
            if i in self.block_indices:
                modules.append(block.attn)
        return modules

    def _inject_lora(self, attn_module):
        # patch q_proj, k_proj, v_proj, out as LoRA linear layers
        for name in ["query", "key", "value", "out"]:
            orig = getattr(attn_module, name)
            lora_layer = LinearLoRA(orig, self.rank, self.alpha, self.dropout)
            setattr(attn_module, name, lora_layer)

    def forward(self, mel, tokens):
        return self.model(mel, tokens)


class LinearLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        
        in_dim = linear.in_features
        out_dim = linear.out_features
        
        # LoRA matrices
        self.A = nn.Parameter(torch.zeros(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        
        # CRITICAL: Initialize A with small values, B stays at zero
        # This ensures LoRA starts with near-zero contribution
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Freeze base weights
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        # Base linear transformation
        base_output = F.linear(x, self.linear.weight, self.linear.bias)
        
        # LoRA path: x -> A -> dropout -> B -> scale
        lora_out = x @ self.A.T  # (batch, seq, rank)
        lora_out = self.dropout(lora_out)  # Apply dropout in LoRA path, not to input!
        lora_out = lora_out @ self.B.T  # (batch, seq, out_dim)
        lora_out = lora_out * self.scaling  # Scale down
        
        return base_output + lora_out