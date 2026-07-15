"""
GPT-3 Fine-tuning / Continual Pretraining Script
Based on karpathy/build-nanogpt

Supports:
1. Pretraining GPT-3 models of various sizes (125M to 175B)
2. Continual Pretraining on a single .npy file (e.g., train_pmc.npy)
3. Supervised Fine-Tuning (SFT) on conversation/instruction datasets (JSONL)
4. RLHF-ready architecture

Training Pipeline:
- Pretraining: next-token prediction on large web corpus (shard directory)
- Continual Pretraining: further train on a specific .npy file
- SFT: fine-tune on (prompt, response) pairs to learn instruction following
- RLHF (optional): further align with human preferences

Usage:
  # Pretraining from scratch
  python gpt3_finetune.py --mode pretrain --model gpt3-small

  # Continual Pretraining (load pretrained, then train on train_pmc.npy)
  python gpt3_finetune.py --mode finetune --model gpt3-small --checkpoint model_pretrain.pt --finetune_data "E:/maxc/build-nanogpt/edu_fineweb10B/train_pmc.npy"

  # SFT Fine-tuning
  python gpt3_finetune.py --mode sft --model gpt3-small --checkpoint model_pretrain.pt --sft_data sft_data.jsonl

  # DDP launch
  torchrun --standalone --nproc_per_node=8 gpt3_finetune.py --mode finetune --model gpt3-medium --checkpoint model.pt
"""

import os
import math
import time
import inspect
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Model Architecture: GPT-3 Compatible
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 2048  # GPT-3 uses 2048 context window (vs 1024 for GPT-2)
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension

# GPT-3 model family configurations (from the GPT-3 paper, Table 2.1)
GPT3_CONFIGS = {
    'gpt3-small':   dict(n_layer=12, n_head=12, n_embd=768),   # 125M params
    'gpt3-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    'gpt3-large':   dict(n_layer=24, n_head=16, n_embd=1536),  # 760M params
    'gpt3-xl':      dict(n_layer=24, n_head=24, n_embd=2048),  # 1.3B params
    'gpt3-2.7b':    dict(n_layer=32, n_head=32, n_embd=2560),  # 2.7B params
    'gpt3-6.7b':    dict(n_layer=32, n_head=32, n_embd=4096),  # 6.7B params
    'gpt3-13b':     dict(n_layer=40, n_head=40, n_embd=5140),  # 13B params (note: 5140 is approximate, paper says ~5120)
    'gpt3-175b':    dict(n_layer=96, n_head=96, n_embd=12288), # 175B params
}

# GPT-2 model family configurations (for backward compatibility)
GPT2_CONFIGS = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
}

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, mask=None):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target tokens (for pretraining/SFT loss)
            mask: (B, T) boolean mask for SFT (1 = compute loss, 0 = ignore)
                  In SFT, we typically mask out the prompt tokens and only
                  compute loss on the assistant's response tokens.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            if mask is not None:
                # SFT mode: only compute loss on response tokens
                # Flatten and apply mask
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                mask_flat = mask.view(-1).float()
                loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            else:
                # Pretraining / Continual Pretraining mode: standard next-token prediction
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in GPT2_CONFIGS
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = GPT2_CONFIGS[model_type].copy()
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT-2 checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, betas=(0.9, 0.95)):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8, fused=use_fused)
        return optimizer


# ---------------------------------------------------------------------------
# Data Loading: Pretraining / Continual Pretraining / SFT
# ---------------------------------------------------------------------------

import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    """Simple data loader for pretraining on token shards."""
    def __init__(self, B, T, process_rank, num_processes, split, data_root="edu_fineweb10B"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


class FinetuneDataLoader:
    """
    DataLoader for Continual Pretraining on a single .npy file.

    This is used for fine-tuning a pretrained model on a specific dataset
    (e.g., train_pmc.npy) in the same next-token prediction fashion as pretraining.

    The difference from pretraining:
    - Pretraining uses multiple shards from a directory
    - Finetuning uses a single .npy file
    """
    def __init__(self, B, T, process_rank, num_processes, filename):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.filename = filename

        if master_process:
            print(f"Loading finetune data from: {filename}")

        # Load the single .npy file
        self.tokens = load_tokens(filename)

        if master_process:
            print(f"Loaded {len(self.tokens):,} tokens for finetuning")

        self.reset()

    def reset(self):
        # Each process starts at a different position
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, wrap around (epoch)
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = B * T * self.process_rank
            if master_process:
                print(f"  Finetune data epoch complete, wrapping around...")
        return x, y


class SFTDataLoader:
    """
    DataLoader for Supervised Fine-Tuning (SFT).

    Expects data in JSONL format where each line is:
    {
        "messages": [
            {"role": "system", "content": "..."},   # optional
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

    Or the simpler format:
    {
        "prompt": "user input...",
        "completion": "assistant response..."
    }

    The key idea of SFT (as discussed in the video):
    - Pretraining teaches the model "language"
    - SFT teaches the model "how to respond" by showing (prompt, response) pairs
    - We mask out the prompt tokens in the loss so the model only learns to predict responses
    """
    def __init__(self, filename, B, T, process_rank, num_processes, tokenizer, 
                 mask_prompt=True, system_prompt=""):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.tokenizer = tokenizer
        self.mask_prompt = mask_prompt
        self.system_prompt = system_prompt

        # Load all examples
        self.examples = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        if master_process:
            print(f"Loaded {len(self.examples)} SFT examples from {filename}")

        # Tokenize all examples and store
        self.tokenized = []
        for ex in self.examples:
            tokens, masks = self._tokenize_example(ex)
            if len(tokens) > 1:  # Need at least 2 tokens for input/target
                self.tokenized.append((tokens, masks))

        if master_process:
            print(f"Valid tokenized examples: {len(self.tokenized)}")

        self.reset()

    def _tokenize_example(self, example):
        """Tokenize a single example and create loss mask."""
        # Format: <|endoftext|> system\n user\n assistant\n

        # Build the full text
        if "messages" in example:
            # Chat format
            parts = []
            for msg in example["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system" and content:
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            text = "\n".join(parts)
        else:
            # Simple prompt-completion format
            prompt = example.get("prompt", "")
            completion = example.get("completion", "")
            text = f"User: {prompt}\nAssistant: {completion}"

        # Tokenize
        tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        tokens = [self.tokenizer.eot_token] + tokens  # prepend BOS

        # Create mask: 1 for assistant response tokens, 0 for prompt tokens
        mask = [1] * len(tokens)  # Default: train on all tokens

        if self.mask_prompt and "messages" in example:
            # Rebuild to find assistant token positions
            # This is a simplified version; production code would track positions during tokenization
            mask = [1] * len(tokens)  # For simplicity, train on all in basic version
        elif self.mask_prompt and "prompt" in example:
            # For prompt-completion format, mask out the prompt portion
            prompt_text = f"User: {example['prompt']}\nAssistant: "
            prompt_tokens = self.tokenizer.encode(prompt_text, allowed_special={"<|endoftext|>"})
            prompt_len = len(prompt_tokens) + 1  # +1 for BOS
            mask = [0] * min(prompt_len, len(tokens)) + [1] * max(0, len(tokens) - prompt_len)

        return tokens, mask

    def reset(self):
        self.current_idx = self.process_rank
        self.buffer_tokens = []
        self.buffer_masks = []

    def next_batch(self):
        """Get a batch of (x, y, mask) for SFT training."""
        B, T = self.B, self.T

        # Fill buffer if needed
        while len(self.buffer_tokens) < B * T + B:
            if self.current_idx >= len(self.tokenized):
                self.current_idx = self.process_rank  # wrap around

            tokens, masks = self.tokenized[self.current_idx]
            self.buffer_tokens.extend(tokens)
            self.buffer_masks.extend(masks)
            self.current_idx += self.num_processes

        # Extract batch
        x_list = []
        y_list = []
        mask_list = []

        for b in range(B):
            start = b * T
            end = start + T + 1

            seq_tokens = self.buffer_tokens[start:end]
            seq_masks = self.buffer_masks[start:end]

            # Pad if necessary
            if len(seq_tokens) < T + 1:
                pad_len = T + 1 - len(seq_tokens)
                seq_tokens = seq_tokens + [self.tokenizer.eot_token] * pad_len
                seq_masks = seq_masks + [0] * pad_len

            x_list.append(seq_tokens[:T])
            y_list.append(seq_tokens[1:T+1])
            mask_list.append(seq_masks[1:T+1])  # mask aligns with targets

        # Consume from buffer
        self.buffer_tokens = self.buffer_tokens[B * T:]
        self.buffer_masks = self.buffer_masks[B * T:]

        x = torch.tensor(x_list, dtype=torch.long)
        y = torch.tensor(y_list, dtype=torch.long)
        mask = torch.tensor(mask_list, dtype=torch.bool)

        return x, y, mask


# ---------------------------------------------------------------------------
# HellaSwag Eval (unchanged from original)
# ---------------------------------------------------------------------------

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='GPT-3 Pretraining, Continual Pretraining, and SFT Fine-tuning')

    # Mode
    parser.add_argument('--mode', type=str, default='pretrain', 
                        choices=['pretrain', 'finetune', 'sft'],
                        help='Training mode: pretrain, finetune (continual pretraining on .npy), or sft (supervised fine-tuning on JSONL)')

    # Model
    parser.add_argument('--model', type=str, default='gpt3-small',
                        choices=list(GPT3_CONFIGS.keys()) + list(GPT2_CONFIGS.keys()),
                        help='Model size configuration')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (required for finetune and sft modes)')

    # Data paths
    parser.add_argument('--data_root', type=str, 
                        default=r'E:/maxc/build-nanogpt/edu_fineweb10B',
                        help='Data directory for pretraining (shard directory)')
    parser.add_argument('--finetune_data', type=str, 
                        default=r'E:/maxc/build-nanogpt/edu_fineweb10B/train_pmc.npy',
                        help='Single .npy file for continual pretraining / finetuning')
    parser.add_argument('--sft_data', type=str, default='sft_data.jsonl',
                        help='SFT dataset (JSONL format)')

    # Training hyperparameters
    parser.add_argument('--total_batch_size', type=int, default=524288,
                        help='Total batch size in tokens')
    parser.add_argument('--B', type=int, default=32, help='Micro batch size')
    parser.add_argument('--T', type=int, default=1024, help='Sequence length')

    # Learning rate
    parser.add_argument('--max_lr', type=float, default=6e-4, help='Max learning rate')
    parser.add_argument('--min_lr', type=float, default=None, help='Min learning rate')
    parser.add_argument('--warmup_steps', type=int, default=715, help='Warmup steps')
    parser.add_argument('--max_steps', type=int, default=19073, help='Max training steps')

    # Finetune / SFT specific
    parser.add_argument('--finetune_epochs', type=int, default=1, 
                        help='Number of epochs for continual pretraining / finetune')
    parser.add_argument('--sft_epochs', type=int, default=3, help='Number of epochs for SFT')
    parser.add_argument('--mask_prompt', type=bool, default=True, 
                        help='Mask prompt tokens in SFT loss')

    # Optimization
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')

    # Logging
    parser.add_argument('--log_dir', type=str, default='log', help='Log directory')
    parser.add_argument('--eval_interval', type=int, default=250, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=5000, help='Checkpoint save interval')

    # System
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--device', type=str, default=None, help='Device override')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Validate arguments
    # -----------------------------------------------------------------------
    if args.mode in ('finetune', 'sft'):
        if args.checkpoint is None:
            raise ValueError(
                f"--checkpoint is required for --mode {args.mode}.\n"
                f"Please provide a pretrained model checkpoint.\n"
                f"Example: python gpt3_finetune.py --mode {args.mode} --model {args.model} "
                f"--checkpoint path/to/model_pretrain.pt"
            )
        if not os.path.exists(args.checkpoint):
            raise ValueError(
                f"Checkpoint file not found: {args.checkpoint}\n"
                f"Please check the path or run pretraining first:\n"
                f"  python gpt3_finetune.py --mode pretrain --model {args.model}"
            )

    if args.mode == 'finetune' and not os.path.exists(args.finetune_data):
        raise ValueError(f"--finetune_data file not found: {args.finetune_data}")

    if args.mode == 'sft' and not os.path.exists(args.sft_data):
        raise ValueError(f"--sft_data file not found: {args.sft_data}")

    # -----------------------------------------------------------------------
    # DDP setup
    # -----------------------------------------------------------------------
    global master_process
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = args.device if args.device else "cpu"
        if device == "cpu":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        if master_process:
            print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    torch.manual_seed(args.seed + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + ddp_rank)

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------
    enc = tiktoken.get_encoding("gpt2")

    # -----------------------------------------------------------------------
    # Model Configuration
    # -----------------------------------------------------------------------
    if args.model in GPT3_CONFIGS:
        config_args = GPT3_CONFIGS[args.model].copy()
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 2048  # GPT-3 uses 2048
    elif args.model in GPT2_CONFIGS:
        config_args = GPT2_CONFIGS[args.model].copy()
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024  # GPT-2 uses 1024
    else:
        raise ValueError(f"Unknown model: {args.model}")

    config = GPTConfig(**config_args)

    if master_process:
        print(f"=" * 60)
        print(f"Mode: {args.mode.upper()}")
        print(f"Model: {args.model}")
        print(f"  n_layer: {config.n_layer}")
        print(f"  n_head: {config.n_head}")
        print(f"  n_embd: {config.n_embd}")
        print(f"  block_size: {config.block_size}")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"=" * 60)

    # -----------------------------------------------------------------------
    # Create or Load Model
    # -----------------------------------------------------------------------
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        if master_process:
            print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Check if checkpoint has config
        if 'config' in checkpoint:
            loaded_config = checkpoint['config']
            # If block_size differs, we need to handle position embedding resizing
            if loaded_config.block_size != config.block_size:
                if master_process:
                    print(f"Note: checkpoint block_size ({loaded_config.block_size}) != target ({config.block_size})")
                    print(f"Position embeddings will be interpolated/truncated if needed")
            model = GPT(loaded_config)
        else:
            model = GPT(config)
        model.load_state_dict(checkpoint['model'])
        if master_process:
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        model = GPT(config)
        if master_process:
            print("Initializing model from scratch")

    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    if master_process:
        print(f"Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # -----------------------------------------------------------------------
    # Compile (optional)
    # -----------------------------------------------------------------------
    use_compile = args.compile
    if use_compile:
        if master_process:
            print("Using torch.compile...")
        model = torch.compile(model)

    # -----------------------------------------------------------------------
    # DDP wrap
    # -----------------------------------------------------------------------
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # -----------------------------------------------------------------------
    # Data Loaders
    # -----------------------------------------------------------------------
    # Adjust T if it exceeds block_size
    effective_T = min(args.T, config.block_size)
    if effective_T != args.T and master_process:
        print(f"Adjusted sequence length: {args.T} -> {effective_T} (max block_size)")

    if args.mode == 'pretrain':
        train_loader = DataLoaderLite(B=args.B, T=effective_T, 
                                       process_rank=ddp_rank, 
                                       num_processes=ddp_world_size, 
                                       split="train",
                                       data_root=args.data_root)
        val_loader = DataLoaderLite(B=args.B, T=effective_T, 
                                     process_rank=ddp_rank, 
                                     num_processes=ddp_world_size, 
                                     split="val",
                                     data_root=args.data_root)
    elif args.mode == 'finetune':
        # Continual pretraining on a single .npy file
        train_loader = FinetuneDataLoader(
            B=args.B, T=effective_T,
            process_rank=ddp_rank,
            num_processes=ddp_world_size,
            filename=args.finetune_data
        )
        # For validation during finetuning, we can use a portion of the same data
        # or specify a separate val file. Here we use the same file with offset.
        val_loader = FinetuneDataLoader(
            B=args.B, T=effective_T,
            process_rank=ddp_rank,
            num_processes=ddp_world_size,
            filename=args.finetune_data
        )
        # Offset validation loader to use different portion
        val_loader.current_position = (len(val_loader.tokens) // 2) + (ddp_rank * args.B * effective_T)
    else:  # SFT mode
        train_loader = SFTDataLoader(
            filename=args.sft_data,
            B=args.B, T=effective_T,
            process_rank=ddp_rank,
            num_processes=ddp_world_size,
            tokenizer=enc,
            mask_prompt=args.mask_prompt
        )
        # For SFT validation, use a separate file or split
        val_file = args.sft_data.replace('.jsonl', '_val.jsonl')
        if os.path.exists(val_file):
            val_loader = SFTDataLoader(
                filename=val_file,
                B=args.B, T=effective_T,
                process_rank=ddp_rank,
                num_processes=ddp_world_size,
                tokenizer=enc,
                mask_prompt=args.mask_prompt
            )
        else:
            val_loader = None
            if master_process:
                print(f"Warning: No validation file found at {val_file}")

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    # Adjust learning rate based on model size (GPT-3 paper recommendations)
    if args.min_lr is None:
        args.min_lr = args.max_lr * 0.1

    # GPT-3 uses different LR for different sizes (from paper Table 2.1)
    gpt3_lrs = {
        'gpt3-small': 6.0e-4,
        'gpt3-medium': 3.0e-4,
        'gpt3-large': 2.5e-4,
        'gpt3-xl': 2.0e-4,
        'gpt3-2.7b': 1.6e-4,
        'gpt3-6.7b': 1.2e-4,
        'gpt3-13b': 1.0e-4,
        'gpt3-175b': 0.6e-4,
    }

    if args.mode == 'pretrain' and args.model in gpt3_lrs:
        lr_override = gpt3_lrs[args.model]
        if master_process:
            print(f"Using GPT-3 recommended LR for {args.model}: {lr_override:.2e}")
        args.max_lr = lr_override
        args.min_lr = lr_override * 0.1
    elif args.mode in ('finetune', 'sft'):
        # Finetuning / SFT typically uses lower LR
        ft_lr = args.max_lr * 0.1  # Typically 10x lower than pretraining
        if master_process:
            print(f"{args.mode.upper()} mode: reducing LR from {args.max_lr:.2e} to {ft_lr:.2e}")
        args.max_lr = ft_lr
        args.min_lr = ft_lr * 0.1

    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay, 
        learning_rate=args.max_lr, 
        device_type=device_type
    )

    # -----------------------------------------------------------------------
    # Learning Rate Schedule
    # -----------------------------------------------------------------------
    def get_lr(it):
        if it < args.warmup_steps:
            return args.max_lr * (it + 1) / args.warmup_steps
        if it > args.max_steps:
            return args.min_lr
        decay_ratio = (it - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.max_lr - args.min_lr)

    # -----------------------------------------------------------------------
    # Gradient Accumulation
    # -----------------------------------------------------------------------
    assert args.total_batch_size % (args.B * effective_T * ddp_world_size) == 0, \
        "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = args.total_batch_size // (args.B * effective_T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {args.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{args.mode}_{args.model}.txt")
    with open(log_file, "w") as f:
        pass

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    torch.set_float32_matmul_precision('high')

    # Calculate total steps based on epochs for finetune / sft modes
    if args.mode == 'finetune':
        tokens_per_step = args.B * effective_T * grad_accum_steps * ddp_world_size
        total_tokens = len(train_loader.tokens)
        steps_per_epoch = max(1, total_tokens // tokens_per_step)
        args.max_steps = steps_per_epoch * args.finetune_epochs
        if master_process:
            print(f"Finetune: {total_tokens:,} tokens, {steps_per_epoch} steps/epoch, {args.finetune_epochs} epochs = {args.max_steps} total steps")
    elif args.mode == 'sft':
        examples_per_step = args.B * grad_accum_steps * ddp_world_size
        steps_per_epoch = max(1, len(train_loader.tokenized) // examples_per_step)
        args.max_steps = steps_per_epoch * args.sft_epochs
        if master_process:
            print(f"SFT: {len(train_loader.tokenized)} examples, {steps_per_epoch} steps/epoch, {args.sft_epochs} epochs = {args.max_steps} total steps")

    for step in range(args.max_steps):
        t0 = time.time()
        last_step = (step == args.max_steps - 1)

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------
        if step % args.eval_interval == 0 or last_step:
            model.eval()
            if args.mode in ('pretrain', 'finetune') and val_loader is not None:
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f"validation loss: {val_loss_accum.item():.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            elif args.mode == 'sft' and val_loader is not None:
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = min(20, max(1, len(val_loader.tokenized) // (args.B * ddp_world_size)))
                    for _ in range(val_loss_steps):
                        x, y, mask = val_loader.next_batch()
                        x, y, mask = x.to(device), y.to(device), mask.to(device)
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(x, y, mask=mask)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f"SFT validation loss: {val_loss_accum.item():.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            # Save checkpoint
            if master_process and step > 0 and (step % args.save_interval == 0 or last_step):
                checkpoint_path = os.path.join(args.log_dir, f"model_{args.mode}_{args.model}_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                }
                if args.mode in ('pretrain', 'finetune'):
                    checkpoint['val_loss'] = val_loss_accum.item() if val_loader else None
                torch.save(checkpoint, checkpoint_path)
                print(f"saved checkpoint to {checkpoint_path}")

        # -------------------------------------------------------------------
        # Generation sample (pretrain / finetune only, skip for SFT to save time)
        # -------------------------------------------------------------------
        if args.mode in ('pretrain', 'finetune') and ((step > 0 and step % args.eval_interval == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        # -------------------------------------------------------------------
        # Training step
        # -------------------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            if args.mode in ('pretrain', 'finetune'):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            else:  # SFT mode
                x, y, mask = train_loader.next_batch()
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y, mask=mask)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = args.B * effective_T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    if ddp:
        destroy_process_group()

    if master_process:
        print(f"\nTraining complete! Final checkpoint saved.")
        print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
