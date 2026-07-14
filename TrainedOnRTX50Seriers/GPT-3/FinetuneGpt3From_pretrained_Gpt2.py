
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Causal Self-Attention with Flash Attention

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
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


# ---------------------------------------------------------------------------
# MLP (Feed-Forward Network)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block

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


# ---------------------------------------------------------------------------
# GPT-3 Configuration

@dataclass
class GPTConfig:
    block_size: int = 2048      # max sequence length (GPT-3 uses 2048)
    vocab_size: int = 50257     # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    n_layer: int = 12           # number of layers (12 for GPT-3 Small 125M)
    n_head: int = 12            # number of heads (12 for GPT-3 Small 125M)
    n_embd: int = 768           # embedding dimension (768 for GPT-3 Small 125M)


# ---------------------------------------------------------------------------
# GPT Model

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

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, config=None, checkpoint_path=None):
        """Loads pretrained weights for GPT-3 fine-tuning / transfer learning.

        Supports two modes:
          1. checkpoint_path: load a local checkpoint to resume or fine-tune.
          2. model_type: 
             OpenAI never released GPT-3 weights, so GPT-3 sizes fall back to
             from-scratch initialization (matching the GPT-3 reproduction setup
             described in the GPT-3 paper and the build-nanogpt FineWeb/HellaSwag
             workflow).
        """
        # ------------------------------------------------------------------
        # Mode 1: load from a local checkpoint for fine-tuning / resuming
        # ------------------------------------------------------------------
        if checkpoint_path is not None:
            print(f"loading weights from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_config = checkpoint['config']
            model = GPT(checkpoint_config)
            state_dict = checkpoint['model']
            # strip optional DDP / torch.compile prefix if present
            unwanted_prefix = '_orig_mod.'
            if any(k.startswith(unwanted_prefix) for k in state_dict):
                state_dict = {
                    k[len(unwanted_prefix):] if k.startswith(unwanted_prefix) else k: v
                    for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict)
            return model

        # ------------------------------------------------------------------
        # Mode 2: load from HuggingFace GPT-2 checkpoints ( and load GPT-2 weights as initialization).
        # ------------------------------------------------------------------
        assert model_type in {
            'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
            'gpt3-small', 'gpt3-medium', 'gpt3-large', 'gpt3-xl', 'gpt3-2.7B'
        }
        """
        gpt2_configs = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }
        """
        gpt3_configs = {
            'gpt3-small':  dict(n_layer=12, n_head=12, n_embd=768),   # 125M params
            'gpt3-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt3-large':  dict(n_layer=24, n_head=16, n_embd=1536),  # 760M params
            'gpt3-xl':     dict(n_layer=24, n_head=24, n_embd=2048),  # 1.3B params
            'gpt3-2.7B':   dict(n_layer=32, n_head=32, n_embd=2560),  # 2.7B params
        }

        is_gpt3 = model_type.startswith('gpt3')
        config_args = gpt3_configs[model_type]
        #config_args = (gpt3_configs if is_gpt3 else gpt2_configs)[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 2048 if is_gpt3 else 1024

        if config is None:
            config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # GPT-3 public weights are not available; return from-scratch init
        if is_gpt3:
            print(f"no pretrained weights available for {model_type}; using from-scratch initialization")
            return model

        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # OpenAI checkpoints use a Conv1D module, but we use vanilla Linear,
        # so we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, betas=(0.9, 0.95), eps=1e-8):
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
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=use_fused)
        return optimizer


# ---------------------------------------------------------------------------
# Data Loader for FineWeb-Edu

import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "pmc"
        shards = os.listdir(data_root) if os.path.exists(data_root) else []
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split} in {data_root}. Please download FineWeb-Edu dataset."
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
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y



def get_most_likely_row(tokens, mask, logits):
    """takes tokens, mask, and logits, returns the index of the completion with the lowest loss"""
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# ---------------------------------------------------------------------------
# Distributed Training Setup

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"


# ---------------------------------------------------------------------------
# GPT-3 Model Configurations
# Uncomment the model size you want to train

# GPT-3 Small (125M parameters) - Good for single GPU
GPT3_SMALL = dict(n_layer=12, n_head=12, n_embd=768)

# GPT-3 Medium (350M parameters)
GPT3_MEDIUM = dict(n_layer=24, n_head=16, n_embd=1024)

# GPT-3 Large (760M parameters)
GPT3_LARGE = dict(n_layer=24, n_head=16, n_embd=1536)

# GPT-3 XL (1.3B parameters)
GPT3_XL = dict(n_layer=24, n_head=24, n_embd=2048)

# GPT-3 2.7B
GPT3_2_7B = dict(n_layer=32, n_head=32, n_embd=2560)

# Select model size (default to Small for testing)
MODEL_SIZE = "small"  # Change to: "medium", "large", "xl", "2.7B"

MODEL_CONFIGS = {
    "small": GPT3_SMALL,
    "medium": GPT3_MEDIUM,
    "large": GPT3_LARGE,
    "xl": GPT3_XL,
    "2.7B": GPT3_2_7B,
}


# ---------------------------------------------------------------------------
# Training Configuration

# Set random seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

# Model hyperparameters - MODIFY THESE FOR YOUR SETUP
# For single GPU/local testing, use smaller values
# For full GPT-3 reproduction, use the commented values

# Batch size configuration
# GPT-3 paper: 3.2M tokens per batch
# For local/single GPU testing, use smaller batch
B = 16          # micro batch size (use 12 for local testing, 64+ for multi-GPU)
T = 2048         # sequence length (GPT-3 uses 2048)

total_batch_size = 3276800  # (use 3.2M for full GPT-3)
# For full GPT-3: total_batch_size = 3276800  # 3.2M tokens

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# Create data loaders
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# Enable TF32 for faster training on Ampere GPUs
torch.set_float32_matmul_precision('high')

# Create model with selected GPT-3 config
selected_config = MODEL_CONFIGS[MODEL_SIZE]
config = GPTConfig(
    block_size=T,  # Use T (256 or 2048) for sequence length
    vocab_size=50304,  # Nice number for efficiency (divisible by many numbers)
    **selected_config
)

if master_process:
    print(f"Training GPT-3 {MODEL_SIZE.upper()} config:")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_head: {config.n_head}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  block_size: {config.block_size}")

model = GPT(config)
# model = GPT.from_pretrained("gpt2")  # or init from OpenAI GPT-2 for transfer learning
model.to(device)

# Compile model for faster training (disabled by default due to compatibility)
use_compile = False  # torch.compile interferes with some eval. Set to True if you don't need eval during training
if use_compile:
    if hasattr(torch, 'compile'):
        print("compiling the model...")
        model = torch.compile(model)
    else:
        print("torch.compile not available in this PyTorch version")

# Wrap model for DDP if using distributed training
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model


# ---------------------------------------------------------------------------
# Learning Rate Schedule

# GPT-3 uses:
# - Max LR: 6e-4 for most models (0.6e-4 for 13B, 0.1e-4 for 175B)
# - Warmup: 375M tokens
# - Cosine decay to 10% of max
# - Training: 300B tokens total

max_lr = 6e-4
min_lr = max_lr * 0.1

# Calculate warmup and max steps based on tokens
# For full training on 10B tokens with 0.5M batch size:
tokens_per_step = total_batch_size
warmup_tokens = 375_000_000  # 375M tokens for warmup
max_tokens = 10_000_000_000  # 10B tokens (FineWeb-Edu)
# For 300B tokens (full GPT-3 scale): max_tokens = 300_000_000_000

warmup_steps = warmup_tokens // tokens_per_step
max_steps = max_tokens // tokens_per_step

if master_process:
    print(f"Training schedule:")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  max_steps: {max_steps}")
    print(f"  tokens per step: {tokens_per_step:,}")


def get_lr(it):
    """Learning rate schedule with linear warmup and cosine decay"""
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Optimizer

# GPT-3 uses Adam with β1=0.9, β2=0.95, ε=10^-8
# Weight decay: 0.1
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=max_lr,
    device_type=device_type,
    betas=(0.9, 0.95),
    eps=1e-8
)


# ---------------------------------------------------------------------------
# Logging

log_dir = "log_gpt3"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_{MODEL_SIZE}.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    f.write(f"GPT-3 {MODEL_SIZE} training log\n")
    f.write(f"Config: {config}\n")
    f.write("="*50 + "\n")


# ---------------------------------------------------------------------------
# Training Loop

if master_process:
    print("\n" + "="*50)
    print("Starting GPT-3 training!")
    print("="*50 + "\n")

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # -----------------------------------------------------------------------
    # Validation
    if step % 250 == 0 or last_step:
        model.eval()
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

        # Save checkpoints
        if step > 0 and (step % 5000 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{MODEL_SIZE}_{step:05d}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }
            torch.save(checkpoint, checkpoint_path)
            if master_process:
                print(f"saved checkpoint to {checkpoint_path}")

    # -----------------------------------------------------------------------
    # Generation (sample from model)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
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
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        if master_process:
            print(f"\n--- Generation at step {step} ---")
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"sample {i}: {decoded}")
            print("-" * 30 + "\n")

    # -----------------------------------------------------------------------
    # Training Step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # Synchronize gradients only on last micro-step for DDP
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    # Average loss across DDP processes
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    # Synchronize for timing (CUDA only)
    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")


# ---------------------------------------------------------------------------
# Cleanup

if ddp:
    destroy_process_group()

if master_process:
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
