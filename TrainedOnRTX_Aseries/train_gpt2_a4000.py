import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
# DataLoaderLite 里 shakespeare 分支维持原样，不要删
with open("input.txt", "r") as f:
    text = f.read()
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)
import numpy as np
# =============================================================================
# 模型定义（与原版完全一致）
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


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
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
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
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# =============================================================================
# 数据加载器 - 支持两种模式
# =============================================================================

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        self.split = split

        # 检查是否有 edu_fineweb10B 数据集
        data_root = "edu_fineweb10B"
        if os.path.exists(data_root):
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards
            self.mode = "fineweb"
            print(f"[{split}] 使用 edu_fineweb10B，找到 {len(shards)} 个 shard")
        else:
            # 回退到 tiny shakespeare
            self.mode = "shakespeare"
            with open("input.txt", "r") as f:
                text = f.read()
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            tokens = enc.encode(text)
            # 简单划分 train/val (90/10)
            split_idx = int(len(tokens) * 0.9)
            self.tokens_list = tokens[:split_idx] if split == "train" else tokens[split_idx:]
            self.shards = []
            print(f"[{split}] 使用 tiny shakespeare，{len(self.tokens_list)} tokens")

        self.reset()

    def reset(self):
        if self.mode == "fineweb":
            self.current_shard = 0
            self.tokens = self._load_tokens(self.shards[self.current_shard])
        else:
            import numpy as np
            self.tokens = torch.tensor(self.tokens_list, dtype=torch.long)
        self.current_position = self.B * self.T

    def _load_tokens(self, filename):
        
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        return torch.tensor(npt, dtype=torch.long)

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            if self.mode == "fineweb":
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = self._load_tokens(self.shards[self.current_shard])
            # shakespeare 就循环
            self.current_position = 0
        return x, y


# =============================================================================
# HellaSwag eval
# =============================================================================

def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# =============================================================================
# 训练主流程
# =============================================================================


# 设备
device = "cuda"
device_type = "cuda"
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

# ============================================================
# ★★★ 关键优化参数（针对 RTX A4000 15GB 显存） ★★★
# ============================================================

# Batch size: A4000 只有 15GB 显存
# GPT-2 124M 在 T=1024 下, logits=(B,1024,50304) 是显存大头
B = 64          
T = 512        # 序列长度

# total_batch_size: 保持原版的 524288 tokens
total_batch_size = 524288
grad_accum_steps = total_batch_size // (B * T)
print(f"配置: B={B}, T={T}, grad_accum_steps={grad_accum_steps}")
print(f"total_batch_size: {total_batch_size} tokens")
max_steps = 2000

# 学习率
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# 数据加载器
train_loader = DataLoaderLite(B=B, T=T, split="train")
val_loader = DataLoaderLite(B=B, T=T, split="val")

# 精度设置 - 使用 bf16 加速
torch.set_float32_matmul_precision('high')

# 创建模型
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# ============================================================
# ★★★ 开启 torch.compile (原版 use_compile=False) ★★★
# torch.compile 可以带来 1.3-2x 加速，但首次编译会慢一些
# ============================================================
use_compile = False  # Triton 在 Windows 上不可用，关闭 torch.compile
if use_compile:
    print("正在编译模型 (torch.compile)，首次运行会慢一些...")
    model = torch.compile(model)

# 优化器
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# 日志
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass

# 训练循环
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # ===== 评估 (降低频率: 每 500 步一次，原版 250) =====
    if step % 500 == 0 or last_step:
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
        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    # ===== HellaSwag eval =====
    if (step % 500 == 0 or last_step) and (not use_compile):
        try:
            from hellaswag import render_example, iterate_examples
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            acc_norm = num_correct_norm / num_total
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
        except Exception as e:
            print(f"HellaSwag eval skipped: {e}")

    # ===== 生成样本 =====
    if (step > 0 and step % 500 == 0) or last_step:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
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
            print(f"sample {i}: {decoded}")

    # ===== 训练步骤 =====
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # 注意：torch.cuda.synchronize() 会降低吞吐量
    # 只在需要精确计时的时候才调用
    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")

print("训练完成！")
