#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的Sokoban RWKV训练脚本
特点：
1. 不使用ROSA，纯RWKV7
2. 简化的模型配置
3. 更好的训练监控
4. 针对简单数据优化
5. 所有日志输出到文件
"""

import os
import sys
import math
import json
import time
import random
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from types import SimpleNamespace


# -------------------------
# Logger Setup
# -------------------------

class Logger:
    """日志类：将所有输出写入文件"""
    def __init__(self, log_file: str, console: bool = False):
        self.log_file = log_file
        self.console = console
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        self.file = open(log_file, 'w', encoding='utf-8')
        self.file.write(f"=== Training Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        self.file.flush()

    def log(self, msg: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        line = f"[{timestamp}] {msg}\n"
        self.file.write(line)
        self.file.flush()
        if self.console:
            print(msg)

    def log_raw(self, msg: str):
        """不带时间戳的日志"""
        self.file.write(msg + "\n")
        self.file.flush()
        if self.console:
            print(msg)

    def close(self):
        self.file.write(f"\n=== Training Log Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.file.close()

# Global logger
logger: Optional[Logger] = None

def log(msg: str):
    if logger:
        logger.log(msg)
    else:
        print(msg)

def log_raw(msg: str):
    if logger:
        logger.log_raw(msg)
    else:
        print(msg)


# -------------------------
# Reproducibility
# -------------------------

def set_seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# -------------------------
# Tokenizer (vocab-based)
# -------------------------

class VocabTokenizer:
    """基于词表文件的Tokenizer"""
    def __init__(self, vocab_file: str = "vocab_simple.txt"):
        self.vocab_file = vocab_file
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.max_token_len = 1
        self._load_vocab()

        assert "<PAD>" in self.token_to_id, "Vocab must contain <PAD>"
        assert "<BOS>" in self.token_to_id, "Vocab must contain <BOS>"
        assert "<EOS>" in self.token_to_id, "Vocab must contain <EOS>"

        self.PAD = self.token_to_id["<PAD>"]
        self.BOS = self.token_to_id["<BOS>"]
        self.EOS = self.token_to_id["<EOS>"]
        self.vocab_size = len(self.token_to_id)

    def _load_vocab(self):
        if not os.path.isfile(self.vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_file}")

        idx = 0
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                token = line.rstrip('\n').rstrip('\r')
                if token == "":
                    continue
                if token == "\\n":
                    token = "\n"
                if token in self.token_to_id:
                    continue
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                self.max_token_len = max(self.max_token_len, len(token))
                idx += 1

        log(f"[Tokenizer] Loaded {len(self.token_to_id)} tokens")

    def encode(self, s: str) -> List[int]:
        tokens = []
        i = 0
        while i < len(s):
            matched = False
            max_len = min(self.max_token_len, len(s) - i)
            for length in range(max_len, 0, -1):
                candidate = s[i:i+length]
                if candidate in self.token_to_id:
                    tokens.append(self.token_to_id[candidate])
                    i += length
                    matched = True
                    break
            if not matched:
                # 跳过未知字符
                i += 1
        return tokens

    def decode(self, ids: List[int]) -> str:
        result = []
        for t in ids:
            if t == self.PAD:
                continue
            if t in self.id_to_token:
                result.append(self.id_to_token[t])
        return "".join(result)

    def get_vocab_size(self) -> int:
        return self.vocab_size


# -------------------------
# RWKV7 CUDA op
# -------------------------

from torch.utils.cpp_extension import load

HEAD_SIZE = 16
CHUNK_LEN = 16


def load_rwkv7_extension():
    flags = [
        "-res-usage",
        f"-D_C_={HEAD_SIZE}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]
    load(
        name="wind_backstepping",
        sources=["cuda/wkv7_cuda.cu", "cuda/wkv7_op.cpp"],
        is_python_module=False,
        verbose=False,
        extra_cuda_cflags=flags,
    )


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        original_dtype = w.dtype

        w_bf = w.to(torch.bfloat16).contiguous()
        q_bf = q.to(torch.bfloat16).contiguous()
        k_bf = k.to(torch.bfloat16).contiguous()
        v_bf = v.to(torch.bfloat16).contiguous()
        z_bf = z.to(torch.bfloat16).contiguous()
        b_bf = b.to(torch.bfloat16).contiguous()

        y_bf = torch.empty_like(v_bf)
        s = torch.zeros(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.zeros(B, T, H, C, dtype=torch.float32, device=w.device)

        torch.ops.wind_backstepping.forward(w_bf, q_bf, k_bf, v_bf, z_bf, b_bf, y_bf, s, sa)

        ctx.save_for_backward(w_bf, q_bf, k_bf, v_bf, z_bf, b_bf, s, sa)
        ctx.original_dtype = original_dtype

        return y_bf.to(original_dtype)

    @staticmethod
    def backward(ctx, dy):
        w_bf, q_bf, k_bf, v_bf, z_bf, b_bf, s, sa = ctx.saved_tensors
        original_dtype = ctx.original_dtype

        dy_bf = dy.to(torch.bfloat16).contiguous()

        dw_bf = torch.empty_like(w_bf)
        dq_bf = torch.empty_like(q_bf)
        dk_bf = torch.empty_like(k_bf)
        dv_bf = torch.empty_like(v_bf)
        dz_bf = torch.empty_like(z_bf)
        db_bf = torch.empty_like(b_bf)

        torch.ops.wind_backstepping.backward(
            w_bf, q_bf, k_bf, v_bf, z_bf, b_bf, dy_bf, s, sa,
            dw_bf, dq_bf, dk_bf, dv_bf, dz_bf, db_bf
        )

        return (dw_bf.to(original_dtype), dq_bf.to(original_dtype),
                dk_bf.to(original_dtype), dv_bf.to(original_dtype),
                dz_bf.to(original_dtype), db_bf.to(original_dtype))


def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC // 16, 16).contiguous() for i in [q, w, k, v, a, b]]
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)


# -------------------------
# RWKV block
# -------------------------

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(1, args.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                elif len(shape) == 3:
                    gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                    for i in range(shape[0]):
                        nn.init.orthogonal_(x[i], gain=gain * scale)
                return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = 8
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

            D_AAA_LORA = 8
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

            D_MV_LORA = 8
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

            D_GATE_LORA = 8
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v_mix = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            v = v + (v_first - v) * v_mix

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0, eps=1e-8).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


class FFN(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.key = nn.Linear(C, C * 4, bias=False)
        self.value = nn.Linear(C * 4, C, bias=False)
        with torch.no_grad():
            self.value.weight.data.zero_()
            nn.init.orthogonal_(self.key.weight.data, gain=(4**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        x = x + xx * self.x_k
        x = torch.relu(self.key(x)) ** 2
        return self.value(x)


# -------------------------
# Model
# -------------------------

class RWKV_Model(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int):
        super().__init__()
        assert n_layer >= 2
        assert n_embd % HEAD_SIZE == 0
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer

        args = SimpleNamespace()
        args.head_size = HEAD_SIZE
        args.n_embd = n_embd
        args.dim_att = n_embd
        args.n_layer = n_layer

        self.emb = nn.Embedding(vocab_size, n_embd)
        self.ln_a = nn.ModuleList([nn.LayerNorm(n_embd) for _ in range(n_layer)])
        self.ln_b = nn.ModuleList([nn.LayerNorm(n_embd) for _ in range(n_layer)])
        self.rwkv = nn.ModuleList([RWKV_Tmix_x070(args, i) for i in range(n_layer)])
        self.ffn = nn.ModuleList([FFN(n_embd) for _ in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx).float()
        v_first = torch.empty_like(x)

        for i in range(self.n_layer):
            xx, v_first = self.rwkv[i](self.ln_a[i](x), v_first if i > 0 else torch.empty_like(x))
            x = x + xx
            x = x + self.ffn[i](self.ln_b[i](x))

        x = self.head(self.ln_out(x))
        return x


# -------------------------
# Dataset
# -------------------------

@dataclass
class Sample:
    tokens: List[int]
    prompt_len: int  # prompt部分的长度（用于mask）


class SokobanDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer: VocabTokenizer, ctx_len: int,
                 mask_prompt: bool = True):
        self.path = path
        self.tok = tokenizer
        self.ctx_len = ctx_len
        self.mask_prompt = mask_prompt
        self.samples: List[Sample] = []
        self.raw_data: List[dict] = []
        self._build()

    def _build(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "input" not in obj or "cot" not in obj:
                    continue

                inp = obj["input"]
                cot = obj["cot"]

                # 构建完整序列
                prompt_tokens = [self.tok.BOS] + self.tok.encode(inp + "\n")
                cot_tokens = self.tok.encode(cot) + [self.tok.EOS]
                full = prompt_tokens + cot_tokens

                # 截断到ctx_len
                if len(full) > self.ctx_len + 1:
                    full = full[:self.ctx_len + 1]

                self.samples.append(Sample(
                    tokens=full,
                    prompt_len=len(prompt_tokens)
                ))
                self.raw_data.append(obj)

        log(f"[Dataset] Loaded {len(self.samples)} samples from {self.path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]


def collate_fn(batch: List[Sample], ctx_len: int, pad_id: int, mask_prompt: bool):
    B = len(batch)

    # 找到最大长度
    max_len = max(len(s.tokens) for s in batch)
    max_len = min(max_len, ctx_len + 1)

    # 对齐到CHUNK_LEN
    T = ((max_len - 1 + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
    T = max(T, CHUNK_LEN)
    T = min(T, ctx_len)

    tokens_pad = torch.full((B, T + 1), pad_id, dtype=torch.long)
    for i, s in enumerate(batch):
        L = min(len(s.tokens), T + 1)
        tokens_pad[i, :L] = torch.tensor(s.tokens[:L], dtype=torch.long)

    x = tokens_pad[:, :T].contiguous()
    y = tokens_pad[:, 1:T + 1].contiguous()
    labels = y.clone()
    labels[labels == pad_id] = -100

    if mask_prompt:
        for i, s in enumerate(batch):
            # mask prompt部分（不学习预测prompt）
            prompt_end = min(s.prompt_len - 1, T)
            if prompt_end > 0:
                labels[i, :prompt_end] = -100

    return x, labels


# -------------------------
# Training / Eval
# -------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(x)
        if not torch.isfinite(logits).all():
            continue
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        n_valid = (flat_labels != -100).sum().item()
        if n_valid == 0:
            continue
        loss_sum = F.cross_entropy(
            flat_logits, flat_labels, ignore_index=-100, reduction="sum"
        ).item()
        total_loss += loss_sum
        total_tokens += n_valid
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else 0.0


@torch.no_grad()
def generate(model: nn.Module, tokenizer: VocabTokenizer, prompt: str, device: str,
             ctx_len: int, max_new: int = 256, temperature: float = 0.8) -> str:
    model.eval()
    tokens = [tokenizer.BOS] + tokenizer.encode(prompt)

    for _ in range(max_new):
        if len(tokens) >= ctx_len:
            break

        # Padding to CHUNK_LEN multiple
        padded_len = ((len(tokens) + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
        padded_len = min(padded_len, ctx_len)
        inp = tokens + [tokenizer.PAD] * (padded_len - len(tokens))

        x = torch.tensor([inp], dtype=torch.long, device=device)
        logits = model(x)
        last_logits = logits[0, len(tokens) - 1].float() / max(temperature, 1e-6)
        probs = F.softmax(last_logits, dim=-1)

        # Top-p sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum > 0.9
        cutoff[0] = False
        sorted_probs[cutoff] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        next_id = sorted_idx[torch.multinomial(sorted_probs, num_samples=1)].item()

        tokens.append(int(next_id))
        if next_id == tokenizer.EOS:
            break

    gen_part = tokens[len([tokenizer.BOS] + tokenizer.encode(prompt)):]
    return tokenizer.decode(gen_part)


# -------------------------
# Sokoban Simulator for Validation
# -------------------------

DIRS_SIM = {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0)}

def parse_map(map_str: str):
    """解析地图字符串，返回walls, boxes, goals, player, width, height"""
    lines = [l for l in map_str.strip().split('\n') if l.strip()]
    height = len(lines)
    width = max(len(l) for l in lines) if lines else 0

    walls = set()
    boxes = set()
    goals = set()
    player = None

    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            pos = (x, y)
            if ch == '#':
                walls.add(pos)
            elif ch == '@':
                player = pos
            elif ch == '+':  # player on goal
                player = pos
                goals.add(pos)
            elif ch == '$':
                boxes.add(pos)
            elif ch == '*':  # box on goal
                boxes.add(pos)
                goals.add(pos)
            elif ch == '.':
                goals.add(pos)
            # '_' is floor, ignore

    return walls, boxes, goals, player, width, height


def get_reachable_positions(walls, boxes, player, width, height):
    """获取玩家可达的所有位置"""
    from collections import deque
    visited = {player}
    queue = deque([player])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS_SIM.values():
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and (nx, ny) not in walls and (nx, ny) not in boxes:
                if 0 <= nx < width and 0 <= ny < height:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return visited


def simulate_move(walls, boxes, goals, player, direction, width, height):
    """
    模拟一次移动
    返回: (success, new_player, new_boxes, error_msg)
    """
    if direction not in DIRS_SIM:
        return False, player, boxes, f"无效方向: {direction}"

    dx, dy = DIRS_SIM[direction]
    new_x, new_y = player[0] + dx, player[1] + dy
    front = (new_x, new_y)

    # 检查是否撞墙
    if front in walls:
        return False, player, boxes, "撞墙"

    # 检查边界
    if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
        return False, player, boxes, "越界"

    new_boxes = set(boxes)

    # 检查是否推箱子
    if front in boxes:
        box_dest = (new_x + dx, new_y + dy)
        # 检查箱子目标位置
        if box_dest in walls:
            return False, player, boxes, "箱子撞墙"
        if box_dest in boxes:
            return False, player, boxes, "箱子撞箱子"
        if box_dest[0] < 0 or box_dest[0] >= width or box_dest[1] < 0 or box_dest[1] >= height:
            return False, player, boxes, "箱子越界"
        # 移动箱子
        new_boxes.remove(front)
        new_boxes.add(box_dest)

    return True, front, new_boxes, ""


def simulate_teleport(walls, boxes, player, target, width, height):
    """
    模拟传送
    返回: (success, new_player, error_msg)
    """
    # 检查目标位置是否可达
    reachable = get_reachable_positions(walls, boxes, player, width, height)
    if target not in reachable:
        return False, player, f"传送目标不可达: {target}"
    return True, target, ""


def check_solution_validity(generated: str, raw_data: dict) -> dict:
    """
    完整验证生成的解是否有效
    直接从<done>标签解析最终状态来验证
    """
    import re
    result = {"valid": False, "reason": "", "score": 0.0, "details": []}

    # 1. 解析输入地图
    input_str = raw_data.get("input", "")
    input_match = re.search(r'<input>\s*(.*?)\s*</input>', input_str, re.DOTALL)
    if not input_match:
        result["reason"] = "无法解析输入地图"
        return result

    map_str = input_match.group(1)
    walls, init_boxes, goals, init_player, width, height = parse_map(map_str)

    if init_player is None:
        result["reason"] = "输入地图中找不到玩家"
        return result

    if not goals:
        result["reason"] = "输入地图中找不到目标"
        return result

    # 2. 直接从<done>标签解析最终状态
    done_match = re.search(r'<done>\s*(.*?)\s*</done>', generated, re.DOTALL)
    if not done_match:
        # 没有<done>标签，尝试从最后一个<state>解析
        state_matches = list(re.finditer(r'<state>\s*(.*?)\s*</state>', generated, re.DOTALL))
        if state_matches:
            final_map = state_matches[-1].group(1)
        else:
            result["reason"] = "未找到<done>或<state>标签"
            return result
    else:
        final_map = done_match.group(1)

    # 3. 解析最终状态
    final_walls, final_boxes, final_goals, final_player, fw, fh = parse_map(final_map)

    # 4. 验证地图结构一致性（墙和目标位置应该不变）
    if final_walls != walls:
        result["reason"] = "地图结构被篡改（墙位置变化）"
        result["score"] = 0.0
        return result

    if final_goals != goals:
        result["reason"] = "地图结构被篡改（目标位置变化）"
        result["score"] = 0.0
        return result

    # 5. 检查箱子数量
    if len(final_boxes) != len(init_boxes):
        result["reason"] = f"箱子数量不对（应{len(init_boxes)}个，实际{len(final_boxes)}个）"
        result["score"] = 0.0
        return result

    # 6. 检查是否完成（所有箱子都在目标上）
    is_solved = (final_boxes == goals)
    boxes_on_goal = len(final_boxes & goals)

    # 7. 统计动作数量
    actions = re.findall(r'<action>([UDLR]):[^<]*</action>', generated)
    num_actions = len(actions)

    # 8. 计算得分
    if is_solved:
        result["valid"] = True
        result["score"] = 1.0
        result["reason"] = f"完成! {num_actions}步"
    else:
        partial = boxes_on_goal / len(goals) if goals else 0
        result["score"] = partial * 0.5
        result["reason"] = f"未完成，{boxes_on_goal}/{len(goals)}箱子到位"

    return result


def evaluate_and_show(model, tokenizer, val_loader, raw_data, device, ctx_len,
                      num_samples=3):
    """评测并显示生成样例"""
    model.eval()

    # 计算验证loss
    val_loss = evaluate(model, val_loader, device)
    val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')

    log_raw("\n" + "="*70)
    log_raw("评测结果")
    log_raw("="*70)
    log(f"验证Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

    # 显示生成样例
    log_raw("\n" + "-"*70)
    log_raw("生成样例")
    log_raw("-"*70)

    sample_indices = random.sample(range(len(raw_data)), min(num_samples, len(raw_data)))
    total_score = 0.0

    for idx, sample_idx in enumerate(sample_indices):
        data = raw_data[sample_idx]
        prompt = data["input"] + "\n"
        ground_truth = data["cot"]

        log_raw(f"\n【样例 {idx + 1}】")
        log_raw(f"输入:")
        log_raw(data["input"])

        generated = generate(model, tokenizer, prompt, device, ctx_len, max_new=1024)

        log_raw(f"\n模型生成:")
        log_raw(generated)

        # 检查有效性
        validity = check_solution_validity(generated, data)
        log_raw(f"\n评估: {validity['reason']} (得分: {validity['score']:.2f})")
        total_score += validity["score"]

        log_raw(f"\n真实答案:")
        log_raw(ground_truth)
        log_raw("-"*50)

    avg_score = total_score / len(sample_indices) if sample_indices else 0
    log_raw(f"\n平均得分: {avg_score:.2f}")
    log_raw("="*70)

    model.train()
    return val_loss


def save_checkpoint(path: str, model, optimizer, step: int, config: dict):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
        "config": config
    }, path)


def main():
    global logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="simple_train.jsonl")
    parser.add_argument("--vocab_file", type=str, default="vocab_simple.txt")
    parser.add_argument("--save_dir", type=str, default="ckpt_simple")
    parser.add_argument("--log_file", type=str, default="", help="Log file path (default: save_dir/train.log)")
    parser.add_argument("--load_model", type=str, default="", help="Load checkpoint to continue training")
    parser.add_argument("--console", action="store_true", help="Also print to console")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # 模型配置 - 简化
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension (smaller)")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of layers (smaller)")
    parser.add_argument("--ctx_len", type=int, default=512, help="Context length (smaller)")

    # 训练配置
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=5, help="Eval every N epochs")
    parser.add_argument("--mask_prompt", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    args = parser.parse_args()

    # 初始化日志
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = args.log_file if args.log_file else os.path.join(args.save_dir, "train.log")
    logger = Logger(log_file, console=args.console)

    log(f"Log file: {log_file}")
    log(f"Arguments: {vars(args)}")

    set_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        log("CUDA not available, using CPU")

    if device == "cuda":
        load_rwkv7_extension()

    # Tokenizer
    tokenizer = VocabTokenizer(vocab_file=args.vocab_file)
    vocab_size = tokenizer.get_vocab_size()
    log(f"Vocab size: {vocab_size}")

    # Model
    model = RWKV_Model(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # 加载已有模型（如果指定）
    start_epoch = 0
    if args.load_model and os.path.isfile(args.load_model):
        ckpt = torch.load(args.load_model, map_location=device)
        model.load_state_dict(ckpt["model"])
        log(f"Loaded model from {args.load_model}")

    # Dataset
    ds = SokobanDataset(
        path=args.train_file,
        tokenizer=tokenizer,
        ctx_len=args.ctx_len,
        mask_prompt=bool(args.mask_prompt)
    )

    n = len(ds)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_fn(b, args.ctx_len, tokenizer.PAD, bool(args.mask_prompt)),
        drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_fn(b, args.ctx_len, tokenizer.PAD, bool(args.mask_prompt)),
        drop_last=False
    )

    log(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(100, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    # Training loop
    model.train()
    global_step = 0
    best_val_loss = float('inf')

    log(f"Starting training for {args.epochs} epochs...")
    log(f"Model: n_embd={args.n_embd}, n_layer={args.n_layer}, ctx_len={args.ctx_len}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(x)
            if not torch.isfinite(logits).all():
                log(f"Warning: NaN in logits at step {global_step}")
                continue

            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            n_valid = (flat_labels != -100).sum()

            if n_valid.item() == 0:
                continue

            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Update LR
            lr_mult = lr_schedule(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_mult

            optimizer.step()

            epoch_loss += loss.item() * n_valid.item()
            epoch_tokens += n_valid.item()
            global_step += 1

        # Epoch stats
        avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        elapsed = time.time() - t0
        log(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}, time={elapsed:.1f}s")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_loss = evaluate_and_show(
                model, tokenizer, val_loader, ds.raw_data, device, args.ctx_len, num_samples=2
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.save_dir, "model_best.pt")
                save_checkpoint(save_path, model, optimizer, global_step, vars(args))
                log(f"[Save] Best model saved to {save_path}")

    # Final save
    save_path = os.path.join(args.save_dir, "model_final.pt")
    save_checkpoint(save_path, model, optimizer, global_step, vars(args))
    log(f"[Save] Final model saved to {save_path}")
    log("Training complete!")

    # 关闭日志
    logger.close()


if __name__ == "__main__":
    main()
