#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立评估脚本：加载模型并用正确的验证器测试
"""

import os
import sys
import json
import random
import argparse
import torch

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_simple import (
    VocabTokenizer, RWKV_Model, load_rwkv7_extension,
    generate, check_solution_validity, parse_map,
    log, log_raw, Logger, set_seed_all
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_file", type=str, required=True, help="Data file to evaluate")
    parser.add_argument("--vocab_file", type=str, default="vocab_simple.txt")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ctx_len", type=int, default=2048)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=6)

    args = parser.parse_args()

    set_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if device == "cuda":
        load_rwkv7_extension()

    # Tokenizer
    tokenizer = VocabTokenizer(vocab_file=args.vocab_file)
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Model
    model = RWKV_Model(
        vocab_size=tokenizer.get_vocab_size(),
        n_embd=args.n_embd,
        n_layer=args.n_layer,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from {args.model_path}")

    # Load data
    raw_data = []
    with open(args.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))

    print(f"Loaded {len(raw_data)} samples from {args.data_file}")

    # Evaluate
    model.eval()
    sample_indices = random.sample(range(len(raw_data)), min(args.num_samples, len(raw_data)))

    total_score = 0.0
    solved = 0
    errors = 0

    print("\n" + "="*80)
    print("评估结果（使用模拟验证器）")
    print("="*80)

    for idx, sample_idx in enumerate(sample_indices):
        data = raw_data[sample_idx]
        prompt = data["input"] + "\n"

        print(f"\n【样例 {idx + 1}/{len(sample_indices)}】")
        print(f"输入:")
        print(data["input"])

        # Generate
        with torch.no_grad():
            generated = generate(model, tokenizer, prompt, device, args.ctx_len, max_new=1024)

        print(f"\n模型生成:")
        print(generated)

        # Validate using simulator
        validity = check_solution_validity(generated, data)

        print(f"\n验证结果: {validity['reason']} (得分: {validity['score']:.2f})")
        if validity.get('details'):
            for detail in validity['details']:
                print(f"  - {detail}")

        total_score += validity["score"]
        if validity["valid"] and validity["score"] >= 0.9:
            solved += 1
        if validity["score"] == 0:
            errors += 1

        print(f"\n真实答案:")
        print(data["cot"][:500] + "..." if len(data["cot"]) > 500 else data["cot"])
        print("-"*60)

    # Summary
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print(f"测试样本数: {len(sample_indices)}")
    print(f"完全正确: {solved} ({100*solved/len(sample_indices):.1f}%)")
    print(f"执行错误: {errors} ({100*errors/len(sample_indices):.1f}%)")
    print(f"平均得分: {total_score/len(sample_indices):.2f}")


if __name__ == "__main__":
    main()
