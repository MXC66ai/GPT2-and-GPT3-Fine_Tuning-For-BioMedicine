import os
import json
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

output_dir = "edu_fineweb10B"
os.makedirs(output_dir, exist_ok=True)

all_train_tokens = []
all_val_tokens = []

print("=" * 60)
print("处理 PMC-OA...")
pmc_oa_path = r"E:\maxc\PMC-OA"

# train.jsonl - 1GB, 有大量 caption 文本
train_file = os.path.join(pmc_oa_path, "train.jsonl")
val_file = os.path.join(pmc_oa_path, "valid.jsonl")

def process_jsonl(filepath, token_list, max_items=None):
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                caption = obj.get("caption", "")
                if caption and len(caption) > 20:
                    # 编码文本, 加上换行分隔
                    tokens = enc.encode(caption)
                    token_list.extend(tokens)
                    token_list.append(198)  # \n token in GPT-2 BPE
                    count += 1
            except (json.JSONDecodeError, KeyError):
                continue
            if max_items and count >= max_items:
                break
    print(f"  {filepath}: {count} 条文本")

# 训练集 - 处理全部 (1GB 文件, 会有很多文本)
print(f"  处理 {train_file} ...")
process_jsonl(train_file, all_train_tokens)
print(f"  处理 {val_file} ...")
process_jsonl(val_file, all_val_tokens)

print("=" * 60)
print("处理 PMC-VQA / Slake1.0...")
slake_path = r"E:\maxc\PMC-VQA\Slake1.0"

def process_slake(filepath, token_list, max_items=None):
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        # 英文问答
        if item.get("q_lang") == "en":
            q = item.get("question", "")
            a = item.get("answer", "")
            if q and a:
                text = f"Question: {q}\nAnswer: {a}"
                tokens = enc.encode(text)
                token_list.extend(tokens)
                token_list.append(198)
                count += 1
        # 中文问答 (尝试解码)
        elif item.get("q_lang") == "zh":
            q = item.get("question", "")
            a = item.get("answer", "")
            if q and a:
                # 中文文本用 GPT-2 BPE 也能编码, 只是用 bytes 编码
                text = f"Question: {q}\nAnswer: {a}"
                tokens = enc.encode(text)
                token_list.extend(tokens)
                token_list.append(198)
                count += 1
        if max_items and count >= max_items:
            break
    print(f"  {filepath}: {count} 条问答")

train_json = os.path.join(slake_path, "train.json")
val_json = os.path.join(slake_path, "validate.json")
test_json = os.path.join(slake_path, "test.json")

if os.path.exists(train_json):
    print(f"  处理 {train_json} ...")
    process_slake(train_json, all_train_tokens)
if os.path.exists(val_json):
    print(f"  处理 {val_json} ...")
    process_slake(val_json, all_val_tokens)
if os.path.exists(test_json):
    print(f"  处理 {test_json} ...")
    process_slake(test_json, all_val_tokens)

print("=" * 60)
print("保存为 .npy 文件...")

if len(all_train_tokens) > 0:
    train_arr = np.array(all_train_tokens, dtype=np.int32)
    train_path = os.path.join(output_dir, "train_0000.npy")
    np.save(train_path, train_arr)
    print(f"  train_0000.npy: {len(all_train_tokens):,} tokens ({train_arr.nbytes / 1e6:.1f} MB)")

if len(all_val_tokens) > 0:
    val_arr = np.array(all_val_tokens, dtype=np.int32)
    val_path = os.path.join(output_dir, "val_0000.npy")
    np.save(val_path, val_arr)
    print(f"  val_0000.npy: {len(all_val_tokens):,} tokens ({val_arr.nbytes / 1e6:.1f} MB)")

print("=" * 60)
print(f"完成! 数据已保存到 {output_dir}/")
print(f"  训练集: {len(all_train_tokens):,} tokens")
print(f"  验证集: {len(all_val_tokens):,} tokens")
print(f"\n现在可以运行 train_gpt2.py 了!")
