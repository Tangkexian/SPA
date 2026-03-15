import argparse
import json
import logging
import pathlib
import random
import sys
from typing import Any, Dict, List
from tqdm import tqdm
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, set_seed
from ..utils_tools.utils import build_train_sequences

LOG = logging.getLogger(__name__)
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, nargs='+')
    p.add_argument("--output_dir", required=True, help="Directory to save tokenized dataset and eval data")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", help="Tokenizer name")
    p.add_argument("--tag", default="CPT_tokenized_dataset")
    p.add_argument("--k_completions", type=int, nargs='+', default=[5])
    p.add_argument("--n_articles", type=int, default=-1)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=43)
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    set_seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.tag
    
    # --- Load Data ---
    if len(args.k_completions) == 1 and len(args.dataset) > 1:
        args.k_completions = args.k_completions * len(args.dataset)
    if len(args.k_completions) != len(args.dataset):
        sys.exit(f"[!] Error: Provided {len(args.dataset)} datasets but {len(args.k_completions)} k_completions values.")
    dataset: List[Dict[str, Any]] = []
    for i, d_path_str in enumerate(args.dataset):
        data_path = pathlib.Path(d_path_str)
        target_k = args.k_completions[i]
        if data_path.is_dir():
            LOG.info(f"Loading data from directory: {data_path}")
            files = sorted(list(data_path.glob("*.json")) + list(data_path.glob(".*.json")))
            for f_path in tqdm(files, desc=f"Reading dir {d_path_str}", leave=False):
                try:
                    content = json.loads(f_path.read_text(encoding="utf-8"))
                    if isinstance(content, list) and len(content) > 1:
                        for txt in content[1:]:
                            dataset.append({"body": txt, "_target_k": target_k})
                except Exception as e:
                    LOG.warning(f"Error reading {f_path}: {e}")
            continue
        if not data_path.exists():
            sys.exit(f"[!] Dataset not found: {data_path}")
        all_data = json.loads(data_path.read_text(encoding="utf-8"))
        for item in all_data:
            item["_target_k"] = target_k
        dataset.extend(all_data)
    if args.n_articles > 0:
        dataset = dataset[: args.n_articles]

    train_sequences: List[str] = []
    for item_idx, item in enumerate(dataset):
        title = item.get("title", "")
        context = item.get("body", "") if "context" not in item else item["context"]
        target_k = item.get("_target_k", 5)
        
        if "completions" in item:
            if item["completions"] and isinstance(item["completions"][0], list):
                all_comp_rows = item["completions"]
                selected_rows = random.sample(all_comp_rows, min(target_k, len(all_comp_rows))) if target_k > 0 else []
                for i, comp_row in enumerate(selected_rows):
                    for j, comp_text in enumerate(comp_row):
                        if comp_text is None:
                            continue
                        if comp_text.strip():
                            train_sequences.extend(build_train_sequences(comp_text, context, title, add_context=False))

    # --- Tokenize ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    LOG.info(f"Tokenizing (splitting long sequences max_len={args.max_seq_length})...")
    rows = []
    for seq in tqdm(train_sequences, desc="Tokenizing"):
        tok = tokenizer(seq, truncation=False, add_special_tokens=True)
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"] 
        labels = list(input_ids)     
        for i in range(0, len(input_ids), args.max_seq_length):
            chunk_ids = input_ids[i : i + args.max_seq_length]
            chunk_mask = attention_mask[i : i + args.max_seq_length]
            chunk_labels = labels[i : i + args.max_seq_length]
            if len(chunk_ids) < args.max_seq_length:
                pad_len = args.max_seq_length - len(chunk_ids)
                chunk_ids = chunk_ids + [tokenizer.pad_token_id] * pad_len
                chunk_mask = chunk_mask + [0] * pad_len
                chunk_labels = chunk_labels + [-100] * pad_len
            
            rows.append({
                "input_ids": chunk_ids,
                "attention_mask": chunk_mask,
                "labels": chunk_labels,
            })
    train_ds = HFDataset.from_list(rows)
    LOG.info(f"Created {len(train_ds)} samples after splitting.")
        
    row_tokens = [sum(x) for x in train_ds["attention_mask"]]
    total_tokens = sum(row_tokens)
    LOG.info("Total tokens in train set (excluding padding): %d", total_tokens)
    print(f"Total tokens in train set (excluding padding): {total_tokens}")

    # --- Save ---
    train_ds_path = out_path / "train_dataset"
    LOG.info(f"Saving dataset to {train_ds_path}")
    train_ds.save_to_disk(str(train_ds_path))
            
    # save args
    args_file = out_path / "tokenization_args.json"
    LOG.info(f"Saving args to {args_file}")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)
        json.dump({"article_number": len(dataset)}, f, indent=2)

if __name__ == "__main__":
    main()
