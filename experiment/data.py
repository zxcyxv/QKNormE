import os
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from .config import ExperimentConfig


class ChunkedTextDataset(Dataset):
    """Tokenized text chunked into fixed-length sequences for LM training."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        n = len(token_ids) // (seq_len + 1)
        self.data = torch.tensor(token_ids[: n * (seq_len + 1)], dtype=torch.long)
        self.data = self.data.view(n, seq_len + 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.data[idx]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def get_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # Disk cache: dataset + tokenizer + seq_len 조합으로 고유 키 생성     #
    # 최초 실행 시만 토크나이징하고, 이후엔 .pt 파일에서 바로 로드        #
    # ------------------------------------------------------------------ #
    cache_key = (
        f"{config.dataset_name}_{config.dataset_config}"
        f"_{config.tokenizer_name}_{config.max_seq_len}"
    )
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    train_cache = cache_dir / f"train_{cache_hash}.pt"
    val_cache   = cache_dir / f"val_{cache_hash}.pt"

    if train_cache.exists() and val_cache.exists():
        print("Loading tokenized data from cache...", flush=True)
        train_dataset = torch.load(train_cache, weights_only=False)
        val_dataset   = torch.load(val_cache,   weights_only=False)
    else:
        eos_id  = tokenizer.eos_token_id
        dataset = load_dataset(config.dataset_name, config.dataset_config)

        # Windows는 fork 없이 spawn → num_proc > 1이면 오히려 느림
        num_proc = 1 if os.name == "nt" else 2

        def tokenize_fn(examples):
            out = tokenizer(examples["text"], add_special_tokens=False)
            result = []
            for ids in out["input_ids"]:
                if ids:
                    result.append(ids + [eos_id])
                else:
                    result.append([])
            return {"input_ids": result}

        def flatten_ids(split_name: str) -> list[int]:
            ds = dataset[split_name].filter(
                lambda ex: bool(ex["text"].strip()),
                num_proc=num_proc,
                desc=f"Filtering {split_name}",
            )
            ds = ds.map(
                tokenize_fn,
                batched=True,
                batch_size=1000,
                remove_columns=ds.column_names,
                num_proc=num_proc,
                desc=f"Tokenizing {split_name}",
            )
            all_ids: list[int] = []
            for row in ds:
                all_ids.extend(row["input_ids"])
            return all_ids

        print("Preparing train split...", flush=True)
        train_ids = flatten_ids("train")
        print(f"  {len(train_ids):,} tokens", flush=True)

        print("Preparing validation split...", flush=True)
        val_ids = flatten_ids("validation")
        print(f"  {len(val_ids):,} tokens", flush=True)

        train_dataset = ChunkedTextDataset(train_ids, config.max_seq_len)
        val_dataset   = ChunkedTextDataset(val_ids,   config.max_seq_len)

        print("Saving tokenized data to cache...", flush=True)
        torch.save(train_dataset, train_cache)
        torch.save(val_dataset,   val_cache)

    print(f"Train: {len(train_dataset):,} chunks, Val: {len(val_dataset):,} chunks")

    n_workers = 0 if os.name == "nt" else 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
