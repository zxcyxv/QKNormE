import sys
import json
import math
import time
import random
from pathlib import Path
from itertools import islice

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from .config import ExperimentConfig
from .model import GPTQKNorm
from .data import get_dataloaders
from .metrics import collect_diagnostics, collect_gradient_norms, collect_value_hidden_alignment


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def infinite_loader(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model: GPTQKNorm, val_loader, config: ExperimentConfig) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
    model.train()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))  # clamp to prevent overflow
    return avg_loss, ppl


def train(config: ExperimentConfig):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Log directory
    run_name = f"{config.attn_output_mode}_seed{config.seed}"
    log_path = Path(config.log_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {k: v for k, v in config.__dict__.items()}
    (log_path / "config.json").write_text(json.dumps(config_dict, indent=2, default=str))

    # Data
    train_loader, val_loader = get_dataloaders(config)

    # Model
    model = GPTQKNorm(config).to(device)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} parameters", flush=True)
    print(f"Variant: {config.attn_output_mode}", flush=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Cosine annealing with warmup
    def lr_schedule(step: int) -> float:
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / max(
            1, config.max_steps - config.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Mixed precision
    scaler = GradScaler(enabled=config.mixed_precision)

    # Training log
    log_entries = []
    train_iter = infinite_loader(train_loader)

    print(f"Training for {config.max_steps:,} steps...", flush=True)
    t0 = time.time()

    for step in range(config.max_steps):
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward + backward
        with autocast(device_type="cuda", dtype=torch.float16, enabled=config.mixed_precision):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
            scaled_loss = loss / config.grad_accum_steps

        scaler.scale(scaled_loss).backward()

        # Gradient accumulation step
        if (step + 1) % config.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # Collect gradient norms before optimizer step (at diagnostic intervals)
            grad_metrics = {}
            if (step + 1) % config.diagnostic_interval == 0:
                grad_metrics = collect_gradient_norms(model)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # --- Logging ---
        if (step + 1) % config.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            entry = {
                "step": step + 1,
                "train_loss": loss.item(),
                "lr": lr,
                "steps_per_sec": steps_per_sec,
            }
            log_entries.append(entry)
            print(
                f"  step {step+1:>6d} | loss {loss.item():.4f} | lr {lr:.2e} | {steps_per_sec:.1f} steps/s",
                flush=True,
            )

        # --- Evaluation ---
        if (step + 1) % config.eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, config)
            entry = {
                "step": step + 1,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            }
            log_entries.append(entry)
            print(
                f"  [eval] step {step+1:>6d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}",
                flush=True,
            )

        # --- Diagnostics ---
        if (step + 1) % config.diagnostic_interval == 0:
            diag_batch = {k: v.to(device) for k, v in batch.items()}
            diag = collect_diagnostics(model, diag_batch)
            align = collect_value_hidden_alignment(model, diag_batch)
            all_metrics = {**diag, **align, **grad_metrics, "step": step + 1}
            log_entries.append(all_metrics)

        # --- Checkpoint ---
        if (step + 1) % config.save_interval == 0:
            ckpt_path = log_path / f"ckpt_{step+1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                },
                ckpt_path,
            )
            print(f"  [save] {ckpt_path}", flush=True)

    # Final evaluation
    val_loss, val_ppl = evaluate(model, val_loader, config)
    print(f"\n=== Final: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f} ===", flush=True)
    log_entries.append({"step": config.max_steps, "final_val_loss": val_loss, "final_val_ppl": val_ppl})

    # Save logs
    log_file = log_path / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry, default=str) + "\n")
    print(f"Logs saved to {log_file}", flush=True)

    # Save final model
    torch.save(model.state_dict(), log_path / "model_final.pt")
    print(f"Model saved to {log_path / 'model_final.pt'}", flush=True)

    return val_loss, val_ppl


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="baseline", choices=["baseline", "postnorm", "postnorm_pvh", "postnorm_pvh_full"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="runs")
    args = parser.parse_args()

    overrides = {"seed": args.seed, "log_dir": args.log_dir}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.grad_accum is not None:
        overrides["grad_accum_steps"] = args.grad_accum

    from .config import baseline_config, postnorm_config, postnorm_pvh_config, postnorm_pvh_full_config

    configs = {
        "baseline": baseline_config,
        "postnorm": postnorm_config,
        "postnorm_pvh": postnorm_pvh_config,
        "postnorm_pvh_full": postnorm_pvh_full_config,
    }

    config = configs[args.variant](**overrides)
    train(config)
