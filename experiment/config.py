from dataclasses import dataclass, field
import math


@dataclass
class ExperimentConfig:
    # --- Variant ---
    attn_output_mode: str = "baseline"  # baseline, postnorm, postnorm_pvh

    # --- Model ---
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    vocab_size: int = 50257
    max_seq_len: int = 512
    dropout: float = 0.1

    # --- QK-Norm ---
    qk_norm: bool = True

    # --- Training ---
    batch_size: int = 16
    grad_accum_steps: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 40_000
    grad_clip: float = 1.0
    mixed_precision: bool = True
    seed: int = 42

    # --- Data ---
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    tokenizer_name: str = "gpt2"

    # --- Logging ---
    log_dir: str = "runs"
    log_interval: int = 50
    eval_interval: int = 500
    diagnostic_interval: int = 500
    save_interval: int = 5000

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


def baseline_config(**overrides) -> ExperimentConfig:
    return ExperimentConfig(attn_output_mode="baseline", **overrides)


def postnorm_config(**overrides) -> ExperimentConfig:
    return ExperimentConfig(attn_output_mode="postnorm", **overrides)


def postnorm_pvh_config(**overrides) -> ExperimentConfig:
    return ExperimentConfig(attn_output_mode="postnorm_pvh", **overrides)


def postnorm_pvh_full_config(**overrides) -> ExperimentConfig:
    return ExperimentConfig(attn_output_mode="postnorm_pvh_full", **overrides)
