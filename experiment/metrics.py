import torch
import torch.nn.functional as F

from .model import GPTQKNorm
from .config import ExperimentConfig


@torch.no_grad()
def collect_diagnostics(
    model: GPTQKNorm,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Collect diagnostic metrics from a single forward pass."""
    model.eval()
    logits, diag = model(batch["input_ids"], return_diagnostics=True)
    model.train()

    metrics = {}

    for key, val in diag.items():
        if "attn_entropy" in key:
            # val is (n_heads,) tensor
            layer = key.split("/")[0]
            for h_idx, ent in enumerate(val):
                metrics[f"diag/{layer}/head{h_idx}/entropy"] = ent.item()
            metrics[f"diag/{layer}/entropy_mean"] = val.mean().item()

        elif "token_norm" in key:
            metrics[f"diag/{key}"] = val.item()

        elif "eta_attn" in key:
            layer = key.split("/")[0]
            metrics[f"diag/{layer}/eta_attn"] = val if isinstance(val, float) else val.item()

        elif "eta_ffn" in key:
            layer = key.split("/")[0]
            metrics[f"diag/{layer}/eta_ffn"] = val if isinstance(val, float) else val.item()

        elif "tau" in key and "eta" not in key:
            layer = key.split("/")[0]
            for h_idx, t in enumerate(val):
                metrics[f"diag/{layer}/head{h_idx}/tau"] = t.item()

    return metrics


@torch.no_grad()
def collect_gradient_norms(model: GPTQKNorm) -> dict[str, float]:
    """Collect per-layer gradient norms after backward pass."""
    metrics = {}
    for i, block in enumerate(model.blocks):
        attn_grad_norm = 0.0
        ff_grad_norm = 0.0

        for p in block.attn.parameters():
            if p.grad is not None:
                attn_grad_norm += p.grad.data.norm(2).item() ** 2
        for p in block.ff.parameters():
            if p.grad is not None:
                ff_grad_norm += p.grad.data.norm(2).item() ** 2

        metrics[f"grad_norm/layer{i}/attn"] = attn_grad_norm ** 0.5
        metrics[f"grad_norm/layer{i}/ff"] = ff_grad_norm ** 0.5

    return metrics


@torch.no_grad()
def collect_value_hidden_alignment(
    model: GPTQKNorm,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Measure cosine similarity between v_i*W_O and h_i per layer."""
    model.eval()

    B, T = batch["input_ids"].shape
    device = batch["input_ids"].device
    config = model.config

    pos = torch.arange(T, device=device)
    h = model.drop(model.tok_emb(batch["input_ids"]) + model.pos_emb(pos))

    metrics = {}
    for i, block in enumerate(model.blocks):
        h_before = h  # residual stream before this layer

        # For Pre-LN (baseline), Q/K/V are derived from normed h
        # For Post-Norm variants, Q/K/V are derived from raw h
        if block.mode == "baseline":
            attn_input = block.attn_norm(h)
        else:
            attn_input = h

        # Compute V = attn_input @ W_V
        qkv = block.attn.W_qkv(attn_input)
        _, _, v = qkv.split(config.d_model, dim=-1)

        # V @ W_O
        v_wo = block.attn.W_o(v)  # (B, T, d_model)

        # Cosine similarity between v_wo and h_before
        cos_sim = F.cosine_similarity(v_wo, h_before, dim=-1)  # (B, T)
        metrics[f"diag/layer{i}/value_hidden_cos"] = cos_sim.mean().item()

        # Continue forward pass
        h, _ = block(h, return_diagnostics=False)

    model.train()
    return metrics
