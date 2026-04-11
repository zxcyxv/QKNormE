import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention with optional QK-Norm.

    Always outputs PV·W_O. Residual connection strategy is handled by
    TransformerBlock, not here.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        self.W_qkv = nn.Linear(d, 3 * d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # QK-Norm: RMSNorm applied to Q and K (per d_head dimension)
        if config.qk_norm:
            self.q_norm = RMSNorm(config.d_head)
            self.k_norm = RMSNorm(config.d_head)
            # Learnable temperature per head
            # RMSNorm makes ||q||=||k||=√d_head, so Q·K^T max = d_head.
            # Fixed 1/√d_head would cap logits at √d_head ≈ 8, too flat.
            # Instead, learn τ per head. Init at 1/√d_head for stable start.
            self.log_tau = nn.Parameter(
                torch.full((config.n_heads,), math.log(1.0 / math.sqrt(config.d_head)))
            )

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        B, T, D = x.shape

        # QKV projection
        qkv = self.W_qkv(x)
        q, k, v = qkv.split(D, dim=-1)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # QK-Norm: RMSNorm on Q and K + learnable temperature
        if self.config.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            tau = self.log_tau.exp()  # (n_heads,)
            scale = tau.view(1, self.n_heads, 1, 1)
        else:
            scale = 1.0 / math.sqrt(self.d_head)

        # Attention scores + causal mask
        scores = scale * (q @ k.transpose(-2, -1))  # (B, n_heads, T, T)
        causal_mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        P = F.softmax(scores, dim=-1)
        P = self.attn_dropout(P)

        # PV → W_O
        pv = P @ v
        out = pv.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)
        out = self.resid_dropout(out)

        diagnostics = {}
        if return_diagnostics:
            with torch.no_grad():
                # Attention entropy: -sum(P * log(P))
                P_clean = P.clamp(min=1e-8)
                entropy = -(P_clean * P_clean.log()).sum(dim=-1)  # (B, n_heads, T)
                diagnostics["attn_entropy"] = entropy.mean(dim=(0, 2))  # (n_heads,)

                if self.config.qk_norm:
                    diagnostics["tau"] = tau.detach()

        return out, diagnostics


class TransformerBlock(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.mode = config.attn_output_mode
        self.attn = CausalSelfAttention(config)
        self.ff = FeedForward(config)

        if self.mode == "baseline":
            # Pre-LN: norm before sublayer
            self.attn_norm = RMSNorm(config.d_model)
            self.ff_norm = RMSNorm(config.d_model)
        elif self.mode in ("rezero", "rezero_pvh"):
            # Pure ReZero: no norms in block, η init=0 → Jacobian = I
            self.eta_attn = nn.Parameter(torch.zeros(1))
            self.eta_ffn = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(
        self, h: torch.Tensor, return_diagnostics: bool = False
    ) -> tuple[torch.Tensor, dict]:

        if self.mode == "baseline":
            # ── Pre-LN ──
            h_normed = self.attn_norm(h)
            attn_out, diag = self.attn(
                h_normed, return_diagnostics=return_diagnostics
            )
            h = h + attn_out
            h = h + self.ff(self.ff_norm(h))

        elif self.mode == "rezero":
            # ── ReZero: H + η · PVW_O ──
            attn_out, diag = self.attn(
                h, return_diagnostics=return_diagnostics
            )
            h = h + self.eta_attn * attn_out
            h = h + self.eta_ffn * self.ff(h)

        elif self.mode == "rezero_pvh":
            # ── ReZero + Mean-Shift: H + η · (PVW_O - H) = (1-η)H + η·PVW_O ──
            # dH/dH = I at η=0, perfect identity mapping
            attn_out, diag = self.attn(
                h, return_diagnostics=return_diagnostics
            )
            h = (1 - self.eta_attn) * h + self.eta_attn * attn_out
            h = h + self.eta_ffn * self.ff(h)

        if return_diagnostics:
            diag["token_norm"] = h.detach().norm(dim=-1).mean()
            if self.mode in ("rezero", "rezero_pvh"):
                diag["eta_attn"] = self.eta_attn.detach().item()
                diag["eta_ffn"] = self.eta_ffn.detach().item()

        return h, diag


class GPTQKNorm(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_norm = RMSNorm(config.d_model)  # project onto hypersphere before block 0
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)

        # Weight tying: output head shares weights with token embedding
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len

        pos = torch.arange(T, device=input_ids.device)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)
        h = self.emb_norm(h)   # onto hypersphere before first block
        h = self.drop(h)

        all_diag = {}
        for i, block in enumerate(self.blocks):
            h, diag = block(h, return_diagnostics=return_diagnostics)
            if return_diagnostics:
                for k, v in diag.items():
                    all_diag[f"layer{i}/{k}"] = v

        h = self.final_norm(h)
        logits = self.lm_head(h)

        return logits, all_diag

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
