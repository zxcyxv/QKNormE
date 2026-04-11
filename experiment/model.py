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
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.mode = config.attn_output_mode
        d = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        self.W_qkv = nn.Linear(d, 3 * d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # QK-Norm: learnable log-temperature per head
        if config.qk_norm:
            self.log_tau = nn.Parameter(
                torch.full((config.n_heads,), config.log_tau_init)
            )

        # pvwo_minus_h: learnable eta per layer
        if self.mode == "pvwo_minus_h":
            # sigmoid^{-1}(eta_init)
            raw = math.log(config.eta_init / (1.0 - config.eta_init))
            self.raw_eta = nn.Parameter(torch.tensor(raw))

        # gated: learnable alpha per layer
        if self.mode == "gated":
            raw = math.log(config.gate_init / (1.0 - config.gate_init))
            self.raw_alpha = nn.Parameter(torch.tensor(raw))

    def forward(
        self,
        h_normed: torch.Tensor,
        h_residual: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        B, T, D = h_normed.shape

        # QKV projection
        qkv = self.W_qkv(h_normed)
        q, k, v = qkv.split(D, dim=-1)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # QK-Norm: L2 normalize + learnable temperature
        if self.config.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
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

        # PV: (B, n_heads, T, d_head)
        pv = P @ v

        # --- Switchable attention output ---
        if self.mode == "baseline":
            # Standard: output = PV @ W_O
            out = pv.transpose(1, 2).contiguous().view(B, T, D)
            out = self.W_o(out)

        elif self.mode == "piv":
            # (P-I)V = PV - V, then project
            piv = pv - v
            out = piv.transpose(1, 2).contiguous().view(B, T, D)
            out = self.W_o(out)

        elif self.mode == "pvwo_minus_h":
            # eta * (PVW_O - h_residual)
            pv_wo = self.W_o(pv.transpose(1, 2).contiguous().view(B, T, D))
            eta = torch.sigmoid(self.raw_eta)
            out = eta * (pv_wo - h_residual)

        elif self.mode == "gated":
            # PVW_O - alpha * VW_O
            pv_flat = pv.transpose(1, 2).contiguous().view(B, T, D)
            v_flat = v.transpose(1, 2).contiguous().view(B, T, D)
            pv_wo = self.W_o(pv_flat)
            v_wo = self.W_o(v_flat)
            alpha = torch.sigmoid(self.raw_alpha)
            out = pv_wo - alpha * v_wo

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

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

                if self.mode == "pvwo_minus_h":
                    diagnostics["eta"] = torch.sigmoid(self.raw_eta).detach()
                elif self.mode == "gated":
                    diagnostics["alpha"] = torch.sigmoid(self.raw_alpha).detach()

        return out, diagnostics


class TransformerBlock(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(
        self, h: torch.Tensor, return_diagnostics: bool = False
    ) -> tuple[torch.Tensor, dict]:
        # Pre-LN: norm → attn → residual
        h_normed = self.attn_norm(h)
        attn_out, diag = self.attn(
            h_normed, h_residual=h, return_diagnostics=return_diagnostics
        )
        h = h + attn_out

        # Pre-LN: norm → ff → residual
        h = h + self.ff(self.ff_norm(h))

        if return_diagnostics:
            diag["token_norm"] = h.detach().norm(dim=-1).mean()

        return h, diag


class GPTQKNorm(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
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
        h = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

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
