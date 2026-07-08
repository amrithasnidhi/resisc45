"""
Hybrid Quantum-Classical Fusion Models for RESISC-45.

E11 – MobileViT-S (frozen) + ResNet-18+PQC  → Concatenation
E12 – MobileViT-S (frozen) + ResNet-18+PQC  → Cross-Attention
E13 – DenseNet-121 (frozen) + MobileViT-S+PQC → Multi-Head Cross-Attention
E14 – DenseNet-121 (frozen) + MobileViT-S+PQC → Concatenation

All models accept (B, 3, 64, 64) input (internally resized to 224×224 for
pretrained backbones) and output (B, 45) logits.

Feature caching: frozen Branch-A backbones expose `extract_branch_a(x)` so
callers can cache features on disk and pass them as `cached_a=` in forward().
This is critical for CPU training speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.quantum_models import _make_sel_layer
from shared_config import N_QUBITS, NUM_CLASSES, BACKBONE_SIZE


def _resize(x: torch.Tensor, size: int = BACKBONE_SIZE) -> torch.Tensor:
    """Bilinear upsample x to (size × size) if needed."""
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = False
    return module


# ── E11: MobileViT-S + (ResNet-18 + 4Q PQC) – Concatenation ─────────────────

class E11_MobileViT_ResNet_Concat(nn.Module):
    """
    Branch A: MobileViT-S (4.94 M, frozen)  → (B, 640)
    Branch B: ResNet-18 + 4Q PQC            → (B, 4)
    Fusion  : concat → [640+4=644] → MLP → 45
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2,
                 n_classes: int = NUM_CLASSES):
        super().__init__()

        # Branch A – frozen
        self.branch_a   = _freeze(
            timm.create_model('mobilevit_s', pretrained=True,
                              num_classes=0, global_pool='avg')
        )
        branch_a_dim = 640

        # Branch B – trainable
        self.branch_b   = timm.create_model('resnet18', pretrained=True,
                                            num_classes=0, global_pool='avg')
        self.quantum_fc = nn.Sequential(nn.Linear(512, n_qubits), nn.Tanh())
        self.qlayer     = _make_sel_layer(n_qubits, n_layers)

        # Fusion head
        fusion_dim = branch_a_dim + n_qubits   # 644
        self.head  = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def extract_branch_a(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen Branch A – call once per dataset to cache features."""
        with torch.no_grad():
            return self.branch_a(_resize(x))

    def forward(self, x: torch.Tensor,
                cached_a: torch.Tensor | None = None) -> torch.Tensor:
        feat_a = cached_a if cached_a is not None else self.extract_branch_a(x)
        feat_b = self.branch_b(_resize(x))          # (B, 512)
        angles = self.quantum_fc(feat_b)             # (B, 4)
        q_out  = self.qlayer(angles)                 # (B, 4)
        fused  = torch.cat([feat_a, q_out], dim=1)  # (B, 644)
        return self.head(fused)


# ── E12: MobileViT-S + (ResNet-18 + 4Q PQC) – Cross-Attention ───────────────

class E12_MobileViT_ResNet_CrossAttn(nn.Module):
    """
    Branch A: MobileViT-S (frozen) → (B, 640)
    Branch B: ResNet-18 + 4Q PQC  → (B, 4)
    Fusion  : Cross-Attention (Q=quantum, K=V=CNN, dim=128)
              + residual + LayerNorm → Linear(128, 45)
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2,
                 n_classes: int = NUM_CLASSES, attn_dim: int = 128):
        super().__init__()

        self.branch_a   = _freeze(
            timm.create_model('mobilevit_s', pretrained=True,
                              num_classes=0, global_pool='avg')
        )

        self.branch_b   = timm.create_model('resnet18', pretrained=True,
                                            num_classes=0, global_pool='avg')
        self.quantum_fc = nn.Sequential(nn.Linear(512, n_qubits), nn.Tanh())
        self.qlayer     = _make_sel_layer(n_qubits, n_layers)

        # Cross-attention projections (single query/key/value pair → gating)
        self.q_proj  = nn.Linear(n_qubits, attn_dim)
        self.k_proj  = nn.Linear(640,      attn_dim)
        self.v_proj  = nn.Linear(640,      attn_dim)
        self.ln      = nn.LayerNorm(attn_dim)
        self.scale   = attn_dim ** 0.5

        self.classifier = nn.Linear(attn_dim, n_classes)

    def extract_branch_a(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.branch_a(_resize(x))

    def forward(self, x: torch.Tensor,
                cached_a: torch.Tensor | None = None) -> torch.Tensor:
        feat_a = cached_a if cached_a is not None else self.extract_branch_a(x)
        feat_b = self.branch_b(_resize(x))
        angles = self.quantum_fc(feat_b)
        q_out  = self.qlayer(angles)                 # (B, 4)

        Q = self.q_proj(q_out)    # (B, 128)
        K = self.k_proj(feat_a)   # (B, 128)
        V = self.v_proj(feat_a)   # (B, 128)

        # Scalar attention gate: sigmoid(Q·K / √d)
        gate     = torch.sigmoid((Q * K).sum(dim=-1, keepdim=True) / self.scale)  # (B, 1)
        attended = gate * V                             # (B, 128)
        out      = self.ln(Q + attended)               # (B, 128) residual + LN
        return self.classifier(out)


# ── E13: DenseNet-121 + (MobileViT-S + 4Q PQC) – MH Cross-Attention ─────────

class E13_DenseNet_MobileViT_CrossAttn(nn.Module):
    """
    Branch A: DenseNet-121 (6.96 M, frozen) → (B, 1024)
    Branch B: MobileViT-S + 4Q PQC         → (B, 4)
    Fusion  : Multi-Head Cross-Attention (4 heads, dim=256)
              + residual + LayerNorm → Linear(256, 45)
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2,
                 n_classes: int = NUM_CLASSES, n_heads: int = 4,
                 attn_dim: int = 256):
        super().__init__()

        self.branch_a   = _freeze(
            timm.create_model('densenet121', pretrained=True,
                              num_classes=0, global_pool='avg')
        )

        self.branch_b   = timm.create_model('mobilevit_s', pretrained=True,
                                            num_classes=0, global_pool='avg')
        self.quantum_fc = nn.Sequential(nn.Linear(640, n_qubits), nn.Tanh())
        self.qlayer     = _make_sel_layer(n_qubits, n_layers)

        self.q_proj  = nn.Linear(n_qubits, attn_dim)
        self.k_proj  = nn.Linear(1024,     attn_dim)
        self.v_proj  = nn.Linear(1024,     attn_dim)
        self.mha     = nn.MultiheadAttention(attn_dim, n_heads, batch_first=True)
        self.ln      = nn.LayerNorm(attn_dim)

        self.classifier = nn.Linear(attn_dim, n_classes)

    def extract_branch_a(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.branch_a(_resize(x))

    def forward(self, x: torch.Tensor,
                cached_a: torch.Tensor | None = None) -> torch.Tensor:
        feat_a = cached_a if cached_a is not None else self.extract_branch_a(x)
        feat_b = self.branch_b(_resize(x))          # (B, 640)
        angles = self.quantum_fc(feat_b)             # (B, 4)
        q_out  = self.qlayer(angles)                 # (B, 4)

        Q = self.q_proj(q_out).unsqueeze(1)          # (B, 1, 256)
        K = self.k_proj(feat_a).unsqueeze(1)         # (B, 1, 256)
        V = self.v_proj(feat_a).unsqueeze(1)         # (B, 1, 256)

        attended, _ = self.mha(Q, K, V)              # (B, 1, 256)
        out = self.ln(Q + attended).squeeze(1)       # (B, 256)
        return self.classifier(out)


# ── E14: DenseNet-121 + (MobileViT-S + 4Q PQC) – Concatenation ──────────────

class E14_DenseNet_MobileViT_Concat(nn.Module):
    """
    Branch A: DenseNet-121 (frozen) → (B, 1024)
    Branch B: MobileViT-S + 4Q PQC → (B, 4)
    Fusion  : concat [1024+4=1028] → MLP → 45
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2,
                 n_classes: int = NUM_CLASSES):
        super().__init__()

        self.branch_a   = _freeze(
            timm.create_model('densenet121', pretrained=True,
                              num_classes=0, global_pool='avg')
        )

        self.branch_b   = timm.create_model('mobilevit_s', pretrained=True,
                                            num_classes=0, global_pool='avg')
        self.quantum_fc = nn.Sequential(nn.Linear(640, n_qubits), nn.Tanh())
        self.qlayer     = _make_sel_layer(n_qubits, n_layers)

        fusion_dim = 1024 + n_qubits   # 1028
        self.head  = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def extract_branch_a(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.branch_a(_resize(x))

    def forward(self, x: torch.Tensor,
                cached_a: torch.Tensor | None = None) -> torch.Tensor:
        feat_a = cached_a if cached_a is not None else self.extract_branch_a(x)
        feat_b = self.branch_b(_resize(x))          # (B, 640)
        angles = self.quantum_fc(feat_b)             # (B, 4)
        q_out  = self.qlayer(angles)                 # (B, 4)
        fused  = torch.cat([feat_a, q_out], dim=1)  # (B, 1028)
        return self.head(fused)
