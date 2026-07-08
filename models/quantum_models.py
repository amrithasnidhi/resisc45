"""
Pure Quantum Models for RESISC-45.

Q1_QNN4EO      – ESA baseline quantum model (LeNet encoder + 4Q SEL)
Q2_AngleSEL    – Quadrant PQC with cross-attention fusion
Q3_DataReupload – Data reuploading quantum circuit

All models accept (B, 3, 64, 64) input and output (B, 45) logits.
Optimised for CPU inference on i5-1335U (lightning.qubit adjoint).
"""

import math
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer

from shared_config import N_QUBITS, NUM_CLASSES, QUANTUM_BACKEND, DIFF_METHOD


# ── Quantum layer factories ───────────────────────────────────────────────────

def _make_sel_layer(n_qubits: int, n_layers: int) -> TorchLayer:
    """StronglyEntanglingLayers circuit wrapped as a TorchLayer."""
    nq = n_qubits   # Capture in local scope for closure safety

    dev = qml.device(QUANTUM_BACKEND, wires=nq)

    @qml.qnode(dev, interface='torch', diff_method=DIFF_METHOD)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(nq), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(nq))
        return [qml.expval(qml.PauliZ(i)) for i in range(nq)]

    weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers, nq)}
    return TorchLayer(circuit, weight_shapes)


def _make_reupload_layer(n_qubits: int, n_layers: int) -> TorchLayer:
    """Data reuploading circuit: AngleEmbed → ring-CNOT → RY/RZ (repeated)."""
    nq = n_qubits
    nl = n_layers

    dev = qml.device(QUANTUM_BACKEND, wires=nq)

    @qml.qnode(dev, interface='torch', diff_method=DIFF_METHOD)
    def circuit(inputs, ry_weights, rz_weights):
        for layer in range(nl):
            qml.AngleEmbedding(inputs, wires=range(nq), rotation='Y')
            for i in range(nq):
                qml.CNOT(wires=[i, (i + 1) % nq])
            for i in range(nq):
                qml.RY(ry_weights[layer, i], wires=i)
                qml.RZ(rz_weights[layer, i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(nq)]

    weight_shapes = {
        "ry_weights": (nl, nq),
        "rz_weights": (nl, nq),
    }
    return TorchLayer(circuit, weight_shapes)


# ── Q1: QNN4EO (ESA Baseline) ─────────────────────────────────────────────────

class Q1_QNN4EO(nn.Module):
    """
    LeNet-style CNN encoder (3 conv layers + AdaptiveAvgPool) → 4 angles
    → 4-qubit AngleEmbedding + StronglyEntanglingLayers (2 layers)
    → Linear(4, 45)

    Params: ~1.3 K classical + 24 quantum
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2,
                 n_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3), nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                   # → (B, 8)
        )
        self.angle_fc   = nn.Linear(8, n_qubits)
        self.qlayer     = _make_sel_layer(n_qubits, n_layers)
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat   = self.encoder(x)           # (B, 8)
        angles = self.angle_fc(feat)        # (B, 4)
        q_out  = self.qlayer(angles)        # (B, 4)
        return self.classifier(q_out)       # (B, 45)


# ── Q2: AngleSEL (Quadrant PQC + Cross-Attention) ────────────────────────────

class Q2_AngleSEL(nn.Module):
    """
    Split 64×64 image into 4 quadrants (32×32 each).
    Shared CNN per quadrant → angles → shared 4Q PQC.
    Cross-Attention: Q=quantum (4×4), K=V=CNN avg-pool (4×32) → (B, 16).
    Linear(16, 45).

    Params: ~40 K classical + 24 quantum
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2,
                 n_classes: int = NUM_CLASSES, attn_dim: int = 16):
        super().__init__()
        # Shared CNN applied to each 32×32 quadrant
        self.quad_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                # 32→16 spatial
        )
        # After MaxPool: (B, 32, 16, 16) → flatten 8192
        self.angle_fc = nn.Linear(32 * 16 * 16, n_qubits)
        self.qlayer   = _make_sel_layer(n_qubits, n_layers)

        # Cross-attention projections
        self.q_proj = nn.Linear(n_qubits, attn_dim)    # Q: quantum features
        self.k_proj = nn.Linear(32,       attn_dim)    # K: CNN avg-pool features
        self.v_proj = nn.Linear(32,       attn_dim)    # V: same
        self.scale  = attn_dim ** 0.5

        self.classifier = nn.Linear(attn_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into 4 non-overlapping 32×32 quadrants
        quads = [
            x[:, :, :32, :32],
            x[:, :, :32, 32:],
            x[:, :, 32:, :32],
            x[:, :, 32:, 32:],
        ]

        quantum_outs = []
        cnn_avgs     = []
        for q in quads:
            f = self.quad_cnn(q)                    # (B, 32, 16, 16)
            cnn_avgs.append(f.mean(dim=[2, 3]))     # (B, 32) global avg-pool
            angles = self.angle_fc(f.flatten(1))    # (B, 4)
            quantum_outs.append(self.qlayer(angles)) # (B, 4)

        # Stack: 4 quadrants as sequence dimension
        Q = torch.stack(quantum_outs, dim=1)  # (B, 4, 4)
        K = torch.stack(cnn_avgs,    dim=1)  # (B, 4, 32)

        Qp = self.q_proj(Q)         # (B, 4, 16)
        Kp = self.k_proj(K)         # (B, 4, 16)
        Vp = self.v_proj(K)         # (B, 4, 16)

        attn = torch.bmm(Qp, Kp.transpose(1, 2)) / self.scale  # (B, 4, 4)
        attn = torch.softmax(attn, dim=-1)
        out  = torch.bmm(attn, Vp).mean(dim=1)                  # (B, 16)

        return self.classifier(out)  # (B, 45)


# ── Q3: DataReupload ──────────────────────────────────────────────────────────

class Q3_DataReupload(nn.Module):
    """
    CNN (3 conv + BN + AdaptiveAvgPool) → 4 angles (tanh-scaled to [-π, π])
    → 4Q data-reuploading circuit (3 layers: AngleEmbed + ring-CNOT + RY/RZ)
    → Linear(4, 45)

    Params: ~3.5 K classical + 24 quantum
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 3,
                 n_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  8, 3), nn.BatchNorm2d(8),  nn.ReLU(),
            nn.Conv2d(8, 16, 3), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                             # → (B, 16)
        )
        self.angle_fc   = nn.Linear(16, n_qubits)
        self.qlayer     = _make_reupload_layer(n_qubits, n_layers)
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat   = self.encoder(x)                              # (B, 16)
        angles = torch.tanh(self.angle_fc(feat)) * math.pi   # scale to [-π, π]
        q_out  = self.qlayer(angles)                          # (B, 4)
        return self.classifier(q_out)                         # (B, 45)
