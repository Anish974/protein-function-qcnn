"""Hybrid QNN protein classifier — reusable model definitions + loader.

Extracted from notebooks/hybrid_qnn_final.ipynb so scripts (tests, app) can
import the model instead of re-running the notebook.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# --- Amino acid encoding (must match training) ---
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_LIST)}  # 0 = pad/unknown

# Label index -> class name (from training LabelEncoder, alphabetical)
CLASSES = [
    "HYDROLASE", "HYDROLASE/HYDROLASE INHIBITOR", "IMMUNE SYSTEM", "LYASE",
    "OXIDOREDUCTASE", "RIBOSOME", "TRANSCRIPTION", "TRANSFERASE",
    "VIRAL PROTEIN", "VIRUS",
]


def seq_to_indices(seq: str, max_len: int = 300):
    seq = str(seq).upper().strip()
    idx = [AA_TO_IDX.get(ch, 0) for ch in seq[:max_len]]
    idx += [0] * (max_len - len(idx))
    return idx


class CNNEncoder(nn.Module):
    def __init__(self, num_embeddings=21, emb_dim=64, out_dim=128, kernel_sizes=(3, 5, 7, 11)):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, emb_dim, padding_idx=0)
        out_ch = out_dim // len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, out_ch, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])
        self.output_dim = out_ch * len(kernel_sizes)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        outs = [self.pool(F.relu(conv(emb))).squeeze(-1) for conv in self.convs]
        return torch.cat(outs, dim=1)


def create_quantum_torchlayer(n_qubits=10, n_layers=4, dev_name="default.qubit", shots=None):
    dev = qml.device(dev_name, wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(inputs, weights):
        for i in range(n_qubits):
            qml.RX(inputs[:, i], wires=i)
        k = 0
        for _ in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(weights[k, 0], weights[k, 1], weights[k, 2], wires=i)
                k += 1
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers * n_qubits, 3)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)


class HybridVQNN(nn.Module):
    def __init__(self, encoder, n_qubits=10, q_layers=4, classical_to_quantum_dim=10,
                 num_classes=10, use_compressor=True):
        super().__init__()
        self.encoder = encoder
        with torch.no_grad():
            enc_dim = self.encoder(torch.zeros(1, 512, dtype=torch.long)).shape[1]
        self.use_compressor = use_compressor
        if use_compressor:
            mid = max(enc_dim // 2, classical_to_quantum_dim)
            self.compressor = nn.Sequential(
                nn.Linear(enc_dim, mid), nn.ReLU(),
                nn.Linear(mid, classical_to_quantum_dim), nn.Tanh(),
            )
        elif enc_dim != classical_to_quantum_dim:
            raise ValueError("no compressor: enc_dim must equal classical_to_quantum_dim")
        self.qlayer = create_quantum_torchlayer(n_qubits=n_qubits, n_layers=q_layers)
        self.classical_head = nn.Sequential(
            nn.Linear(n_qubits, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        enc = self.encoder(x)
        q_in = self.compressor(enc) * math.pi if self.use_compressor else enc
        return self.classical_head(self.qlayer(q_in))


def load_model(weights_path="models/hybrid_qnn_protein_model.pt", device="cpu"):
    """Build model with shipped-checkpoint config (q_layers=4) and load weights."""
    model = HybridVQNN(CNNEncoder(), n_qubits=10, q_layers=4,
                       classical_to_quantum_dim=10, num_classes=10, use_compressor=True)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model, sequence: str, device="cpu", max_len=300):
    x = torch.tensor(seq_to_indices(sequence, max_len), dtype=torch.long).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]), probs.tolist()
