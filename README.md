# Protein Function Prediction Using a Hybrid Quantum–Classical Neural Network

**Hybrid Quantum–Classical Modeling Using CNN Embeddings and a Multi-Layer Deep
Variational Quantum Classifier for Accurate Multi-Class Protein Sequence
Classification.**

A classical Convolutional Neural Network (CNN) encodes local motifs in an amino-acid
sequence, a Variational Quantum Circuit (VQC) models non-linear feature interactions,
and a classical head predicts the protein functional class (10-class classification).

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-classical-red)
![PennyLane](https://img.shields.io/badge/PennyLane-quantum-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
<!-- DOI badge: add after creating a Zenodo release, e.g.
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg) -->

## Authors

Anish Patankar, Krunal Nagendra, Ishant Patle, Manas Chintawar.
**Guide:** Dr. Amit Pimpalkar.
Department of Computer Science and Engineering (AI & Cyber Security),
Shri Ramdeobaba College of Engineering & Management, Nagpur, India (2025–2026).

## Abstract

Protein function prediction is central to bioinformatics and drug discovery. Classical
deep networks handle large biological datasets but struggle with the extremely
high-dimensional patterns in protein sequences. This work investigates a hybrid
quantum–classical architecture: a CNN encoder captures local sequence motifs, its
compressed features are embedded into a variational quantum circuit using angle
encoding, and entanglement-based quantum rotations model complex non-linear
relationships before a classical classifier produces the final prediction. Training is
restricted to the ten most frequent protein classes for faster, clearer convergence.
The hybrid model learns discriminative sequence patterns while keeping a compact
parameter footprint, supporting the potential of quantum-enhanced models for
biological sequence analysis.

## Data Availability

The dataset is publicly available on Kaggle and is downloaded programmatically via
`kagglehub` inside the notebook (no manual download required).

- **Dataset:** *Protein Classification Dataset* — derived from the RCSB Protein Data Bank (PDB).
- **URL:** https://www.kaggle.com/datasets/anish1137/protein-classification-dataset
- **`kagglehub` id:** `anish1137/protein-classification-dataset`
- **Files used:** `pdb_data_no_dups.csv` (141,401 rows), `pdb_data_seq.csv` (467,304 rows).

**Preprocessing (in this repo):** the two files are merged on `structureId`
(471,149 rows), null/empty sequences are dropped (471,117 rows), and the data is
filtered to the **Top-10 most frequent classes**, giving **250,775 samples**.

| Idx | Class | Idx | Class |
|----|-------|----|-------|
| 0 | HYDROLASE | 5 | RIBOSOME |
| 1 | HYDROLASE/HYDROLASE INHIBITOR | 6 | TRANSCRIPTION |
| 2 | IMMUNE SYSTEM | 7 | TRANSFERASE |
| 3 | LYASE | 8 | VIRAL PROTEIN |
| 4 | OXIDOREDUCTASE | 9 | VIRUS |

The class distribution is imbalanced (RIBOSOME 60,710 → VIRAL PROTEIN 8,875).

## Repository Structure

```
protein-function-qcnn/
├── README.md
├── LICENSE                                  # MIT
├── requirements.txt
├── notebooks/
│   └── hybrid_qnn_final.ipynb               # full pipeline: download, EDA, train, infer
├── src/
│   └── model.py                             # reusable model + load_model() + predict()
├── models/
│   └── hybrid_qnn_protein_model.pt          # trained weights (state_dict)
├── test_model.py                            # load model + predict on a sequence
└── docs/
    └── mini_project_report.pdf              # dissertation report
```

## Installation

```bash
git clone https://github.com/Anish974/protein-function-qcnn.git
cd protein-function-qcnn
pip install -r requirements.txt
```

Requires Python 3.10+. Training/inference run on CPU (the quantum circuit uses
PennyLane's `default.qubit` simulator); no GPU or real quantum hardware is needed.

## Usage

**Run the full pipeline** (data download, EDA, training, evaluation, inference):
open `notebooks/hybrid_qnn_final.ipynb` and run the cells top to bottom.

**Predict on a sequence with the trained model:**

```bash
python test_model.py "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
```

**Use the model in your own code:**

```python
from src.model import load_model, predict

model = load_model("models/hybrid_qnn_protein_model.pt")
label, confidence, probs = predict(model, "MKTAYIAK...")
print(label, confidence)
```

> Note: the model always outputs one of the 10 trained classes and is calibrated for
> full-length PDB sequences (`max_len = 300`). Very short or non-PDB sequences are
> out-of-distribution and give low-confidence predictions.

## Method

```
sequence indices (batch, 300)
  │
  ▼  CNN Encoder
  Embedding(21 → 64, padding_idx=0)
  4 parallel Conv1d (kernels 3, 5, 7, 11; 32 channels each) + ReLU + AdaptiveMaxPool1d(1)
  → concatenate → 128-dim feature vector
  │
  ▼  Compressor (classical → quantum bridge)
  Linear(128 → 64) → ReLU → Linear(64 → 10) → Tanh → × π
  │
  ▼  Variational Quantum Circuit  (PennyLane default.qubit, 10 qubits)
  angle encoding: RX(feature_i) on each qubit
  per layer: Rot(θ, φ, ω) on each qubit + CNOT ring entanglement (0-1-…-9-0)
  measurement: ⟨PauliZ⟩ on each qubit → 10 outputs
  interface = torch, diff_method = backprop
  │
  ▼  Classical Head
  Linear(10 → 64) → ReLU → Dropout(0.2) → Linear(64 → 10)
  → logits over 10 classes
```

**Encoding:** 20 canonical amino acids `ACDEFGHIKLMNPQRSTVWY` map to indices `1..20`;
index `0` is reserved for padding/unknown. Sequences are truncated/padded to
`max_len = 300`.

**Training configuration**

| Setting | Value |
|--------|-------|
| Epochs | 30 |
| Optimizer | AdamW, learning rate `3e-4` |
| Loss | Cross-Entropy |
| Batch size | 128 |
| Train / validation split | 80 / 20 |
| Quantum layers (shipped checkpoint) | 4 |
| Device | CPU (PennyLane `default.qubit`) |

## Results

| Metric | Train | Validation |
|--------|-------|-----------|
| Accuracy | 87.84% | **86.55%** |
| Loss | 0.3976 | 0.4675 |

Convergence is smooth and monotonic with a small train/validation gap, indicating no
notable overfitting. Loss/accuracy curves, evaluation metrics, and the confusion
matrix are reported in `docs/mini_project_report.pdf` (Chapter 5).

## Reproducibility Notes

- The shipped checkpoint `models/hybrid_qnn_protein_model.pt` was trained with a
  quantum circuit depth of **4 variational layers** (`qlayer.weights` shape `(40, 3)`);
  `src/model.py` and the load cell in the notebook use `q_layers=4` to match it.
- `random_state=42` / `np.random.seed(42)` are set for the split, but the PennyLane
  simulator and PyTorch initialization are not globally seeded, so exact numbers may
  vary slightly across runs and machines.

## Code & Data Availability

The complete source code, trained model, and dissertation report are available in this
repository: **https://github.com/Anish974/protein-function-qcnn**.
The dataset is publicly available at
**https://www.kaggle.com/datasets/anish1137/protein-classification-dataset**.
A citable, versioned archive with a DOI can be minted from a GitHub release via Zenodo
(add the DOI badge above once created).

## Citation

If you use this code, please cite the associated manuscript:

```
Patankar A., Nagendra K., Patle I., Chintawar M., Pimpalkar A.
"Hybrid Quantum–Classical Modeling Using CNN Embeddings and a Multi-Layer Deep
Variational Quantum Classifier for Accurate Multi-Class Protein Sequence
Classification." 2025–2026.
```

## License

Released under the [MIT License](LICENSE).
