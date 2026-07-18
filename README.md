# Protein Function Prediction Using Quantum CNN (Hybrid QNN)

Full project overview — generated for review.

## 1. What this project is

A **Hybrid Quantum Neural Network (HQNN)** that classifies protein amino-acid
sequences into their protein family. A classical CNN extracts local sequence
motifs, a **variational quantum circuit (VQC)** models non-linear feature
interactions, and a classical head produces the final 10-class prediction.

- **Type:** B.Tech (VII Sem) mini-project / dissertation
- **Title (report):** *Protein Function Prediction Using Quantum CNN*
- **Institute:** Shri Ramdeobaba College of Engineering & Management, Nagpur — CSE (AI & Cyber Security)
- **Guide:** Dr. Amit Pimpalkar
- **Authors:** Ishant Patle (42), Manas Chintawar (46), Krunal Nagendra (45), Anish Patankar (30)
- **Academic year:** 2025–2026

## 2. Files in the repo

| File | What it is |
|------|-----------|
| `Hybrid QNN Final.ipynb` | Full pipeline: data download, EDA, model, training, inference (13 cells) |
| `Mini Project Report .pdf` | 60-page dissertation report |
| `hybrid_qnn_protein_model.pt.zip` | Trained model weights (`state_dict`) |

## 3. Dataset

- **Source:** Kaggle `shahir/protein-data-set` (RCSB PDB structures), auto-downloaded via `kagglehub`.
- **Merge:** `pdb_data_no_dups.csv` (141,401) + `pdb_data_seq.csv` (467,304) on `structureId` → 471,149 rows.
- **Clean:** drop null/empty sequences → 471,117 rows.
- **Filter:** keep **Top 10 most frequent classes** → **250,775 samples**.

**10 target classes (label index):**

| Idx | Class | Idx | Class |
|----|-------|----|-------|
| 0 | HYDROLASE | 5 | RIBOSOME |
| 1 | HYDROLASE/HYDROLASE INHIBITOR | 6 | TRANSCRIPTION |
| 2 | IMMUNE SYSTEM | 7 | TRANSFERASE |
| 3 | LYASE | 8 | VIRAL PROTEIN |
| 4 | OXIDOREDUCTASE | 9 | VIRUS |

Class distribution is imbalanced (RIBOSOME 60,710 → VIRAL PROTEIN 8,875).

## 4. Preprocessing / encoding

- 20 canonical amino acids `ACDEFGHIKLMNPQRSTVWY` → index `1..20`; `0` reserved for pad/unknown.
- Each sequence truncated/padded to `max_len = 300`.
- `seq_to_indices()` (cell 7) is the correct encoder used in training.

## 5. Model architecture

```
sequence indices (batch, 300)
  │
  ▼  CNNEncoder
  Embedding(21 → 64, padding_idx=0)
  4 × parallel Conv1d (kernel = 3,5,7,11, 32 ch each) + ReLU + AdaptiveMaxPool1d(1)
  → concat → 128-dim feature vector
  │
  ▼  Compressor MLP (classical → quantum bridge)
  Linear(128→64) → ReLU → Linear(64→10) → Tanh → × π
  │
  ▼  Variational Quantum Circuit  (PennyLane, default.qubit simulator)
  10 qubits, 5 layers
  angle encoding: RX(input_i) on each qubit
  per layer: Rot(θ,φ,ω) on each qubit + CNOT ring entanglement (0-1-…-9-0)
  measure: ⟨PauliZ⟩ on each qubit → 10 outputs
  diff_method = 'backprop', interface = 'torch'
  │
  ▼  Classical head
  Linear(10→64) → ReLU → Dropout(0.2) → Linear(64→10)
  → logits (10 classes)
```

- Quantum weight shape: `(n_layers × n_qubits, 3)` = `(50, 3)`.
- Encoder output dim auto-detected in `HybridVQNN.__init__`.

## 6. Training configuration

| Setting | Value |
|--------|-------|
| Epochs | 30 |
| Optimizer | AdamW, lr `3e-4` |
| Loss | CrossEntropyLoss |
| Batch size | 128 |
| Train/val split | 80 / 20 |
| Device | CPU (quantum sim, no GPU) |

## 7. Results

- **Final Train Acc: 87.84%** (loss 0.3976)
- **Final Val Acc: 86.55%** (loss 0.4675)
- Smooth monotonic convergence, small train/val gap → no notable overfitting.
- Inference sanity check: sample sequence → predicted `TRANSFERASE`.

Report also covers loss/accuracy curves, evaluation metrics, and confusion matrix (Ch. 5).

## 8. Tech stack

`torch`, `pennylane`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `kagglehub`.

## 9. Known issues / notes (for review)

1. **`q_layers` mismatch between train cell & shipped checkpoint.** The training cell
   (cell 8) is written with `q_layers=5` (would give `qlayer.weights` shape `(50,3)`),
   but the **committed checkpoint** (`models/hybrid_qnn_protein_model.pt`) has
   `qlayer.weights` = **`(40,3)`** → it was actually trained with `q_layers=4`. So the
   load cell (cell 9, `q_layers=4`) is correct; the shipped weights come from a
   `q_layers=4` run, **not** the `q_layers=5` training log printed in cell 8. Keep
   `q_layers=4` when loading. To reproduce the shipped model, set the training cell to
   `q_layers=4` as well so the log matches the checkpoint.
2. **Dead / inconsistent encoder.** Cell 4 uses `ord(c) % 20` (lossy, collides amino
   acids) and `LabelEncoder`; the actual training path (cell 7 `seq_to_indices` +
   `ProteinSequenceDataset`) uses the correct `AA_TO_IDX` mapping. Cell 4 is unused
   leftover — safe to delete.
3. **CPU-only quantum sim** → training is slow; not a bug, just a scaling limit.
4. Class imbalance not handled (no weighting/resampling) — accuracy is the only
   reported metric in the notebook; per-class F1 would be more honest given imbalance.
