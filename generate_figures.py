"""Generate result figures + performance metrics for the hybrid QNN.

Produces (into ./figures/):
  - confusion_matrix.png           counts heatmap on the validation split
  - confusion_matrix_normalized.png  row-normalized (recall) heatmap
  - per_class_metrics.png          precision / recall / F1 bar chart
  - class_distribution.png         class balance of the Top-10 dataset
  - classification_report.txt      full precision/recall/F1/support table
  - metrics.csv                    same, machine-readable

All figures are computed from the shipped q_layers=4 checkpoint: the script
rebuilds the exact validation split used in training and runs the model on it,
so it needs the dataset (local `data/` CSVs, else kagglehub) and the weights in
models/.

Loss/accuracy-vs-epoch curves are intentionally not generated here: the shipped
checkpoint's per-epoch history was not logged, so honest curves require running
the training notebook end-to-end.

Run:  python generate_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")  # headless: save files, no display needed
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def rebuild_val_split():
    """Reproduce the exact top-10 dataframe and validation split from training."""
    import pandas as pd

    # Prefer local CSVs (data/); fall back to kagglehub download.
    if os.path.exists("data/pdb_data_no_dups.csv") and os.path.exists("data/pdb_data_seq.csv"):
        path = "data"
    else:
        import kagglehub
        path = kagglehub.dataset_download("anish1137/protein-classification-dataset")
    df1 = pd.read_csv(os.path.join(path, "pdb_data_no_dups.csv"))
    df2 = pd.read_csv(os.path.join(path, "pdb_data_seq.csv"))

    df = pd.merge(df1[["structureId", "classification"]],
                  df2[["structureId", "sequence"]], on="structureId", how="inner")
    df = df.dropna(subset=["sequence", "classification"])
    df = df[df["sequence"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    top_10 = df["classification"].value_counts().nlargest(10).index
    df = df[df["classification"].isin(top_10)].reset_index(drop=True)

    # same encoding/shuffle as prepare_loaders_from_df (seed=42, 80/20)
    codes = pd.Categorical(df["classification"]).codes
    sequences = df["sequence"].fillna("").tolist()
    labels = np.asarray(codes, dtype=int)
    idxs = np.arange(len(sequences))
    np.random.seed(42)
    np.random.shuffle(idxs)
    split = int(0.8 * len(sequences))
    val_idx = idxs[split:]
    val_seqs = [sequences[i] for i in val_idx]
    val_labels = labels[val_idx]
    return df, val_seqs, val_labels


def plot_class_distribution(df):
    import pandas as pd
    counts = df["classification"].value_counts()
    plt.figure(figsize=(10, 4.5))
    plt.bar(range(len(counts)), counts.values, color="#4c72b0")
    plt.xticks(range(len(counts)), counts.index, rotation=45, ha="right")
    plt.title("Protein Class Distribution (Top-10)")
    plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=150); plt.close()
    print("saved: class_distribution.png")


def predict_all(val_seqs, batch_size=256):
    import torch
    from src.model import load_model, seq_to_indices
    model = load_model()
    preds = []
    with torch.no_grad():
        for i in range(0, len(val_seqs), batch_size):
            chunk = val_seqs[i:i + batch_size]
            X = torch.tensor([seq_to_indices(s, 300) for s in chunk], dtype=torch.long)
            preds.extend(model(X).argmax(dim=1).tolist())
            if (i // batch_size) % 20 == 0:
                print(f"  inferred {min(i + batch_size, len(val_seqs))}/{len(val_seqs)}")
    return np.asarray(preds)


def plot_confusion_and_metrics(y_true, y_pred):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    from src.model import CLASSES

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar_kws={"label": "Count"})
    plt.title("Confusion Matrix (Validation Set)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150); plt.close()

    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar_kws={"label": "Recall"})
    plt.title("Normalized Confusion Matrix (row = true class)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/confusion_matrix_normalized.png", dpi=150); plt.close()
    print("saved: confusion_matrix.png, confusion_matrix_normalized.png")

    report_txt = classification_report(y_true, y_pred, labels=range(len(CLASSES)),
                                       target_names=CLASSES, digits=4)
    with open(f"{FIG_DIR}/classification_report.txt", "w") as f:
        acc = (y_true == y_pred).mean()
        f.write(f"Validation accuracy: {acc:.4f}  (n={len(y_true)})\n\n")
        f.write(report_txt)
    print("saved: classification_report.txt")

    report = classification_report(y_true, y_pred, labels=range(len(CLASSES)),
                                   target_names=CLASSES, output_dict=True)
    prec = [report[c]["precision"] for c in CLASSES]
    rec = [report[c]["recall"] for c in CLASSES]
    f1 = [report[c]["f1-score"] for c in CLASSES]

    with open(f"{FIG_DIR}/metrics.csv", "w") as f:
        f.write("class,precision,recall,f1_score,support\n")
        for c in CLASSES:
            r = report[c]
            f.write(f"{c},{r['precision']:.4f},{r['recall']:.4f},{r['f1-score']:.4f},{int(r['support'])}\n")

    x = np.arange(len(CLASSES)); w = 0.27
    plt.figure(figsize=(12, 5))
    plt.bar(x - w, prec, w, label="Precision", color="#4c72b0")
    plt.bar(x, rec, w, label="Recall", color="#dd8452")
    plt.bar(x + w, f1, w, label="F1-score", color="#55a868")
    plt.xticks(x, CLASSES, rotation=45, ha="right")
    plt.ylim(0, 1.05); plt.ylabel("Score")
    plt.title("Per-Class Performance Metrics (Validation Set)")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/per_class_metrics.png", dpi=150); plt.close()
    print("saved: per_class_metrics.png, metrics.csv")


if __name__ == "__main__":
    try:
        df, val_seqs, val_labels = rebuild_val_split()
        plot_class_distribution(df)
        y_pred = predict_all(val_seqs)
        plot_confusion_and_metrics(val_labels, y_pred)
        print("\nAll figures written to ./figures/")
    except Exception as e:
        print(f"\n[skipped] {type(e).__name__}: {e}")
        print("Need the dataset (local data/*.csv or kagglehub) and models/*.pt.")
