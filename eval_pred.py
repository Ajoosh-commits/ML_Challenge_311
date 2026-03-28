import sys
import os
import numpy as np
import pandas as pd
import pred


LABEL_TO_ID = {
    "The Persistence of Memory": 0,
    "The Starry Night": 1,
    "The Water Lily Pond": 2,
}

ID_TO_LABEL = {
    0: "The Persistence of Memory",
    1: "The Starry Night",
    2: "The Water Lily Pond",
}


def _load_truth_ids(df):
    if "Painting" in df.columns:
        return df["Painting"].map(LABEL_TO_ID).to_numpy(dtype=int)
    if "Painting_Target" in df.columns:
        return pd.to_numeric(df["Painting_Target"], errors="coerce").to_numpy(dtype=int)
    raise RuntimeError("CSV must contain either 'Painting' or 'Painting_Target' ground-truth column.")


def _confusion_matrix(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def _stratified_split_indices(y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    train_parts = []
    val_parts = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = idx.copy()
        rng.shuffle(idx)

        n_val = max(1, int(round(len(idx) * val_ratio)))
        n_val = min(n_val, len(idx) - 1) if len(idx) > 1 else 1

        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _evaluate_df(df):
    y_true = _load_truth_ids(df)
    model = pred._load_model()
    X_df = pred._build_feature_matrix(df, model)
    y_pred = pred._predict_class_ids(X_df.to_numpy(dtype=np.float32), model)

    valid = (y_true >= 0) & (y_pred >= 0)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    acc = float(np.mean(y_true == y_pred))
    cm = _confusion_matrix(y_true, y_pred, n_classes=3)
    return y_true, y_pred, acc, cm


def main():
    csv_path = "ml_challenge_dataset.csv"
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]

    full_df = pd.read_csv(csv_path)
    y_all = _load_truth_ids(full_df)

    holdout_info = "holdout_20_info.csv"
    if os.path.exists(holdout_info):
        info_df = pd.read_csv(holdout_info)
        if "index" not in info_df.columns:
            raise RuntimeError("holdout_20_info.csv exists but has no 'index' column.")
        val_idx = info_df["index"].to_numpy(dtype=int)
        all_idx = np.arange(len(full_df))
        val_set = set(val_idx.tolist())
        train_idx = np.array([i for i in all_idx if i not in val_set], dtype=int)
    else:
        train_idx, val_idx = _stratified_split_indices(y_all, val_ratio=0.2, seed=42)

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    _, _, train_acc, train_cm = _evaluate_df(train_df)
    _, _, val_acc, val_cm = _evaluate_df(val_df)

    print("Training Set (80%)")
    print(f"Rows: {len(train_df)}")
    print(f"Accuracy: {train_acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(train_cm)

    print("\nValidation Set (20%)")
    print(f"Rows: {len(val_df)}")
    print(f"Accuracy: {val_acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(val_cm)


if __name__ == "__main__":
    main()
