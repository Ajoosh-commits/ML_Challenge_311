import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pred


LABEL_TO_ID = {
    "The Persistence of Memory": 0,
    "The Starry Night": 1,
    "The Water Lily Pond": 2,
}

# Final tuned RF configuration used for export.
RF_PARAMS = {
    "n_estimators": 400,
    "criterion": "entropy",
    "max_depth": 24,
    "max_features": 6,
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "bootstrap": True,
    "max_samples": 0.9,
    "class_weight": "balanced_subsample",
    "random_state": 7,
    "n_jobs": 1,
}

# Optional decision-time class bias used by pred.py
CLASS_BIAS = np.array([-0.03, -0.03, 0.02], dtype=np.float32)


def _load_feature_schema(schema_npz_path):
    # pred._build_feature_matrix expects a loaded schema model.
    pred.MODEL_FILE = schema_npz_path
    pred._MODEL_CACHE = None
    return pred._load_model()


def _build_train_matrix(raw_df, schema_model):
    X_df = pred._build_feature_matrix(raw_df, schema_model)
    X = X_df.to_numpy(dtype=np.float32)
    y = raw_df["Painting"].map(LABEL_TO_ID).to_numpy(dtype=int)
    return X_df, X, y


def _export_npz(output_path, feature_cols, numeric_fill, clf, class_bias):
    n_trees = len(clf.estimators_)
    max_nodes = max(est.tree_.node_count for est in clf.estimators_)
    n_classes = len(clf.classes_)

    children_left = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    children_right = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    feature = np.full((n_trees, max_nodes), -2, dtype=np.int16)
    threshold = np.zeros((n_trees, max_nodes), dtype=np.float32)
    leaf_proba = np.zeros((n_trees, max_nodes, n_classes), dtype=np.float32)

    for t, est in enumerate(clf.estimators_):
        tree = est.tree_
        n = tree.node_count

        children_left[t, :n] = tree.children_left.astype(np.int32)
        children_right[t, :n] = tree.children_right.astype(np.int32)
        feature[t, :n] = tree.feature.astype(np.int16)
        threshold[t, :n] = tree.threshold.astype(np.float32)

        vals = tree.value[:, 0, :].astype(np.float32)
        sums = vals.sum(axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0
        leaf_proba[t, :n, :] = vals / sums

    np.savez_compressed(
        output_path,
        feature_cols=np.array(feature_cols, dtype="<U128"),
        numeric_fill=numeric_fill.astype(np.float32),
        classes=clf.classes_.astype(np.int8),
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        leaf_proba=leaf_proba,
        class_bias=class_bias.astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest and export rf_model_params.npz")
    parser.add_argument("--raw-csv", default="ml_challenge_dataset.csv")
    parser.add_argument("--schema-npz", default="rf_model_params.npz")
    parser.add_argument("--output-npz", default="rf_model_params.npz")
    parser.add_argument(
        "--mode",
        choices=["full", "holdout80"],
        default="full",
        help="full: train on 100%% data; holdout80: train on 80%% and keep 20%% validation split.",
    )
    parser.add_argument("--holdout-info", default="holdout_20_info.csv")
    parser.add_argument("--holdout-raw", default="holdout_20_raw.csv")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.raw_csv)
    if "Painting" not in raw_df.columns:
        raise RuntimeError("Input CSV must include a 'Painting' column for training.")

    schema_model = _load_feature_schema(args.schema_npz)
    X_df, X_all, y_all = _build_train_matrix(raw_df, schema_model)

    all_idx = np.arange(len(raw_df))
    if args.mode == "holdout80":
        train_idx, val_idx = train_test_split(
            all_idx,
            test_size=0.2,
            random_state=args.split_seed,
            stratify=y_all,
        )
        raw_df.iloc[val_idx].reset_index(drop=True).to_csv(args.holdout_raw, index=False)
        pd.DataFrame({"index": val_idx, "y_true": y_all[val_idx]}).to_csv(args.holdout_info, index=False)
    else:
        train_idx = all_idx
        val_idx = np.array([], dtype=int)

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    train_X_df = X_df.iloc[train_idx].reset_index(drop=True)

    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)

    numeric_fill = np.array(
        [
            float(train_X_df[col].median()) if col in train_X_df.columns else 0.0
            for col in pred.NUMERIC_FEATURES
        ],
        dtype=np.float32,
    )

    _export_npz(
        output_path=args.output_npz,
        feature_cols=schema_model["feature_cols"],
        numeric_fill=numeric_fill,
        clf=clf,
        class_bias=CLASS_BIAS,
    )

    print("Saved:", args.output_npz)
    print("Mode:", args.mode)
    print("Train rows:", len(train_idx))
    print("Validation rows:", len(val_idx))
    print("RF params:", RF_PARAMS)
    print("Class bias:", CLASS_BIAS.tolist())
    if args.mode == "holdout80":
        print("Saved holdout raw:", args.holdout_raw)
        print("Saved holdout info:", args.holdout_info)


if __name__ == "__main__":
    main()
