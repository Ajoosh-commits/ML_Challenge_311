import sys
import numpy as np
import pandas as pd


LABEL_MAP = {
    0: "The Persistence of Memory",
    1: "The Starry Night",
    2: "The Water Lily Pond",
}

INTENSITY_COL = "On a scale of 1–10, how intense is the emotion conveyed by the artwork?"
SOMBRE_COL = "This art piece makes me feel sombre."
CONTENT_COL = "This art piece makes me feel content."
CALM_COL = "This art piece makes me feel calm."
UNEASY_COL = "This art piece makes me feel uneasy."
COLOURS_COL = "How many prominent colours do you notice in this painting?"
OBJECTS_COL = "How many objects caught your eye in the painting?"
PRICE_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"

ROOM_COL = "If you could purchase this painting, which room would you put that painting in?"
WITH_COL = "If you could view this art in person, who would you want to view it with?"
SEASON_COL = "What season does this art piece remind you of?"

FEELINGS_TEXT_COL = "Describe how this painting makes you feel."
FOOD_TEXT_COL = "If this painting was a food, what would be?"
SOUND_TEXT_COL = "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."

NUMERIC_FEATURES = [
    INTENSITY_COL,
    SOMBRE_COL,
    CONTENT_COL,
    CALM_COL,
    UNEASY_COL,
    COLOURS_COL,
    OBJECTS_COL,
    PRICE_COL,
]

MODEL_FILE = "rf_model_params.npz"
_MODEL_CACHE = None


def _normalize_name(name):
    s = str(name).strip().lower()
    out = []
    for ch in s:
        if ("a" <= ch <= "z") or ch.isdigit():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def _normalize_token(text):
    s = str(text).strip().lower()
    out = []
    for ch in s:
        if ("a" <= ch <= "z") or ch.isdigit():
            out.append(ch)
    return "".join(out)


def _extract_first_number(value):
    if pd.isna(value):
        return np.nan

    s = str(value)
    token = []
    started = False
    used_dot = False

    for ch in s:
        if ch.isdigit():
            token.append(ch)
            started = True
        elif ch == "." and started and not used_dot:
            token.append(ch)
            used_dot = True
        elif started:
            break

    if not token:
        return np.nan

    try:
        return float("".join(token))
    except Exception:
        return np.nan


def _clean_price(value):
    if pd.isna(value):
        return np.nan

    s = str(value).replace(",", "")
    cleaned = []
    for ch in s:
        if ch.isdigit() or ch == ".":
            cleaned.append(ch)
        else:
            cleaned.append(" ")

    parts = "".join(cleaned).split()
    if not parts:
        return np.nan

    number = parts[0]
    if "." not in number and number.isdigit():
        i = 1
        while i < len(parts) and parts[i].isdigit() and len(parts[i]) == 3:
            number += parts[i]
            i += 1

    try:
        return float(number)
    except Exception:
        return np.nan


def _split_multi(value):
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _tokenize_text(value, phrase_replacements=None):
    if pd.isna(value):
        return []

    text = str(value).lower()
    if phrase_replacements is not None:
        for src, dst in phrase_replacements.items():
            text = text.replace(src, dst)

    chars = []
    for ch in text:
        if ("a" <= ch <= "z") or ch == "_":
            chars.append(ch)
        else:
            chars.append(" ")

    tokens = "".join(chars).split()
    return [tok for tok in tokens if len(tok) > 2]


def _normalized_column_lookup(df):
    lookup = {}
    for col in df.columns:
        lookup[_normalize_name(col)] = col
    return lookup


def _lookup_col(raw_df, target_name):
    lookup = _normalized_column_lookup(raw_df)
    return lookup.get(_normalize_name(target_name))


def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    try:
        packed = np.load(MODEL_FILE, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load {MODEL_FILE}. Include it with pred.py.") from exc

    feature_cols = packed["feature_cols"].astype(str).tolist()
    numeric_fill_values = packed["numeric_fill"].astype(float)

    numeric_fill = {}
    for i, col in enumerate(NUMERIC_FEATURES):
        if i < len(numeric_fill_values):
            numeric_fill[col] = float(numeric_fill_values[i])
        else:
            numeric_fill[col] = 0.0

    model = {
        "feature_cols": feature_cols,
        "numeric_fill": numeric_fill,
        "classes": packed["classes"].astype(int),
        "children_left": packed["children_left"].astype(np.int32),
        "children_right": packed["children_right"].astype(np.int32),
        "feature": packed["feature"].astype(np.int16),
        "threshold": packed["threshold"].astype(np.float32),
    }

    # New format: per-leaf class probabilities + optional class biases.
    if "leaf_proba" in packed:
        model["leaf_proba"] = packed["leaf_proba"].astype(np.float32)
    elif "leaf_class" in packed:
        # Backward compatibility with older exported models.
        leaf_class = packed["leaf_class"].astype(np.int16)
        n_trees, max_nodes = leaf_class.shape
        n_classes = len(model["classes"])
        leaf_proba = np.zeros((n_trees, max_nodes, n_classes), dtype=np.float32)
        for t in range(n_trees):
            for node in range(max_nodes):
                cls_idx = int(leaf_class[t, node])
                if 0 <= cls_idx < n_classes:
                    leaf_proba[t, node, cls_idx] = 1.0
        model["leaf_proba"] = leaf_proba
    else:
        raise RuntimeError("Model file missing both leaf_proba and leaf_class.")

    if "class_bias" in packed:
        model["class_bias"] = packed["class_bias"].astype(np.float32)
    else:
        model["class_bias"] = np.zeros(len(model["classes"]), dtype=np.float32)

    model["room_features"] = [c for c in feature_cols if c.startswith("room_")]
    model["with_features"] = [c for c in feature_cols if c.startswith("with_")]
    model["season_features"] = [c for c in feature_cols if c.startswith("season_")]
    model["feelings_features"] = [c for c in feature_cols if c.startswith("feelings_")]
    model["food_features"] = [c for c in feature_cols if c.startswith("food_")]
    model["sound_features"] = [c for c in feature_cols if c.startswith("sound_")]

    _MODEL_CACHE = model
    return _MODEL_CACHE


def _build_feature_matrix(raw_df, model):
    feature_cols = model["feature_cols"]
    out = pd.DataFrame(0.0, index=raw_df.index, columns=feature_cols)

    if set(feature_cols).issubset(set(raw_df.columns)):
        aligned = raw_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        return aligned.fillna(0.0)

    numeric_extractors = {
        INTENSITY_COL: _extract_first_number,
        SOMBRE_COL: _extract_first_number,
        CONTENT_COL: _extract_first_number,
        CALM_COL: _extract_first_number,
        UNEASY_COL: _extract_first_number,
        COLOURS_COL: _extract_first_number,
        OBJECTS_COL: _extract_first_number,
        PRICE_COL: _clean_price,
    }

    for feature_name, parser in numeric_extractors.items():
        if feature_name not in out.columns:
            continue

        source_col = _lookup_col(raw_df, feature_name)
        if source_col is None:
            values = pd.Series(np.nan, index=raw_df.index)
        else:
            values = raw_df[source_col].apply(parser)

        fill_value = model["numeric_fill"].get(feature_name, 0.0)
        out[feature_name] = values.fillna(fill_value)

    def fill_multi(raw_col_name, feature_list, prefix):
        if not feature_list:
            return

        source_col = _lookup_col(raw_df, raw_col_name)
        if source_col is None:
            return

        normalized_targets = {}
        for feat in feature_list:
            suffix = feat[len(prefix):]
            normalized_targets[_normalize_token(suffix)] = feat

        for idx, value in raw_df[source_col].items():
            categories = _split_multi(value)
            norms = {_normalize_token(cat) for cat in categories}
            for norm_cat in norms:
                feat = normalized_targets.get(norm_cat)
                if feat is not None:
                    out.at[idx, feat] = 1.0

    fill_multi(ROOM_COL, model["room_features"], "room_")
    fill_multi(WITH_COL, model["with_features"], "with_")
    fill_multi(SEASON_COL, model["season_features"], "season_")

    def fill_text_counts(raw_col_name, feature_list, prefix, phrase_replacements=None):
        if not feature_list:
            return

        source_col = _lookup_col(raw_df, raw_col_name)
        if source_col is None:
            return

        vocab = [feat[len(prefix):] for feat in feature_list]
        for idx, value in raw_df[source_col].items():
            tokens = _tokenize_text(value, phrase_replacements=phrase_replacements)
            counts = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for word, feat in zip(vocab, feature_list):
                out.at[idx, feat] = float(counts.get(word, 0))

    fill_text_counts(FEELINGS_TEXT_COL, model["feelings_features"], "feelings_")
    fill_text_counts(
        FOOD_TEXT_COL,
        model["food_features"],
        "food_",
        phrase_replacements={"ice cream": "ice_cream", "mac and cheese": "mac_and_cheese"},
    )
    fill_text_counts(SOUND_TEXT_COL, model["sound_features"], "sound_")

    return out


def _predict_class_ids(X, model):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)

    children_left = model["children_left"]
    children_right = model["children_right"]
    feature = model["feature"]
    threshold = model["threshold"]
    leaf_proba = model["leaf_proba"]
    classes = model["classes"]
    class_bias = model["class_bias"]

    n_trees = children_left.shape[0]
    n_classes = len(classes)

    preds = np.zeros(X.shape[0], dtype=classes.dtype)

    for i in range(X.shape[0]):
        prob_sum = np.zeros(n_classes, dtype=np.float32)
        row = X[i]

        for t in range(n_trees):
            node = 0
            while True:
                left = children_left[t, node]
                if left == -1:
                    prob_sum += leaf_proba[t, node]
                    break

                f = feature[t, node]
                if row[f] <= threshold[t, node]:
                    node = left
                else:
                    node = children_right[t, node]

        pred_scores = (prob_sum / float(n_trees)) + class_bias
        preds[i] = classes[int(np.argmax(pred_scores))]

    return preds


def predict_all(filename):
    model = _load_model()
    test_df = pd.read_csv(filename)
    X_test_df = _build_feature_matrix(test_df, model)
    pred_ids = _predict_class_ids(X_test_df.to_numpy(dtype=np.float32), model)

    preds = []
    for cls in pred_ids:
        preds.append(LABEL_MAP.get(int(cls), "The Starry Night"))
    return preds


def _main():
    if len(sys.argv) < 2:
        print("Usage: py pred.py <test_csv>")
        return

    predictions = predict_all(sys.argv[1])
    for pred in predictions:
        print(pred)


if __name__ == "__main__":
    _main()
