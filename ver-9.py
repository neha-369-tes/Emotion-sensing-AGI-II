import os
import re
import json
import pickle
import numpy as np
import pandas as pd

from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import f_classif

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ---------------------------
# Small utilities
# ---------------------------
def log_to_file(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def sanitize_colname(c):
    # Remove leading hash/whitespace; collapse spaces
    c = str(c).strip()
    if c.startswith("#"):
        c = c[1:].strip()
    c = re.sub(r"\s+", "_", c)
    return c

def normalize_label(lbl):
    if lbl is None:
        return "unknown"
    s = str(lbl).strip().lower()
    mapping = {
        "aroused": "arousal",
        "arousal": "arousal",
        "excited": "excitement",
        "excitement": "excitement",
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    }
    # map to known; if not in mapping, keep normalized token
    return mapping.get(s, s)

def feature_family(name):
    # family is the first token before first underscore: e.g., mean_*, std_*, fft_*
    m = re.match(r"^([a-zA-Z0-9]+)_", name)
    return m.group(1).lower() if m else "other"

def feature_suffix_group(name):
    # group is the last underscore token: _a, _b, _a2, _b2
    m = re.search(r"_([A-Za-z0-9]+)$", name)
    return m.group(1).lower() if m else "unk"


# ---------------------------
# Data loading tailored to your CSV
# ---------------------------
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # find the label column (case-insensitive)
    label_col = None
    for c in df.columns:
        if str(c).strip().lower() == "label":
            label_col = c
            break
    assert label_col is not None, "CSV must contain a 'label' column"

    # sanitize all column names (handle '# mean_0_a', spaces, etc.)
    new_cols = []
    for c in df.columns:
        if c == label_col:
            new_cols.append("label")
        else:
            new_cols.append(sanitize_colname(c))
    df.columns = new_cols

    # labels
    y_raw = df["label"].apply(normalize_label).values

    # features = all numeric non-label columns
    feat_cols = [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    assert len(feat_cols) > 0, "No numeric feature columns found."

    X = df[feat_cols].astype(np.float32).values
    return X, y_raw, feat_cols


# ---------------------------
# Model builder (MLP for tabular)
# ---------------------------
def build_mlp(input_dim, n_classes, lr=1e-3):
    model = Sequential([
        # Block 1
        Dense(512, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.35),

        # Block 2
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        # Block 3
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.15),

        Dense(n_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------
# Feature importance (ANOVA F-score)
# ---------------------------
def compute_feature_importance_anova(X, y_enc, feat_cols):
    # f_classif expects y as integer classes
    F, p = f_classif(X, y_enc)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    imp = pd.DataFrame({
        "feature": feat_cols,
        "F_score": F
    }).sort_values("F_score", ascending=False)
    return imp


def aggregate_importance(imp_df):
    # Add family and group columns
    imp_df = imp_df.copy()
    imp_df["family"] = imp_df["feature"].map(feature_family)
    imp_df["group"] = imp_df["feature"].map(feature_suffix_group)

    group_imp = imp_df.groupby("group")["F_score"].mean().reset_index().sort_values("F_score", ascending=False)
    fam_imp = imp_df.groupby("family")["F_score"].mean().reset_index().sort_values("F_score", ascending=False)
    return imp_df, group_imp, fam_imp


# ---------------------------
# Main training pipeline
# ---------------------------
def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")

    csv_path = "emotions.csv"
    assert os.path.exists(csv_path), f"{csv_path} not found."

    X, y_raw, feat_cols = load_dataset(csv_path)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    # Train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save encoders
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Build + train
    n_classes = len(le.classes_)
    model = build_mlp(input_dim=X_train.shape[1], n_classes=n_classes, lr=1e-3)

    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    # Save model + architecture
    model.save("emotion_model.h5")
    with open("model_summary.json", "w", encoding="utf-8") as f:
        f.write(model.to_json())

    # Evaluate
    y_pred = model.predict(X_test, verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1)

    report = classification_report(y_test, y_pred_cls, target_names=list(le.classes_))
    cm = confusion_matrix(y_test, y_pred_cls)

    log_to_file("classification_report.txt", report)
    log_to_file("classification_report.txt", "Confusion Matrix:\n" + str(cm))
    print(report)

    # Explainability (fast on CPU): ANOVA F-score
    imp_df = compute_feature_importance_anova(
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        feat_cols
    )
    # add family/group & aggregate
    imp_df, group_imp, fam_imp = aggregate_importance(imp_df)

    # Save top-k features and aggregates
    imp_df.sort_values("F_score", ascending=False).head(50).to_csv("feature_importance_top50.csv", index=False)
    group_imp.to_csv("group_importance.csv", index=False)
    fam_imp.to_csv("family_importance.csv", index=False)


if __name__ == "__main__":
    main()
