"""
Run all models, collect all results, and generate a professional PDF report.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from io import StringIO

# ─── Setup ───────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "pubmed-rct",
                        "PubMed_20k_RCT_numbers_replaced_with_at_sign")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

# ─── Data Loading ────────────────────────────────────────────────────────────
def load_pubmed_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    samples = []
    abstract_lines = ""
    for line in lines:
        if line.startswith("###"):
            abstract_lines = ""
        elif line.isspace():
            split = abstract_lines.splitlines()
            for i, al in enumerate(split):
                parts = al.split("\t")
                if len(parts) == 2:
                    samples.append({
                        "target": parts[0],
                        "text": parts[1].lower(),
                        "line_number": i,
                        "total_lines": len(split) - 1
                    })
        else:
            abstract_lines += line
    return samples

print("=" * 60)
print("LOADING DATA...")
print("=" * 60)
train_df = pd.DataFrame(load_pubmed_data(os.path.join(DATA_DIR, "train.txt")))
val_df = pd.DataFrame(load_pubmed_data(os.path.join(DATA_DIR, "dev.txt")))
test_df = pd.DataFrame(load_pubmed_data(os.path.join(DATA_DIR, "test.txt")))

print(f"Train: {len(train_df):,} samples")
print(f"Val:   {len(val_df):,} samples")
print(f"Test:  {len(test_df):,} samples")

# ─── EDA stats ───────────────────────────────────────────────────────────────
class_counts = train_df["target"].value_counts()
class_pcts = (class_counts / len(train_df) * 100).round(2)
imbalance_ratio = class_counts.max() / class_counts.min()

sent_lengths = [len(s.split()) for s in train_df["text"]]
eda_stats = {
    "mean": round(np.mean(sent_lengths), 1),
    "median": round(float(np.median(sent_lengths)), 0),
    "std": round(np.std(sent_lengths), 1),
    "min": int(np.min(sent_lengths)),
    "max": int(np.max(sent_lengths)),
    "p95": int(np.percentile(sent_lengths, 95)),
    "p99": int(np.percentile(sent_lengths, 99)),
}

print(f"\nClass distribution:")
for cls in CLASS_NAMES:
    print(f"  {cls:15s}: {class_counts.get(cls, 0):>6,}  ({class_pcts.get(cls, 0):.1f}%)")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
print(f"\nSentence length: mean={eda_stats['mean']}, median={eda_stats['median']}, "
      f"95th%={eda_stats['p95']}, max={eda_stats['max']}")

# ─── MODEL 1: Baseline (TF-IDF + Naive Bayes) ───────────────────────────────
print("\n" + "=" * 60)
print("MODEL 1: TF-IDF + Multinomial Naive Bayes")
print("=" * 60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, y_train = train_df["text"].to_numpy(), train_df["target"].to_numpy()
X_val, y_val = val_df["text"].to_numpy(), val_df["target"].to_numpy()
X_test, y_test = test_df["text"].to_numpy(), test_df["target"].to_numpy()

baseline_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                               min_df=2, max_df=0.95, sublinear_tf=True)),
    ("clf", MultinomialNB(alpha=1.0))
])
baseline_model.fit(X_train, y_train)

vocab_size_baseline = len(baseline_model.named_steps['tfidf'].vocabulary_)
val_preds_bl = baseline_model.predict(X_val)
test_preds_bl = baseline_model.predict(X_test)
val_acc_bl = accuracy_score(y_val, val_preds_bl)
test_acc_bl = accuracy_score(y_test, test_preds_bl)

baseline_report = classification_report(y_test, test_preds_bl, target_names=CLASS_NAMES, digits=4, output_dict=True)
baseline_report_str = classification_report(y_test, test_preds_bl, target_names=CLASS_NAMES, digits=4)
baseline_cm = confusion_matrix(y_test, test_preds_bl, labels=CLASS_NAMES)

print(f"Vocabulary size: {vocab_size_baseline:,}")
print(f"Validation accuracy: {val_acc_bl*100:.2f}%")
print(f"Test accuracy:       {test_acc_bl*100:.2f}%")
print(baseline_report_str)

# Error analysis
errors_bl = pd.DataFrame({"text": X_test, "true": y_test, "pred": test_preds_bl})
errors_bl = errors_bl[errors_bl["true"] != errors_bl["pred"]]
confusion_pairs = errors_bl.groupby(["true", "pred"]).size().sort_values(ascending=False)
top_confusions_bl = confusion_pairs.head(5)
print(f"Total errors: {len(errors_bl)} / {len(X_test)} ({len(errors_bl)/len(X_test)*100:.2f}%)")
print("Top confusions:")
for (t, p), cnt in top_confusions_bl.items():
    print(f"  {t} -> {p}: {cnt}")

# ─── MODEL 2: GloVe Embeddings ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2: GloVe Embeddings + Dense")
print("=" * 60)

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)
tf.random.set_seed(42)

MAX_LENGTH = 55

train_sentences = train_df["text"].to_numpy()
val_sentences = val_df["text"].to_numpy()
test_sentences = test_df["text"].to_numpy()

encoder = OneHotEncoder(sparse_output=False)
train_labels = encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels = encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels = encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

max_tokens = 68000
text_vectorizer = layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=MAX_LENGTH)
text_vectorizer.adapt(train_sentences)
vocab = text_vectorizer.get_vocabulary()
print(f"Vocabulary size: {len(vocab)}")

# Load GloVe
embedding_dim = 100
glove_path = os.path.join(PROJECT_DIR, "data", "glove.6B.100d.txt")

embeddings_index = {}
if os.path.exists(glove_path):
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    print(f"Loaded {len(embeddings_index)} GloVe vectors")
else:
    print(f"GloVe not found at {glove_path} - using random embeddings")

embedding_matrix = np.zeros((len(vocab), embedding_dim))
found = 0
for i, word in enumerate(vocab):
    vec = embeddings_index.get(word)
    if vec is not None:
        embedding_matrix[i] = vec
        found += 1
print(f"GloVe coverage: {found}/{len(vocab)} ({found/len(vocab)*100:.1f}%)")

# Build model
inputs = layers.Input(shape=[], dtype="string")
x = text_vectorizer(inputs)
x = layers.Embedding(input_dim=len(vocab), output_dim=embedding_dim,
                     embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                     trainable=False)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

glove_model = Model(inputs, outputs, name="glove_model")
glove_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Training GloVe model...")
history_glove = glove_model.fit(
    train_sentences, train_labels,
    epochs=10, batch_size=32,
    validation_data=(val_sentences, val_labels),
    verbose=1
)

test_loss_glove, test_acc_glove = glove_model.evaluate(test_sentences, test_labels, verbose=0)
preds_glove = np.argmax(glove_model.predict(test_sentences, verbose=0), axis=1)
true_glove = np.argmax(test_labels, axis=1)

# Map indices to class names using encoder categories
encoder_classes = list(encoder.categories_[0])
true_names_glove = [encoder_classes[i] for i in true_glove]
pred_names_glove = [encoder_classes[i] for i in preds_glove]

glove_report_str = classification_report(true_names_glove, pred_names_glove, target_names=CLASS_NAMES, digits=4, zero_division=0)
glove_report = classification_report(true_names_glove, pred_names_glove, target_names=CLASS_NAMES, digits=4, output_dict=True, zero_division=0)
glove_cm = confusion_matrix(true_names_glove, pred_names_glove, labels=CLASS_NAMES)

# Extract training history
glove_history = {
    "train_acc": [round(x, 4) for x in history_glove.history["accuracy"]],
    "val_acc": [round(x, 4) for x in history_glove.history["val_accuracy"]],
    "train_loss": [round(x, 4) for x in history_glove.history["loss"]],
    "val_loss": [round(x, 4) for x in history_glove.history["val_loss"]],
}

print(f"\nTest loss:     {test_loss_glove:.4f}")
print(f"Test accuracy: {test_acc_glove*100:.2f}%")
print(glove_report_str)

# ─── MODEL 3: Bidirectional LSTM ────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 3: Bidirectional LSTM")
print("=" * 60)

from tensorflow.keras import callbacks

np.random.seed(42)
tf.random.set_seed(42)

text_vectorizer2 = layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=MAX_LENGTH)
text_vectorizer2.adapt(train_sentences)
vocab2 = text_vectorizer2.get_vocabulary()

embedding_dim2 = 128
lstm_units = 64

inputs2 = layers.Input(shape=[], dtype="string")
x2 = text_vectorizer2(inputs2)
x2 = layers.Embedding(input_dim=len(vocab2), output_dim=embedding_dim2, mask_zero=True)(x2)
x2 = layers.Bidirectional(layers.LSTM(lstm_units, recurrent_dropout=0.2))(x2)
x2 = layers.Dense(64, activation="relu")(x2)
x2 = layers.Dropout(0.5)(x2)
outputs2 = layers.Dense(len(CLASS_NAMES), activation="softmax")(x2)

bilstm_model = Model(inputs2, outputs2, name="bilstm_model")
bilstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)

print("Training Bi-LSTM model...")
history_bilstm = bilstm_model.fit(
    train_sentences, train_labels,
    epochs=10, batch_size=32,
    validation_data=(val_sentences, val_labels),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

test_loss_bilstm, test_acc_bilstm = bilstm_model.evaluate(test_sentences, test_labels, verbose=0)
preds_bilstm = np.argmax(bilstm_model.predict(test_sentences, verbose=0), axis=1)
true_bilstm = np.argmax(test_labels, axis=1)

true_names_bilstm = [encoder_classes[i] for i in true_bilstm]
pred_names_bilstm = [encoder_classes[i] for i in preds_bilstm]

bilstm_report_str = classification_report(true_names_bilstm, pred_names_bilstm, target_names=CLASS_NAMES, digits=4, zero_division=0)
bilstm_report = classification_report(true_names_bilstm, pred_names_bilstm, target_names=CLASS_NAMES, digits=4, output_dict=True, zero_division=0)
bilstm_cm = confusion_matrix(true_names_bilstm, pred_names_bilstm, labels=CLASS_NAMES)

bilstm_history = {
    "train_acc": [round(x, 4) for x in history_bilstm.history["accuracy"]],
    "val_acc": [round(x, 4) for x in history_bilstm.history["val_accuracy"]],
    "train_loss": [round(x, 4) for x in history_bilstm.history["loss"]],
    "val_loss": [round(x, 4) for x in history_bilstm.history["val_loss"]],
}

print(f"\nTest loss:     {test_loss_bilstm:.4f}")
print(f"Test accuracy: {test_acc_bilstm*100:.2f}%")
print(bilstm_report_str)

# ─── SAVE ALL RESULTS TO JSON ───────────────────────────────────────────────
all_results = {
    "dataset": {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "class_distribution": {cls: int(class_counts.get(cls, 0)) for cls in CLASS_NAMES},
        "class_percentages": {cls: float(class_pcts.get(cls, 0)) for cls in CLASS_NAMES},
        "imbalance_ratio": round(imbalance_ratio, 2),
        "sentence_length_stats": eda_stats,
    },
    "baseline": {
        "name": "TF-IDF + Multinomial Naive Bayes",
        "vocab_size": vocab_size_baseline,
        "val_accuracy": round(val_acc_bl, 4),
        "test_accuracy": round(test_acc_bl, 4),
        "classification_report": baseline_report,
        "confusion_matrix": baseline_cm.tolist(),
        "top_confusions": {f"{t}->{p}": int(cnt) for (t, p), cnt in top_confusions_bl.items()},
        "error_count": len(errors_bl),
        "error_rate": round(len(errors_bl)/len(X_test)*100, 2),
    },
    "glove": {
        "name": "GloVe Embeddings (100d) + Dense",
        "vocab_size": len(vocab),
        "glove_coverage": found,
        "glove_coverage_pct": round(found/len(vocab)*100, 1),
        "test_accuracy": round(test_acc_glove, 4),
        "test_loss": round(test_loss_glove, 4),
        "classification_report": glove_report,
        "confusion_matrix": glove_cm.tolist(),
        "training_history": glove_history,
    },
    "bilstm": {
        "name": "Bidirectional LSTM (128d + 64 units)",
        "vocab_size": len(vocab2),
        "test_accuracy": round(test_acc_bilstm, 4),
        "test_loss": round(test_loss_bilstm, 4),
        "classification_report": bilstm_report,
        "confusion_matrix": bilstm_cm.tolist(),
        "training_history": bilstm_history,
    },
}

results_path = os.path.join(RESULTS_DIR, "all_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print(f"ALL RESULTS SAVED to {results_path}")
print(f"{'='*60}")
print(f"\nFINAL COMPARISON:")
print(f"  Baseline (TF-IDF+NB): {test_acc_bl*100:.2f}%")
print(f"  GloVe Embeddings:     {test_acc_glove*100:.2f}%")
print(f"  Bi-LSTM:              {test_acc_bilstm*100:.2f}%")
best_name = max([("Baseline", test_acc_bl), ("GloVe", test_acc_glove), ("Bi-LSTM", test_acc_bilstm)], key=lambda x: x[1])
print(f"\n  Best model: {best_name[0]} ({best_name[1]*100:.2f}%)")
