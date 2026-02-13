# This script implements a full training pipeline for financial sentiment classification
# using XLM-RoBERTa with the following steps:
#
# 1) DATA LOADING:
#    - Loads 4 local text files with fixed sentiment labels.
#    - Loads multiple HuggingFace datasets and normalizes their labels.
#    - Combines all sources into a single cleaned dataset.
#
# 2) TOKENIZATION:
#    - Uses XLM-RoBERTa tokenizer with padding + truncation (max_length=128).
#
# 3) CROSS-VALIDATION (5-Fold Stratified K-Fold):
#    - Maintains class distribution in each fold.
#    - Ensures stable evaluation across splits.
#
# 4) CLASS WEIGHTING:
#    - Automatically computes balanced class weights to fix class imbalance.
#
# 5) TWO-PHASE TRAINING:
#    Phase 1: Train only the classification head (encoder frozen).
#    Phase 2: Fine-tune the full transformer model (encoder unfrozen).
#    - Uses Adam optimizers with different learning rates for each phase.
#    - Early stopping monitors validation loss.
#
# 6) CHECKPOINTING:
#    - Saves best model weights within each fold.
#    - Tracks best validation accuracy across all folds.
#
# 7) AUTO-RESUME SYSTEM:
#    - Stores "training_state.json" containing:
#         * completed_fold
#         * best_overall_accuracy
#         * predictions + labels
#    - Allows training to resume EXACTLY from the next fold if interrupted.
#
# 8) REPORTING:
#    - Saves training curves per fold.
#    - Generates final confusion matrix + classification report.
#
# 9) FINAL MODEL EXPORT:
#    - Reloads the best weights found across folds.
#    - Saves final model + tokenizer to disk.
#    - Cleans temporary checkpoints and state file after successful run.

import os
import pandas as pd
import numpy as np
import logging
import shutil
import json # RESUME CHANGE: Import json for state management
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.optimizers import Adam

# implmented early Stopping
# === Setup Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Main Configuration ===
class ModelConfig:
    MODEL_CHECKPOINT = "xlm-roberta-base"
    N_SPLITS = 5
    BATCH_SIZE = 8

    # Learning Rates and Epochs
    HEAD_LEARNING_RATE = 5e-5
    FULL_MODEL_LEARNING_RATE = 1e-5
    HEAD_EPOCHS = 2
    FULL_MODEL_EPOCHS = 3
    EARLY_STOPPING_PATIENCE = 1

    # Paths (using relative paths for portability)
    OUTPUT_DIR = "results"
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_tf_classification_model")
    PLOTS_PATH = os.path.join(OUTPUT_DIR, "plots")
    TEMP_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "temp_checkpoints")
    BEST_MODEL_WEIGHTS_PATH = os.path.join(TEMP_CHECKPOINT_DIR, "best_model.weights.h5")

    # RESUME CHANGE: Add path for the state file
    STATE_FILE_PATH = os.path.join(OUTPUT_DIR, "training_state.json")

    os.makedirs(PLOTS_PATH, exist_ok=True)
    os.makedirs(TEMP_CHECKPOINT_DIR, exist_ok=True)

# === Dataset Configuration ===
PHRASE_FILES = [
    "Sentences_50Agree.txt", "Sentences_66Agree.txt",
    "Sentences_75Agree.txt", "Sentences_AllAgree.txt"
]

ID2LABEL = {0: "Neutral", 1: "Bullish", 2: "Bearish", 3: "Strongly Bullish"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

def load_all_data():
    """Loads and combines data from local files and multiple online sources with consistent labeling."""
    logging.info("Loading and combining datasets...")
    local_rows = []
    file_to_label_id = [LABEL2ID["Neutral"], LABEL2ID["Bullish"], LABEL2ID["Bearish"], LABEL2ID["Strongly Bullish"]]
    for path, label_id in zip(PHRASE_FILES, file_to_label_id):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if text: local_rows.append({"text": text, "label": label_id})
        except FileNotFoundError:
            logging.error(f"Local file not found: {path}. Skipping.")
    local_df = pd.DataFrame(local_rows)

    online_dfs = []
    try:
        hf_df1 = pd.DataFrame(load_dataset("TimKoornstra/financial-tweets-sentiment")["train"])
        hf_df1 = hf_df1.rename(columns={"tweet": "text", "sentiment": "label_str"})
        label_map1 = {"neutral": LABEL2ID["Neutral"], "bullish": LABEL2ID["Bullish"], "bearish": LABEL2ID["Bearish"]}
        hf_df1["label"] = hf_df1["label_str"].map(label_map1)
        online_dfs.append(hf_df1[["text", "label"]])
    except Exception as e:
        logging.error(f"Could not load TimKoornstra/financial-tweets-sentiment: {e}")

    try:
        hf_df2 = pd.DataFrame(load_dataset("zeroshot/twitter-financial-news-sentiment")["train"])
        zeroshot_map = {0: LABEL2ID["Bearish"], 1: LABEL2ID["Bullish"], 2: LABEL2ID["Neutral"]}
        hf_df2["label"] = hf_df2["label"].map(zeroshot_map)
        online_dfs.append(hf_df2[["text", "label"]])
        logging.info("Successfully loaded and remapped 'zeroshot/twitter-financial-news-sentiment'.")
    except Exception as e:
        logging.error(f"Could not load zeroshot/twitter-financial-news-sentiment: {e}")

    combined_df = pd.concat([local_df] + online_dfs, ignore_index=True).dropna(subset=["text", "label"])
    combined_df["label"] = combined_df["label"].astype(int)
    logging.info(f"Combined dataset created with {len(combined_df)} records.")
    logging.info(f"Final class distribution:\n{combined_df['label'].map(ID2LABEL).value_counts(normalize=True)}")
    return combined_df

def plot_training_history(history, fold, phase):
    """Plots training history for a specific phase and fold."""
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f"Fold {fold+1} - Phase: {phase}")
    plt.grid(True)
    plt.savefig(os.path.join(ModelConfig.PLOTS_PATH, f'history_fold_{fold+1}_{phase}.png'))
    plt.close()

def plot_confusion_matrix_report(y_true, y_pred):
    """Plots a final aggregated confusion matrix and classification report."""
    if not y_true or not y_pred:
        logging.warning("Cannot generate report with no data.")
        return

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ID2LABEL.values(), yticklabels=ID2LABEL.values())
    plt.title('Aggregated Confusion Matrix (All Folds)'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(ModelConfig.PLOTS_PATH, 'final_confusion_matrix.png'))
    plt.close()

    logging.info("\n--- Aggregated Classification Report (All Folds) ---")
    report = classification_report(y_true, y_pred, target_names=ID2LABEL.values(), zero_division=0)
    logging.info(f"\n{report}")
    with open(os.path.join(ModelConfig.OUTPUT_DIR, "final_classification_report.txt"), "w") as f:
        f.write(report)

def main():
    """Main training function implementing two-phase fine-tuning and cross-validation."""
    # RESUME CHANGE: Load state from file if it exists
    start_fold = 0
    best_overall_accuracy = 0.0
    all_true_labels, all_predictions = [], []

    if os.path.exists(ModelConfig.STATE_FILE_PATH):
        try:
            with open(ModelConfig.STATE_FILE_PATH, 'r') as f:
                state = json.load(f)
                start_fold = state.get('completed_fold', -1) + 1
                best_overall_accuracy = state.get('best_overall_accuracy', 0.0)
                all_true_labels = state.get('all_true_labels', [])
                all_predictions = state.get('all_predictions', [])
                logging.info(f"--- Resuming training from Fold {start_fold + 1} ---")
                logging.info(f"Loaded state: Best accuracy so far is {best_overall_accuracy:.4f}")
        except (json.JSONDecodeError, KeyError):
            logging.warning("State file is corrupted. Starting a new session.")
            start_fold = 0

    if start_fold >= ModelConfig.N_SPLITS:
        logging.info("All folds have already been completed. Proceeding to final reporting.")

    df = load_all_data()
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_CHECKPOINT)

    texts, labels = df["text"].tolist(), df["label"].to_numpy()

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(class_weights))
    logging.info(f"Calculated class weights: {class_weight_dict}")

    kfold = StratifiedKFold(n_splits=ModelConfig.N_SPLITS, shuffle=True, random_state=42)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(texts, labels)):
        # RESUME CHANGE: Skip folds that have already been completed
        if fold < start_fold:
            continue

        logging.info(f"\n{'='*20} Starting Fold {fold+1}/{ModelConfig.N_SPLITS} {'='*20}")

        train_texts, val_texts = [texts[i] for i in train_ids], [texts[i] for i in val_ids]
        train_labels, val_labels = labels[train_ids], labels[val_ids]
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
        train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(1000).batch(ModelConfig.BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(ModelConfig.BATCH_SIZE)

        model = TFAutoModelForSequenceClassification.from_pretrained(
            ModelConfig.MODEL_CHECKPOINT, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID)

        early_stopping_callback = EarlyStopping(
            monitor='val_loss', patience=ModelConfig.EARLY_STOPPING_PATIENCE,
            verbose=1, restore_best_weights=True)

        logging.info("--- Phase 1: Training the classification head ---")
        model.layers[0].trainable = False
        model.compile(optimizer=Adam(learning_rate=ModelConfig.HEAD_LEARNING_RATE),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        head_history = model.fit(train_dataset, epochs=ModelConfig.HEAD_EPOCHS, validation_data=val_dataset,
                                 class_weight=class_weight_dict, callbacks=[early_stopping_callback])
        plot_training_history(head_history, fold, "Phase1_Head")

        logging.info("--- Phase 2: Fine-tuning the entire model ---")
        model.layers[0].trainable = True
        model.compile(optimizer=Adam(learning_rate=ModelConfig.FULL_MODEL_LEARNING_RATE),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        full_history = model.fit(train_dataset, epochs=ModelConfig.FULL_MODEL_EPOCHS, validation_data=val_dataset,
                                 class_weight=class_weight_dict, callbacks=[early_stopping_callback])
        plot_training_history(full_history, fold, "Phase2_Full")

        _, final_accuracy = model.evaluate(val_dataset)
        logging.info(f"Fold {fold+1} final validation accuracy: {final_accuracy:.4f}")

        predictions = np.argmax(model.predict(val_dataset).logits, axis=1)
        all_true_labels.extend(val_labels.tolist()) # Ensure lists are extended correctly
        all_predictions.extend(predictions.tolist())

        if final_accuracy > best_overall_accuracy:
            logging.info(f"*** New best model found! Accuracy: {final_accuracy:.4f}. Saving weights. ***")
            best_overall_accuracy = final_accuracy
            model.save_weights(ModelConfig.BEST_MODEL_WEIGHTS_PATH)

        # RESUME CHANGE: Save the current state after a fold is completed
        state_to_save = {
            'completed_fold': fold,
            'best_overall_accuracy': best_overall_accuracy,
            'all_true_labels': all_true_labels,
            'all_predictions': all_predictions
        }
        with open(ModelConfig.STATE_FILE_PATH, 'w') as f:
            json.dump(state_to_save, f, indent=4)
        logging.info(f"Successfully completed and saved state for Fold {fold+1}.")


    # --- Final Reporting ---
    logging.info("--- Cross-validation complete. Generating final reports. ---")
    plot_confusion_matrix_report(all_true_labels, all_predictions)
    logging.info(f"Overall best validation accuracy across all folds: {best_overall_accuracy:.4f}")

    # --- SAVE THE FINAL MODEL AT THE END ---
    if os.path.exists(ModelConfig.BEST_MODEL_WEIGHTS_PATH):
        logging.info("--- Loading best weights and saving the final model ---")
        final_model = TFAutoModelForSequenceClassification.from_pretrained(
            ModelConfig.MODEL_CHECKPOINT, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID)
        final_model.load_weights(ModelConfig.BEST_MODEL_WEIGHTS_PATH)
        final_model.save_pretrained(ModelConfig.BEST_MODEL_PATH)
        tokenizer.save_pretrained(ModelConfig.BEST_MODEL_PATH)
        logging.info(f"Final model and tokenizer saved to {ModelConfig.BEST_MODEL_PATH}")

        # --- Final Cleanup ---
        logging.info("Cleaning up temporary files...")
        shutil.rmtree(ModelConfig.TEMP_CHECKPOINT_DIR)
        # RESUME CHANGE: Clean up the state file after a fully successful run
        os.remove(ModelConfig.STATE_FILE_PATH)
        logging.info("Temporary checkpoint files and state file cleaned up.")
    else:
        logging.warning("No best model weights found. This could happen if training was interrupted before any model was saved.")


if __name__ == "__main__":
    main()
