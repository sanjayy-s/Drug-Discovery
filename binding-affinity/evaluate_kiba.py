import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# ==============================
# LOAD DATA
# ==============================
X_prot = np.load(os.path.join(MODEL_DIR, "X_prot.npy"))
X_lig = np.load(os.path.join(MODEL_DIR, "X_lig.npy"))
Y = np.load(os.path.join(MODEL_DIR, "Y.npy"))

split_index = int(len(Y) * 0.8)

X_prot_test = X_prot[split_index:]
X_lig_test = X_lig[split_index:]
Y_true = Y[split_index:]

# ==============================
# LOAD VOCAB
# ==============================
with open(os.path.join(MODEL_DIR, "protein_vocab.json")) as f:
    prot_vocab_size = len(json.load(f)) + 1

with open(os.path.join(MODEL_DIR, "ligand_vocab.json")) as f:
    lig_vocab_size = len(json.load(f)) + 1

# ==============================
# BUILD MODEL (same as training)
# ==============================
def build_model():

    prot_input = Input(shape=(X_prot.shape[1],))
    x_prot = Embedding(prot_vocab_size, 128, mask_zero=True)(prot_input)
    x_prot = Conv1D(64, 5, activation='relu', padding='same')(x_prot)
    x_prot = Conv1D(128, 5, activation='relu', padding='same')(x_prot)
    x_prot = Conv1D(256, 7, activation='relu', padding='same')(x_prot)
    x_prot = GlobalMaxPooling1D()(x_prot)
    x_prot = Dense(256, activation='relu')(x_prot)

    lig_input = Input(shape=(X_lig.shape[1],))
    x_lig = Embedding(lig_vocab_size, 128, mask_zero=True)(lig_input)
    x_lig = Conv1D(64, 3, activation='relu', padding='same')(x_lig)
    x_lig = Conv1D(128, 5, activation='relu', padding='same')(x_lig)
    x_lig = GlobalMaxPooling1D()(x_lig)
    x_lig = Dense(256, activation='relu')(x_lig)

    interaction = tf.keras.layers.Multiply()([x_prot, x_lig])
    merged = Concatenate()([x_prot, x_lig, interaction])

    fc = Dense(512, activation='relu')(merged)
    fc = Dropout(0.3)(fc)
    fc = Dense(256, activation='relu')(fc)

    output = Dense(1)(fc)

    model = Model([prot_input, lig_input], output)
    model.compile(optimizer=Adam(0.0005), loss='mse')

    return model

# ==============================
# LOAD WEIGHTS
# ==============================
model = build_model()
model.load_weights(os.path.join(MODEL_DIR, "kiba_best_model.h5"))

print("✅ Model loaded via weights")

# ==============================
# PREDICT
# ==============================
Y_pred = model.predict([X_prot_test, X_lig_test], verbose=1).flatten()

# ==============================
# METRICS
# ==============================
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

mse = mean_squared_error(Y_true, Y_pred)
rmse = np.sqrt(mse)
pearson_corr, _ = pearsonr(Y_true, Y_pred)
spearman_corr, _ = spearmanr(Y_true, Y_pred)
r2 = r2_score(Y_true, Y_pred)

print("\n===== RESULTS =====")
print(f"MSE       : {mse:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"Pearson R : {pearson_corr:.4f}")
print(f"Spearman  : {spearman_corr:.4f}")
print(f"R² Score  : {r2:.4f}")

import matplotlib.pyplot as plt

print("\nGenerating plots...")

# ==============================
# SCATTER PLOT
# ==============================
plt.figure(figsize=(7,7))
plt.scatter(Y_true, Y_pred, alpha=0.3, s=10)

# Perfect prediction line
min_val = min(Y_true.min(), Y_pred.min())
max_val = max(Y_true.max(), Y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

plt.xlabel("Actual KIBA Score")
plt.ylabel("Predicted KIBA Score")
plt.title(f"Actual vs Predicted\nR={pearson_corr:.3f}, RMSE={rmse:.3f}")
plt.grid(True)
plt.tight_layout()
plt.show()