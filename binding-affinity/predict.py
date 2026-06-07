import os
import json
import numpy as np
import deepsmiles

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# ==============================
# PATH SETUP (FIXED)
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_FILE = os.path.join(MODEL_DIR, "kiba_best_model.h5")

# CONSTANTS
PROTEIN_K = 3
LIGAND_K = 8
MAX_PROT_LEN = 1000
MAX_LIG_LEN = 85
EMBEDDING_DIM = 128


# ==============================
# MODEL ARCHITECTURE
# ==============================
def build_widedta_model(prot_vocab_size, lig_vocab_size):

    prot_input = Input(shape=(MAX_PROT_LEN,))
    x_prot = Embedding(prot_vocab_size, EMBEDDING_DIM, mask_zero=True)(prot_input)
    x_prot = Conv1D(64, 5, activation='relu', padding='same')(x_prot)
    x_prot = Conv1D(128, 5, activation='relu', padding='same')(x_prot)
    x_prot = Conv1D(256, 7, activation='relu', padding='same')(x_prot)
    x_prot = GlobalMaxPooling1D()(x_prot)
    x_prot = Dense(256, activation='relu')(x_prot)

    lig_input = Input(shape=(MAX_LIG_LEN,))
    x_lig = Embedding(lig_vocab_size, EMBEDDING_DIM, mask_zero=True)(lig_input)
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

    model = Model(inputs=[prot_input, lig_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

    return model


# ==============================
# PREDICTOR CLASS
# ==============================
class KibaPredictor:

    def __init__(self):

        print("\nInitializing KIBA Predictor...")

        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file not found at {MODEL_FILE}")
            exit()

        # Load vocabularies
        try:
            with open(os.path.join(MODEL_DIR, 'protein_vocab.json'), 'r') as f:
                self.prot_vocab = json.load(f)

            with open(os.path.join(MODEL_DIR, 'ligand_vocab.json'), 'r') as f:
                self.lig_vocab = json.load(f)

        except FileNotFoundError:
            print("Error: Vocabulary files not found.")
            exit()

        prot_vocab_size = len(self.prot_vocab) + 1
        lig_vocab_size = len(self.lig_vocab) + 1

        # Build model + load weights
        try:
            self.model = build_widedta_model(prot_vocab_size, lig_vocab_size)
            self.model.load_weights(MODEL_FILE)
            self.model.trainable = False
        except Exception as e:
            print(f"Error loading model weights: {e}")
            exit()

        # Load normalization stats
        self.normalized = False

        y_mean_path = os.path.join(MODEL_DIR, 'y_mean.npy')
        y_std_path = os.path.join(MODEL_DIR, 'y_std.npy')

        if os.path.exists(y_mean_path) and os.path.exists(y_std_path):

            self.y_mean = float(np.load(y_mean_path))
            self.y_std = float(np.load(y_std_path))

            self.normalized = True

            print(f"KIBA normalization detected (mean={self.y_mean:.4f}, std={self.y_std:.4f})")

        else:
            print("Warning: No normalization stats found.")

        self.converter = deepsmiles.Converter(rings=True, branches=True)

        print("✅ KIBA Model Loaded Successfully.")


    def get_k_mers(self, sequence, k):
        if len(sequence) < k:
            return [sequence]
        return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


    def encode_sequence(self, seq_of_words, vocab):
        return [vocab.get(word, 0) for word in seq_of_words]


    def preprocess_inputs(self, smile, protein_seq):

        try:
            deep_smile = self.converter.encode(smile)
        except Exception as e:
            print(f"Error converting SMILES: {e}")
            return None, None

        lig_words = self.get_k_mers(deep_smile, LIGAND_K)
        prot_words = self.get_k_mers(protein_seq, PROTEIN_K)

        lig_ints = self.encode_sequence(lig_words, self.lig_vocab)
        prot_ints = self.encode_sequence(prot_words, self.prot_vocab)

        lig_padded = pad_sequences([lig_ints], maxlen=MAX_LIG_LEN, padding='post', truncating='post')
        prot_padded = pad_sequences([prot_ints], maxlen=MAX_PROT_LEN, padding='post', truncating='post')

        return prot_padded, lig_padded


    def predict(self, smile, protein_seq):

        prot_padded, lig_padded = self.preprocess_inputs(smile, protein_seq)

        if prot_padded is None:
            return None, None

        raw_pred = self.model.predict([prot_padded, lig_padded], verbose=0)[0][0]

        if self.normalized and abs(raw_pred) < 5:
            kiba_score = (raw_pred * self.y_std) + self.y_mean
        else:
            kiba_score = raw_pred

        return float(kiba_score), float(raw_pred)


# ==============================
# CLI LOOP
# ==============================
if __name__ == "__main__":

    predictor = KibaPredictor()

    while True:

        print("\n--- NEW QUERY ---")

        smile_input = input("Ligand SMILES: ").strip()
        if smile_input.lower() == "exit":
            break

        prot_input = input("Protein Sequence: ").strip()
        if prot_input.lower() == "exit":
            break

        kiba_score, raw_output = predictor.predict(smile_input, prot_input)

        if kiba_score is not None:
            print("\nRESULTS:")
            print(f"Predicted KIBA Score (Real Scale): {kiba_score:.4f}")

            if kiba_score >= 12.1:
                print("Interpretation: HIGH BINDING AFFINITY")
            elif kiba_score >= 10.0:
                print("Interpretation: MODERATE BINDING")
            else:
                print("Interpretation: LOW / NO BINDING")
   