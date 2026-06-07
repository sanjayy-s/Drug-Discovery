import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#CONFIGURATION
EMBEDDING_DIM = 128
FILTER_1 = 32
FILTER_2 = 64
KERNEL_SIZE = 3
DROPOUT = 0.3
LR = 0.001

#LOAD DATA 
try:
    X_prot = np.load('X_prot.npy')
    X_lig = np.load('X_lig.npy')
    Y = np.load('Y.npy')  # Now this is normalized
    y_mean = np.load('y_mean.npy')
    y_std = np.load('y_std.npy')

    print(f"Normalized Target Range: {Y.min():.2f} - {Y.max():.2f}")

    
    with open('protein_vocab.json', 'r') as f:
        prot_vocab_size = len(json.load(f)) + 1
    with open('ligand_vocab.json', 'r') as f:
        lig_vocab_size = len(json.load(f)) + 1
        
    print(f"Data Loaded: {len(Y)} samples.")
    print(f"Score Range: {np.min(Y):.2f} - {np.max(Y):.2f}")

except FileNotFoundError:
    print("Error: .npy files not found. Run 'process_kiba.py' first.")
    exit()

# BUILD MODEL
def build_widedta_model():
    # Protein Branch
    prot_input = Input(shape=(X_prot.shape[1],), name='Protein_Input')
    x_prot = Embedding(prot_vocab_size, EMBEDDING_DIM, mask_zero=True)(prot_input)
    x_prot = Conv1D(64, 5, activation='relu', padding='same')(x_prot)
    x_prot = Conv1D(128, 5, activation='relu', padding='same')(x_prot)
    x_prot = Conv1D(256, 7, activation='relu', padding='same')(x_prot)
    x_prot = GlobalMaxPooling1D()(x_prot)
    x_prot = Dense(256, activation='relu')(x_prot)

    # Ligand Branch
    lig_input = Input(shape=(X_lig.shape[1],), name='Ligand_Input')
    x_lig = Embedding(lig_vocab_size, EMBEDDING_DIM, mask_zero=True)(lig_input)
    x_lig = Conv1D(64, 3, activation='relu', padding='same')(x_lig)
    x_lig = Conv1D(128, 5, activation='relu', padding='same')(x_lig)
    x_lig = GlobalMaxPooling1D()(x_lig)
    x_lig = Dense(256, activation='relu')(x_lig)

    # Interaction Fusion (CRITICAL)
    interaction = tf.keras.layers.Multiply()([x_prot, x_lig])
    merged = Concatenate()([x_prot, x_lig, interaction])

    fc = Dense(512, activation='relu')(merged)
    fc = Dropout(0.3)(fc)
    fc = Dense(256, activation='relu')(fc)

    output = Dense(1, name='Prediction')(fc)

    model = Model(inputs=[prot_input, lig_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mse'])
    return model

model = build_widedta_model()
print("\nModel Compiled.")

# TRAIN
print("\nStarting Training...")

checkpoint = ModelCheckpoint(
    'kiba_best_model.h5',  
    monitor='val_loss', 
    save_best_only=True, 
    mode='min',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
    x=[X_prot, X_lig],
    y=Y,
    batch_size=64,      
    epochs=100,         
    validation_split=0.2,
    callbacks=[checkpoint, early_stop]
)

print("\n Training Complete.") 