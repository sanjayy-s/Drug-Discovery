import pandas as pd
import numpy as np
import deepsmiles
import json
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

# SETTINGS FOR KIBA 
INPUT_FILE = 'kiba.txt'
PROTEIN_K = 3        
LIGAND_K = 8          
MAX_PROT_LEN = 1000   
MAX_LIG_LEN = 85      

def get_k_mers(sequence, k):
    if len(sequence) < k: return [sequence]
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def build_vocab(all_seqs):
    unique_words = set()
    for seq in all_seqs: unique_words.update(seq)
    # Start indexing at 1 (0 is padding)
    return {word: i+1 for i, word in enumerate(sorted(unique_words))}

def encode_sequence(seq_of_words, vocab):
    return [vocab.get(word, 0) for word in seq_of_words]

# LOAD DATA
print(f"LOADING {INPUT_FILE}")
if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found!")
    exit()

# KIBA format
df = pd.read_csv(INPUT_FILE, sep=' ', header=None, names=['ID1', 'ID2', 'SMILES', 'Sequence', 'Score'])
print(f"Loaded {len(df)} rows.")

# DEEPSMILES CONVERSION
converter = deepsmiles.Converter(rings=True, branches=True)

def to_deepsmiles(s):
    try: return converter.encode(s)
    except: return None

# Using a loop with progress print for large datasets
deep_smiles = []
for i, s in enumerate(df['SMILES']):
    if i % 10000 == 0: print(f"   Processed {i}/{len(df)}")
    deep_smiles.append(to_deepsmiles(s))

df['DeepSMILES'] = deep_smiles
df.dropna(subset=['DeepSMILES'], inplace=True)

# TOKENIZATION
df['prot_words'] = df['Sequence'].apply(lambda x: get_k_mers(x, PROTEIN_K))
df['lig_words'] = df['DeepSMILES'].apply(lambda x: get_k_mers(x, LIGAND_K))

# VOCABULARY
prot_vocab = build_vocab(df['prot_words'])
lig_vocab = build_vocab(df['lig_words'])
print(f"Protein Vocab: {len(prot_vocab)} words")
print(f"Ligand Vocab:  {len(lig_vocab)} words")

# ENCODE & PAD
df['prot_int'] = df['prot_words'].apply(lambda x: encode_sequence(x, prot_vocab))
df['lig_int'] = df['lig_words'].apply(lambda x: encode_sequence(x, lig_vocab))

X_prot = pad_sequences(df['prot_int'], maxlen=MAX_PROT_LEN, padding='post', truncating='post')
X_lig = pad_sequences(df['lig_int'], maxlen=MAX_LIG_LEN, padding='post', truncating='post')
Y = df['Score'].values.astype(np.float32)

# NORMALIZE TARGET
y_mean = Y.mean()
y_std = Y.std()

Y_norm = (Y - y_mean) / (y_std + 1e-8)

print(f"KIBA Mean: {y_mean:.4f}, Std: {y_std:.4f}")

# Save normalization stats
np.save('y_mean.npy', y_mean)
np.save('y_std.npy', y_std)

# Save normalized targets
np.save('Y.npy', Y_norm)


#SHUFFLE
X_prot, X_lig, Y = shuffle(X_prot, X_lig, Y, random_state=42)

# SAVE
np.save('X_prot.npy', X_prot)
np.save('X_lig.npy', X_lig)
np.save('Y.npy', Y)
with open('protein_vocab.json', 'w') as f: json.dump(prot_vocab, f)
with open('ligand_vocab.json', 'w') as f: json.dump(lig_vocab, f)

print(f"KIBA Processing Completed. Value Range: {Y.min():.2f} - {Y.max():.2f}")