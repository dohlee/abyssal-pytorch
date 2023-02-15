import torch

import pandas as pd
import numpy as np
import esm
import argparse

def mutate(seq, pos, aa_mut):
    seq = list(seq)
    seq[pos - 1] = aa_mut  # `pos` is 1-based.
    return ''.join(seq)

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='data/mega.train.csv')
parser.add_argument('--output-dir', '-o', type=str, default='data/embeddings')
parser.add_argument('--batch-size', '-b', type=int, default=496)
args = parser.parse_args()

df = pd.read_csv(args.input)
print(f'Loaded {len(df)} rows.')

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.cuda()

# Save embeddings
for i in range(0, len(df), args.batch_size):
    batch = df.iloc[i:i + args.batch_size]
    _positions_0based = batch['pos'].values - 1

    # Get ESM2 embeddings.
    data = [(i, seq) for i, seq in enumerate(batch['wt_seq'].values)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.cuda()
    with torch.no_grad():
        result = model(tokens, repr_layers=[33])

    # Index embeddings by position. 
    h = result['representations'][33][range(len(batch)), _positions_0based].cpu().numpy()

    # Save embeddings to `{wt_name}_{pos}{aa_wt}.pt`.
    for h, wt_name, wt_seq, pos, mut_type in zip(h, batch['WT_name'].values, batch['wt_seq'], _positions_0based, batch['mut_type'].values):
        aa_wt = mut_type[0]
        assert aa_wt == wt_seq[pos], f'WT sequence mismatch: {aa_wt} != {wt_seq[pos]}, {wt_name}, {pos}, {mut_type}'

        torch.save(h, f'{args.output_dir}/{wt_name}_{pos+1}{aa_wt}.pt')  # Save as 1-based.

    print(f'Processed {i + len(batch)} / {len(df)}')