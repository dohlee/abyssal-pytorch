import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

class MegaDataset(Dataset):
    def __init__(self, df, emb_dir):
        """df: pandas dataframe for MegaTrain/MegaValidation/MegaTest datasets.
        """
        self.df = df.to_records()
        self.emb_dir = emb_dir

        print('Preloading embeddings..')
        wt_embs, mut_embs = [], []
        for r in tqdm(self.df):
            wt_emb = torch.load(f'{self.emb_dir}/{r["WT_name"]}_{r["pos"]}{r["wt_seq"][r["pos"] - 1]}.pt')
            mut_emb = torch.load(f'{self.emb_dir}/{r["WT_name"]}_{r["pos"]}{r["aa_mut"]}.pt')

            wt_embs.append(wt_emb)
            mut_embs.append(mut_emb)
        
        self.wt_embs = np.stack(wt_embs)
        self.mut_embs = np.stack(mut_embs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df[idx]

        wt_emb = self.wt_embs[idx]
        mut_emb = self.mut_embs[idx]

        # wt_emb = torch.load(f'{self.emb_dir}/{r["WT_name"]}_{r["pos"]}{r["wt_seq"][r["pos"] - 1]}.pt')
        # mut_emb = torch.load(f'{self.emb_dir}/{r["WT_name"]}_{r["pos"]}{r["aa_mut"]}.pt')

        return {
            'wt_emb': wt_emb,
            'mut_emb': mut_emb,
            'label': torch.tensor([r['ddG_ML']]).float(),
        }

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../data/mega.train.csv')
    train = MegaDataset(df)

    train_loader = DataLoader(train, batch_size=8, shuffle=False)
    for batch in train_loader:
        print(batch['wt_emb'].shape)
        print(batch['mut_emb'].shape)
        print(batch['label'].shape)
        break