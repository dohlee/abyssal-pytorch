import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import tqdm
import os
import wandb

from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr

from abyssal_pytorch import Abyssal
from abyssal_pytorch.data import MegaDataset

def train(model, train_loader, optimizer, criterion, metrics_f):
    model.train()
    running_output, running_label = [], []

    # Training loop with progressbar.
    bar = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)
    for idx, batch in enumerate(bar):
        wt_emb, mut_emb = batch['wt_emb'].cuda(), batch['mut_emb'].cuda()
        label = batch['label'].cuda().flatten()

        optimizer.zero_grad()
        output = model(wt_emb, mut_emb).flatten()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_output.append(output.detach().cpu())
        running_label.append(label.detach().cpu())

        if idx % 100 == 0:
            running_output = torch.cat(running_output, dim=0)
            running_label = torch.cat(running_label, dim=0)

            running_loss = criterion(running_output, running_label)
            running_metrics = {k: f(running_output, running_label) for k, f in metrics_f.items()}

            loss = running_loss.item()
            pearson = running_metrics['pearson']
            spearman = running_metrics['spearman']
            bar.set_postfix(loss=loss, pearson=pearson, spearman=spearman)
            wandb.log({
                'train/loss': loss,
                'train/pearson': pearson,
                'train/spearman': spearman,
            })

            running_output, running_label = [], []

def validate(model, val_loader, criterion, metrics_f):
    model.eval()

    out_fwd, out_rev, label = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            wt_emb, mut_emb = batch['wt_emb'].cuda(), batch['mut_emb'].cuda()
            _label = batch['label'].cuda().flatten()

            _out_fwd = model(wt_emb, mut_emb).flatten()
            _out_rev = model(mut_emb, wt_emb).flatten()  # Swap wt_emb and mut_emb.

            out_fwd.append(_out_fwd.cpu())
            out_rev.append(_out_rev.cpu())

            label.append(_label.cpu())
        
    out_fwd = torch.cat(out_fwd, dim=0)
    out_rev = torch.cat(out_rev, dim=0)
    label = torch.cat(label, dim=0)

    loss = criterion(out_fwd, label).item()
    metrics = {k: f(out_fwd, label) for k, f in metrics_f.items()}

    # Add antisymmetry metrics.
    metrics['pearson_fr'] = pearsonr(out_fwd, out_rev)[0] 
    metrics['delta'] = torch.cat([out_fwd, out_rev], dim=0).mean()

    wandb.log({
        'val/loss': loss,
        'val/pearson': metrics['pearson'],
        'val/spearman': metrics['spearman'],
        'val/pearson_fr': metrics['pearson_fr'],
        'val/delta': metrics['delta'],
    })

    return loss, metrics

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Performance drops, so commenting out for now.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def main():
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--emb-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=72)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use-bn', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    args = parser.parse_args()

    seed_everything(args.seed)
    if not args.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    
    wandb.init(project='abyssal-pytorch', config=args, reinit=True)

    train_df = pd.read_csv(args.train)
    train_set = MegaDataset(train_df, emb_dir=args.emb_dir)

    val_df = pd.read_csv(args.val)
    val_set = MegaDataset(val_df, emb_dir=args.emb_dir)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=16, pin_memory=True)

    model = Abyssal()
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
    criterion = nn.MSELoss()

    metrics_f = {
        'pearson': lambda x, y: pearsonr(x, y)[0],
        'spearman': lambda x, y: spearmanr(x, y)[0],
    }

    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, metrics_f)
        val_loss, val_metrics = validate(model, val_loader, criterion, metrics_f)

        message = f'Epoch {epoch} Validation: loss {val_loss:.4f},'
        message += ', '.join([f'{k} {v:.4f}' for k, v in val_metrics.items()])
        print(message)

        scheduler.step()

if __name__ == '__main__':
    main()
