import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import tqdm
from torch.utils.data import Dataset, DataLoader

from abyssal_pytorch import Abyssal
from data import MegaDataset

def train(model, train_loader, optimizer, criterion, log_interval, device):
    model.train()
    running_loss = 0.0

    # Training loop with tqdm progressbar
    bar = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)
    for idx, batch in enumerate(bar):
        seq = batch['seq']
        pos = batch['pos']
        aa_mut = batch['aa_mut']
        label = batch['label']

        pos = pos.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(seq, pos, aa_mut).flatten()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0:
            bar.set_postfix(loss=f'{loss.item():.4f}')

        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, log_interval, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            seq = batch['seq']
            pos = batch['pos']
            aa_mut = batch['aa_mut']
            label = batch['label']

            pos = pos.to(device)
            label = label.to(device)

            output = model(seq, pos, aa_mut).flatten()
            loss = criterion(output, label)

            running_loss += loss.item()
    return running_loss / len(val_loader)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def main():
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=72)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=10)
    args = parser.parse_args()

    seed_everything(args.seed)

    train_df = pd.read_csv('../data/mega.train.csv')
    train_set = MegaDataset(train_df)

    val_df = pd.read_csv('../data/mega.val.csv')
    val_set = MegaDataset(val_df)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = Abyssal()
    freeze(model.embedder)
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, args.log_interval, args.device)
        val_loss = validate(model, val_loader, criterion, args.log_interval, args.device)

        if epoch % args.log_interval == 0:
            print(f'Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')

        scheduler.step()

if __name__ == '__main__':
    main()