import esm
import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange

def mutate(seq, pos, aa_mut):
    seq = list(seq)
    seq[pos - 1] = aa_mut  # `pos` is 1-based.
    return ''.join(seq)

class LightAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.att_conv = nn.Conv1d(1, 1, 9, 1, padding='same')
        self.feat_conv = nn.Conv1d(1, 1, 9, 1, padding='same')
    
    def forward(self, x):
        return self.att_conv(x).softmax(dim=-1) * self.feat_conv(x)

class Abyssal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()
        
        self.light_attention = LightAttention()
        self.fc_block = nn.Sequential(
            nn.Linear(2560, 2048),
            nn.ReLU(),  # Not sure
            nn.Linear(2048, 1024),
            nn.ReLU(),  # Not sure
            nn.Linear(1024, 1),
        )

    def forward(self, seq, pos, aa_mut):
        bsz = len(seq)
        seq_orig, seq_mut = seq, map(mutate, seq, pos, aa_mut)

        data = [('seq_orig', o) for o in seq_orig] + [('seq_mut', m) for m in seq_mut]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        with torch.no_grad():
            results = self.embedder(batch_tokens, repr_layers=[33], return_contacts=False)

        x_orig = results['representations'][33][:bsz][range(bsz), pos-1].unsqueeze(1)
        x_mut = results['representations'][33][bsz:][range(bsz), pos-1].unsqueeze(1)

        x_orig, x_mut = map(self.light_attention, [x_orig, x_mut])
        x = torch.cat([x_orig, x_mut], dim=-1)
        x = rearrange(x, 'b c h -> b (c h)')

        return self.fc_block(x)

if __name__ == '__main__':
    seq = [
        'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
        'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE',
    ]
    pos = torch.tensor([10, 11]).long()
    aa_mut = ['W', 'W']

    model = Abyssal()
    x = model(seq, pos, aa_mut)

    print(x.shape)
