import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange

class LightAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.att_conv = nn.Conv1d(1, 1, 9, 1, padding='same')
        self.feat_conv = nn.Conv1d(1, 1, 9, 1, padding='same')
    
    def forward(self, x):
        return self.att_conv(x).softmax(dim=-1) * self.feat_conv(x)

class Abyssal(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        
        self.light_attention = LightAttention()
        self.fc_block = nn.Sequential(
            nn.Linear(2560, 2048),
            nn.BatchNorm1d(2048) if use_bn else nn.Identity(),
            nn.ReLU(),  # Not sure
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024) if use_bn else nn.Identity(),
            nn.ReLU(),  # Not sure
            nn.Linear(1024, 1),
        )

    def forward(self, wt_emb, mut_emb):
        x_wt, x_mut = map(lambda t: rearrange(t, 'b h -> b 1 h'), [wt_emb, mut_emb])
        x_wt, x_mut = map(self.light_attention, [x_wt, x_mut])
        x = torch.cat([x_wt, x_mut], dim=-1)
        x = rearrange(x, 'b c h -> b (c h)')

        return self.fc_block(x)

if __name__ == '__main__':
    wt_emb = torch.randn([2, 1280])
    mut_emb = torch.randn([2, 1280])

    model = Abyssal()
    x = model(wt_emb, mut_emb)

    print(x.shape)
