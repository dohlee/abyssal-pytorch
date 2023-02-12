import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F

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
        self.light_attention = LightAttention()

        self.fc_block = nn.Sequential(
            nn.Linear(2560, 2048),
            nn.ReLU(),  # Not sure
            nn.Linear(2048, 1024),
            nn.ReLU(),  # Not sure
            nn.Linear(1024, 1),
        )

    def forward(self, x1, x2):
        x1, x2 = map(self.light_attention, [x1, x2])
        x = torch.cat([x1, x2], dim=-1)
        x = rearrange(x, 'b c d -> b (c d)')

        return self.fc_block(x)

if __name__ == '__main__':
    x1 = torch.randn([16, 1, 1280])
    x2 = torch.randn([16, 1, 1280])

    model = Abyssal()
    x = model(x1, x2)
    print(x.shape)
