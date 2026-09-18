"""
M1_DenseNet3D
--------------
M1's direct 6D-regression head, with a DenseNet3D encoder instead of
M1's own ResNet3D encoder. Encoder-only (no decoder), matching M1's own
"no decoder at all" design -- GAP straight off the final DenseNet3D
feature map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from densenet3d_encoder import DenseNet3DEncoder


class M1_DenseNet3D(nn.Module):
    def __init__(self, dropout_p: float = 0.3, growth_rate: int = 16, block_layers=(4, 4, 4)):
        super().__init__()
        self.encoder = DenseNet3DEncoder(growth_rate=growth_rate, block_layers=block_layers)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(self.encoder.out_channels, 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.gap(feats[-1]).view(feats[-1].size(0), -1)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    model = M1_DenseNet3D()
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    print(f"output shape: {out.shape} (expect [2, 6])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert out.shape == (2, 6)
    print("PASS")
