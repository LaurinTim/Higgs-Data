import torch
from torch import nn
from HIGGS_utils import DenseBlock

class Deep(nn.Module):
    def __init__(self, units: int = 28, p: float = 0.1):
        super().__init__()
        self.linear_stack = nn.Sequential(
            DenseBlock(28, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            nn.Linear(units, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear_stack(x)
        return logits
    
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(28, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear_stack(x)
        return logits
    
class DeepWide(nn.Module):
    def __init__(self, deep: 'Deep', wide: 'Wide', deep_ratio: float = 0.5):
        super().__init__()
        self.deep: Deep = deep
        self.wide: Wide = wide
        self.deep_ratio: float = deep_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deep_logits = self.deep(x)
        wide_logits = self.wide(x)
        logits = self.deep_ratio * deep_logits + (1 - self.deep_ratio) * wide_logits
        return nn.Sigmoid()(logits)

class Deep_test(nn.Module):
    def __init__(self, units: int = 28, p: float = 0.1):
        super().__init__()
        self.linear_stack = nn.Sequential(
            DenseBlock(28, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.GELU(), p),
            DenseBlock(units, units, nn.Tanh(), p),
            nn.Linear(units, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear_stack(x)
        return logits