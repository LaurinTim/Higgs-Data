from torch import nn
from HIGGS_utils import DenseBlock

class Deep(nn.Module):
    def __init__(self, units=28, p=0.1):
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

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits
    
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(28, 1),
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits
    
class DeepWide(nn.Module):
    def __init__(self, deep, wide, deep_ratio=0.5):
        super().__init__()
        self.deep = deep
        self.wide = wide
        self.deep_ratio = deep_ratio

    def forward(self, x):
        deep_logits = self.deep(x)
        wide_logits = self.wide(x)
        logits = self.deep_ratio * deep_logits + (1 - self.deep_ratio) * wide_logits
        return nn.Sigmoid()(logits)

class Deep_test(nn.Module):
    def __init__(self, units=28, p=0.1):
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

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits