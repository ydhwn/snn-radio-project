import torch
import torch.nn as nn
import snntorch as snn

class SNNModulator(nn.Module):
    def __init__(self, input_dim: int = 512, hidden1: int = 512, hidden2: int = 256, num_classes: int = 4, drop_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.dropout1 = nn.Dropout(p=drop_p)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.3)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(p=drop_p)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.lif2 = snn.Leaky(beta=0.9, threshold=0.3)
        self.out = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor):
        mem1 = self.lif1.init_leaky()
        h1 = self.dropout1(self.fc1(x))
        h1 = self.bn1(h1)
        spk1, mem1 = self.lif1(h1, mem1)
        mem2 = self.lif2.init_leaky()
        h2 = self.dropout2(self.fc2(spk1))
        h2 = self.bn2(h2)
        spk2, mem2 = self.lif2(h2, mem2)
        out = self.out(spk2)
        return out, spk2
