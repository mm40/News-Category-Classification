import torch.nn as nn
import torch.nn.functional as F


class DummyNeural(nn.Module):
    def __init__(self, seq_length, num_categories):
        super().__init__()
        self.fc1 = nn.Linear(seq_length, num_categories)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # TODO : softmas here
        return x
