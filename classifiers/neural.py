import torch.nn as nn
import torch.nn.functional as F


class DummyNeural(nn.Module):
    def __init__(self, seqLength, numCategories):
        super().__init__()
        self.fc1 = nn.Linear(seqLength, numCategories)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # TODO : softmas here
        return x
