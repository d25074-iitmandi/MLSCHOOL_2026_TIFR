import sys
print("Python:", sys.version.split()[0])

import numpy, scipy, sklearn
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)

import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

import torch_geometric
print("PyTorch Geometric:", torch_geometric.__version__)

import lightning as L
print("Lightning:", L.__version__)

# Tiny end-to-end smoke test: build a graph + a Lightning module
from torch_geometric.data import Data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1.0], [0.0], [1.0]])
data = Data(x=x, edge_index=edge_index)
print("Sample graph:", data)

class TinyModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.layer(x)

module = TinyModule()
print("Sample Lightning module instantiated:", type(module).__name__)
print("\n✅ Environment ready.")
