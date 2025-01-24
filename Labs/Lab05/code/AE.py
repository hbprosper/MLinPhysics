# ------------------------------------------------------------------
# Tutorial05: Autoencoder (AE)
# Created Aug. 2024 for the FSU Course: Machine Learning in Physics
# H. B. Prosper
# ------------------------------------------------------------------
import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, nodes=20):
        # initial base class (nn.Module)
        super().__init__()
        self.NN = nn.Sequential(nn.Linear(nodes, nodes), nn.SiLU(),
                                nn.Linear(nodes, nodes), nn.SiLU())    
    def forward(self, x):
        return self.NN(x) + x

def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

H_NODES = 40
encoder = nn.Sequential(
    nn.Linear(      5, H_NODES), nn.SiLU(), nn.LayerNorm(H_NODES),
    nn.Linear(H_NODES, H_NODES), nn.SiLU(), nn.LayerNorm(H_NODES), 
    nn.Linear(H_NODES, H_NODES), nn.SiLU(), nn.LayerNorm(H_NODES),
    nn.Linear(H_NODES, H_NODES), nn.SiLU(), nn.LayerNorm(H_NODES), 
    nn.Linear(H_NODES, 2), 
        )

decoder = nn.Sequential(
    nn.Linear(      2, H_NODES), nn.SiLU(), nn.LayerNorm(H_NODES),
    nn.Linear(H_NODES, H_NODES), nn.SiLU(), nn.LayerNorm(H_NODES),
    nn.Linear(H_NODES, H_NODES), nn.SiLU(), nn.LayerNorm(H_NODES),
    nn.Linear(H_NODES, H_NODES), nn.SiLU(), nn.LayerNorm(H_NODES),
    nn.Linear(H_NODES,  5)
        )

class AutoEncoder(nn.Module):
    
    def __init__(self, encoder, decoder):
        # call constructor of base (or super, or parent) class
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y
        
model = AutoEncoder(encoder, decoder)
