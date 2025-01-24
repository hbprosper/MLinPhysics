# ------------------------------------------------------------------
# Tutorial05: Autoencoder (AE)
# Created Aug. 2024 for the FSU Course: Machine Learning in Physics
# H. B. Prosper
# ------------------------------------------------------------------
import torch
import torch.nn as nn

def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

encoder = nn.Sequential(

    # LAYER 0	 (-1, 3, 96, 96))	=>	(-1, 6, 48, 48)
    nn.Conv2d(in_channels=3,
              out_channels=6,
              kernel_size=3,
              stride=1,
              padding=1),
    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    nn.ReLU(),
    
    # LAYER 1	 (-1, 6, 48, 48))	=>	(-1, 12, 24, 24)
    nn.Conv2d(in_channels=6,
              out_channels=12,
              kernel_size=3,
              stride=1,
              padding=1),
    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    nn.ReLU(),
    
    # LAYER 2	 (-1, 12, 24, 24))	=>	(-1, 24, 12, 12)
    nn.Conv2d(in_channels=12,
              out_channels=24,
              kernel_size=3,
              stride=1,
              padding=1),
    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    nn.ReLU(),
    
    # LAYER 3	 (-1, 24, 12, 12))	=>	(-1, 16, 6, 6)
    nn.Conv2d(in_channels=24,
              out_channels=16,
              kernel_size=3,
              stride=1,
              padding=1),
    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    nn.ReLU()
        )

decoder = nn.Sequential(

    # LAYER 0	 (-1, 16, 6, 6))	=>	(-1, 24, 12, 12)
    nn.ConvTranspose2d(in_channels=16, 
                       out_channels=24, 
                       kernel_size=2, 
                       stride=2),
    nn.ReLU(),

    # LAYER 1	 (-1, 24, 12, 12))	=>	(-1, 12, 24, 24)
    nn.ConvTranspose2d(in_channels=24, 
                       out_channels=12, 
                       kernel_size=2, 
                       stride=2),
    nn.ReLU(),

    # LAYER 2	 (-1, 12, 12, 12))	=>	(-1, 6, 48, 48)
    nn.ConvTranspose2d(in_channels=12, 
                       out_channels=6, 
                       kernel_size=2, 
                       stride=2),
    nn.ReLU(),

    # LAYER 4    (-1, 6, 48, 48))	=>	(-1, 3, 96, 96)
    nn.ConvTranspose2d(in_channels=6, 
                       out_channels=3, 
                       kernel_size=2, 
                       stride=2),    
    nn.Sigmoid()
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
