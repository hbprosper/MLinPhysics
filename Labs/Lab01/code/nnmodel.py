import torch.nn as nn
H = 15
model = nn.Sequential(nn.Linear(2, H), nn.SiLU(), nn.LayerNorm(H),
                      nn.Linear(H, H), nn.SiLU(), nn.LayerNorm(H),
                      nn.Linear(H, H), nn.SiLU(), nn.LayerNorm(H),
                      nn.Linear(H, 1), nn.Sigmoid())
