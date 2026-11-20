"""Small visual JEPA baselines with a shared backbone and momentum target."""
import copy
import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, dim=64, objects=False):
        super().__init__()
        self.objects = objects
        self.net = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, dim, 3, 2, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d((8,8) if objects else (1,1)))
        if objects:
            self.positions = nn.Parameter(torch.randn(1,64,dim)*0.02)
            self.queries = nn.Parameter(torch.randn(1,8,dim)*0.02)
            self.slot_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
    def forward(self, x):
        b,t,c,h,w=x.shape
        z=self.net(x.reshape(b*t,c,h,w)).flatten(2).transpose(1,2)
        if self.objects:
            z = z + self.positions
            z, _ = self.slot_attention(self.queries.expand(b*t,-1,-1), z, z, need_weights=False)
        return z.reshape(b,t,z.shape[1],z.shape[2])

class JEPA(nn.Module):
    def __init__(self, variant='a', dim=64):
        super().__init__()
        if variant not in ('a','b','c'): raise ValueError(variant)
        self.variant=variant
        self.encoder=Encoder(dim, variant != 'a')
        self.target=copy.deepcopy(self.encoder)
        self.target.requires_grad_(False)
        self.relation=nn.TransformerEncoderLayer(dim, 4, dim*2, dropout=0, batch_first=True) if variant != 'a' else nn.Identity()
        self.predictor=nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.action=nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim)) if variant == 'c' else None
    def forward(self, video, interventions):
        # Predict each next-frame representation from the current frame.
        b,t=video.shape[:2]
        z=self.encoder(video[:,:-1]); slots=z.shape[2]
        h=self.relation(z.reshape(b*(t-1),slots,-1)).reshape(b,t-1,slots,-1)
        pred=self.predictor(h)
        if self.action is not None: pred=pred+self.action(interventions[:,:t-1]).unsqueeze(2)
        with torch.no_grad(): target=self.target(video[:,1:])
        return pred, target
    def loss(self, video, interventions):
        pred, target = self(video, interventions)
        return F.mse_loss(pred, target)
    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for online, target in zip(self.encoder.parameters(), self.target.parameters()): target.lerp_(online, 1-momentum)
