try:
 import torch
 import torch.nn as nn
 class CAE(nn.Module):
  def __init__(self,input_dim=128,latent_dim=16):
   super().__init__(); self.encoder=nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,latent_dim)); self.decoder=nn.Sequential(nn.Linear(latent_dim,64),nn.ReLU(),nn.Linear(64,input_dim))
  def forward(self,x): z=self.encoder(x); return self.decoder(z),z
 class Generator(nn.Module):
  def __init__(self,noise_dim=64,condition_dim=32,out_dim=32):
   super().__init__(); self.net=nn.Sequential(nn.Linear(noise_dim+condition_dim,128),nn.ReLU(),nn.Linear(128,out_dim),nn.Tanh())
  def forward(self,z,c): return self.net(torch.cat([z,c],-1))
 class Critic(nn.Module):
  def __init__(self,condition_dim=32,data_dim=32):
   super().__init__(); self.net=nn.Sequential(nn.Linear(condition_dim+data_dim,128),nn.LeakyReLU(.2),nn.Linear(128,1))
  def forward(self,x,c): return self.net(torch.cat([x,c],-1))
except ImportError: pass
