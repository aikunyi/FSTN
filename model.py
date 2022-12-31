import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class FSTN(nn.Module):
    def __init__(self, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size
        self.sparsity_threshold = sparsity_threshold
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size, self.embed_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size, self.embed_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size,self.embed_size))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size ))
        self.w4 = nn.Parameter(
            self.scale * torch.randn(2, self.embed_size, self.embed_size))
        self.b4 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.fc = nn.Sequential(
                nn.Linear(self.seq_length*self.embed_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.pre_length)
            )
        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y

    '''
    FGCN:fourier graph convolution network
    '''
    def fourierGCN(self, x, B, N, L):
        bias = x
        o1_real = torch.zeros([B, N, L // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, N, L // 2 + 1, self.embed_size ],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)
        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        total_modes = L // 2 + 1
        kept_modes = int(total_modes)

        # complex-valued feed-forward layer
        o1_real[:, :, :kept_modes] = (
            torch.einsum('...i,io->...o', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...i,io->...o', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = (
            torch.einsum('...i,io->...o', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...i,io->...o', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        # layer 2
        o2_real[:, :, :kept_modes] = (
            torch.einsum('...i,io->...o', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...i,io->...o', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...i,io->...o', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...i,io->...o', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        # layer 3
        o3_real[:, :, :kept_modes] = (
                torch.einsum('...i,io->...o', o2_real[:, :, :kept_modes], self.w3[0]) - \
                torch.einsum('...i,io->...o', o2_imag[:, :, :kept_modes], self.w3[1]) + \
                self.b3[0]
        )

        o3_imag[:, :, :kept_modes] = (
                torch.einsum('...i,io->...o', o2_imag[:, :, :kept_modes], self.w3[0]) + \
                torch.einsum('...i,io->...o', o2_real[:, :, :kept_modes], self.w3[1]) + \
                self.b3[1]
        )

        out = torch.stack([o3_real, o3_imag], dim=-1)
        out = F.softshrink(out, lambd=self.sparsity_threshold)
        out = torch.view_as_complex(out)
        # residual
        out = out + bias
        return out

    '''
    FSA: fourier attention network
    '''
    def fourierSA(self, x, B, N, L):
        bias = x
        o1_real = torch.zeros([B, N, L // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, N, L // 2 + 1, self.embed_size],
                              device=x.device)

        total_modes = L // 2 + 1
        kept_modes = int(total_modes)

        # complex-valued feed-forward network
        o1_real[:, :, :kept_modes] = (
                torch.einsum('...i,io->...o', x[:, :, :kept_modes].real, self.w4[0]) - \
                torch.einsum('...i,io->...o', x[:, :, :kept_modes].imag, self.w4[1]) + \
                self.b4[0]
        )

        o1_imag[:, :, :kept_modes] = (
                torch.einsum('...i,io->...o', x[:, :, :kept_modes].imag, self.w4[0]) + \
                torch.einsum('...i,io->...o', x[:, :, :kept_modes].real, self.w4[1]) + \
                self.b4[1]
        )

        out = torch.stack([o1_real, o1_imag], dim=-1)
        out = F.softshrink(out, lambd=self.sparsity_threshold)
        out = torch.view_as_complex(out)
        # residual
        out = out + bias
        return out

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()

        # embedding B*N*L ==> B*N*L*D
        x = self.tokenEmb(x)

        B, N, L, D = x.shape

        # FFT
        x = torch.fft.rfft(x, dim=1, norm='ortho') # conduct fourier transform along N dimension

        x = x.reshape(B, N//2+1, L, self.embed_size)

        x = x.permute(0, 2, 1, 3)

        # fourier graph convolution network
        x = self.fourierGCN(x, B, L, N)

        x = x.permute(0, 2, 1, 3)

        # ifft
        x = x.reshape(B, N//2+1, L, D)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")

        # FFT
        x = torch.fft.rfft(x, dim=2, norm='ortho') # conduct fourier transform along time dimension

        x = x.reshape(B, N, L//2+1, self.embed_size)

        # fourier self attention
        x = self.fourierSA(x, B, N, L)

        x = x.reshape(B, N, L//2+1, D)
        x = torch.fft.irfft(x, n=L, dim=2, norm="ortho")

        # forecast B*N*L*D ==> B*N*P
        x = x.reshape(B, N, -1)
        x = self.fc(x)

        return x

    def viz_adj(self, x):
        x = x.permute(0, 2, 1, 3) # [B, L, N, D]
        x = x[:, :, 170:200, :]
        x = torch.mean(x, dim=1)
        adp = torch.mm(x[0], x[0].T)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp * (1 / np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(data=df, cmap="Oranges")
        plt.savefig("./emb" + '.pdf')