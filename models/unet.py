import torch
import torch.nn as nn
from unet_parts import *


# La U-Net ha il compito di effettuare una regressione pixel per pixel, prendendo
# un'immagine rumorosa (con 3 canali) e predicendo il rumore che è stato aggiunto.
## Poiché il rumore viene aggiunto a ogni singolo canale di colore dell'immagine,
# il tensore del rumore deve avere la stessa forma dimensionale dell'immagine (3)
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, context_dim):
        super().__init__()

        time_emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.inc = DoubleConv(in_channels, base_channels, time_emb_dim)
        self.attn_inc = SpatialCrossAttention(base_channels, context_dim)

        self.down1 = Down(base_channels, base_channels * 2, time_emb_dim)
        self.attn_down1 = SpatialCrossAttention(base_channels * 2, context_dim)

        self.down2 = Down(base_channels * 2, base_channels * 4, time_emb_dim)
        self.attn_down2 = SpatialCrossAttention(base_channels * 4, context_dim)

        # La bottleneck non effettua ulteriori riduzioni di risoluzione. L'immagine in questo
        # punto contiene il massimo dell'informazione sul suo contesto, tuttavia ha perso completamente
        # l'informazioni sui dettagli fini. E' il momento migliore per effettuare la cross attention,
        # perché qui si ragiona su concetti globali, fondendo la rappresentazione visiva compressa con i token testuali
        self.bott1 = DoubleConv(base_channels * 4, base_channels * 4, time_emb_dim)
        self.attn_bott1 = SpatialCrossAttention(base_channels * 4, context_dim)
        self.bott2 = DoubleConv(base_channels * 4, base_channels * 4, time_emb_dim)

        self.up1 = Up(base_channels * 6, base_channels * 2, time_emb_dim, bilinear=True)
        self.attn_up1 = SpatialCrossAttention(base_channels * 2, context_dim)
        self.up2 = Up(base_channels * 3, base_channels, time_emb_dim, bilinear=True)
        self.attn_up2 = SpatialCrossAttention(base_channels, context_dim)

        self.out = OutConv(base_channels, out_channels)

    def forward(self, x, time, context):
        t = self.time_mlp(time)

        # ENCODER
        skip1 = self.attn_inc(self.inc(x, t), context)

        skip2 = self.attn_down1(self.down1(skip1, t), context)

        skip3 = self.attn_down2(self.down2(skip2, t), context)

        # BOTTLENECK
        bott = self.bott1(skip3, t)
        bott = self.attn_bott1(bott, context)
        bott = self.bott2(bott, t)

        # DECODER
        x = self.up1(bott, skip2, t)
        x = self.attn_up1(x, context)
        x = self.up2(x, skip1, t)
        x = self.attn_up2(x, context)

        # OUTPUT
        out = self.out(x)

        return out
