import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # contiene la dimensione del vettore per ogni timestep
        self.dim = dim

    def forward(self, time):
        # estraggo il device su cui si trova il tensore
        device = time.device
        # si divide a metà la dimensione target perché l'embedding finale verrà costruito
        # concatenando una metà calcolata con la funzione seno e un'altra metà con il coseno
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)

        # qui avviene la generazione della banda di frequenze scalate, che permettono alla rete
        # di cogliere relazioni ad alta frequenza (tempi vicini) e a bassa frequenza (twmpi lontani)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # broadcasting: il primo tensore viene espanso in una colonna di forma [batch_size, 1],
        # mentre il secondo diventa [1, half_dim]
        embeddings = time[:, None] * embeddings[None, :]
        
        # Applicazione di seno e coseno seguiti da una concatenazione
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Durante le operazioni di convoluzione, che si tratti del blocco down o del blocco up, vengono
# effettuate modifiche sulla risoluzione spaziale dell'immagine e sui canali di profondità
# La risoluzione mantiene le informazioni relative alla posizione esatta, ai contorni, alla geometria
# e ai dettagli a livello di singolo pixel. Durante il passaggio nell'encoder , l'operazione MaxPool2d
# dimezza progressivamente questa griglia, comprimendo l'immagine per avere una visione d'insieme
# più ampia per comprendere il contesto globale
# I canali definiscono la terza dimensione del tensore, cioè il numero di strati di informazione
# impilati su ogni singola coordinata spaziale. Ogni canale impara a riconoscere e isolare una caratteristica
# specifica. Man mano che si scende nell'encoder, i canali raddoppiano e l'informazione diventa
# ricca e profonda.
# Se mantenessimo la stessa risoluzione spaziale, aumentando il numero di canali, la quantità di calcoli esaurirebbe
# la memoria di qualsiasi GPU


# Nel blocco UP, si impostano dei mid_channels perché il decoder richiede una compressione temporanea
# dei canali, per gestire la concatenazione delle skip connecions.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU()
        )

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, mid_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        # Viene introdotta una convoluzione con kernel 1x1 perché non è possibile sommare due tensori con un numero diverso di canali.
        # Una convoluzione di questo tipo non guarda i pixel adiacenti e non altera la risoluzione spaziale, ma agisce come una
        # proiezione lineare per ogni singolo pixel: moltiplica i canali in ingresso per dei pesi addestrabili e restituisce 
        # l'esatto numero di out_channels richiesto, rendendo i due tensori matematicamente sommabili.
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()


    def forward(self, x, t_emb):
        residual = self.residual_conv(x)
        x = self.conv1(x)

        t_emb = self.time_emb_proj(t_emb)
        t_emb = t_emb[:, :, None, None]

        x = x + t_emb
        x = self.conv2(x)

        return x + residual



class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        # Applica un'operazione di Max Pooling che scorre l'immagine con una griglia 2x2. Seleziona solo il valore di attivazione 
        # massimo all'interno di ogni quadrante, riducendo esattamente della metà l'altezza e la larghezza del tensore. 
        # Questo concentra l'informazione sulle feature dominanti, scartando il rumore posizionale.

        self.maxpool = nn.MaxPool2d(2)

        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x, t_emb):
        x = self.maxpool(x)

        return self.conv(x, t_emb)


# Il termine bilinear si riferisce all'interpolazione bilineare, un classico algoritmo matematico utilizzato per calcolare 
# i nuovi valori dei pixel quando una matrice o un'immagine viene ingrandita (upsampling).
# 1. Se bilinear=True (Interpolazione Bilineare)
# La rete usa un classico algoritmo matematico di ridimensionamento delle immagini (lo stesso che usa Photoshop quando ingrandisci una foto).

# Calcola i nuovi pixel facendo una media ponderata dei pixel vicini.

# Pro: È un'operazione fissa, non ha pesi da addestrare. Questo rende la rete più leggera, più veloce da allenare e riduce il rischio di overfitting.

# Contro: Essendo un'operazione matematica standard, la rete non "impara" il modo migliore per ingrandire l'immagine, lo fa e basta.

# 2. Se bilinear=False (Convoluzione Trasposta)
# La rete smette di usare la matematica standard e usa un livello chiamato ConvTranspose2d (spesso chiamato impropriamente "deconvoluzione").

# Applica un filtro (kernel) che "spalma" un singolo pixel in ingresso su più pixel in uscita, ingrandendo di fatto l'immagine.

# Pro: I pesi di questo filtro vengono aggiornati durante l'addestramento. La rete impara da sola qual è il modo ottimale per ricostruire e ingrandire l'immagine specifica per il tuo problema.

# Contro: Aumenta il numero di parametri della rete, richiedendo più memoria e più tempo per l'addestramento, dettagli da tenere a mente per i progetti pratici del tuo corso magistrale.

# In sintesi, il parametro permette di scegliere tra un ingrandimento "matematico e leggero" (True) e un ingrandimento "addestrabile ma più pesante" (False). Spesso si parte con bilinear=True per avere un modello base veloce, per poi passare a False se si cerca la massima precisione possibile.

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, time_emb_dim, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

# x1 (Il flusso dal basso / Contesto): È il tensore che proviene dal livello precedente del decoder (dal basso della "U") e che è appena stato ingrandito (upsampled).
# Questo input ha viaggiato attraverso tutta la rete, quindi ha un fortissimo significato semantico (la rete ha capito cosa rappresenta l'immagine), ma i suoi contorni sono "sfocati" o imprecisi a causa dei ripetuti passaggi di pooling.

# x2 (Il flusso laterale / Dettaglio Spaziale): È la famosa skip connection. È il tensore che proviene direttamente, in orizzontale, dal livello corrispondente dell'encoder (fase di discesa).
# Poiché non è mai sceso nei livelli più profondi, non ha subito le pesanti riduzioni di risoluzione. Contiene dettagli precisi, bordi netti e informazioni spaziali ad alta risoluzione (sa esattamente dove si trovano i bordi).

    def forward(self, x1, x2, t_emb):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

#A causa delle approssimazioni, a volte x1 e x2 possono differire di 1 pixel. Questa riga aggiunge padding (bordi) per farli combaciare perfettamente.
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)

        return self.conv(x, t_emb)

# Una convoluzione 1x1 non guarda i pixel vicini, ma mappa semplicemente il numero di canali in entrata nel numero di classi desiderate per
# l'output (ad esempio, 1 canale se stai distinguendo solo "sfondo" e "oggetto", oppure N canali per N classi diverse).

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# query_dim: canali dell'immagine
# context_dim: dimensione degli embedding testuali
class SpatialCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        
        # Dimensione latente totale
        inner_dim = dim_head * heads
        self.heads = heads
        
        # Norm layer essenziale per stabilizzare i gradienti nei modelli di diffusione
        self.norm = nn.GroupNorm(8, query_dim)
        
        # Proiezioni lineari: 
        # to_q riceve la dimensione dei canali dell'immagine
        # to_k e to_v ricevono la dimensione dell'embedding testuale
        # Si imposta il bias a False perché l'obiettivo del meccanismo di attenzione è 
        # misurare la similarità angolare  (il prodotto scalare) tra vettori nello spazio latente
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # Proiezione finale per ripristinare la dimensionalità iniziale
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )


    # Prende in ingresso il tensore dell'immagine x e il testo
    def forward(self, x, context):

        # Batch, canali, altezza, larghezza
        b, c, h, w = x.shape
        
        # Normalizzazione spaziale
        x_norm = self.norm(x)
        
        # Flatten spaziale: [B, C, H, W] -> [B, C, H*W] -> trasposizione in [B, H*W, C].
        # Appiattisce altezza e larghezza in una sola dimensione
        # I pixel diventano i "token" della sequenza visuale
        # Permute inverte gli assi, fornendo il formato richiesto dalle proiezioni lineari
        x_flat = x_norm.view(b, c, -1).permute(0, 2, 1)
        
        # 3. Proiezioni lineari per Q, K, V
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 4. Reshape per la Multi-Head Attention
        # Dividiamo l'inner_dim nel numero di testine di attenzione (heads)
        # Forma attesa: [Batch, Heads, Sequenza, Dim_Head]
        q = q.view(b, -1, self.heads, q.shape[-1] // self.heads).transpose(1, 2)
        k = k.view(b, -1, self.heads, k.shape[-1] // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, v.shape[-1] // self.heads).transpose(1, 2)
        
        # 5. Calcolo dell'Attenzione ottimizzata (FlashAttention)
        # Sostituisce l'implementazione manuale (Q @ K.T -> softmax -> @ V) 
        # riducendo l'impronta di memoria e massimizzando l'uso dei Tensor Core
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.to_out[1].p if self.training else 0.0
        )
        
        # 6. Riassemblaggio degli Head
        out = out.transpose(1, 2).reshape(b, -1, out.shape[-1] * self.heads)
        
        # 7. Proiezione lineare finale
        out = self.to_out(out)
        
        # 8. Unflatten spaziale e Connessione Residuale
        # Ripristiniamo la forma spaziale [B, C, H, W], usando l'altezza e la larghezza originali
        # per scartare l'array piatto
        out = out.permute(0, 2, 1).view(b, c, h, w)
        
        return x + out  # connessione residua