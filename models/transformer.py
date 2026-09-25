import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    # d_model indica la dimensione di embedding
    # vocab_size indica la dimensione del vocabolario contenente i token
    def __init__(self, vocab_size, d_model=128):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model) # matrice di embedding

    def forward(self, x):
        # il tensore di embedding viene normalizzato, moltiplicando per un fattore di normalizzazione
        # questo evita che gli input di embedding diventino estremamente piccoli
        return self.embedding(x) * math.sqrt(self.d_model)

# il positional encoding serve ad immettere un significato relativo alla posizione della parole
# all'interno della frase
class PositionalEncoding(nn.Module):
    def __init__(self, dropout, seq_len, d_model=128):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # questa matrice conterrà le codifiche posizionali per ogni possibile posizione
        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # applica la funzione seno alle colonne pari
        pe[:, 0::2] = torch.sin(position * div_term)

        # applica la funzione coseno a quelle dispari
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # si registra 'pe' come buffer anziché come parametro addestrabile. I buffer sono tensori
        # che vengono salvati e caricati insieme ai pesi e spostati sulla GPU
        self.register_buffer('pe', pe)

    def forward(self, x):

        # estraggo dalla matrice solo le posizioni necessarie per la sequenza corrente
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

        # applicando questo dropout, forzo il modello a non fare affidamento eccessivo su specifici token o posizioni
        # esatte per comprendere la frase
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()

        self.eps = eps

        # questi due parametri apprendibili introducono flessibilità, permettendo anche di annullare
        # la normalizzazione stessa. Se, durante l'addestramento, il modello rileva che per un 
        # determinato token la normalizzazione pura degrada le performance, può teoricamente impostare 
        # alpha uguale alla deviazione standard originale e bias uguale alla media originale, 
        # restituendo di fatto i dati originari non alterati.
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # il calcolo avviene lungo l'ultima dimensione (dim=-1), ovvero la dimensione di embedding
        # keepdim assicura che il tensore risultante mantenga lo stesso numero di dimensioni
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# la Feed Forward Network funge da estrattore di feature per consolidare i risultati ottenuti 
# nel meccanismo di attention. Inoltre introduce non-linearità
# d_ff: dimensione del layer interno nascosto
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        # espande la rappresentazione
        self.linear_1 = nn.Linear(d_model, d_ff)

        self.dropout = nn.Dropout(dropout)

        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

# Questo blocco riceve in dati in input, li divide in q, k, v organizzandoli in matrici. Da ciascuna matrice,
# si ottiene la rispettiva matrice dei pesi
# h: numero di teste

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()

        # la dimensione di embedding deve essere divisibile per il numero di teste
        assert d_model % h == 0

        self.d_k  = d_model // h

        self.h = h

        # si istanziano 3 layer per creare le matrici
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # layer finale per concatenare tutte le attenzioni derivanti dalle teste
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    # riceve i tensori già divisi per teste
    def attention(query, key, value, mask, dropout):

        # dimensione di ogni singola testa
        d_k = query.shape[-1]

        # si calcolano i punteggi di attenzione, dividendoli per la radice di d_k per prevenire
        # gradienti troppo deboli
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # applicazione della maschera per:
        #   - non considerare i token di padding
        #   - non permettere al modello di guardare i token futuri

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)

        # applicata lungo l'ultima dimensione che corrisponde al key_len
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # il prodotto con la matrice value, combina i concetti semantici in base alla forza
        # della connessione
        return attention_scores @ value, attention_scores

    # i tensori possono provenire sia dallo stesso flusso (self-attention), che da flussi diversi (cross-attention)
    def forward(self, q, k, v, mask):
        # proiezione lineare dell'input
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # divisione nelle varie teste
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)

        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)


        # per ottimizzare il calcolo, si scambia la dimensione della seq_len con la dimensione delle teste
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        x, attention_scores = MultiHeadAttentionBlock.attention(
            query,
            key,
            value,
            mask,
            self.dropout
        )

        x = x.transpose(1, 2)
        x = x.contiguous()  # forza a ricostruire una matrice allineata perfettamente a livello hardware
        x = x.view(x.shape[0], -1, self.h * self.d_k)   # ricompatta le varie teste

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # stabilizza l'input prima che entri nel sublayer
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):

        # Il sublayer può essere una blocco di attenzione o feed forward.
        # Posizionare la normalizzazione prima del sublayer, stabilizza i gradienti
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
            self,
            self_attention_block: MultiHeadAttentionBlock,
            feed_forward_block: FeedForwardBlock,
            dropout
    ):
        super().__init__()

        # q, k, v provengono dalla stesso input
        self.self_attention_block = self_attention_block

        # si applica ad ogni token dopo l'attenzione per incrementare la rappresentazione
        # del modello, tramite due trasformazioni lineari ed un'attivazione (non-lineare)
        self.feed_forward_block = feed_forward_block

        # l'encoder ha due connessioni residue, una attorno il self-attention sublayer
        # ed una attorno il feed-forward sublayer
        self.residual_connections = nn.ModuleList([
                    ResidualConnection(dropout),
                    ResidualConnection(dropout)
        ])

    def forward(self, x, src_mask):

        # src_mask è usata per ignorare i PAD token
        x = self.residual_connections[0](
                x,

                # si usa la funzione lambda perché la connessione residua aspetta
                # una funzione che riceve x come input
                lambda x: self.self_attention_block(x, x, x, src_mask)
            )

        x = self.residual_connections[1](
            x,
            self.feed_forward_block
        )

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers    # aspetta un EncoderBlock
        self.norm = LayerNormalization()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)