import torch
import torch.nn as nn


"""
Nei modelli di diffusione, l'aggiunta di rumore ad ogni step t è governata da un parametro
chiamato beta_t (varianza).
Invece di aggiungere rumore iterativamente, gli autori di DDPM hanno dimostrato matematicamente
che possiamo calcolare l'immagine rumorosa al tempo t direttamente dall'immagine originale x_0 in
un solo passaggio
"""
class DiffusionForwardProcess:
    def __init__(self, beta_start=1e-4, beta_end=0.02, num_time_steps=1000):
        
        """
        Inizializzazione dello schedule di rumore. Nei modelli di diffusione non si aggiunge rumore
        nella stessa quantità ogni volta. Per questo si utilizza un'interpolazione tra due estremi.
        Ai primi step temporali, si aggiunge un rumore microscopico per non distruggere le informazioni troppo
        in fretta. In questa fase, la rete viene addestrata a rimuovere lievi imperfezioni e ricostruire dettagli
        ad alta frequenza.
        Verso gli ultimi step, il rumore aggiunto diventa molto più aggressivo, assicurando che l'immagine diventi puro
        rumore Gaussiano. In questa fase la rete impara a generare la struttura globale dal nulla
        
        """
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Questo tensore rappresenta il peso del segnale, ovvero determina quanta parte dell'immagine sopravvive ad un determinato step t
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        # Quest'altro invece, rappresenta il peso del rumore. Determina quanto rumore viene aggiunto
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    # Riceve in input un intero batch di immagini, il rumore da applicare e un tensore t
    def add_noise(self, original, noise, t):
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.to(original.device)[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.to(original.device)[t]

        # Broadcasting
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]

        return (sqrt_alpha_bar_t * original) + (sqrt_one_minus_alpha_bar_t * noise)
    

"""

Qui si definisce lo step opposto, partendo dal rumore puro fino ad arrivare all'immagine nitida al tempo t = 0
Dobbiamo procedere in modo probabilistico, calcolando:
- La media: la direzione più probabile verso cui muoversi per togliere il rumore
- La varianza: una piccola quantità di rumore casuale da iniettare per mantenere stabili i dettagli
               dell'immagine, per evitare che l'immagine generata risulti innaturalmente liscia o sfocata

"""
class DiffusionReverseProcess:
    def __init__(self, beta_start=1e-4, beta_end=0.02, num_time_steps=1000):
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps)
        self.alphas