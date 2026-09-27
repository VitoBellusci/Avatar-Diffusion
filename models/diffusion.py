import torch
import math


class DiffusionScheduler:
    """
    Classe base per gestire lo scheduling del rumore.
    Implementa il Cosine Scheduler per evitare che il rumore
    venga aggiunto troppo velocemente, preservando l'informazione dell'immagine.
    """
    def __init__(self, num_time_steps=1000, s=0.008, device="cpu"):
        self.num_time_steps = num_time_steps
        self.device = torch.device(device)

        # Calcolo di alpha_bar usando la formula del cosine scheduler:
        # f(t) = cos(((t/T + s) / (1 + s)) * (pi/2))^2
        steps = torch.arange(num_time_steps + 1, dtype=torch.float32, device=self.device)
        f_t = torch.cos(((steps / num_time_steps + s) / (1 + s)) * (math.pi / 2))**2

        # Questo tensore viene normalizzato, assicurando che allo step t = 0 il valore sia esattamente 1.0.
        # Esso contiene la frazione totale del segnale originale, sopravvissuta dopo t iniezioni di rumore
        ab_full = f_t / f_t[0]    

        # Prendiamo gli step da 1 a T (escludendo lo step 0)
        self.alpha_bars = ab_full[1:].to(self.device)

        # alpha_bar_t = alpha_bar_{t-1} * alpha_t  => alpha_t = alpha_bar_t / alpha_bar_{t-1}
        # Escludendo l'ultimo elemento, si prendono tutti i valori fino a T - 1
        ab_prev = ab_full[: -1].to(self.device)
        alphas = self.alpha_bars / ab_prev
        betas = 1 - alphas    # betas contiene le quote di informazione visiva che vengono cancellate e sostituite dal rumore per ogni time step

        # Clipping delle beta per evitare instabilità numeriche (singolarità vicino a t=T)
        self.betas = torch.clamp(betas, max=0.999).to(self.device)

        # Ricalcolo di alphas e alpha_bars dopo il clipping per coerenza matematica
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Pre-calcolo per il Forward Process (add_noise)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

        # Pre-calcolo per il Reverse Process (sample)
        self.inv_sqrt_alphas = 1 / torch.sqrt(self.alphas)
        self.beta_over_sqrt_one_minus_alpha_bar = self.betas / torch.sqrt(1 - self.alpha_bars)

class DiffusionForwardProcess(DiffusionScheduler):

    # Questo metodo riceve l'immagine originale, il rumore gaussiano e il tensore t
    def add_noise(self, original, noise, t):
        """
        Applica il rumore direttamente al tempo t partendo da x_0.
        Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        # Assicuriamoci che i coefficienti siano sullo stesso device dell'immagine
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].to(original.device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].to(original.device)

        # Broadcasting per [batch, channel, height, width]
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]

        return (sqrt_alpha_bar_t * original) + (sqrt_one_minus_alpha_bar_t * noise)

class DiffusionReverseProcess(DiffusionScheduler):
    """
    Gestisce la rimozione del rumore (Processo di Generazione).
    """
    def sample(self, model, x, t, context=None, uncond_context=None, guidance_scale=3.0, noise_free=False):
        """
        Esegue un singolo step di campionamento inverso con Classifier-Free Guidance.
        Questa tecnica forza la rete a rispettare il prompt testuale. Si effettuano due
        forward pass simultanei per stimare il rumore in presenza o assenza dell'embedding testuale

        Args:
            model: U-Net (text-conditioned)
            x: Immagine rumorosa [B, C, H, W]
            t: Tensore dei timestamp [B]
            context: Embeddings testuali [B, Seq_Len, Embed_Dim]
            uncond_context: contiene l'embedding di un prompt testuale vuoto o nullo.
                            Rappresenta la direzione statistica generica del dataset. 
                            Senza alcuna informazione il modello genererebbe un'immagine
                            coerente al dataset di addestramento.
            guidance_scale: Fattore di guida (1.0 = no CFG, >1.0 = guida verso il testo)
            noise_free: Se True, rimuove il rumore stocastico.
        """
        # --- Implementazione Classifier-Free Guidance ---
        if guidance_scale > 1.0 and context is not None and uncond_context is not None:
            
            # Concateniamo condizionati e non condizionati calcolati dal Transformer
            # per ottimizzare l'uso della VRAM
            x_input = torch.cat([x, x], dim=0)
            t_input = torch.cat([t, t], dim=0)
            context_input = torch.cat([context, uncond_context], dim=0)

            # Predizione singola per entrambi i rami
            all_noise = model(x_input, t_input, context=context_input)

            # Dividiamo i risultati. chunk lo fa esattamente a metà
            eps_cond, eps_uncond = torch.chunk(all_noise, 2, dim=0)

            # Applichiamo la formula CFG: eps = eps_uncond + w * (eps_cond - eps_uncond)
            # Il cuore del CFG sta in questa differenza che rappresenta l'essenza pure del testo
            predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            # Se guidance_scale == 1.0 o non c'è contesto, usiamo la predizione standard
            predicted_noise = model(x, t, context=context)

        # --- Calcolo dello step di denoising ---
        inv_sqrt_alpha_t = self.inv_sqrt_alphas[t].to(x.device)[:, None, None, None]
        beta_over_sqrt_one_minus_alpha_bar_t = self.beta_over_sqrt_one_minus_alpha_bar[t].to(x.device)[:, None, None, None]
        beta_t = self.betas[t].to(x.device)[:, None, None, None]

        # Calcolo della media mu_theta
        # Questa operazione sottrae dall'immagine attuale x_t una porzione del predicted noidee ridimensiona
        # l'energia complessiva dei pixel. Rappresenta la stima matematica (media) di come dovrebbe apparire
        # l'immagine allo step t - 1
        mean = inv_sqrt_alpha_t * (x - beta_over_sqrt_one_minus_alpha_bar_t * predicted_noise)

        # Controllo sicuro del timestep: verifichiamo se TUTTO il batch è a t=0 in modo da restituire la media
        if (t == 0).all() or noise_free:
            return mean

        # 3. Langevin dynamics
        # In questa fase si campiona del rumore stocastico. Questo rumore serve alla rete a spingersi oltre la media sfocata.
        # Ad uno step intermedio, la media calcolata consiste in una media statistica delle possibili immagini nel dataset
        # con quella specifica macchia di rumore che caratterizza il tensore x allo step t. Questa media tuttavia è insicura e sfocata.
        # Inserendo questo nuovo rumore campionato, si costringe la traiettoria latente a prendere una decisione e a scivolare verso un'immagine
        # specifica e definita.
        # L'aggiunta di questo rumore è scalata con un cosine scheduler che permette di iniettare sempre meno rumore, man mano che si raggiunge t = 0
        z = torch.randn_like(x)

        # TRUCCO DI STABILITA': usare un clamp minimo per evitare sqrt(0) o instabilità numeriche
        sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))

        return mean + sigma_t * z