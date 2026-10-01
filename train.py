import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np


def set_seed(seed: int = 42):
    """
    Imposta un seed fisso per garantire la totale riproducibilità degli esperimenti.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_unconditional_context(text_encoder, tokenizer, batch_size, max_seq_len, device):
    """
    Genera l'embedding per il contesto incondizionato (testo vuoto o solo PAD).
    Utile sia per la baseline incondizionata che per il Classifier-Free Guidance.
    """
    # Creiamo un tensore pieno di token <PAD> (o un token <UNK> vuoto)
    pad_token_id = tokenizer.vocab.get("<PAD>", 0)
    uncond_tokens = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long, device=device)
    
    # La maschera per il PAD (solitamente 0 per ignorare)
    mask = (uncond_tokens != pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
    
    # Estrazione dell'embedding
    uncond_context = text_encoder(uncond_tokens, mask)
    return uncond_context

def train(
    unet: nn.Module,
    text_encoder: nn.Module,
    forward_process,
    dataloader: DataLoader,
    tokenizer,
    epochs: int,
    device: str,
    checkpoint_dir: str = "checkpoints",
    conditional: bool = True,
    cfg_drop_rate: float = 0.1,
    lr: float = 1e-4
):
    """
    Ciclo di addestramento unificato per:
    - unconditional neural baseline (conditional = False)
    - conditional model (conditional = True)
    """

    # creazione della cartella destinata ai pesi del modello
    os.makedirs(checkpoint_dir, exist_ok=True)
    unet.to(device)
    text_encoder.to(device)

    # Ottimizzatore congiunto per U-Net e Text Encoder, per gestire l'aggiornamento simultaneo dei parametri
    # Viene scelto AdamW per offrire maggiore stabilità, garantendo che il weight decay venga applicato con la stessa
    # efficacia a tutti i layer
    optimizer = optim.AdamW(list(unet.parameters()) + list(text_encoder.parameters()), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Scheduler per il learning rate. Riduce progressivamente il lr seguendo una curva cosinoidale
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    unet.train()
    text_encoder.train()

    print(f"Inizio Addestramento - Modalità: {'Condizionata' if conditional else 'Incondizionata (Baseline)'}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, text_tokens in progress_bar:
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            batch_size = images.shape[0]

            optimizer.zero_grad()

            """            
            Si campiona un timestep t uniformemente per ogni immagine nel batch, da 0 al numero di timestep.
            (batch_size,) definisce il livello di rumore, diverso per ogni immagine processata presente nel batch_size, esponendo
            il modello a vari livelli di rumore (quante le immagini nel batch) in un singolo step di ottimizzazione
            """         
            timesteps = torch.randint(0, forward_process.num_time_steps, (batch_size,), device=device).long()

            # Generazione del rumore target (epsilon)
            noise = torch.randn_like(images).to(device)

            # Processo Forward: aggiunta del rumore all'immagine pulita
            noisy_images = forward_process.add_noise(images, noise, timesteps)

            # Estrazione del contesto testuale
            if conditional:
                # Creazione della maschera (1 per i token validi, 0 per i PAD)
                pad_token_id = tokenizer.vocab.get("<PAD>", 0)
                mask = (text_tokens != pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
                context = text_encoder(text_tokens, mask)

                # Classifier-Free Guidance (CFG) Dropout
                # Con probabilità `cfg_drop_rate`, scartiamo il testo e passiamo un contesto vuoto.
                # Questo permette alla rete di imparare simultaneamente la generazione condizionata e incondizionata.
                if cfg_drop_rate > 0.0:
                    # Viene inizialmente generato un tensore monodimensionale lungo quanto il batch_size e popolato solo da numeri
                    # pseudo-casuali distribuiti tra 0.0 e 1.0. Dopo, con un'operazione di confronto, gli elementi che hanno un valore
                    # minore del drop rate assumeranno il valore True nella maschera, altrimenti ci sarà False
                    drop_mask = torch.rand(batch_size, device=device) < cfg_drop_rate
                    if drop_mask.any():     # se la maschera contiene almeno un True (.any())
                        uncond_context = get_unconditional_context(
                            text_encoder, tokenizer, batch_size, text_tokens.shape[1], device
                        )
                        # Sostituiamo le righe "droppate" con il contesto incondizionato
                        context = torch.where(drop_mask.unsqueeze(1).unsqueeze(2), uncond_context, context)
            else:
                # Per la baseline incondizionata, passiamo sempre un contesto vuoto
                context = get_unconditional_context(
                    text_encoder, tokenizer, batch_size, text_tokens.shape[1], device
                )

            # 5. Predizione del rumore target (epsilon-prediction)
            predicted_noise = unet(noisy_images, timesteps, context)

            # 6. Calcolo della Loss e Backpropagation
            loss = criterion(predicted_noise, noise)
            loss.backward()
            
            # Gradient clipping per prevenire l'esplosione dei gradienti nel Transformer/U-Net.
            # Nei primi step, i gradienti possono esplodere verso l'infinito. Si impone così un tetto massimo
            torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(text_encoder.parameters()), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})

        # Riduce il learning rate
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completata | Loss Media: {avg_loss:.4f}")

        # 7. Salvataggio del Checkpoint (permettendo di riprendere l'addestramento)
        # Salvare lo stato dell'ottimizzatore e del seed è essenziale per la riproducibilità
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'unet_state_dict': unet.state_dict(),
            'text_encoder_state_dict': text_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'random_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }, checkpoint_path)
        print(f"Checkpoint salvato: {checkpoint_path}")