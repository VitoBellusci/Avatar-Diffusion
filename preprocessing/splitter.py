from typing import List, Dict, Tuple, Any
from preprocessing.config import PreprocessingConfig
import random

class CompositionalSplitter:
    """
    Questa classe isola specifiche combinazioni semantiche per creare una valutazione
    rigorosa. Si applica il concetto di OOD per testare la generalizzazione composizionale del modello.
    Durante la fase di test sul set OOD, viene chiesto al modello di generare un'immagine partendo proprio
    dalla combinazione bloccata che non ha mai visto per intero.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def split(self, metadata_list: List[Dict[str, Any]]) -> Tuple[List[int], List[int], List[int]]:
        """
        Restituisce gli indici delle immagini in tre set distinti (Train, Validation, OOD test),
        ricevendo le liste dei dizionari dei metadati. Ogni dizionario all'interno di questa lista
        rappresenta i metadati associati a una singola immagine del dataset
        """
        
        train_indices = []
        ood_indices = []
        val_indices = []

        for idx, meta in enumerate(metadata_list):
            # Ogni dizionario viene convertito in un set contenente tuple di coppie chiave-valore
            current_comb = { (k, v) for k, v in meta.items() }

            is_ood = False
            
            # Viene recuperata la lista delle combinazioni vietate. Ogni set di questa lista
            # è a sua volta un set di tuple che rappresenta un incrocio di attributi da isolare
            for blocked_set in self.config.ood_blocked_combinations:
                if blocked_set.issubset(current_comb):

                    # Diventa true se tutti gli elementi di questa lista sono presenti nei metadati dell'immagine
                    is_ood = True
                    break

            if is_ood:
                ood_indices.append(idx)
            else:
                train_indices.append(idx)

        # Estrazione del validation set dal training set
        random.shuffle(train_indices)
        val_indices = train_indices[:int(len(train_indices) * 0.1)]
        train_indices = train_indices[int(len(train_indices) * 0.1):]

        return train_indices, val_indices, ood_indices

    def verify_ood_isolation(self, train_meta: List[Dict]) -> bool:
        """
        Questa metodo verifica che non ci siano combinazioni OOD nei due dataset. Certifica che i dati finali
        siano corretti. Potrebbero esserci errori di slicing, per questosi inserisce questo sanity check.
        """
        for meta in train_meta:
            current_comb = { (k, v) for k, v in meta.items() }
            for blocked_set in self.config.ood_blocked_combinations:
                if blocked_set.issubset(current_comb):
                    return False # Leakage trovato
        return True
