import json
from typing import List, Tuple, Set

class PreprocessingConfig:
    """
    Classe wrapper che carica la configurazione da un file JSON,
    ricostruisce le strutture dati native di Python e valida gli iperparametri.
    """
    def __init__(self, json_path: str):
        # 1. Caricamento del file JSON
        with open(json_path, 'r') as f:
            raw_config = json.load(f)

        # 2. Mappatura diretta e Type Hinting
        self.resolution: Tuple[int, int] = tuple(raw_config["resolution"])
        self.norm_min: float = raw_config["norm_min"]
        self.norm_max: float = raw_config["norm_max"]
        self.mean: List[float] = raw_config["mean"]
        self.std: List[float] = raw_config["std"]
        self.max_seq_len: int = raw_config["max_seq_len"]
        self.vocab_path: str = raw_config["vocab_path"]

        # 3. Ricostruzione delle strutture complesse non gestibili nel file JSON
        # Converte [[["color", "blue"], ["proportion", "exaggerated"]]] 
        # in [{('color', 'blue'), ('proportion', 'exaggerated')}]
        self.ood_blocked_combinations: List[Set[Tuple[str, str]]] = []
        for combination in raw_config.get("ood_blocked_combinations", []):
            rebuilt_set = set(tuple(attr) for attr in combination)
            self.ood_blocked_combinations.append(rebuilt_set)

        # 4. Validazione istantanea (Fail-Fast)
        self._validate()

    def _validate(self):
        """Verifica la coerenza matematica degli iperparametri."""
        if self.norm_min >= self.norm_max:
            raise ValueError("norm_min must be strictly less than norm_max")