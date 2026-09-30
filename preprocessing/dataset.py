import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Any, Tuple
from preprocessing.config import PreprocessingConfig
from preprocessing.caption_generator import CaptionGenerator
from preprocessing.tokenizer import AvatarTokenizer

class AvatarDataset(Dataset):
    """
    Creiamo il dataset effettivo che integra l'immagine ed il testo. Questa classe si assicura che l'immagine sia
    normalizzata tra [-1, 1] e che la caption sia tokenizzata usando soltanto il vocabolario costruito con il set di training
    """

    def __init__(
        self,
        image_paths: List[str],
        metadata: List[Dict[str, Any]],
        tokenizer: AvatarTokenizer,
        config: PreprocessingConfig
    ):
        self.image_paths = image_paths
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.config = config
        self.caption_gen = CaptionGenerator()

        self.transform = transforms.Compose([
            transforms.Resize(self.config.resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.mean,
                std=self.config.std
            )
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        meta = self.metadata[idx]
        caption_text = self.caption_gen.generate(meta)
        caption_tokens = self.tokenizer.encode(
            caption_text,
            max_len=self.config.max_seq_len
        )
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

        return image_tensor, caption_tensor
