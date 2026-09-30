from typing import Dict, Any

class CaptionGenerator:
    """
    Questa classe trasforma i metadati grezzi in stringhe di testo da trasformare in embedding.
    Le didascalie generate sono deterministiche:
        - La struttura della frase è fissa e determinata da un template
        - Se nei metadati manca un'informazione, si utilizza sempre la stessa parola sostitutiva
    """

    def __init__(self, template: str = None):
        # Default template
        self.template = template or (
            "a {gender} with {color} colors, {eyes} eyes, and {proportion} proportions"
        )

    def generate(self, metadata: Dict[str, Any]) -> str:
        
        try:
            # .format() assicura che i valori siano posizionati esattamente dove richiesti dal template
            return self.template.format(
                gender=metadata.get('gender', 'avatar'),
                color=metadata.get('color', 'neutral'),
                eyes=metadata.get('eyes', 'standard'),
                proportion=metadata.get('proportion', 'normal')
            )
        except KeyError:
            # Fallback in presenza di un errore inatteso con le chiavi
            return f"an avatar with {metadata.get('color', 'unknown')} colors"

    # Sovrascrive la rappresentazione standard dell'oggetto
    def __repr__(self):
        return f"CaptionGenerator(template='{self.template}')"