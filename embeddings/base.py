from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    model_name: str
    normalize: bool = True 