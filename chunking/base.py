from dataclasses import dataclass

@dataclass
class ChunkConfig:
    chunk_size: int
    overlap: int