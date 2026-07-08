from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Document:
    text: str 
    metadata: Dict[str, Any]

"""
Defines the Document dataclass with text and metadata fields. Every file loaded in the system becomes a Document object.
"""