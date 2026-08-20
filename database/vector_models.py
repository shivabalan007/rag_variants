from datetime import datetime

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import JSON
from sqlalchemy import UniqueConstraint

from pgvector.sqlalchemy import Vector

from database.connection import Base


class DocumentChunk(Base):

    __tablename__ = "document_chunks"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunk"
        ),
    )

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    document_id = Column(
        String(100),
        nullable=False,
        index=True
    )

    filename = Column(
        String(255),
        nullable=False,
        index=True
    )

    content = Column(
        Text,
        nullable=False
    )

    chunk_index = Column(
        Integer,
        nullable=False
    )

    embedding = Column(
        Vector(384),
        nullable=False
    )

    chunk_metadata = Column(
        JSON,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    
"""
V3 PostgreSQL table that stores document chunks, metadata, and their 384-dimensional embeddings for pgvector similarity search.
"""