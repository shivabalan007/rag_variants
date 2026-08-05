import os

from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# Database credentials
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD"))

# PostgreSQL connection URL
DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/"
    f"{DB_NAME}"
)

# SQLAlchemy Engine
engine = create_engine(
    DATABASE_URL,
    echo=False,          # True -> shows SQL queries in terminal
    pool_pre_ping=True   # Automatically reconnect if connection is lost
)

# Session Factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for all ORM models
Base = declarative_base()


def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


def test_connection():


    try:
        with engine.connect() as connection:
            print("✅ PostgreSQL connection successful!")

    except Exception as e:
        print("❌ Connection failed")
        print(e)


if __name__ == "__main__":
    test_connection()

"""
Creates a reusable PostgreSQL connection using SQLAlchemy.
"""