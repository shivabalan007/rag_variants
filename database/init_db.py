from database.connection import Base, engine

from database.models import Conversation


def init_database():
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully!")
    print("✅ Tables created (if they did not already exist).")

if __name__ == "__main__":
    init_database()

"""
Creates all database tables defined in models.py.
"""