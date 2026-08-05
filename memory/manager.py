from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


class MemoryManager:

    def __init__(self,short_memory: ShortTermMemory,long_memory: LongTermMemory,session_id: str,user_id: str = "default_user"):
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.session_id = session_id
        self.user_id = user_id

        self._loaded = False

    def load_session(self):  # Load conversation from PostgreSQL into short-term memory
        if self._loaded:
            return

        history = self.long_memory.get_session_history(
            self.session_id
        )

        print("INSIDE load_session()")

        self.short_memory.clear()

        for message in history:
            self.short_memory.add_message(
                role=message["role"],
                content=message["content"]
            )

        self._loaded = True


    def get_history(self):
        # Current session history.
        return self.short_memory.get_history()

    def add_message(self,role: str,content: str):
        # Save in short-term memory
        self.short_memory.add_message(
            role=role,
            content=content
        )

        # Save in PostgreSQL
        self.long_memory.save_message(
            session_id=self.session_id,
            user_id=self.user_id,
            role=role,
            message=content
        )

    def clear(self):  
        self.short_memory.clear()   # Clear current session
        self.long_memory.clear_session(self.session_id)
        self._loaded = False

    def get_recent_messages(self, n=5):
        return self.short_memory.get_recent_messages(n)

"""
Combines Short-Term and Long-Term memory.
"""