from collections import deque
from typing import Dict, List


class ShortTermMemory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages = deque(maxlen=max_messages)

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({
            "role": role,
            "content": content
        })

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._messages)

    def get_recent_messages(self, n: int = 5) -> List[Dict[str, str]]:
        return list(self._messages)[-n:]

    def clear(self) -> None:
        self._messages.clear()

    def size(self) -> int:
        return len(self._messages)

    def is_empty(self) -> bool:
        return len(self._messages) == 0