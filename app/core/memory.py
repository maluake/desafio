"""Short-term conversation memory (per user, bounded, TTL).

In-process and thread-safe; swap for Redis in a multi-replica deployment (same interface).
Only *redacted* text turns are stored - never raw PII, never tool payloads.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class ConversationMemory:
    def __init__(self, max_turns: int = 6, ttl_s: int = 1800) -> None:
        self.max_turns = max_turns
        self.ttl_s = ttl_s
        self._data: dict[str, tuple[float, deque[dict[str, str]]]] = {}
        self._lock = threading.Lock()

    def get(self, user_id: str) -> list[dict[str, str]]:
        with self._lock:
            entry = self._data.get(user_id)
            if entry is None:
                return []
            ts, turns = entry
            if time.time() - ts > self.ttl_s:
                del self._data[user_id]
                return []
            return list(turns)

    def append(self, user_id: str, user_msg: str, assistant_msg: str) -> None:
        with self._lock:
            _, turns = self._data.get(user_id, (0.0, deque(maxlen=self.max_turns * 2)))
            turns.append({"role": "user", "content": user_msg})
            turns.append({"role": "assistant", "content": assistant_msg})
            self._data[user_id] = (time.time(), turns)

    def clear(self, user_id: str) -> None:
        with self._lock:
            self._data.pop(user_id, None)
