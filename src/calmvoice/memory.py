"""Per-session conversation memory: opt-in, bounded and never written to disk."""

from __future__ import annotations

from collections import deque


class SessionMemory:
    """Keeps the last ``max_turns`` user/assistant pairs in memory only.

    The Streamlit page stores one instance in ``st.session_state``, so the history survives each
    script rerun but ends with the browser session. With ``enabled=False`` nothing is kept.
    """

    def __init__(self, enabled: bool = False, max_turns: int = 6):
        if max_turns < 1:
            raise ValueError("max_turns must be 1 or more")
        self.enabled = enabled
        self.max_turns = max_turns
        self._turns: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, user: str, assistant: str) -> None:
        if self.enabled:
            self._turns.append((user, assistant))

    def history(self) -> list[tuple[str, str]]:
        """Return the history as (role, content) messages, oldest first."""
        out: list[tuple[str, str]] = []
        for user, assistant in self._turns:
            out.append(("user", user))
            out.append(("assistant", assistant))
        return out

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
