"""Secure in-memory audio buffer with optional encryption."""

from __future__ import annotations

import logging
from typing import List, Optional

from cryptography.fernet import Fernet, InvalidToken

LOGGER = logging.getLogger(__name__)


class SecureAudioBuffer:
    """Accumulates PCM audio chunks with optional Fernet encryption."""

    def __init__(self, encryption_key: Optional[str] = None) -> None:
        self._chunks: List[bytes] = []
        self._fernet: Optional[Fernet] = None
        if encryption_key:
            try:
                self._fernet = Fernet(encryption_key)
            except (ValueError, TypeError) as exc:
                LOGGER.error("Invalid AUDIO_ENCRYPTION_KEY provided: %s", exc)
                self._fernet = None

    def append(self, data: bytes) -> None:
        if not data:
            return
        if self._fernet:
            data = self._fernet.encrypt(data)
        self._chunks.append(data)

    def reset(self) -> None:
        self._chunks.clear()

    def to_bytes(self) -> bytes:
        if not self._chunks:
            return b""
        if not self._fernet:
            return b"".join(self._chunks)
        decrypted = []
        for chunk in self._chunks:
            try:
                decrypted.append(self._fernet.decrypt(chunk))
            except InvalidToken as exc:
                LOGGER.error("Failed to decrypt audio chunk: %s", exc)
                raise
        return b"".join(decrypted)

