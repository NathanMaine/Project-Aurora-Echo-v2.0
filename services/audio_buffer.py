"""Secure in-memory audio buffer with optional encryption."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from cryptography.fernet import Fernet, InvalidToken

LOGGER = logging.getLogger(__name__)

# 100 MB default cap — prevents unbounded memory growth from a long session
MAX_BUFFER_BYTES = int(os.environ.get("MAX_AUDIO_BUFFER_BYTES", str(100 * 1024 * 1024)))


class SecureAudioBuffer:
    """Accumulates PCM audio chunks with optional Fernet encryption."""

    def __init__(self, encryption_key: Optional[str] = None) -> None:
        self._chunks: List[bytes] = []
        self._total_bytes: int = 0
        self._fernet: Optional[Fernet] = None
        if encryption_key:
            try:
                self._fernet = Fernet(encryption_key)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid AUDIO_ENCRYPTION_KEY: {exc}") from exc

    def append(self, data: bytes) -> None:
        if not data:
            return
        if self._total_bytes + len(data) > MAX_BUFFER_BYTES:
            raise MemoryError(
                f"Audio buffer would exceed {MAX_BUFFER_BYTES} byte limit"
            )
        if self._fernet:
            data = self._fernet.encrypt(data)
        self._chunks.append(data)
        self._total_bytes += len(data)

    def reset(self) -> None:
        self._chunks.clear()
        self._total_bytes = 0

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently held."""
        return len(self._chunks)

    def snapshot(self, from_chunk: int = 0) -> bytes:
        """Return audio bytes from *from_chunk* onwards without clearing.

        This allows periodic processing to grab new audio since the last
        snapshot while the buffer continues to accumulate.
        """
        if from_chunk >= len(self._chunks):
            return b""
        sliced = self._chunks[from_chunk:]
        if not self._fernet:
            return b"".join(sliced)
        decrypted = []
        for chunk in sliced:
            try:
                decrypted.append(self._fernet.decrypt(chunk))
            except InvalidToken as exc:
                LOGGER.error("Failed to decrypt audio chunk: %s", exc)
                raise
        return b"".join(decrypted)

    def to_bytes(self) -> bytes:
        if not self._chunks:
            return b""
        return self.snapshot(0)

