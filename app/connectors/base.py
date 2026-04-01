"""
ConnectorBase — abstract base class for all external data-source connectors.

Each connector must implement:
- list_files()       → List available files (ids, names, sizes, modified times)
- fetch_content()    → Fetch raw file content as text
- test_connection()  → Validate credentials / connectivity
- sync()             → Trigger an ingestion sync for this connector's files

Connectors are configured via ConnectorConfig DB records and instantiated
through ConnectorRegistry.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class FileEntry:
    """Represents a single file exposed by a connector."""
    id: str                    # Unique file identifier within the connector
    name: str                  # Display name / filename
    size_bytes: Optional[int]  # File size in bytes, if known
    modified_at: Optional[datetime]  # Last modified timestamp
    mime_type: Optional[str]  # MIME type hint, if known
    metadata: dict[str, Any] = None  # Additional connector-specific metadata

    def __post_init__(self):
        self.metadata = self.metadata or {}


class ConnectorBase(ABC):
    """
    Abstract base for all data-source connectors.

    Sub-classes must implement the four core methods and expose
    `connector_type` as a class attribute.
    """

    connector_type: str = "base"  # Override in subclass, e.g. "filesystem", "s3"

    def __init__(self, config: dict[str, Any]):
        """
        Initialise the connector with its configuration dict.

        Args:
            config: Raw JSONB config stored in the connectors_config table.
                   Secrets (api keys, tokens, etc.) are stored encrypted at rest
                   and decrypted before being passed here.
        """
        self.config = config
        self._logger = logger

    # ── Core abstract methods ────────────────────────────────────────────────

    @abstractmethod
    async def list_files(self, path: Optional[str] = None) -> list[FileEntry]:
        """
        List files available in the connector's data source.

        Args:
            path: Optional sub-path / folder within the connector to list.

        Returns:
            List of FileEntry objects.
        """

    @abstractmethod
    async def fetch_content(self, file_id: str) -> str:
        """
        Fetch the full text content of a file by its ID.

        Args:
            file_id: The file identifier returned by list_files().

        Returns:
            Raw text content of the file.
        """

    @abstractmethod
    async def test_connection(self) -> dict[str, Any]:
        """
        Validate that the connector is correctly configured and can reach
        its data source.

        Returns:
            Dict with at minimum {"ok": bool, "message": str}.
            Additional keys (latency_ms, file_count, etc.) are welcome.
        """

    @abstractmethod
    async def sync(self, file_ids: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Trigger an ingestion sync for files from this connector.

        If file_ids is None, sync all files. Otherwise, sync only the
        specified ones.

        Returns:
            Dict describing the sync result (files_queued, errors, etc.).
        """

    # ── Shared utilities ─────────────────────────────────────────────────────

    def mask_secret(self, value: str, visible_chars: int = 4) -> str:
        """Return a masked version of a secret for safe logging."""
        if not value:
            return ""
        if len(value) <= visible_chars * 2:
            return "*" * len(value)
        return value[:visible_chars] + "*" * (len(value) - visible_chars * 2) + value[-visible_chars:]

    def _safe_config(self) -> dict[str, Any]:
        """
        Return a sanitised copy of self.config with secret keys masked.
        Subclasses can override to list additional secret key names.
        """
        secret_keys = {"api_key", "apiKey", "secret", "secret_key", "token", "password", "access_token", "refresh_token"}
        safe = {}
        for k, v in self.config.items():
            if any(sk in k.lower() for sk in secret_keys):
                safe[k] = self.mask_secret(str(v)) if v else None
            else:
                safe[k] = v
        return safe
