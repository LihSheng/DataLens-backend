"""
FilesystemConnector — reads files from the local file system.

connector_type: "filesystem"

Config schema:
{
    "base_path": "/data/docs",          # Root directory to walk (required)
    "allowed_extensions": [".txt", ".md", ".pdf", ".docx"],  # Optional whitelist
    "recursive": true,                  # Whether to recurse into subdirectories
    "encoding": "utf-8"                # Text encoding fallback
}
"""
import os
import mimetypes
from pathlib import Path
from typing import Any, Optional

from app.connectors.base import ConnectorBase, FileEntry


class FilesystemConnector(ConnectorBase):
    connector_type = "filesystem"

    # Extensions considered "text" for inline reading.
    TEXT_EXTENSIONS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".csv", ".xml", ".html", ".htm", ".rst", ".log"}
    # Extensions that need special parsing (not implemented here — treated as binary blobs).
    BINARY_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".gif", ".zip"}

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.base_path = Path(self.config.get("base_path", "/data/docs"))
        self.allowed_extensions: set[str] = set(self.config.get("allowed_extensions", []))
        self.recursive = self.config.get("recursive", True)
        self.encoding = self.config.get("encoding", "utf-8")

    # ── FileEntry helpers ───────────────────────────────────────────────────

    def _path_to_entry(self, path: Path) -> FileEntry:
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(str(path))
        return FileEntry(
            id=str(path.relative_to(self.base_path)) if path.is_relative_to(self.base_path) else str(path),
            name=path.name,
            size_bytes=stat.st_size,
            modified_at=path.stat().st_mtime,
            mime_type=mime_type,
        )

    def _is_readable_text(self, path: Path) -> bool:
        """Heuristic: text if extension is a known text type or not a known binary."""
        ext = path.suffix.lower()
        if self.allowed_extensions:
            return ext in self.allowed_extensions
        return ext in self.TEXT_EXTENSIONS or ext not in self.BINARY_EXTENSIONS

    # ── Core implementations ────────────────────────────────────────────────

    async def list_files(self, path: Optional[str] = None) -> list[FileEntry]:
        root = self.base_path if path is None else (self.base_path / path)
        if not root.exists():
            self._logger.warning(f"[filesystem] path does not exist: {root}")
            return []

        if not root.is_dir():
            return [self._path_to_entry(root)]

        entries = []
        for dir_suffix, dir_names, file_names in os.walk(root):
            dir_path = Path(dir_suffix)
            for name in file_names:
                file_path = dir_path / name
                if not self._is_readable_text(file_path):
                    continue
                try:
                    entries.append(self._path_to_entry(file_path))
                except OSError as e:
                    self._logger.warning(f"[filesystem] cannot stat {file_path}: {e}")

            if not self.recursive:
                break

        return entries

    async def fetch_content(self, file_id: str) -> str:
        """
        Read a text file from the filesystem.
        Binary files raise NotImplementedError (add parser support in Stage 2).
        """
        path = self.base_path / file_id
        if not path.exists():
            raise FileNotFoundError(f"[filesystem] file not found: {path}")

        ext = path.suffix.lower()
        if ext in self.BINARY_EXTENSIONS and ext not in self.TEXT_EXTENSIONS:
            raise NotImplementedError(
                f"[filesystem] binary file {file_id} requires a parser "
                f"(PDF/DOCX support is in Stage 2 ingestion pipeline)."
            )

        try:
            return path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            # Fallback: try latin-1 for mixed-encoding files
            return path.read_text(encoding="latin-1")

    async def test_connection(self) -> dict[str, Any]:
        """Check that base_path exists and is readable."""
        try:
            if not self.base_path.exists():
                return {"ok": False, "message": f"base_path does not exist: {self.base_path}"}
            if not self.base_path.is_dir():
                return {"ok": False, "message": f"base_path is not a directory: {self.base_path}"}
            # Count files
            files = await self.list_files()
            return {
                "ok": True,
                "message": f"base_path is accessible ({len(files)} files found)",
                "file_count": len(files),
            }
        except Exception as exc:
            return {"ok": False, "message": f"connection test failed: {exc}"}

    async def sync(self, file_ids: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Returns a manifest of files that should be queued for ingestion.
        Actual ingestion is handled by the Celery ingestion worker.
        """
        all_files = await self.list_files()
        to_sync = [f for f in all_files if file_ids is None or f.id in file_ids]
        return {
            "connector_type": self.connector_type,
            "files_queued": len(to_sync),
            "file_ids": [f.id for f in to_sync],
            "message": f"sync manifest prepared for {len(to_sync)} files",
        }
