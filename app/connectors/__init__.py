"""Connectors module — pluggable external data source integrations."""

from app.connectors.base import ConnectorBase, FileEntry
from app.connectors.filesystem import FilesystemConnector
from app.connectors.registry import ConnectorRegistry, ConnectorRegistryError

__all__ = [
    "ConnectorBase",
    "FileEntry",
    "FilesystemConnector",
    "ConnectorRegistry",
    "ConnectorRegistryError",
]
