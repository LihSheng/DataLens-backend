"""
ConnectorRegistry — factory that instantiates concrete ConnectorBase subclasses
by connector_type name.

Usage:
    connector = ConnectorRegistry.get("filesystem", config_dict)
    connector = ConnectorRegistry.get("s3", config_dict)
"""
from typing import Any

from app.connectors.base import ConnectorBase
from app.connectors.filesystem import FilesystemConnector

_REGISTRY: dict[str, type[ConnectorBase]] = {
    "filesystem": FilesystemConnector,
    # S3, Google Drive, Notion — implement in future stages
    # "s3": S3Connector,
    # "googledrive": GoogleDriveConnector,
    # "notion": NotionConnector,
}


class ConnectorRegistryError(ValueError):
    """Raised when an unknown connector_type is requested."""
    pass


class ConnectorRegistry:
    """Factory for connector instances."""

    @classmethod
    def register(cls, connector_type: str, cls_: type[ConnectorBase]) -> None:
        """Register a new connector type (for future use)."""
        _REGISTRY[connector_type] = cls_

    @classmethod
    def get(cls, connector_type: str, config: dict[str, Any]) -> ConnectorBase:
        """
        Instantiate the connector for the given type.

        Args:
            connector_type: The connector type string, e.g. "filesystem".
            config: The JSONB config dict from the connectors_config table.

        Returns:
            An instance of the appropriate ConnectorBase subclass.

        Raises:
            ConnectorRegistryError: If connector_type is not registered.
        """
        if connector_type not in _REGISTRY:
            available = list(_REGISTRY.keys())
            raise ConnectorRegistryError(
                f"Unknown connector type '{connector_type}'. "
                f"Available: {available}"
            )
        return _REGISTRY[connector_type](config)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return list of registered connector type names."""
        return list(_REGISTRY.keys())
