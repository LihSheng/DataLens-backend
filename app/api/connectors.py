"""
Connectors API — configure and manage external data source connectors.

Endpoints (admin only):
- GET    /api/connectors                        — list all connectors
- POST   /api/connectors                        — add connector config
- PATCH  /api/connectors/{connector_type}       — update connector config
- DELETE /api/connectors/{connector_type}        — remove connector
- POST   /api/connectors/{connector_type}/test   — test connection
- POST   /api/connectors/{connector_type}/sync   — trigger sync
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import require_admin
from app.models.user import User
from app.models.connector_config import ConnectorConfig
from app.connectors import ConnectorRegistry, FileEntry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/connectors", tags=["connectors"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class ConnectorConfigCreate(BaseModel):
    connector_type: str = Field(..., description="'filesystem', 's3', 'googledrive', 'notion'")
    config: dict[str, Any] = Field(..., description="Connector configuration (credentials, paths, etc.)")
    is_active: bool = Field(False, description="Enable this connector immediately")


class ConnectorConfigUpdate(BaseModel):
    config: dict[str, Any] = Field(None, description="Partial or full config update")
    is_active: bool = Field(None, description="Enable/disable connector")


class ConnectorConfigResponse(BaseModel):
    id: str
    connector_type: str
    config: dict[str, Any]  # sanitised — secrets masked by connector
    is_active: bool
    created_by: str | None
    created_at: Any
    updated_at: Any


class ConnectionTestResult(BaseModel):
    ok: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SyncResult(BaseModel):
    connector_type: str
    files_queued: int
    file_ids: list[str]
    message: str


class FileEntryResponse(BaseModel):
    id: str
    name: str
    size_bytes: int | None
    modified_at: str | None
    mime_type: str | None


# ── Connector instance helper ─────────────────────────────────────────────────

async def _load_connector_config(
    connector_type: str,
    db: AsyncSession,
) -> ConnectorConfig:
    """Load a ConnectorConfig row, raising 404 if not found."""
    result = await db.execute(
        select(ConnectorConfig).where(ConnectorConfig.connector_type == connector_type)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector '{connector_type}' not found",
        )
    return row


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ConnectorConfigResponse])
async def list_connectors(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    List all configured connectors (admin only).

    Secrets are masked in the returned config dict.
    """
    result = await db.execute(select(ConnectorConfig))
    rows = result.scalars().all()

    responses = []
    for row in rows:
        try:
            conn = ConnectorRegistry.get(row.connector_type, dict(row.config))
            safe_config = conn._safe_config()
        except Exception:
            # Unknown connector type — return config as-is (mask any string values)
            safe_config = {
                k: ("***" if any(s in k.lower() for s in ["key", "token", "secret", "password"]) else v)
                for k, v in dict(row.config).items()
            }

        responses.append(ConnectorConfigResponse(
            id=row.id,
            connector_type=row.connector_type,
            config=safe_config,
            is_active=row.is_active,
            created_by=row.created_by,
            created_at=row.created_at,
            updated_at=row.updated_at,
        ))

    return responses


@router.post("", response_model=ConnectorConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_connector(
    payload: ConnectorConfigCreate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Add a new connector configuration (admin only).

    connector_type must be unique; use PATCH to update an existing one.
    """
    # Validate connector type is registered
    try:
        ConnectorRegistry.get(payload.connector_type, payload.config)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid connector type '{payload.connector_type}': {exc}",
        )

    # Check for duplicate
    result = await db.execute(
        select(ConnectorConfig).where(ConnectorConfig.connector_type == payload.connector_type)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Connector '{payload.connector_type}' already exists; use PATCH to update",
        )

    row = ConnectorConfig(
        connector_type=payload.connector_type,
        config=payload.config,
        is_active=payload.is_active,
        created_by=admin.id,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    return ConnectorConfigResponse(
        id=row.id,
        connector_type=row.connector_type,
        config=ConnectorRegistry.get(row.connector_type, dict(row.config))._safe_config(),
        is_active=row.is_active,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.patch("/{connector_type}", response_model=ConnectorConfigResponse)
async def update_connector(
    connector_type: str,
    payload: ConnectorConfigUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Update an existing connector's config and/or active status (admin only)."""
    row = await _load_connector_config(connector_type, db)

    if payload.config is not None:
        row.config = payload.config
    if payload.is_active is not None:
        row.is_active = payload.is_active

    await db.commit()
    await db.refresh(row)

    conn = ConnectorRegistry.get(row.connector_type, dict(row.config))
    return ConnectorConfigResponse(
        id=row.id,
        connector_type=row.connector_type,
        config=conn._safe_config(),
        is_active=row.is_active,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.delete("/{connector_type}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connector(
    connector_type: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Remove a connector configuration (admin only)."""
    row = await _load_connector_config(connector_type, db)
    await db.delete(row)
    await db.commit()
    return None


@router.post("/{connector_type}/test", response_model=ConnectionTestResult)
async def test_connector(
    connector_type: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Test connectivity for a connector (admin only)."""
    row = await _load_connector_config(connector_type, db)

    try:
        conn = ConnectorRegistry.get(row.connector_type, dict(row.config))
        result = await conn.test_connection()
        return ConnectionTestResult(**result)
    except Exception as exc:
        logger.exception(f"[connectors] test_connection failed for {connector_type}")
        return ConnectionTestResult(
            ok=False,
            message=f"Connection test failed: {exc}",
            details={},
        )


@router.post("/{connector_type}/sync", response_model=SyncResult)
async def sync_connector(
    connector_type: str,
    file_ids: list[str] | None = None,  # Optional: sync specific files
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Trigger a content sync for a connector (admin only).

    Returns a manifest of files queued for ingestion.
    Actual ingestion is handled by the Celery ingestion worker.
    """
    row = await _load_connector_config(connector_type, db)

    try:
        conn = ConnectorRegistry.get(row.connector_type, dict(row.config))
        result = await conn.sync(file_ids=file_ids)
        return SyncResult(**result)
    except Exception as exc:
        logger.exception(f"[connectors] sync failed for {connector_type}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync failed: {exc}",
        ) from exc


@router.get("/{connector_type}/files", response_model=list[FileEntryResponse])
async def list_connector_files(
    connector_type: str,
    path: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """List files available in a connector (admin only)."""
    row = await _load_connector_config(connector_type, db)

    try:
        conn = ConnectorRegistry.get(row.connector_type, dict(row.config))
        files = await conn.list_files(path=path)
        return [
            FileEntryResponse(
                id=f.id,
                name=f.name,
                size_bytes=f.size_bytes,
                modified_at=f.modified_at.isoformat() if f.modified_at else None,
                mime_type=f.mime_type,
            )
            for f in files
        ]
    except Exception as exc:
        logger.exception(f"[connectors] list_files failed for {connector_type}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files: {exc}",
        ) from exc
