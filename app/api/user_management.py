"""
User Management API — Stage 7 (Governance).

POST /api/users/me/delete   — GDPR "Right to be Forgotten" (soft delete + cascade)
DELETE /api/admin/users/{id} — Hard delete a user (admin only, actual row deletion)

ACL enforcement:
- Regular users can only request deletion of their own account.
- Only admins can perform hard deletes.
"""
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import AdminUser, get_current_user, get_audit_context
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.models.feedback import Feedback
from app.models.experiment import Experiment, ExperimentResult
from app.services.audit_service import audit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])


# ─────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────

class GdprDeletionSummary(BaseModel):
    """Summary of what was deleted for a GDPR deletion request."""
    user_id: str
    conversations_deleted: int = 0
    messages_deleted: int = 0
    feedback_deleted: int = 0
    experiment_results_deleted: int = 0
    experiments_deleted: int = 0
    user_soft_deleted: bool = False


class GdprDeletionResponse(BaseModel):
    message: str
    summary: GdprDeletionSummary


class HardDeleteResponse(BaseModel):
    message: str
    user_id: str


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

async def _cascade_delete_user_data(
    db: AsyncSession,
    user_id: str,
) -> GdprDeletionSummary:
    """
    Delete all data associated with a user (GDPR flow).
    This is a soft delete on the user row itself (sets is_deleted=True, deleted_at=NOW())
    but hard deletes related rows.

    Returns a summary of what was deleted.
    """
    summary = GdprDeletionSummary(user_id=user_id)

    # 1. Count + delete messages via conversations
    conv_result = await db.execute(
        select(Conversation.id).where(Conversation.user_id == user_id)
    )
    conversation_ids = [r for (r,) in conv_result.fetchall()]

    if conversation_ids:
        msg_result = await db.execute(
            delete(Message).where(Message.conversation_id.in_(conversation_ids))
        )
        summary.messages_deleted = msg_result.rowcount or 0

    # 2. Count + delete conversations
    conv_del = await db.execute(
        delete(Conversation).where(Conversation.user_id == user_id)
    )
    summary.conversations_deleted = conv_del.rowcount or 0

    # 3. Count + delete feedback
    fb_result = await db.execute(
        delete(Feedback).where(Feedback.user_id == user_id)
    )
    summary.feedback_deleted = fb_result.rowcount or 0

    # 4. Count + delete experiment results
    exp_result = await db.execute(
        select(Experiment.id).where(Experiment.created_by == user_id)
    )
    experiment_ids = [r for (r,) in exp_result.fetchall()]

    if experiment_ids:
        er_del = await db.execute(
            delete(ExperimentResult).where(ExperimentResult.experiment_id.in_(experiment_ids))
        )
        summary.experiment_results_deleted = er_del.rowcount or 0

    # 5. Count + delete experiments
    if experiment_ids:
        exp_del = await db.execute(
            delete(Experiment).where(Experiment.id.in_(experiment_ids))
        )
        summary.experiments_deleted = exp_del.rowcount or 0

    # 6. Soft-delete the user row
    user_result = await db.execute(
        select(User).where(User.id == user_id, User.is_deleted == False)  # noqa: E712
    )
    user_row = user_result.scalar_one_or_none()
    if user_row:
        user_row.is_deleted = True
        user_row.deleted_at = datetime.utcnow()
        user_row.is_active = False
        summary.user_soft_deleted = True

    await db.commit()
    return summary


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@router.post("/me/delete", response_model=GdprDeletionResponse)
async def gdpr_delete_my_account(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    GDPR "Right to be Forgotten" endpoint.

    Allows any authenticated user to request deletion of all their personal data.
    This performs:
      1. Soft-delete of the user account (is_deleted=True, deleted_at=NOW())
      2. Hard-delete of all their conversations and messages
      3. Hard-delete of all their feedback entries
      4. Hard-delete of all their experiments and experiment results

    Returns a summary of what was deleted. This action is irreversible.
    """
    ip_address, user_agent = get_audit_context(request)

    logger.info(f"GDPR deletion request from user {current_user.id} ({current_user.email})")

    # Audit the deletion request
    await audit(
        db=db,
        user=current_user,
        action="gdpr_delete_account",
        resource="user",
        resource_id=current_user.id,
        details={"email": current_user.email},
        ip_address=ip_address,
        user_agent=user_agent,
    )

    summary = await _cascade_delete_user_data(db, user_id=current_user.id)

    return GdprDeletionResponse(
        message=(
            f"Your account and all associated data have been scheduled for deletion. "
            "This action is irreversible."
        ),
        summary=summary,
    )


@router.delete("/admin/users/{user_id}", response_model=HardDeleteResponse)
async def admin_hard_delete_user(
    request: Request,
    user_id: str,
    current_user: AdminUser = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Permanently delete a user and all their data (hard delete).

    This is a privileged admin action. It cascades to:
      - All conversations and messages
      - All feedback
      - All experiments and results

    The user row itself is hard-deleted from the database.
    Use this only when absolutely required (e.g., complete account removal for
    legal/compliance reasons).
    """
    ip_address, user_agent = get_audit_context(request)

    # Verify target user exists
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    target_user = result.scalar_one_or_none()

    if target_user is None:
        raise HTTPException(404, "User not found")

    # Prevent admin from deleting themselves
    if target_user.id == current_user.id:
        raise HTTPException(400, "Cannot delete your own account via this endpoint")

    # Audit before deletion
    await audit(
        db=db,
        user=current_user,
        action="hard_delete_user",
        resource="user",
        resource_id=user_id,
        details={
            "deleted_email": target_user.email,
            "deleted_role": target_user.role,
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )

    # Cascade delete all user data
    summary = await _cascade_delete_user_data(db, user_id=user_id)

    # Finally, hard-delete the user row itself
    await db.execute(delete(User).where(User.id == user_id))
    await db.commit()

    logger.info(
        f"Admin {current_user.email} hard-deleted user {user_id} "
        f"(convs={summary.conversations_deleted}, "
        f"msgs={summary.messages_deleted}, "
        f"fb={summary.feedback_deleted}, "
        f"exps={summary.experiments_deleted})"
    )

    return HardDeleteResponse(
        message="User and all associated data permanently deleted.",
        user_id=user_id,
    )
