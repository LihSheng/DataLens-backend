"""
Retention Worker — Stage 7 (Governance).

Celery beat task that runs daily to enforce retention policies.
Deletes conversations, messages, and feedback older than the configured
retention period.

Configuration is read from the `retention_policy` DB table.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="retention.run")
def retention_task(self) -> Dict[str, Any]:
    """
    Daily Celery task: enforce all active retention policies.

    For each active policy in the retention_policy table:
      1. Determine the cutoff date (now - retention_days).
      2. Delete related rows older than the cutoff.

    Cascade order (respects FK constraints):
      messages  → conversations → feedback → experiment_results → experiments

    Returns a summary dict with counts of deleted rows per resource.
    """
    import asyncio
    from sqlalchemy import delete, select, func, and_

    async def _run() -> Dict[str, Any]:
        from app.db.session import AsyncSessionLocal
        from app.models.retention_policy import RetentionPolicy
        from app.models.conversation import Conversation, Message
        from app.models.feedback import Feedback
        from app.models.experiment import Experiment, ExperimentResult

        summary: Dict[str, Any] = {
            "run_at": datetime.utcnow().isoformat(),
            "policies_processed": [],
            "errors": [],
        }

        async with AsyncSessionLocal() as db:
            # Load all active retention policies
            result = await db.execute(
                select(RetentionPolicy).where(RetentionPolicy.is_active == True)  # noqa: E712
            )
            policies = list(result.scalars().all())

        if not policies:
            logger.info("No active retention policies found — nothing to do.")
            return summary

        for policy in policies:
            cutoff = datetime.utcnow() - timedelta(days=policy.retention_days)
            policy_result: Dict[str, Any] = {
                "resource": policy.resource,
                "retention_days": policy.retention_days,
                "cutoff": cutoff.isoformat(),
                "deleted": {},
            }

            try:
                async with AsyncSessionLocal() as db:
                    if policy.resource == "conversations":
                        # Find old conversation IDs
                        conv_result = await db.execute(
                            select(Conversation.id).where(
                                Conversation.created_at < cutoff
                            )
                        )
                        old_conv_ids = [r for (r,) in conv_result.fetchall()]

                        if old_conv_ids:
                            # Delete messages in those conversations
                            msg_del = await db.execute(
                                delete(Message).where(
                                    Message.conversation_id.in_(old_conv_ids)
                                )
                            )
                            policy_result["deleted"]["messages"] = msg_del.rowcount or 0

                            # Delete conversations
                            conv_del = await db.execute(
                                delete(Conversation).where(
                                    Conversation.id.in_(old_conv_ids)
                                )
                            )
                            policy_result["deleted"]["conversations"] = conv_del.rowcount or 0
                        else:
                            policy_result["deleted"]["conversations"] = 0
                            policy_result["deleted"]["messages"] = 0

                    elif policy.resource == "messages":
                        # Delete messages older than cutoff (not cascaded from conv)
                        msg_del = await db.execute(
                            delete(Message).where(Message.created_at < cutoff)
                        )
                        policy_result["deleted"]["messages"] = msg_del.rowcount or 0

                    elif policy.resource == "feedback":
                        fb_del = await db.execute(
                            delete(Feedback).where(Feedback.created_at < cutoff)
                        )
                        policy_result["deleted"]["feedback"] = fb_del.rowcount or 0

                    await db.commit()

                logger.info(
                    f"Retention policy '{policy.resource}': "
                    f"deleted {policy_result['deleted']}"
                )

            except Exception as e:
                logger.error(
                    f"Retention policy '{policy.resource}' failed: {e}",
                    exc_info=True,
                )
                policy_result["error"] = str(e)
                summary["errors"].append(policy_result)

            summary["policies_processed"].append(policy_result)

        return summary

    result = asyncio.run(_run())
    logger.info(f"Retention task completed: {result}")
    return result
