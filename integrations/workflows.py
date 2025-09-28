"""Workflow integration stubs for meeting outcomes."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import httpx

LOGGER = logging.getLogger(__name__)


async def run_post_meeting_workflows(summary: str, actions: List[Dict[str, Any]]) -> None:
    """Dispatch meeting outcomes to downstream systems.

    Currently supports:
      * Slack webhook via MEETING_SLACK_WEBHOOK_URL env var.
      * JSONL audit log written to MEETING_LOG_PATH (if set).
    Extend as needed for ticketing or calendar integrations.
    """

    await _notify_slack(summary, actions)
    await _append_local_log(summary, actions)


async def _notify_slack(summary: str, actions: List[Dict[str, Any]]) -> None:
    webhook = os.getenv("MEETING_SLACK_WEBHOOK_URL")
    if not webhook:
        LOGGER.debug("Slack webhook not configured; skipping notification")
        return

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Meeting Summary*\n{summary}"},
        }
    ]

    if actions:
        action_lines = [
            f"• *{item.get('task', 'Task')}* — {item.get('assignee', 'Unassigned')} (due {item.get('due', 'TBD')})"
            for item in actions
        ]
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Action Items*\n" + "\n".join(action_lines),
                },
            }
        )

    payload = {"blocks": blocks}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(webhook, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - runtime logging only
            LOGGER.error("Slack notification failed: %s", exc)


async def _append_local_log(summary: str, actions: List[Dict[str, Any]]) -> None:
    log_path = os.getenv("MEETING_LOG_PATH")
    if not log_path:
        return

    record = {"summary": summary, "actions": actions}
    try:
        with open(log_path, "a", encoding="utf8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")
    except OSError as exc:  # pragma: no cover - runtime logging only
        LOGGER.error("Failed to append meeting log: %s", exc)

