"""Multi-channel alert dispatcher for Telegram bots and HMAC-signed Webhooks."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx

from orbitguard.config import settings

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """Dispatches real-time conjunction alerts to Telegram and ground station webhooks."""

    def __init__(
        self,
        telegram_token: str = settings.TELEGRAM_BOT_TOKEN,
        webhook_secret: str = settings.WEBHOOK_SECRET,
    ):
        self.telegram_token = telegram_token
        self.webhook_secret = webhook_secret

    async def send_telegram_alert(
        self,
        chat_id: str,
        text: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Dispatch a markdown alert to an operator Telegram chat."""
        if not self.telegram_token or not chat_id:
            logger.info("Telegram alert bypassed (no bot token or chat ID configured).")
            return {
                "status": "bypassed_standby",
                "message": "Telegram token not provided (Standby Protocol active)",
                "chat_id": chat_id,
            }

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    return {"status": "delivered", "response": res.json()}
                else:
                    return {
                        "status": "failed",
                        "status_code": res.status_code,
                        "error": res.text,
                    }
        except Exception as ex:
            logger.warning(f"Failed to deliver Telegram alert: {ex}")
            return {"status": "error", "error": str(ex)}

    async def send_webhook_alert(
        self,
        webhook_url: str,
        payload: Dict[str, Any],
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Dispatch HMAC-SHA256 signed JSON payload to an operator webhook endpoint."""
        if not webhook_url:
            return {"status": "bypassed", "message": "No webhook URL supplied"}

        body_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        timestamp = str(int(time.time()))

        # Compute signature: HMAC-SHA256(secret, timestamp + "." + body)
        sig_data = timestamp.encode("utf-8") + b"." + body_bytes
        signature = hmac.new(
            self.webhook_secret.encode("utf-8"),
            sig_data,
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-OrbitGuard-Timestamp": timestamp,
            "X-OrbitGuard-Signature": f"sha256={signature}",
            "User-Agent": "OrbitGuard-AI-Copilot/1.0",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(webhook_url, content=body_bytes, headers=headers)
                return {
                    "status": "delivered" if res.status_code in (200, 201, 202, 204) else "failed",
                    "status_code": res.status_code,
                    "response_text": res.text[:200],
                }
        except Exception as ex:
            logger.warning(f"Webhook delivery exception: {ex}")
            return {"status": "error", "error": str(ex)}
