"""
Stripe scaffold for future credits/subscriptions.

Streamlit Cloud cannot receive webhooks by itself — deploy a small FastAPI/Cloud Function
or a Next.js route and forward events to your backend. See README.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from polymarket_alpha.config import get_config


def stripe_configured() -> bool:
    cfg = get_config()
    return bool(cfg.stripe_secret_key and cfg.stripe_price_credits_id)


def create_checkout_session_stub(
    success_url: str,
    cancel_url: str,
    customer_email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns a dict describing what a real Checkout Session would use.
    If `stripe` is installed and keys are set, creates a live session.
    """
    cfg = get_config()
    if not stripe_configured():
        return {
            "ok": False,
            "error": "Set STRIPE_SECRET_KEY and STRIPE_PRICE_CREDITS_ID",
            "hint": "Webhooks require a separate HTTPS endpoint (not Streamlit alone).",
        }
    try:
        import stripe

        stripe.api_key = cfg.stripe_secret_key
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{"price": cfg.stripe_price_credits_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            customer_email=customer_email,
        )
        return {"ok": True, "url": session.url, "id": session.id}
    except ImportError:
        return {
            "ok": False,
            "error": "Install stripe: pip install stripe",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def construct_webhook_event_stub(payload: bytes, sig_header: str) -> Dict[str, Any]:
    """Verify Stripe webhook signature when STRIPE_WEBHOOK_SECRET is set."""
    cfg = get_config()
    if not cfg.stripe_webhook_secret:
        return {"ok": False, "error": "STRIPE_WEBHOOK_SECRET not set"}
    try:
        import stripe

        event = stripe.Webhook.construct_event(
            payload, sig_header, cfg.stripe_webhook_secret
        )
        return {"ok": True, "type": event.get("type"), "id": event.get("id")}
    except ImportError:
        return {"ok": False, "error": "stripe package not installed"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
