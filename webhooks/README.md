# Webhooks Bridge — comms addon

> Parent: [tritium-addons](../README.md) · Archetype: [COMMS-BRIDGES.md](../COMMS-BRIDGES.md)

**What it's for:** Generic relay — POST Tritium notifications (above a severity threshold) as JSON to any URL.

**Status:** **Functional when configured.** The one comms addon with a real send path: `WebhooksPlugin.send_message_sync()` does `httpx.post(url, json=…)` and returns `True` on a 2xx (`plugin.py:95-132`). Inert until `WEBHOOKS_URL` + `WEBHOOKS_ENABLED` are set, and it is on the live notification dispatch path. No tests yet.

> Manifest note: `tritium_addon.toml` still declares `version 0.1.0` / `chat_bridge, alert_relay`, but the code reports `version 0.2.0` / `{alert_relay, event_bridge}` — and `addon-index.json` still files it as `stub`. See [COMMS-BRIDGES.md](../COMMS-BRIDGES.md#reality-check).

## Wire

- **Entry class:** `WebhooksPlugin` in [`plugin.py`](plugin.py) — a bare, duck-typed
  comms-bridge plugin (no base class), *not* a `SensorAddon`. See
  [COMMS-BRIDGES.md](../COMMS-BRIDGES.md) for the shared contract.
- **Loaded by:** the `CommsDispatcher`
  (`tritium-sc/src/app/comms_dispatcher.py`) — it calls `configure()` →
  `start()` and fans out `notification:new` events above a severity threshold.
- **Routes:** [`routes.py`](routes.py) declares `/api/comms/webhooks`
  (`/config`, `/status`, `/test`, `/send`) — **not currently mounted** by the
  running app (COMMS-BRIDGES.md D2).
- **Frontend:** [`frontend/`](frontend/) — a `comms-container` tab.

## Config keys (`WebhooksPlugin._config`)

| Key | Meaning |
|-----|---------|
| `webhook_url` | incoming webhook URL |
| `secret` | optional shared secret added to the payload |
| `events` | which event types to relay |
| `format` | payload format |
| `enabled` | master switch — real I/O only fires when true |
| `timeout_s` | HTTP request timeout (seconds) |

## To make it real

Extend `send_message_sync` for auth headers, retries, or payload signing.

---

AGPL-3.0 | Copyright 2026 Valpatel Software LLC
