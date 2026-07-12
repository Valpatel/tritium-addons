# SMS Gateway Bridge — comms addon

> Parent: [tritium-addons](../README.md) · Archetype: [COMMS-BRIDGES.md](../COMMS-BRIDGES.md)

**What it's for:** Send alert texts to phone numbers via Twilio or a local GSM modem.

**Status:** **Stub.** `SMSGatewayPlugin.send_message()` logs "Would send" and returns `True` without connecting. It is loaded and `start()`ed by the dispatcher, but never actually invoked — the fan-out only calls plugins that define `send_message_sync`, and this one doesn't (see [COMMS-BRIDGES.md → D3](../COMMS-BRIDGES.md#reality-check)). No tests.

> Identifier note: directory `sms_gateway/`, manifest `id = "sms-gateway"`, route prefix `/api/comms/sms-gateway`. The `addon-index.json` catalog lists it as `sms_gateway` — a known id inconsistency (COMMS-BRIDGES.md).

## Wire

- **Entry class:** `SMSGatewayPlugin` in [`plugin.py`](plugin.py) — a bare, duck-typed
  comms-bridge plugin (no base class), *not* a `SensorAddon`. See
  [COMMS-BRIDGES.md](../COMMS-BRIDGES.md) for the shared contract.
- **Loaded by:** the `CommsDispatcher`
  (`tritium-sc/src/app/comms_dispatcher.py`) — it calls `configure()` →
  `start()` and fans out `notification:new` events above a severity threshold.
- **Routes:** [`routes.py`](routes.py) declares `/api/comms/sms-gateway`
  (`/config`, `/status`, `/test`, `/send`) — **not currently mounted** by the
  running app (COMMS-BRIDGES.md D2).
- **Frontend:** [`frontend/`](frontend/) — a `comms-container` tab.

## Config keys (`SMSGatewayPlugin._config`)

| Key | Meaning |
|-----|---------|
| `provider` | backend provider (e.g. twilio / modem / iridium) |
| `account_sid` | provider account SID |
| `auth_token` | provider auth token |
| `from_number` | sender phone number |
| `alert_numbers` | recipient phone number(s) |
| `enabled` | master switch — real I/O only fires when true |

## To make it real

Use the `twilio` SDK, or AT commands over a serial GSM modem. Then **add a synchronous `send_message_sync(text, payload)`** — without it the dispatcher loads and `start()`s the plugin but never calls it. Follow [COMMS-BRIDGES.md → Making a stub real](../COMMS-BRIDGES.md#making-a-stub-real).

---

AGPL-3.0 | Copyright 2026 Valpatel Software LLC
