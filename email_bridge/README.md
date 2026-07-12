# Email Bridge — comms addon

> Parent: [tritium-addons](../README.md) · Archetype: [COMMS-BRIDGES.md](../COMMS-BRIDGES.md)

**What it's for:** SMTP relay for alert digests and daily reports.

**Status:** **Stub, and currently NOT loaded.** `EmailPlugin.send_message()` logs "Would send" and returns `True` without connecting (a pure stub). On top of that it is **orphaned**: the `CommsDispatcher` looks for a directory named `email`, but this addon lives in `email_bridge/` (renamed to avoid shadowing the Python stdlib). So it never even loads. See [COMMS-BRIDGES.md → D1](../COMMS-BRIDGES.md#reality-check). No tests.

> Identifier note: directory `email_bridge/`, manifest `id = "email"`, route prefix `/api/comms/email`. The `addon-index.json` catalog lists it as `email_bridge` — a known id inconsistency (COMMS-BRIDGES.md).

## Wire

- **Entry class:** `EmailPlugin` in [`plugin.py`](plugin.py) — a bare, duck-typed
  comms-bridge plugin (no base class), *not* a `SensorAddon`. See
  [COMMS-BRIDGES.md](../COMMS-BRIDGES.md) for the shared contract.
- **Loaded by:** the `CommsDispatcher`
  (`tritium-sc/src/app/comms_dispatcher.py`) — it calls `configure()` →
  `start()` and fans out `notification:new` events above a severity threshold.
- **Routes:** [`routes.py`](routes.py) declares `/api/comms/email`
  (`/config`, `/status`, `/test`, `/send`) — **not currently mounted** by the
  running app (COMMS-BRIDGES.md D2).
- **Frontend:** [`frontend/`](frontend/) — a `comms-container` tab.

## Config keys (`EmailPlugin._config`)

| Key | Meaning |
|-----|---------|
| `smtp_host` | SMTP server host |
| `smtp_port` | SMTP server port |
| `username` | SMTP username |
| `password` | SMTP password |
| `from_addr` | sender address |
| `to_addrs` | recipient address(es) |
| `enabled` | master switch — real I/O only fires when true |

## To make it real

Use the stdlib `smtplib` + `email.message` (no third-party dep needed). Then **add a synchronous `send_message_sync(text, payload)`** — without it the dispatcher loads and `start()`s the plugin but never calls it. Follow [COMMS-BRIDGES.md → Making a stub real](../COMMS-BRIDGES.md#making-a-stub-real).

---

AGPL-3.0 | Copyright 2026 Valpatel Software LLC
