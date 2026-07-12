# Signal Bridge — comms addon

> Parent: [tritium-addons](../README.md) · Archetype: [COMMS-BRIDGES.md](../COMMS-BRIDGES.md)

**What it's for:** Secure end-to-end-encrypted messaging for field operators via signal-cli.

**Status:** **Stub, and currently NOT loaded.** `SignalPlugin.send_message()` logs "Would send" and returns `True` without connecting (a pure stub). On top of that it is **orphaned**: the `CommsDispatcher` looks for a directory named `signal`, but this addon lives in `signal_bridge/` (renamed to avoid shadowing the Python stdlib). So it never even loads. See [COMMS-BRIDGES.md → D1](../COMMS-BRIDGES.md#reality-check). No tests.

> Identifier note: directory `signal_bridge/`, manifest `id = "signal"`, route prefix `/api/comms/signal`. The `addon-index.json` catalog lists it as `signal_bridge` — a known id inconsistency (COMMS-BRIDGES.md).

## Wire

- **Entry class:** `SignalPlugin` in [`plugin.py`](plugin.py) — a bare, duck-typed
  comms-bridge plugin (no base class), *not* a `SensorAddon`. See
  [COMMS-BRIDGES.md](../COMMS-BRIDGES.md) for the shared contract.
- **Loaded by:** the `CommsDispatcher`
  (`tritium-sc/src/app/comms_dispatcher.py`) — it calls `configure()` →
  `start()` and fans out `notification:new` events above a severity threshold.
- **Routes:** [`routes.py`](routes.py) declares `/api/comms/signal`
  (`/config`, `/status`, `/test`, `/send`) — **not currently mounted** by the
  running app (COMMS-BRIDGES.md D2).
- **Frontend:** [`frontend/`](frontend/) — a `comms-container` tab.

## Config keys (`SignalPlugin._config`)

| Key | Meaning |
|-----|---------|
| `signal_cli_path` | path to the signal-cli binary |
| `phone_number` | registered Signal number |
| `group_id` | target group ID |
| `bridge_alerts` | relay Tritium alerts (on/off or filter) |
| `enabled` | master switch — real I/O only fires when true |

## To make it real

Shell out to `signal-cli` (JVM CLI) or its JSON-RPC daemon. Then **add a synchronous `send_message_sync(text, payload)`** — without it the dispatcher loads and `start()`s the plugin but never calls it. Follow [COMMS-BRIDGES.md → Making a stub real](../COMMS-BRIDGES.md#making-a-stub-real).

---

AGPL-3.0 | Copyright 2026 Valpatel Software LLC
