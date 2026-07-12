# AGENTS.md — tritium-addons

A **submodule** of the [`Valpatel/tritium`](https://github.com/Valpatel/tritium)
superproject. Active branch: `dev`. The parent repo pins this repo via a gitlink
and bumps it after your commits land here.

> **Read the parent's cross-repo manual first:**
> <https://github.com/Valpatel/tritium/blob/main/AGENTS.md> — commit order,
> push order (submodules before the parent), the pre-push privacy + markdown
> gate, and the merge hazards.

## What belongs here

**Optional connectors to EXTERNAL systems/tools/sensors** that the core runs fine
without, and that drag in heavy or specialized dependencies (Isaac Sim, HackRF,
Meshtastic, a comms bridge). Each addon is a folder with a `tritium_addon.toml`
manifest. The core (lib + sc + edge) must never hard-depend on an addon.

If the core needs it, or it's light + reusable, it belongs in **lib**, not here.
Functional today: `hackrf`, `meshtastic`. Comms connectors (discord, telegram,
slack, …) are stubs until their transport is wired.

> **`../tritium-addon-priv/`** is a SEPARATE PRIVATE repo (e.g. `nav_pro`),
> gitignored and cloned alongside the parent. It is NOT part of this repo. Never
> copy its code in; this public repo only documents the load-path slot.

## Test

```bash
pytest                   # per-addon tests; run inside the addon folder
```

## Style

No Co-Authored-By. AGPL-3.0 / Matthew Valancy / Valpatel Software LLC. The addon
SDK boundary is deliberately Apache-2.0 so closed-source addons can subclass it.
Work on `dev`; `main` advances only via a reviewed dev→main PR.
