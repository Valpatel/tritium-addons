# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Consumer-side Isaac clients — the mirror image of ``connectors/``.

The two tiers exist because they run in different interpreters, and confusing
them is how dependency bleed starts:

``connectors/``
    Run **inside Isaac's own python** on the GPU box.  They may import
    ``isaacsim``/``pxr`` and must **never** import tritium — Isaac's
    interpreter has no tritium installed, and dragging pydantic/paho into the
    Isaac process is exactly the coupling the isaac-bridge rule forbids.  They
    publish over neutral transports (MJPEG, JSON/HTTP, TCP).

``clients/``  (this package)
    Run **anywhere else** — a laptop, the SC host, CI.  They consume those
    transports over the network, never import ``isaacsim``, and **are** free to
    import ``tritium_lib`` for shared models and maths.  That is the whole
    point: the frame conversion, the target model and the codecs stay in one
    tested library instead of being re-derived per connector.

Rule of thumb: if it needs a GPU it is a connector; if it needs tritium it is a
client.  Nothing may need both.
"""
