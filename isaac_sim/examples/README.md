# Isaac Sim connector — examples

Run recipes live in the addon [`../README.md`](../README.md). In short, on the
render host (Isaac's Python 3.12 venv, `OMNI_KIT_ACCEPT_EULA=YES`):

1. `usd_scene_builder.py --scene-url http://<sc>:8000/api/gis/scene3d?bbox=...&ao=dublin --out dublin.usd`
2. `render_city.py --usd dublin.usd --out dublin.png`
3. `camera_server.py --source isaac --scene dublin.usd --port 8100` → register in SC as an `mjpeg` camera.

No-GPU equivalents for CI (plain python3): `usd_scene_builder.py --validate --obj`,
`camera_server.py --selftest`. See [`../tests/test_no_gpu.py`](../tests/test_no_gpu.py).
