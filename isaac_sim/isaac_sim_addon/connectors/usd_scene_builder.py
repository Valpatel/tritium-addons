#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Neutral Scene3D -> a USD stage for NVIDIA Isaac Sim.

The Isaac-side writer of the map -> 3D-scene pipeline.  It consumes a neutral
:class:`tritium_lib.geo.scene3d.Scene3D` (produced by tritium-lib from shared
GIS layers, served by tritium-sc at ``GET /api/gis/scene3d``) and materializes
it as USD prims — terrain heightfield mesh + extruded building meshes, plus any
road/water meshes — so Isaac renders a faithful 3D model of a real Tritium AO.
The Isaac camera (``examples/isaac-camera/isaac_camera_server.py --scene
dublin.usd``) and the robot dogs then operate in that twin.

Render realism
--------------
Every mesh is bound to a per-kind ``UsdPreviewSurface`` material so the twin
reads correctly in Isaac's RTX renderer without any hand-authoring:

  * terrain  → grass/earth green, rough (0.95)
  * building → concrete grey, semi-rough (0.72)
  * road     → asphalt dark grey, rough (0.85)
  * water    → blue, smooth (0.08) so it catches a specular sheen
  * anything else → neutral grey (kinds are handled generically)

If a mesh carries its own ``color`` it is honored (both as a bound material and
as the ``displayColor`` fallback for viewers that do not evaluate materials).
Meshes are flat-shaded (``subdivisionScheme = none``) and grouped under
``/World/<kind>s`` so the stage loads clean and layers can be toggled.

Separation (the standing rule): the geometry is computed in tritium-lib (pure
math, no USD).  This writer is the ONLY place pxr/USD is touched, and it lives
Isaac-side in examples.  tritium-sc never imports pxr; tritium-lib never imports
pxr.  The Scene3D JSON is the neutral contract across the seam.

Modes
-----
  * default (Isaac's python, pxr present): write ``<ao>.usd``.
  * ``--validate`` (plain python3, no pxr): load the JSON, assert it is
    well-formed (no degenerate meshes), print per-kind + AABB stats — a no-GPU
    gate.
  * ``--preview out.png`` (plain python3, matplotlib — no pxr, no GPU): render
    an oblique color-by-kind view so the twin can be eyeballed off-GPU.
  * ``--obj out.obj``: re-emit Wavefront OBJ (any DCC / viewer) — no pxr needed.

Input
-----
  * ``--scene-json path`` a Scene3D JSON file, or
  * ``--scene-url http://<sc-host>:8000/api/gis/scene3d?ao=dublin`` (fetched
    over the LAN with stdlib urllib — no tritium import).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request


# --- Per-kind material scheme --------------------------------------------------
# diffuse RGB (0..1), UsdPreviewSurface roughness + metallic.  Single source of
# truth shared by the USD writer (PBR material) and the matplotlib preview.
KIND_MATERIAL: dict[str, dict] = {
    "terrain":  {"color": (0.34, 0.44, 0.22), "roughness": 0.95, "metallic": 0.0},
    "building": {"color": (0.63, 0.63, 0.61), "roughness": 0.72, "metallic": 0.0},
    "road":     {"color": (0.16, 0.16, 0.18), "roughness": 0.85, "metallic": 0.0},
    "water":    {"color": (0.15, 0.34, 0.55), "roughness": 0.08, "metallic": 0.0},
}
DEFAULT_MATERIAL: dict = {"color": (0.50, 0.50, 0.50), "roughness": 0.6, "metallic": 0.0}

# Preview-only accents: cyan edges make buildings pop against the terrain sheet.
CYAN = (0.0, 0.94, 1.0)  # Tritium #00f0ff
KIND_EDGE: dict[str, tuple] = {"building": CYAN}


def kind_params(kind: str) -> dict:
    """Material params for a kind, generically neutral for unknown kinds."""
    return KIND_MATERIAL.get(kind, DEFAULT_MATERIAL)


def mesh_color(mesh: dict) -> tuple:
    """Mesh's own color if present, else its per-kind default diffuse color."""
    c = mesh.get("color")
    if c:
        return (float(c[0]), float(c[1]), float(c[2]))
    return tuple(kind_params(mesh.get("kind", "other"))["color"])


def load_scene(args) -> dict:
    if args.scene_json:
        with open(args.scene_json) as f:
            return json.load(f)
    if args.scene_url:
        with urllib.request.urlopen(args.scene_url, timeout=30) as r:
            return json.loads(r.read())
    raise SystemExit("provide --scene-json or --scene-url")


def _aabb(scene: dict):
    """AABB over all vertices -> (min[3], max[3]) or (None, None) if empty."""
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    seen = False
    for m in scene.get("meshes", []):
        for v in m.get("vertices", []):
            seen = True
            for k in range(3):
                if v[k] < lo[k]:
                    lo[k] = v[k]
                if v[k] > hi[k]:
                    hi[k] = v[k]
    if not seen:
        return None, None
    return lo, hi


def _stats(scene: dict) -> dict:
    """Per-kind mesh / vertex / face counts + overall AABB size in metres."""
    meshes = scene.get("meshes", [])
    by_kind: dict[str, dict] = {}
    verts = faces = 0
    for m in meshes:
        k = m.get("kind", "?")
        nv = len(m.get("vertices", []))
        nf = len(m.get("faces", []))
        d = by_kind.setdefault(k, {"meshes": 0, "vertices": 0, "faces": 0})
        d["meshes"] += 1
        d["vertices"] += nv
        d["faces"] += nf
        verts += nv
        faces += nf
    lo, hi = _aabb(scene)
    size = [round(hi[i] - lo[i], 2) for i in range(3)] if lo else [0, 0, 0]
    return {"meshes": len(meshes), "by_kind": by_kind, "vertices": verts,
            "faces": faces, "aabb_min": lo, "aabb_max": hi, "size_m": size}


def validate(scene: dict) -> int:
    """No-pxr well-formedness gate: structure, degeneracy, per-kind + AABB stats."""
    assert "origin_lat" in scene and "origin_lng" in scene, "missing origin"
    assert scene.get("up_axis", "Z") == "Z", "expected Z-up scene"

    meshes = scene.get("meshes", [])
    assert meshes, "scene has no meshes"
    for i, m in enumerate(meshes):
        name = m.get("name", f"#{i}")
        vs = m.get("vertices", [])
        fs = m.get("faces", [])
        assert len(vs) >= 3, f"degenerate mesh {name!r}: <3 vertices ({len(vs)})"
        assert len(fs) >= 1, f"degenerate mesh {name!r}: no faces"
        for f in fs:
            assert len(f) == 3, f"mesh {name!r}: face is not a triangle: {f}"
            a, b, c = f
            assert 0 <= a < len(vs) and 0 <= b < len(vs) and 0 <= c < len(vs), \
                f"mesh {name!r}: face index out of range {f} (nverts={len(vs)})"

    st = _stats(scene)
    print(f"SCENE VALID ao={scene.get('ao')} "
          f"origin=({scene['origin_lat']:.5f},{scene['origin_lng']:.5f})")
    print(f"  meshes={st['meshes']} verts={st['vertices']} faces={st['faces']}")
    for k in sorted(st["by_kind"]):
        d = st["by_kind"][k]
        print(f"  kind {k:<9} meshes={d['meshes']:>5} "
              f"verts={d['vertices']:>7} faces={d['faces']:>7}")
    sx, sy, sz = st["size_m"]
    print(f"  AABB size = {sx} x {sy} x {sz} m (X-east, Y-north, Z-up)")
    print("  degeneracy: OK (every mesh >=3 verts, >=1 tri, all indices in range)")
    return 0


def emit_obj(scene: dict, out_path: str) -> int:
    lines = [f"# Tritium AO {scene.get('ao')} scene"]
    offset = 1
    for m in scene.get("meshes", []):
        lines.append(f"g {m.get('kind')}_{m.get('name','')}".replace(" ", "_"))
        vs = m.get("vertices", [])
        for v in vs:
            lines.append(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}")
        for f in m.get("faces", []):
            lines.append(f"f {f[0]+offset} {f[1]+offset} {f[2]+offset}")
        offset += len(vs)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote OBJ {out_path} ({_stats(scene)['by_kind']})")
    return 0


# --- No-GPU matplotlib preview -------------------------------------------------
def render_preview(scene: dict, out_path: str, elev: float = 32.0,
                   azim: float = -58.0) -> int:
    """Render the Scene3D to a PNG with matplotlib (Agg) — no pxr, no GPU.

    Colors meshes by kind, applies cheap normal-based shading so buildings read
    in 3D, and views from an oblique angle so the twin can be eyeballed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    meshes = scene.get("meshes", [])
    light = np.array([-0.3, -0.45, 0.84])
    light /= np.linalg.norm(light)

    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo
    order = ["terrain", "water", "road", "building"]
    kinds_present = [k for k in order if any(m.get("kind") == k for m in meshes)]
    kinds_present += sorted({m.get("kind", "other") for m in meshes}
                            - set(order))

    fig = plt.figure(figsize=(14, 10), facecolor="#0a0e14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0e14")

    # ONE collection for the whole scene: matplotlib depth-sorts polygons within
    # a single Poly3DCollection (painter's algorithm), which correctly interleaves
    # buildings/roads/water on the terrain.  Separate per-kind collections would
    # composite by a single per-collection depth and a large flat terrain quad
    # would paint over everything.  Per-face colors + edges keep the kind read.
    tris: list = []
    cols: list = []
    edges: list = []
    lws: list = []
    for m in meshes:
        V = np.asarray(m.get("vertices", []), dtype=float)
        if V.size == 0:
            continue
        lo = np.minimum(lo, V.min(axis=0))
        hi = np.maximum(hi, V.max(axis=0))
        base = np.array(mesh_color(m))
        kind = m.get("kind", "other")
        edge_rgb = KIND_EDGE.get(kind)
        edge = (edge_rgb[0], edge_rgb[1], edge_rgb[2], 0.9) if edge_rgb \
            else (0.0, 0.0, 0.0, 0.0)
        lw = 0.15 if edge_rgb else 0.0
        for f in m.get("faces", []):
            tri = V[[f[0], f[1], f[2]]]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            ln = float(np.linalg.norm(n))
            shade = 0.6 if ln < 1e-9 else 0.4 + 0.6 * abs(float(np.dot(n / ln, light)))
            tris.append(tri)
            cols.append(np.clip(base * shade, 0.0, 1.0))
            edges.append(edge)
            lws.append(lw)

    if not tris or not np.isfinite(lo).all():
        raise SystemExit("preview: scene has no drawable geometry")

    total_tris = len(tris)
    pc = Poly3DCollection(tris, facecolors=cols, edgecolors=edges,
                          linewidths=lws)
    pc.set_zsort("average")
    ax.add_collection3d(pc)

    size = hi - lo
    # Modest vertical exaggeration so ~10-100 m relief reads over a ~km AO.
    zexag = 1.0
    if size[2] > 1e-6:
        zexag = min(8.0, max(1.0, 0.12 * max(size[0], size[1]) / size[2]))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((max(size[0], 1.0), max(size[1], 1.0),
                       max(size[2] * zexag, max(size[0], size[1]) * 0.02)))
    ax.view_init(elev=elev, azim=azim)

    # Dark cyberpunk styling.
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.04, 0.06, 0.09, 1.0))
        pane.line.set_color((0.3, 0.35, 0.4, 1.0))
    ax.tick_params(colors="#8aa0b0", labelsize=8)
    ax.set_xlabel("East (m)", color="#8aa0b0", fontsize=9)
    ax.set_ylabel("North (m)", color="#8aa0b0", fontsize=9)
    ax.set_zlabel("Up (m)", color="#8aa0b0", fontsize=9)

    ao = scene.get("ao", "scene")
    ax.set_title(
        f"Isaac digital-twin preview — {ao}  ·  {len(meshes)} meshes / "
        f"{total_tris} tris  ·  vert-exag ×{zexag:.1f}\n"
        f"AABB {size[0]:.0f} × {size[1]:.0f} × {size[2]:.0f} m  "
        f"(no-GPU · matplotlib Agg)",
        color="#00f0ff", fontsize=12)

    legend = [Patch(facecolor=tuple(kind_params(k)["color"]),
                    edgecolor=(tuple(KIND_EDGE[k]) if k in KIND_EDGE else "none"),
                    label=k) for k in kinds_present]
    leg = ax.legend(handles=legend, loc="upper left", framealpha=0.2,
                    facecolor="#0a0e14", labelcolor="#c8d6e0", fontsize=9)
    leg.get_frame().set_edgecolor("#00f0ff")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote preview {out_path} "
          f"({len(meshes)} meshes, {total_tris} tris, kinds={kinds_present})")
    return 0


# --- USD writer (Isaac-side, pxr only) ----------------------------------------
def write_usd(scene: dict, out_path: str) -> int:
    """Materialize the Scene3D as a USD stage (needs pxr / Isaac's python)."""
    try:
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt  # type: ignore
    except Exception as exc:  # pragma: no cover - only in Isaac's python
        raise SystemExit(
            f"pxr not importable ({exc}); run under Isaac's python.sh, or use "
            "--validate / --preview / --obj under plain python3"
        )

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")

    # Shared per-(kind,color) UsdPreviewSurface materials — one per kind for the
    # common (no custom color) case; a handful more only when a mesh overrides.
    mat_cache: dict = {}

    def get_material(kind: str, color: tuple):
        params = kind_params(kind)
        key = (kind, tuple(round(c, 4) for c in color))
        if key in mat_cache:
            return mat_cache[key]
        safe_kind = "".join(ch if ch.isalnum() else "_" for ch in kind) or "other"
        name = f"mat_{safe_kind}_{len(mat_cache)}"
        mpath = f"/World/Looks/{name}"
        material = UsdShade.Material.Define(stage, mpath)
        shader = UsdShade.Shader.Define(stage, f"{mpath}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(params["roughness"]))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            float(params["metallic"]))
        # Modern connection form (avoids the deprecated 2-arg ConnectToSource).
        surface_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface_out)
        mat_cache[key] = material
        return material

    scopes: set = set()
    counts: dict[str, int] = {}
    for i, m in enumerate(scene.get("meshes", [])):
        kind = m.get("kind", "other")
        counts[kind] = counts.get(kind, 0) + 1
        group = f"/World/{kind}s"
        if group not in scopes:
            UsdGeom.Scope.Define(stage, group)
            scopes.add(group)
        safe = "".join(ch if ch.isalnum() else "_" for ch in
                       f"{kind}_{m.get('name', i)}") or f"mesh_{i}"
        path = f"{group}/{safe}_{i}"
        mesh = UsdGeom.Mesh.Define(stage, path)

        verts = m.get("vertices", [])
        pts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts]
        faces = m.get("faces", [])
        mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        idx = []
        for f in faces:
            idx.extend([int(f[0]), int(f[1]), int(f[2])])
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))

        # Flat shading + clean load: no subdivision, authored extent.
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
            mesh.CreateExtentAttr(Vt.Vec3fArray([
                Gf.Vec3f(min(xs), min(ys), min(zs)),
                Gf.Vec3f(max(xs), max(ys), max(zs))]))

        color = mesh_color(m)
        # displayColor fallback for viewers that don't evaluate materials.
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([
            Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]))
        # Bind the per-kind PBR material for the RTX render.
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(get_material(kind, color))

    stage.GetRootLayer().Save()
    print(f"wrote USD {out_path} kinds={counts} materials={len(mat_cache)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scene3D -> USD for Isaac Sim")
    ap.add_argument("--scene-json", default="")
    ap.add_argument("--scene-url", default="")
    ap.add_argument("--out", default="", help="output .usd path")
    ap.add_argument("--obj", default="", help="also/instead emit an OBJ here")
    ap.add_argument("--preview", default="", help="render a no-GPU PNG preview here")
    ap.add_argument("--validate", action="store_true",
                    help="no-pxr well-formedness check + stats")
    args = ap.parse_args(argv)

    scene = load_scene(args)

    rc = 0
    other_action = args.validate or bool(args.preview) or bool(args.obj)
    if args.validate:
        rc = validate(scene)
    if args.preview:
        render_preview(scene, args.preview)
    if args.obj:
        emit_obj(scene, args.obj)

    # Write USD only when explicitly asked (--out) or when nothing else was
    # requested (the default action).  Keeps --validate/--preview no-GPU.
    if args.out:
        rc = write_usd(scene, args.out)
    elif not other_action:
        rc = write_usd(scene, f"{scene.get('ao', 'scene')}.usd")
    return rc


if __name__ == "__main__":
    sys.exit(main())
