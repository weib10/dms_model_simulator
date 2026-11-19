"""Car body (configs/body.yaml) as triangles: one geometry for the simulation and the 3D page.

The simulation hides whatever a part stands in front of (world.Cabin builds one "mesh" occluder from every part);
dms-sim view3d draws the same triangles. Vehicle frame, metres; each builder returns a list of (p0, p1, p2),
counter-clockwise seen from the front.
"""
from __future__ import annotations
import numpy as np
from . import geometry as G

SEG = 16                # sides of a cylinder


def _tri_facing(p0, p1, p2, toward):
    n = np.cross(p1 - p0, p2 - p0)
    return (p0, p1, p2) if np.dot(n, toward - (p0 + p1 + p2) / 3) >= 0 else (p0, p2, p1)


def quad(points, toward) -> list:
    """Two triangles of a planar quad, front side towards the point `toward` (None: as given)."""
    p = [np.asarray(q, float) for q in points]
    tris = [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
    return tris if toward is None else [_tri_facing(*t, np.asarray(toward, float)) for t in tris]


def rot_y(deg: float) -> np.ndarray:
    """About Y; deg > 0 tips +Z towards -X (a seat back leaning backwards)."""
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])


def box(center, size, R=None) -> list:
    """Closed box, faces outwards. R: rotation (columns = box axes in the vehicle frame)."""
    c, h = np.asarray(center, float), 0.5 * np.asarray(size, float)
    R = np.eye(3) if R is None else np.asarray(R, float)
    tris = []
    for ax in range(3):
        for s in (-1, 1):
            u, v = [i for i in range(3) if i != ax]
            corners = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                q = np.zeros(3); q[ax] = s * h[ax]; q[u] = a * h[u]; q[v] = b * h[v]
                corners.append(c + R @ q)
            fc = c + R @ (np.eye(3)[ax] * s * h[ax])
            tris += quad(corners, 2 * fc - c)
    return tris


def frame(axis):
    """(unit axis, u, v): a right-handed frame around `axis`."""
    a = G.normalize(np.asarray(axis, float))
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = G.normalize(np.cross(a, ref))
    return a, u, np.cross(a, u)


def cylinder(a, b, radius, caps=True) -> list:
    a, b = np.asarray(a, float), np.asarray(b, float)
    _, u, v = frame(b - a)
    ang = np.linspace(0, 2 * np.pi, SEG + 1)
    ring = [np.cos(t) * u + np.sin(t) * v for t in ang]
    tris = []
    for i in range(SEG):
        tris += quad([a + radius * ring[i], a + radius * ring[i + 1], b + radius * ring[i + 1], b + radius * ring[i]],
                     (a + b) / 2 + 3 * radius * (ring[i] + ring[i + 1]))
    if caps:
        for c, out in ((a, 2 * a - b), (b, 2 * b - a)):
            tris += [_tri_facing(c, c + radius * ring[i], c + radius * ring[i + 1], out) for i in range(SEG)]
    return tris


def panel(center, eye, w, h, lift=0.0) -> list:
    """Flat w x h rectangle at `center` facing `eye` (width horizontal), moved `lift` towards the eye."""
    n = G.normalize(np.asarray(eye, float) - np.asarray(center, float))
    wa = G.normalize(np.cross([0.0, 0.0, 1.0], n)) if abs(n[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    ha = np.cross(n, wa)
    c = np.asarray(center, float) + lift * n
    return quad([c + sw * w / 2 * wa + sh * h / 2 * ha for sw, sh in ((-1, -1), (1, -1), (1, 1), (-1, 1))], c + n)


def parts(body: dict, targets: dict, eye) -> list[dict]:
    """Every configs/body.yaml part as {name, layer, color, alpha, two_sided, closed, tris}. closed: a solid whose
    triangles face outwards (box, cylinder); the rest are open surfaces (quads, panels). Several pieces of one part
    (a mirror's housing, glass and arm) come as several entries with the same name. Panels and housings sit at
    the gaze targets (`targets`, facing the eye point `eye`), so they move with configs/cabin.yaml."""
    toward = np.asarray(body.get("toward", [0.0, 0.0, 0.0]), float)
    eye = np.asarray(eye, float)
    out = []

    def add(p, tris, color=None, alpha=None, two_sided=False, closed=True):
        out.append({"name": p["name"], "layer": p.get("layer", "body"), "color": color or p.get("color", "trim"),
                    "alpha": float(p.get("alpha", 1.0) if alpha is None else alpha), "two_sided": two_sided,
                    "closed": closed, "tris": tris})

    for p in body.get("parts", []):
        t = p["type"]
        if t == "quad":
            shell = p.get("shell", True)
            add(p, quad(p["points"], toward if shell else None), two_sided=not shell, closed=False)
        elif t == "box":
            add(p, box(p["center"], p["size"], rot_y(float(p.get("pitch_deg", 0.0)))))
        elif t == "cylinder":
            add(p, cylinder(p["a"], p["b"], float(p["radius"])))
        elif t == "panel":
            w, h = p["size"]
            tg = np.asarray(targets[p["target"]], float)
            add(p, panel(tg, eye, w, h, lift=0.003), two_sided=True, closed=False)
            if p.get("back_x") is not None and float(p["back_x"]) > tg[0]:
                # housing behind the screen, tilted with it, reaching back to the dash face (x = back_x)
                n = G.normalize(eye - tg)
                wa = G.normalize(np.cross([0.0, 0.0, 1.0], n))
                depth = (float(p["back_x"]) - tg[0]) / max(-n[0], 0.2)
                add(p, box(tg - n * depth / 2, [depth, w, h], np.column_stack([n, wa, np.cross(n, wa)])),
                    color=body.get("housing_color", "dash"), alpha=1.0)
        elif t == "housing":
            tg = np.asarray(targets[p["target"]], float)
            c = tg + np.asarray(p.get("offset", [0.0, 0.0, 0.0]), float)
            sz = p["size"]
            add(p, box(c, sz))
            if p.get("glass"):
                add(p, panel(tg, eye, 0.85 * sz[1], 0.75 * sz[2], lift=0.004), color=p["glass"], alpha=1.0,
                    two_sided=True, closed=False)
            if p.get("arm_to") is not None:
                add(p, cylinder(c, p["arm_to"], 0.012))
        else:
            raise ValueError(f"body part {p.get('name')!r}: unknown type {t!r} (quad | box | cylinder | panel | housing)")
    return out


def occluder(body: dict, targets: dict, eye) -> dict | None:
    """All parts as one "mesh" occluder: tris (N x 3 x 3), the part name of each triangle and two_sided: true
    for open surfaces (quads, panels), which a ray may meet from either side; a closed part is entered through a
    triangle facing the camera. None: no parts."""
    ps = [p for p in parts(body, targets, eye) if p["tris"]]
    if not ps:
        return None
    tris = np.array([t for p in ps for t in p["tris"]], float)
    names = np.array([p["name"] for p in ps for _ in p["tris"]])
    two = np.array([p["two_sided"] or not p["closed"] for p in ps for _ in p["tris"]], bool)
    return {"name": "car_body", "type": "mesh", "tris": tris, "part": names, "two_sided": two}
